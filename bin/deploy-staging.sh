#!/bin/bash
set -e

# Function to log with timestamps
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

# Function to handle errors
handle_error() {
    log "❌ Error occurred at line $1"
    log "📋 Last few lines of system log:"
    sudo tail -20 /var/log/syslog 2>/dev/null || log "ℹ️ Could not access system log"
    log "📋 Current user and permissions:"
    whoami
    id
    log "📋 Available disk space:"
    df -h
    log "📋 Recent deployment logs:"
    sudo journalctl -u ai-trading-bot -n 10 --no-pager 2>/dev/null || true
    exit 1
}

cleanup_on_exit() {
    # Clean up temporary files on exit
    sudo rm -rf /tmp/ai-trading-bot-new /tmp/ai-trading-bot-preserve 2>/dev/null || true
    rm -f /tmp/ai-trading-bot-*.tar.gz 2>/dev/null || true
}

trap 'handle_error $LINENO' ERR
trap 'cleanup_on_exit' EXIT

# Get parameters
COMMIT_SHA="$1"
S3_BUCKET="$2"

if [ -z "$COMMIT_SHA" ] || [ -z "$S3_BUCKET" ]; then
    log "❌ Usage: $0 <commit_sha> <s3_bucket>"
    exit 1
fi

log "🚀 Starting deployment to staging..."
log "📋 Commit: $COMMIT_SHA"
log "📋 S3 Bucket: $S3_BUCKET"

# Check prerequisites
log "🔍 Checking prerequisites..."

# Update package list and install required packages (suppress verbose output)
log "📦 Installing system dependencies..."
sudo apt-get update -qq 2>/dev/null || log "⚠️ APT update had warnings (continuing)"
sudo apt-get install -y python3-venv python3-pip unzip curl >/dev/null 2>&1 || {
    log "❌ Failed to install system dependencies"
    exit 1
}

# Check if AWS CLI is available
if ! command -v aws &> /dev/null; then
    log "❌ AWS CLI not found, installing..."
    curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
    unzip awscliv2.zip
    sudo ./aws/install || sudo ./aws/install --update
    rm -rf aws awscliv2.zip
fi

# Check AWS credentials
log "🔐 Testing AWS credentials..."
if aws sts get-caller-identity >/dev/null 2>&1; then
    log "✅ AWS credentials configured"
else
    log "❌ AWS credentials not configured properly"
    aws sts get-caller-identity || true  # Show error for debugging
    exit 1
fi

# Check if directories exist and create if needed
log "📁 Setting up directories..."
sudo mkdir -p /opt/ai-trading-bot
sudo mkdir -p /opt/ai-trading-bot/data
sudo mkdir -p /opt/ai-trading-bot/logs

# Download deployment package from S3
log "📥 Downloading deployment package..."
cd /tmp
aws s3 cp s3://${S3_BUCKET}/deployments/staging/ai-trading-bot-${COMMIT_SHA}.tar.gz . || {
    log "❌ Failed to download deployment package from S3"
    log "📋 S3 bucket contents:"
    aws s3 ls s3://${S3_BUCKET}/deployments/staging/ || true
    exit 1
}

# Stop the current service
log "⏹️ Stopping current service..."
sudo systemctl stop ai-trading-bot || {
    log "⚠️ Service was not running or failed to stop"
}

# Skip backup for staging (saves disk space)
log "ℹ️ Skipping backup for staging deployment (saves disk space)"

# Extract new deployment
log "📦 Extracting new deployment..."
sudo rm -rf /tmp/ai-trading-bot-new
mkdir -p /tmp/ai-trading-bot-new
tar -xzf ai-trading-bot-${COMMIT_SHA}.tar.gz -C /tmp/ai-trading-bot-new || {
    log "❌ Failed to extract deployment package"
    exit 1
}

# Preserve data and logs
log "🔄 Preserving data and logs..."
sudo mkdir -p /tmp/ai-trading-bot-preserve
sudo cp -r /opt/ai-trading-bot/data /tmp/ai-trading-bot-preserve/ 2>/dev/null || log "ℹ️ No data directory to preserve"
sudo cp -r /opt/ai-trading-bot/logs /tmp/ai-trading-bot-preserve/ 2>/dev/null || log "ℹ️ No logs directory to preserve"
sudo cp /opt/ai-trading-bot/.env /tmp/ai-trading-bot-preserve/ 2>/dev/null || log "ℹ️ No .env file to preserve"

# Replace application files
log "🔄 Updating application files..."
sudo rm -rf /opt/ai-trading-bot/*
sudo cp -r /tmp/ai-trading-bot-new/* /opt/ai-trading-bot/

# Restore preserved data
sudo cp -r /tmp/ai-trading-bot-preserve/data /opt/ai-trading-bot/ 2>/dev/null || log "ℹ️ No data to restore"
sudo cp -r /tmp/ai-trading-bot-preserve/logs /opt/ai-trading-bot/ 2>/dev/null || log "ℹ️ No logs to restore"
sudo cp /tmp/ai-trading-bot-preserve/.env /opt/ai-trading-bot/ 2>/dev/null || log "ℹ️ No .env to restore"

# Set permissions
log "🔧 Setting permissions..."
sudo chown -R ubuntu:ubuntu /opt/ai-trading-bot

# Set up Python virtual environment if it doesn't exist
log "🐍 Setting up Python environment..."
cd /opt/ai-trading-bot

if [ ! -d "venv" ]; then
    log "📚 Creating new virtual environment..."
    sudo -u ubuntu python3 -m venv venv || {
        log "❌ Failed to create virtual environment, trying alternative method..."
        sudo -u ubuntu python3 -m pip install --user virtualenv
        sudo -u ubuntu python3 -m virtualenv venv
    }
else
    log "ℹ️ Virtual environment already exists"
fi

# Ensure venv activation works
if [ ! -f "venv/bin/activate" ]; then
    log "❌ Virtual environment is corrupted, recreating..."
    sudo rm -rf venv
    sudo -u ubuntu python3 -m venv venv
fi

# Update Python dependencies (suppress verbose output)
log "📚 Updating dependencies..."
sudo -u ubuntu ./venv/bin/pip install --upgrade pip --quiet || {
    log "❌ Failed to upgrade pip, trying with python -m pip..."
    sudo -u ubuntu ./venv/bin/python -m pip install --upgrade pip --quiet
}

sudo -u ubuntu ./venv/bin/pip install -r requirements.txt --quiet || {
    log "❌ Failed to install Python dependencies with pip, trying with python -m pip..."
    sudo -u ubuntu ./venv/bin/python -m pip install -r requirements.txt --quiet || {
        log "❌ Failed to install Python dependencies"
        log "📋 Requirements file content:"
        head -20 requirements.txt || true
        log "📋 Python version:"
        sudo -u ubuntu ./venv/bin/python --version || true
        log "📋 Pip version:"
        sudo -u ubuntu ./venv/bin/pip --version || true
        exit 1
    }
}

# Create systemd service file
log "⚙️ Setting up systemd service..."
sudo tee /etc/systemd/system/ai-trading-bot.service > /dev/null << 'EOF'
[Unit]
Description=AI Trading Bot (STAGING)
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/opt/ai-trading-bot
Environment="PATH=/opt/ai-trading-bot/venv/bin"
Environment="ENVIRONMENT=staging"

# Test configuration before starting
ExecStartPre=/opt/ai-trading-bot/venv/bin/python /opt/ai-trading-bot/scripts/test_secrets_access.py

# Start with a simple strategy for staging
ExecStart=/opt/ai-trading-bot/venv/bin/python run_live_trading.py ml_basic

# Restart configuration
Restart=on-failure
RestartSec=10
StartLimitBurst=5
StartLimitInterval=300

# Logging
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

# Test configuration access (suppress verbose output in deployment)
log "🔐 Testing configuration access..."
if sudo -u ubuntu ENVIRONMENT=staging ./venv/bin/python scripts/test_secrets_access.py >/dev/null 2>&1; then
    log "✅ Configuration access test passed"
else
    log "❌ Configuration access test failed, running with verbose output for debugging..."
    sudo -u ubuntu ENVIRONMENT=staging ./venv/bin/python scripts/test_secrets_access.py || {
        log "📋 Available environment variables:"
        env | grep -E "(AWS|ENVIRONMENT)" || true
        log "📋 Script exists check:"
        ls -la scripts/test_secrets_access.py || true
        exit 1
    }
fi

# Update data cache (non-critical)
log "📊 Updating data cache..."
sudo -u ubuntu ./venv/bin/python scripts/download_binance_data.py BTCUSDT >/dev/null 2>&1 || {
    log "⚠️ Data cache update failed, but continuing..."
}

# Restart service
log "🎯 Starting service..."
sudo systemctl daemon-reload
sudo systemctl enable ai-trading-bot
sudo systemctl start ai-trading-bot

# Wait for service to be ready
log "⏳ Waiting for service to start..."
for i in {1..15}; do
    if sudo systemctl is-active --quiet ai-trading-bot; then
        log "✅ Service started after ${i} seconds"
        break
    fi
    if [ $i -eq 15 ]; then
        log "⚠️ Service taking longer than expected to start..."
    fi
    sleep 1
done

# Check service status with more detailed output
log "🔍 Checking service status..."
if sudo systemctl is-active --quiet ai-trading-bot; then
    log "✅ Deployment successful! Service is running."
    log "📊 Service status:"
    sudo systemctl status ai-trading-bot --no-pager --lines=5
else
    log "❌ Deployment failed! Service is not running."
    log "📋 Service status:"
    sudo systemctl status ai-trading-bot --no-pager --lines=10 || true
    log "📋 Recent service logs:"
    sudo journalctl -u ai-trading-bot -n 20 --no-pager || true
    log "📋 System logs:"
    sudo journalctl -n 10 --no-pager || true
    exit 1
fi

log "🎉 Staging deployment completed successfully!"
log "🧹 Cleaning up temporary files..."
# Cleanup will happen automatically via EXIT trap 