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
    tail -20 /var/log/syslog || true
    log "📋 Current user and permissions:"
    whoami
    id
    log "📋 Available disk space:"
    df -h
    exit 1
}

trap 'handle_error $LINENO' ERR

# Get parameters
COMMIT_SHA="$1"
S3_BUCKET="$2"

if [ -z "$COMMIT_SHA" ] || [ -z "$S3_BUCKET" ]; then
    log "❌ Usage: $0 <commit_sha> <s3_bucket>"
    exit 1
fi

log "🚀 Starting PRODUCTION deployment..."
log "⚠️  This is a PRODUCTION deployment - proceeding with extra caution"
log "📋 Commit: $COMMIT_SHA"
log "📋 S3 Bucket: $S3_BUCKET"

# Check prerequisites
log "🔍 Checking prerequisites..."

# Update package list and install required packages
log "📦 Installing system dependencies..."
sudo apt-get update -qq
sudo apt-get install -y python3-venv python3-pip unzip curl

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
aws sts get-caller-identity || {
    log "❌ AWS credentials not configured properly"
    exit 1
}

# Download deployment package from S3
log "📥 Downloading deployment package..."
cd /tmp
aws s3 cp s3://${S3_BUCKET}/deployments/production/ai-trading-bot-${COMMIT_SHA}.tar.gz . || {
    log "❌ Failed to download deployment package from S3"
    log "📋 S3 bucket contents:"
    aws s3 ls s3://${S3_BUCKET}/deployments/production/ || true
    exit 1
}

# Stop the current service gracefully
log "⏹️ Gracefully stopping production service..."
sudo systemctl stop ai-trading-bot || {
    log "⚠️ Service was not running or failed to stop"
}
sleep 5  # Give it time to stop gracefully

# Extract new deployment
log "📦 Extracting new deployment..."
sudo rm -rf /tmp/ai-trading-bot-new
mkdir -p /tmp/ai-trading-bot-new
tar -xzf ai-trading-bot-${COMMIT_SHA}.tar.gz -C /tmp/ai-trading-bot-new || {
    log "❌ Failed to extract deployment package"
    exit 1
}

# Preserve production data and logs
log "🔄 Preserving production data and logs..."
mkdir -p /tmp/ai-trading-bot-preserve
sudo cp -r /opt/ai-trading-bot/data /tmp/ai-trading-bot-preserve/ 2>/dev/null || log "ℹ️ No data directory to preserve"
sudo cp -r /opt/ai-trading-bot/logs /tmp/ai-trading-bot-preserve/ 2>/dev/null || log "ℹ️ No logs directory to preserve"

# Replace application files
log "🔄 Updating application files..."
sudo rm -rf /opt/ai-trading-bot/*
sudo cp -r /tmp/ai-trading-bot-new/* /opt/ai-trading-bot/

# Restore preserved data
sudo cp -r /tmp/ai-trading-bot-preserve/data /opt/ai-trading-bot/ 2>/dev/null || log "ℹ️ No data to restore"
sudo cp -r /tmp/ai-trading-bot-preserve/logs /opt/ai-trading-bot/ 2>/dev/null || log "ℹ️ No logs to restore"

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

# Update Python dependencies
log "📚 Updating dependencies..."
sudo -u ubuntu ./venv/bin/pip install --upgrade pip || {
    log "❌ Failed to upgrade pip, trying with python -m pip..."
    sudo -u ubuntu ./venv/bin/python -m pip install --upgrade pip
}

sudo -u ubuntu ./venv/bin/pip install -r requirements.txt || {
    log "❌ Failed to install Python dependencies with pip, trying with python -m pip..."
    sudo -u ubuntu ./venv/bin/python -m pip install -r requirements.txt || {
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

# Update systemd service for production
log "⚙️ Updating systemd service for production..."
sudo tee /etc/systemd/system/ai-trading-bot.service > /dev/null << 'EOF'
[Unit]
Description=AI Trading Bot (PRODUCTION)
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/opt/ai-trading-bot
Environment="PATH=/opt/ai-trading-bot/venv/bin"
Environment="ENVIRONMENT=production"

# Test configuration before starting
ExecStartPre=/opt/ai-trading-bot/venv/bin/python /opt/ai-trading-bot/scripts/test_secrets_access.py

# Start the bot in LIVE TRADING mode
ExecStart=/opt/ai-trading-bot/venv/bin/python run_live_trading.py adaptive

# Restart configuration (more conservative for production)
Restart=on-failure
RestartSec=30
StartLimitBurst=3
StartLimitInterval=300

# Security settings
PrivateTmp=true
NoNewPrivileges=true
ReadOnlyPaths=/
ReadWritePaths=/opt/ai-trading-bot/data /opt/ai-trading-bot/logs

# Logging
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

# Test configuration access (CRITICAL for production)
log "🔐 Testing production configuration access..."
sudo -u ubuntu ENVIRONMENT=production ./venv/bin/python scripts/test_secrets_access.py || {
    log "❌ Production configuration access test failed"
    log "📋 Available environment variables:"
    env | grep -E "(AWS|ENVIRONMENT)" || true
    log "📋 Script exists check:"
    ls -la scripts/test_secrets_access.py || true
    exit 1
}

# Reload and start service
log "🎯 Starting production service..."
sudo systemctl daemon-reload
sudo systemctl enable ai-trading-bot
sudo systemctl start ai-trading-bot

# Wait longer for production service to start
log "⏳ Waiting for production service to start..."
sleep 30

# Verify service is running
log "🔍 Checking service status..."
if sudo systemctl is-active --quiet ai-trading-bot; then
    log "✅ PRODUCTION deployment successful! Service is running."
    log "📊 Service status:"
    sudo systemctl status ai-trading-bot --no-pager --lines=5
else
    log "❌ PRODUCTION deployment failed! Service is not running."
    log "📋 Service status:"
    sudo systemctl status ai-trading-bot --no-pager --lines=10 || true
    log "📋 Recent service logs:"
    sudo journalctl -u ai-trading-bot -n 20 --no-pager || true
    log "📋 System logs:"
    sudo journalctl -n 10 --no-pager || true
    exit 1
fi

# Cleanup
log "🧹 Cleaning up..."
rm -rf /tmp/ai-trading-bot-new /tmp/ai-trading-bot-preserve
rm /tmp/ai-trading-bot-${COMMIT_SHA}.tar.gz

log "🎉 PRODUCTION deployment completed successfully!"
log "⚠️  Monitor the service closely for the next hour" 