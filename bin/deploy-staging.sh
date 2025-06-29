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

log "🚀 Starting deployment to staging..."
log "📋 Commit: $COMMIT_SHA"
log "📋 S3 Bucket: $S3_BUCKET"

# Check prerequisites
log "🔍 Checking prerequisites..."

# Check if AWS CLI is available
if ! command -v aws &> /dev/null; then
    log "❌ AWS CLI not found, installing..."
    curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
    unzip awscliv2.zip
    sudo ./aws/install || sudo ./aws/install --update
fi

# Check AWS credentials
log "🔐 Testing AWS credentials..."
aws sts get-caller-identity || {
    log "❌ AWS credentials not configured properly"
    exit 1
}

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

# Backup current deployment
log "💾 Creating backup..."
if [ -d "/opt/ai-trading-bot" ] && [ "$(ls -A /opt/ai-trading-bot)" ]; then
    sudo tar -czf /opt/ai-trading-bot-backup-$(date +%Y%m%d_%H%M%S).tar.gz -C /opt/ai-trading-bot . || {
        log "⚠️ Backup failed, but continuing..."
    }
else
    log "ℹ️ No existing deployment to backup"
fi

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
mkdir -p /tmp/ai-trading-bot-preserve
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
    sudo -u ubuntu python3 -m venv venv
fi

# Update Python dependencies
log "📚 Updating dependencies..."
sudo -u ubuntu ./venv/bin/pip install --upgrade pip
sudo -u ubuntu ./venv/bin/pip install -r requirements.txt || {
    log "❌ Failed to install Python dependencies"
    log "📋 Requirements file content:"
    head -20 requirements.txt || true
    exit 1
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
ExecStart=/opt/ai-trading-bot/venv/bin/python run_live_trading.py adaptive

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

# Test configuration access
log "🔐 Testing configuration access..."
sudo -u ubuntu ENVIRONMENT=staging ./venv/bin/python scripts/test_secrets_access.py || {
    log "❌ Configuration access test failed"
    log "📋 Available environment variables:"
    env | grep -E "(AWS|ENVIRONMENT)" || true
    log "📋 Script exists check:"
    ls -la scripts/test_secrets_access.py || true
    exit 1
}

# Update data cache (non-critical)
log "📊 Updating data cache..."
sudo -u ubuntu ./venv/bin/python scripts/download_binance_data.py || {
    log "⚠️ Data cache update failed, but continuing..."
}

# Restart service
log "🎯 Starting service..."
sudo systemctl daemon-reload
sudo systemctl enable ai-trading-bot
sudo systemctl start ai-trading-bot

# Wait for service to be ready
log "⏳ Waiting for service to start..."
sleep 15

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

# Cleanup
log "🧹 Cleaning up..."
rm -rf /tmp/ai-trading-bot-new /tmp/ai-trading-bot-preserve
rm /tmp/ai-trading-bot-${COMMIT_SHA}.tar.gz

log "🎉 Staging deployment completed successfully!" 