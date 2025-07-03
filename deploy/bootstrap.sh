#!/bin/bash
# Bootstrap script for AI Trading Bot EC2 instances
# This script sets up the application environment on a fresh EC2 instance
# Usage: curl -sSL https://raw.githubusercontent.com/yourusername/ai-trading-bot/main/deploy/bootstrap.sh | bash -s [staging|production]

set -e

ENVIRONMENT=${1:-staging}
PROJECT_NAME="ai-trading-bot"
APP_DIR="/opt/${PROJECT_NAME}"
REPO_URL="https://github.com/alexflorisca/ai-trading-bot.git"

echo "🚀 Bootstrapping AI Trading Bot for $ENVIRONMENT environment..."

# Check if running as root or with sudo
if [ "$EUID" -eq 0 ]; then
    echo "❌ Don't run this script as root. Run as ubuntu user with sudo access."
    exit 1
fi

# Check if instance has IAM role
if ! aws sts get-caller-identity &> /dev/null; then
    echo "❌ This instance needs an IAM role with proper permissions"
    exit 1
fi

# Update system
echo "📦 Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install dependencies
echo "🔧 Installing dependencies..."
sudo apt install -y \
    python3.11 python3.11-venv python3-pip \
    build-essential libssl-dev libffi-dev python3-dev \
    git curl wget unzip \
    htop vim

# Install AWS CLI v2 if not present
if ! command -v aws &> /dev/null; then
    echo "☁️ Installing AWS CLI v2..."
    curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
    unzip awscliv2.zip
    sudo ./aws/install
    rm -rf awscliv2.zip aws/
fi

# Create application directory
echo "📁 Setting up application directory..."
sudo mkdir -p "$APP_DIR"
sudo chown $USER:$USER "$APP_DIR"

# Clone repository
echo "📥 Cloning repository..."
if [ -d "$APP_DIR/.git" ]; then
    echo "Repository already exists, pulling latest..."
    cd "$APP_DIR"
    git pull
else
    git clone "$REPO_URL" "$APP_DIR"
    cd "$APP_DIR"
fi

# Create Python virtual environment
echo "🐍 Setting up Python environment..."
python3.11 -m venv venv
source venv/bin/activate

# Install Python dependencies
echo "📚 Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create directories
mkdir -p data logs

# Test secrets access
echo "🔐 Testing secrets access..."
ENVIRONMENT=$ENVIRONMENT python scripts/test_secrets_access.py || {
    echo "❌ Cannot access secrets. Please check IAM role and secrets configuration."
    exit 1
}

# Create systemd service
echo "⚙️ Creating systemd service..."
sudo tee /etc/systemd/system/ai-trading-bot.service > /dev/null << EOF
[Unit]
Description=AI Trading Bot ($ENVIRONMENT)
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$APP_DIR
Environment="PATH=$APP_DIR/venv/bin"
Environment="ENVIRONMENT=$ENVIRONMENT"

# Test configuration before starting
ExecStartPre=$APP_DIR/venv/bin/python $APP_DIR/scripts/test_secrets_access.py

# Start the application
ExecStart=$APP_DIR/venv/bin/python scripts/run_live_trading.py adaptive $([ "$ENVIRONMENT" = "production" ] || echo "--paper-trading")

# Restart configuration
Restart=always
RestartSec=10

# Security settings
PrivateTmp=true
NoNewPrivileges=true
ReadOnlyPaths=/
ReadWritePaths=$APP_DIR/data $APP_DIR/logs

[Install]
WantedBy=multi-user.target
EOF

# Set up log rotation
echo "📝 Setting up log rotation..."
sudo tee /etc/logrotate.d/ai-trading-bot > /dev/null << EOF
$APP_DIR/logs/*.log {
    daily
    rotate 14
    compress
    delaycompress
    missingok
    notifempty
    create 0644 $USER $USER
}
EOF

# Set up cron jobs
echo "⏰ Setting up maintenance cron jobs..."
(crontab -l 2>/dev/null; echo "0 2 * * * cd $APP_DIR && ./venv/bin/python scripts/download_binance_data.py") | crontab -
(crontab -l 2>/dev/null; echo "0 */6 * * * cd $APP_DIR && ./venv/bin/python scripts/cache_manager.py --refresh") | crontab -

# Enable service but don't start yet
echo "🎯 Configuring service..."
sudo systemctl daemon-reload
sudo systemctl enable ai-trading-bot.service

echo "✅ Bootstrap complete!"
echo
echo "🔧 Manual steps required:"
echo "1. Update secrets in AWS Secrets Manager with real API keys"
echo "2. Download initial data: cd $APP_DIR && ./venv/bin/python scripts/download_binance_data.py"
echo "3. Start the service: sudo systemctl start ai-trading-bot"
echo
echo "📊 Useful commands:"
echo "- Check status: sudo systemctl status ai-trading-bot"
echo "- View logs: sudo journalctl -u ai-trading-bot -f"
echo "- Test config: cd $APP_DIR && ENVIRONMENT=$ENVIRONMENT ./venv/bin/python scripts/test_secrets_access.py" 