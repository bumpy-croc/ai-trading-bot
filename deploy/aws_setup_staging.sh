#!/bin/bash
# AWS EC2 Setup Script for Crypto Trading Bot - Staging Environment
# Integrates with AWS Secrets Manager for secure credential storage
# Run this on a fresh Ubuntu 22.04 EC2 instance

set -e

echo "🚀 Setting up Crypto Trading Bot on AWS EC2 (Staging)..."

# Check if instance has IAM role attached
echo "🔐 Checking IAM role..."
if ! aws sts get-caller-identity &> /dev/null; then
    echo "❌ Error: This instance needs an IAM role with Secrets Manager permissions"
    echo "Please attach an IAM role with the following policies:"
    echo "- SecretsManagerReadWrite (for specific secret)"
    echo "- CloudWatchAgentServerPolicy"
    exit 1
fi

# Update system
echo "📦 Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install Python 3.11 and dependencies
echo "🐍 Installing Python 3.11..."
sudo apt install -y python3.11 python3.11-venv python3-pip
sudo apt install -y build-essential libssl-dev libffi-dev python3-dev

# Install PostgreSQL (optional - remove if using SQLite)
echo "🐘 Installing PostgreSQL..."
sudo apt install -y postgresql postgresql-contrib
sudo systemctl start postgresql
sudo systemctl enable postgresql

# Install AWS CLI if not already installed
echo "☁️ Installing AWS CLI..."
if ! command -v aws &> /dev/null; then
    curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
    unzip awscliv2.zip
    sudo ./aws/install
    rm -rf awscliv2.zip aws/
fi

# Create app directory
echo "📁 Creating application directory..."
sudo mkdir -p /opt/ai-trader
sudo chown $USER:$USER /opt/ai-trader
cd /opt/ai-trader

# Clone your repository (replace with your repo URL)
echo "📥 Cloning repository..."
# git clone https://github.com/yourusername/ai-trader.git .
# OR upload your code via SCP/SFTP

# Create virtual environment
echo "🔧 Setting up Python virtual environment..."
python3.11 -m venv venv
source venv/bin/activate

# Install requirements
echo "📚 Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Install boto3 for AWS integration
pip install boto3

# Create data directory if it doesn't exist
mkdir -p data

# Create test script to verify secrets access
echo "🔑 Creating AWS Secrets Manager test script..."
cat > scripts/test_secrets_access.py << 'EOF'
#!/usr/bin/env python3
"""
Test that the configuration system can access secrets from AWS Secrets Manager
"""
import sys
sys.path.insert(0, '/opt/ai-trader')

from core.config import get_config

def main():
    """Test configuration access"""
    print("🔐 Testing configuration access...")
    
    config = get_config()
    
    # Test required configurations
    required_configs = [
        'BINANCE_API_KEY',
        'BINANCE_API_SECRET',
        'DATABASE_URL',
        'TRADING_MODE',
        'INITIAL_BALANCE'
    ]
    
    all_found = True
    for key in required_configs:
        try:
            value = config.get_required(key)
            # Don't print sensitive values
            if 'SECRET' in key or 'PASSWORD' in key:
                print(f"✅ {key}: ***")
            else:
                print(f"✅ {key}: {value}")
        except ValueError:
            print(f"❌ {key}: NOT FOUND")
            all_found = False
    
    if all_found:
        print("\n✅ All required configurations are accessible!")
        return 0
    else:
        print("\n❌ Some configurations are missing!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
EOF

chmod +x scripts/test_secrets_access.py

# Test configuration access
echo "🔐 Testing configuration access..."
ENVIRONMENT=staging python scripts/test_secrets_access.py

# Create systemd service for the trading bot
echo "⚙️ Creating systemd service..."
sudo tee /etc/systemd/system/ai-trader.service > /dev/null << EOF
[Unit]
Description=Crypto Trading Bot (Staging)
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=/opt/ai-trader
Environment="PATH=/opt/ai-trader/venv/bin"
Environment="ENVIRONMENT=staging"

# Test configuration access before starting
ExecStartPre=/opt/ai-trader/venv/bin/python /opt/ai-trader/scripts/test_secrets_access.py

# Start the bot with staging configuration
ExecStart=/opt/ai-trader/venv/bin/python run_live_trading.py adaptive --paper-trading

# Restart configuration
Restart=always
RestartSec=10

# Security settings
PrivateTmp=true
NoNewPrivileges=true
ReadOnlyPaths=/
ReadWritePaths=/opt/ai-trader/data /opt/ai-trader/logs

[Install]
WantedBy=multi-user.target
EOF

# Create log rotation config
echo "📝 Setting up log rotation..."
sudo tee /etc/logrotate.d/ai-trader > /dev/null << EOF
/opt/ai-trader/*.log {
    daily
    rotate 14
    compress
    delaycompress
    missingok
    notifempty
    create 0644 $USER $USER
}
EOF

# Setup cron jobs for maintenance tasks
echo "⏰ Setting up cron jobs..."
(crontab -l 2>/dev/null; echo "0 2 * * * cd /opt/ai-trader && /opt/ai-trader/venv/bin/python scripts/download_binance_data.py") | crontab -
(crontab -l 2>/dev/null; echo "0 */6 * * * cd /opt/ai-trader && /opt/ai-trader/venv/bin/python scripts/cache_manager.py --refresh") | crontab -

# Setup CloudWatch monitoring (optional)
echo "📊 Setting up CloudWatch agent..."
wget https://s3.amazonaws.com/amazoncloudwatch-agent/ubuntu/amd64/latest/amazon-cloudwatch-agent.deb
sudo dpkg -i -E ./amazon-cloudwatch-agent.deb
rm amazon-cloudwatch-agent.deb

# Enable and start the service
echo "🎯 Starting the trading bot service..."
sudo systemctl daemon-reload
sudo systemctl enable ai-trader.service
# sudo systemctl start ai-trader.service  # Uncomment after configuring secrets

echo "✅ Setup complete!"
echo ""
echo "⚠️  IMPORTANT: Before starting the bot, you need to:"
echo ""
echo "1. Create a secret in AWS Secrets Manager:"
echo "   Secret name: ai-trader/staging"
echo ""
echo "2. Run the following AWS CLI command to create the secret:"
echo "   aws secretsmanager create-secret \\"
echo "     --name ai-trader/staging \\"
echo "     --description \"Staging environment secrets for AI Trading Bot\" \\"
echo "     --secret-string '{"
echo "       \"BINANCE_API_KEY\": \"your-staging-api-key\","
echo "       \"BINANCE_API_SECRET\": \"your-staging-api-secret\","
echo "       \"DATABASE_URL\": \"sqlite:///data/trading_bot.db\","
echo "       \"TRADING_MODE\": \"paper\","
echo "       \"INITIAL_BALANCE\": \"1000\","
echo "       \"LOG_LEVEL\": \"INFO\""
echo "     }'"
echo ""
echo "3. Ensure this instance has an IAM role with permission to read the secret"
echo ""
echo "4. Initialize data:"
echo "   cd /opt/ai-trader && ./venv/bin/python scripts/download_binance_data.py"
echo ""
echo "5. Start the bot:"
echo "   sudo systemctl start ai-trader"
echo ""
echo "Useful commands:"
echo "- Check status: sudo systemctl status ai-trader"
echo "- View logs: sudo journalctl -u ai-trader -f"
echo "- Update secrets: aws secretsmanager update-secret --secret-id ai-trader/staging"
echo "- Test config access: cd /opt/ai-trader && ENVIRONMENT=staging ./venv/bin/python scripts/test_secrets_access.py"
echo "- Restart bot: sudo systemctl restart ai-trader"
echo "- Stop bot: sudo systemctl stop ai-trader"
echo ""
echo "Note: The bot now reads secrets directly from AWS Secrets Manager."
echo "No .env file is created - this is more secure!" 