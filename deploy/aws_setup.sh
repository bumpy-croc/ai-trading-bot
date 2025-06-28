#!/bin/bash
# AWS EC2 Setup Script for Crypto Trading Bot
# Run this on a fresh Ubuntu 22.04 EC2 instance

set -e

echo "🚀 Setting up Crypto Trading Bot on AWS EC2..."

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

# Create data directory if it doesn't exist
mkdir -p data

# Setup environment variables
echo "🔐 Setting up environment variables..."
cat > .env << EOF
# Binance API credentials
BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_api_secret_here

# Database (use SQLite for simplicity or PostgreSQL)
DATABASE_URL=sqlite:///data/trading_bot.db
# DATABASE_URL=postgresql://botuser:password@localhost/ai-trader

# Trading settings
TRADING_MODE=paper
INITIAL_BALANCE=1000
EOF

# Create systemd service for the trading bot
echo "⚙️ Creating systemd service..."
sudo tee /etc/systemd/system/ai-trader.service > /dev/null << EOF
[Unit]
Description=Crypto Trading Bot
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=/opt/ai-trader
Environment="PATH=/opt/ai-trader/venv/bin"
ExecStart=/opt/ai-trader/venv/bin/python run_live_trading.py adaptive --paper-trading --balance 1000
Restart=always
RestartSec=10

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
# sudo systemctl start ai-trader.service  # Uncomment after configuring .env

echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit /opt/ai-trader/.env with your API credentials"
echo "2. Run initial data download: cd /opt/ai-trader && ./venv/bin/python scripts/download_binance_data.py"
echo "3. Start the bot: sudo systemctl start ai-trader"
echo "4. Check logs: sudo journalctl -u ai-trader -f"
echo ""
echo "Useful commands:"
echo "- Check status: sudo systemctl status ai-trader"
echo "- View logs: sudo journalctl -u ai-trader -f"
echo "- Restart bot: sudo systemctl restart ai-trader"
echo "- Stop bot: sudo systemctl stop ai-trader" 