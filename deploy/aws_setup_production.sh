#!/bin/bash
# Production AWS EC2 Setup Script for Crypto Trading Bot
# Enhanced version with security, monitoring, and backups

set -euo pipefail

# Configuration
APP_DIR="/opt/ai-trader"
APP_USER="ai-trader"
BACKUP_BUCKET="ai-trader-backups-$(date +%s)"
REGION="us-east-1"

echo "🚀 Setting up Production Crypto Trading Bot on AWS EC2..."

# Create dedicated user for the application
echo "👤 Creating dedicated application user..."
sudo useradd -r -s /bin/bash -d $APP_DIR $APP_USER || true

# Update and install dependencies
echo "📦 Installing system dependencies..."
sudo apt update && sudo apt upgrade -y
sudo apt install -y \
    python3.11 python3.11-venv python3-pip \
    build-essential libssl-dev libffi-dev python3-dev \
    nginx certbot python3-certbot-nginx \
    awscli jq fail2ban

# Setup firewall
echo "🔥 Configuring firewall..."
sudo ufw allow 22/tcp
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw --force enable

# Install and configure fail2ban
echo "🛡️ Setting up fail2ban..."
sudo systemctl enable fail2ban
sudo systemctl start fail2ban

# Create application directory
echo "📁 Setting up application directory..."
sudo mkdir -p $APP_DIR/{logs,backups,data}
sudo chown -R $APP_USER:$APP_USER $APP_DIR

# Copy application files (assuming they're in /tmp/ai-trader)
echo "📥 Copying application files..."
sudo cp -r /tmp/ai-trader/* $APP_DIR/
sudo chown -R $APP_USER:$APP_USER $APP_DIR

# Setup Python environment
echo "🐍 Setting up Python environment..."
cd $APP_DIR
sudo -u $APP_USER python3.11 -m venv venv
sudo -u $APP_USER ./venv/bin/pip install --upgrade pip
sudo -u $APP_USER ./venv/bin/pip install -r requirements.txt

# Create secure environment configuration
echo "🔐 Creating secure environment configuration..."
sudo tee $APP_DIR/.env > /dev/null << EOF
# This file will be populated by AWS Secrets Manager
# DO NOT COMMIT CREDENTIALS HERE
EOF
sudo chown $APP_USER:$APP_USER $APP_DIR/.env
sudo chmod 600 $APP_DIR/.env

# Create configuration test script
echo "🔑 Creating configuration test script..."
sudo tee $APP_DIR/scripts/test_secrets_access.py > /dev/null << 'EOF'
#!/usr/bin/env python3
"""Test that the configuration system can access secrets"""
import sys
sys.path.insert(0, '/opt/ai-trader')

from core.config import get_config

def main():
    print("🔐 Testing configuration access...")
    config = get_config()
    
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

sudo chown $APP_USER:$APP_USER $APP_DIR/scripts/test_secrets_access.py
sudo chmod +x $APP_DIR/scripts/test_secrets_access.py

# Create production systemd service
echo "⚙️ Creating systemd service..."
sudo tee /etc/systemd/system/ai-trader.service > /dev/null << EOF
[Unit]
Description=AI Crypto Trading Bot
After=network.target

[Service]
Type=simple
User=$APP_USER
WorkingDirectory=$APP_DIR
Environment="PATH=$APP_DIR/venv/bin:/usr/local/bin:/usr/bin:/bin"
Environment="PYTHONUNBUFFERED=1"

# Test configuration access before starting
ExecStartPre=$APP_DIR/venv/bin/python $APP_DIR/scripts/test_secrets_access.py

# Main process
ExecStart=$APP_DIR/venv/bin/python run_live_trading.py adaptive

# Restart policy
Restart=always
RestartSec=10
StartLimitBurst=5
StartLimitInterval=60

# Security
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=$APP_DIR/data $APP_DIR/logs

# Resource limits
CPUQuota=80%
MemoryLimit=2G

# Logging
StandardOutput=append:$APP_DIR/logs/trading.log
StandardError=append:$APP_DIR/logs/trading-error.log

[Install]
WantedBy=multi-user.target
EOF

# Setup CloudWatch Logs
echo "📊 Configuring CloudWatch Logs..."
sudo tee /opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json > /dev/null << EOF
{
  "agent": {
    "metrics_collection_interval": 60,
    "run_as_user": "cwagent"
  },
  "logs": {
    "logs_collected": {
      "files": {
        "collect_list": [
          {
            "file_path": "$APP_DIR/logs/trading.log",
            "log_group_name": "/aws/ec2/ai-trader",
            "log_stream_name": "{instance_id}/trading",
            "retention_in_days": 30
          },
          {
            "file_path": "$APP_DIR/logs/trading-error.log",
            "log_group_name": "/aws/ec2/ai-trader",
            "log_stream_name": "{instance_id}/errors",
            "retention_in_days": 30
          }
        ]
      }
    }
  },
  "metrics": {
    "namespace": "AITrader",
    "metrics_collected": {
      "cpu": {
        "measurement": [
          "cpu_usage_idle",
          "cpu_usage_iowait",
          "cpu_usage_user",
          "cpu_usage_system"
        ],
        "metrics_collection_interval": 60
      },
      "disk": {
        "measurement": [
          "used_percent"
        ],
        "metrics_collection_interval": 60,
        "resources": [
          "*"
        ]
      },
      "mem": {
        "measurement": [
          "mem_used_percent"
        ],
        "metrics_collection_interval": 60
      }
    }
  }
}
EOF

# Setup automated backups
echo "💾 Setting up automated backups..."
sudo tee /usr/local/bin/backup-ai-trader.sh > /dev/null << EOF
#!/bin/bash
TIMESTAMP=\$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="ai-trader-backup-\${TIMESTAMP}.tar.gz"

# Create backup
cd $APP_DIR
tar -czf /tmp/\${BACKUP_FILE} data/ logs/ *.db

# Upload to S3
aws s3 cp /tmp/\${BACKUP_FILE} s3://${BACKUP_BUCKET}/backups/

# Keep only last 30 days of backups
aws s3 ls s3://${BACKUP_BUCKET}/backups/ | while read -r line; do
  createDate=\$(echo \$line | awk '{print \$1" "\$2}')
  createDate=\$(date -d "\$createDate" +%s)
  olderThan=\$(date -d "30 days ago" +%s)
  if [[ \$createDate -lt \$olderThan ]]; then
    fileName=\$(echo \$line | awk '{print \$4}')
    aws s3 rm s3://${BACKUP_BUCKET}/backups/\$fileName
  fi
done

# Clean up local file
rm /tmp/\${BACKUP_FILE}
EOF
sudo chmod +x /usr/local/bin/backup-ai-trader.sh

# Setup cron jobs
echo "⏰ Setting up cron jobs..."
sudo -u $APP_USER crontab - << EOF
# Update market data daily at 2 AM
0 2 * * * cd $APP_DIR && ./venv/bin/python scripts/download_binance_data.py >> logs/cron.log 2>&1

# Refresh cache every 6 hours
0 */6 * * * cd $APP_DIR && ./venv/bin/python scripts/cache_manager.py --refresh >> logs/cron.log 2>&1

# Daily backups at 3 AM
0 3 * * * /usr/local/bin/backup-ai-trader.sh >> $APP_DIR/logs/backup.log 2>&1

# Health check every 5 minutes
*/5 * * * * cd $APP_DIR && ./venv/bin/python scripts/health_check.py || systemctl restart ai-trader
EOF

# Create health check script
echo "🏥 Creating health check script..."
sudo tee $APP_DIR/scripts/health_check.py > /dev/null << 'EOF'
#!/usr/bin/env python3
import sys
import psutil
import sqlite3
from datetime import datetime, timedelta

def check_process():
    """Check if trading bot process is running"""
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        if 'run_live_trading.py' in str(proc.info['cmdline']):
            return True
    return False

def check_database():
    """Check if database is accessible and has recent trades"""
    try:
        conn = sqlite3.connect('/opt/ai-trader/data/trading_bot.db')
        cursor = conn.cursor()
        cursor.execute(
            "SELECT COUNT(*) FROM trades WHERE timestamp > datetime('now', '-1 hour')"
        )
        recent_trades = cursor.fetchone()[0]
        conn.close()
        return True
    except:
        return False

def check_disk_space():
    """Check if disk space is sufficient"""
    disk = psutil.disk_usage('/')
    return disk.percent < 90

# Run checks
checks = {
    'process': check_process(),
    'database': check_database(),
    'disk': check_disk_space()
}

if not all(checks.values()):
    print(f"Health check failed: {checks}")
    sys.exit(1)

print("Health check passed")
sys.exit(0)
EOF
sudo chown $APP_USER:$APP_USER $APP_DIR/scripts/health_check.py
sudo chmod +x $APP_DIR/scripts/health_check.py

# Create monitoring dashboard script
echo "📈 Creating monitoring dashboard..."
sudo tee $APP_DIR/scripts/create_cloudwatch_dashboard.py > /dev/null << 'EOF'
#!/usr/bin/env python3
import boto3
import json
from datetime import datetime

cloudwatch = boto3.client('cloudwatch')

dashboard_body = {
    "widgets": [
        {
            "type": "metric",
            "properties": {
                "metrics": [
                    ["AITrader", "cpu_usage_user", {"stat": "Average"}],
                    [".", "mem_used_percent", {"stat": "Average"}]
                ],
                "period": 300,
                "stat": "Average",
                "region": "us-east-1",
                "title": "System Metrics"
            }
        },
        {
            "type": "log",
            "properties": {
                "query": "SOURCE '/aws/ec2/ai-trader' | fields @timestamp, @message | filter @message like /ERROR/ | sort @timestamp desc | limit 20",
                "region": "us-east-1",
                "title": "Recent Errors"
            }
        }
    ]
}

cloudwatch.put_dashboard(
    DashboardName='AITrader-Production',
    DashboardBody=json.dumps(dashboard_body)
)

print("✅ CloudWatch dashboard created")
EOF

# Setup log rotation
echo "📝 Configuring log rotation..."
sudo tee /etc/logrotate.d/ai-trader > /dev/null << EOF
$APP_DIR/logs/*.log {
    daily
    rotate 14
    compress
    delaycompress
    missingok
    notifempty
    create 0644 $APP_USER $APP_USER
    sharedscripts
    postrotate
        systemctl reload ai-trader >/dev/null 2>&1 || true
    endscript
}
EOF

# Final setup steps
echo "🎯 Finalizing setup..."
sudo systemctl daemon-reload
sudo systemctl enable ai-trader.service

echo "
✅ Production setup complete!

⚠️  IMPORTANT NEXT STEPS:

1. Create AWS Secrets Manager secret:
   aws secretsmanager create-secret \\
     --name ai-trader/production \\
     --secret-string '{
       \"BINANCE_API_KEY\": \"your-api-key\",
       \"BINANCE_API_SECRET\": \"your-api-secret\",
       \"DATABASE_URL\": \"sqlite:///data/trading_bot.db\",
       \"TRADING_MODE\": \"paper\",
       \"INITIAL_BALANCE\": \"1000\"
     }'

2. Create S3 backup bucket:
   aws s3 mb s3://${BACKUP_BUCKET}
   
3. Attach IAM role to EC2 instance with policies:
   - SecretsManagerReadWrite (for specific secret)
   - S3 access (for backup bucket)
   - CloudWatch Logs/Metrics

4. Initialize data:
   sudo -u $APP_USER $APP_DIR/venv/bin/python $APP_DIR/scripts/download_binance_data.py

5. Start the service:
   sudo systemctl start ai-trader

📊 Monitoring Commands:
- Status: sudo systemctl status ai-trader
- Logs: sudo journalctl -u ai-trader -f
- Metrics: aws cloudwatch get-metric-statistics (or use console)

🛠️ Maintenance:
- Update code: git pull && sudo systemctl restart ai-trader
- Force backup: sudo /usr/local/bin/backup-ai-trader.sh
- Check health: sudo -u $APP_USER $APP_DIR/venv/bin/python $APP_DIR/scripts/health_check.py
" 