#!/bin/bash
# EC2 User Data Script - Runs automatically on instance launch
# This script prepares the instance for the trading bot deployment

# Log all output
exec > >(tee /var/log/user-data.log)
exec 2>&1

echo "Starting AI Trader instance initialization at $(date)"

# Update system
apt-get update
apt-get upgrade -y

# Install essential tools
apt-get install -y \
    git \
    curl \
    wget \
    htop \
    jq \
    unzip

# Install AWS CLI v2
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
./aws/install
rm -rf awscliv2.zip aws/

# Install CloudWatch Agent
wget https://s3.amazonaws.com/amazoncloudwatch-agent/ubuntu/amd64/latest/amazon-cloudwatch-agent.deb
dpkg -i -E ./amazon-cloudwatch-agent.deb
rm amazon-cloudwatch-agent.deb

# Create swap file (helpful for t3.micro/small instances)
fallocate -l 2G /swapfile
chmod 600 /swapfile
mkswap /swapfile
swapon /swapfile
echo '/swapfile none swap sw 0 0' | tee -a /etc/fstab

# Set system limits for trading bot
cat >> /etc/security/limits.conf << EOF
* soft nofile 65536
* hard nofile 65536
* soft nproc 4096
* hard nproc 4096
EOF

# Configure sysctl for better network performance
cat >> /etc/sysctl.conf << EOF
# Network optimizations for trading
net.core.rmem_max = 134217728
net.core.wmem_max = 134217728
net.ipv4.tcp_rmem = 4096 87380 134217728
net.ipv4.tcp_wmem = 4096 65536 134217728
net.core.netdev_max_backlog = 5000
EOF
sysctl -p

# Create directory for deployment
mkdir -p /tmp/ai-trader

# Install Python 3.11
apt-get install -y software-properties-common
add-apt-repository -y ppa:deadsnakes/ppa
apt-get update
apt-get install -y python3.11 python3.11-venv python3.11-dev

# Install psutil dependencies (for health checks)
apt-get install -y gcc python3.11-dev

# Create marker file to indicate user-data completion
touch /var/lib/cloud/instance/user-data-finished

echo "User data script completed at $(date)"
echo "Instance is ready for AI Trader deployment"
echo "Next step: Copy application files and run setup script" 