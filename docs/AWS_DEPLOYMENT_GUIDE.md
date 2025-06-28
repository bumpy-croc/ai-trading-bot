# AWS Deployment Guide for AI Crypto Trading Bot

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Quick Start](#quick-start)
3. [Production Deployment](#production-deployment)
4. [Security Best Practices](#security-best-practices)
5. [Cost Optimization](#cost-optimization)
6. [Monitoring & Alerts](#monitoring--alerts)
7. [Troubleshooting](#troubleshooting)

## Prerequisites

### AWS Account Setup
- AWS account with billing enabled
- AWS CLI installed locally
- IAM user with appropriate permissions

### Required AWS Services
- EC2 (compute)
- Secrets Manager (credentials)
- S3 (backups)
- CloudWatch (monitoring)
- SNS (optional, for alerts)

### Local Requirements
```bash
# Install AWS CLI
pip install awscli
aws configure  # Enter your AWS credentials
```

## Quick Start

> **Which Setup Should I Use?**
> - **Basic (`aws_setup.sh`)**: Quick testing, development only. Credentials stored in .env file.
> - **Staging (`aws_setup_staging.sh`)**: Recommended for most users. Uses AWS Secrets Manager for secure credential storage.
> - **Production (`aws_setup_production.sh`)**: Full production setup with enhanced security, monitoring, and backups.

### 1. Launch EC2 Instance

```bash
# Create security group
aws ec2 create-security-group \
  --group-name ai-trader-sg \
  --description "Security group for AI Trading Bot"

# Add SSH access (replace YOUR_IP with your IP)
aws ec2 authorize-security-group-ingress \
  --group-name ai-trader-sg \
  --protocol tcp \
  --port 22 \
  --cidr YOUR_IP/32

# Launch instance
aws ec2 run-instances \
  --image-id ami-0c02fb55956c7d316 \  # Ubuntu 22.04 LTS
  --instance-type t3.small \
  --key-name your-key-pair \
  --security-groups ai-trader-sg \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=ai-trader}]'
```

### 2. Connect and Deploy

```bash
# Get instance IP
INSTANCE_IP=$(aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=ai-trader" \
  --query 'Reservations[0].Instances[0].PublicIpAddress' \
  --output text)

# Copy files
scp -r /Users/alex/Sites/ai-trader/* ubuntu@$INSTANCE_IP:/tmp/ai-trader/

# Connect
ssh ubuntu@$INSTANCE_IP

# Run setup (choose one):

# Option 1: Basic setup (stores credentials in .env file)
chmod +x deploy/aws_setup.sh
./deploy/aws_setup.sh

# Option 2: Staging setup with Secrets Manager (recommended)
chmod +x deploy/aws_setup_staging.sh
./deploy/aws_setup_staging.sh

# Option 3: Production setup (full security)
chmod +x deploy/aws_setup_production.sh
./deploy/aws_setup_production.sh
```

## Production Deployment

### 1. Create IAM Role

Create `ai-trader-role.json`:
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "ec2.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
```

Create the role:
```bash
# Create role
aws iam create-role \
  --role-name ai-trader-ec2-role \
  --assume-role-policy-document file://ai-trader-role.json

# Attach policies
aws iam attach-role-policy \
  --role-name ai-trader-ec2-role \
  --policy-arn arn:aws:iam::aws:policy/CloudWatchAgentServerPolicy

# Create instance profile
aws iam create-instance-profile \
  --instance-profile-name ai-trader-profile

aws iam add-role-to-instance-profile \
  --instance-profile-name ai-trader-profile \
  --role-name ai-trader-ec2-role
```

### 2. Create Custom Policy

Create `ai-trader-policy.json`:
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "secretsmanager:GetSecretValue"
      ],
      "Resource": "arn:aws:secretsmanager:*:*:secret:ai-trader/*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "s3:PutObject",
        "s3:GetObject",
        "s3:ListBucket",
        "s3:DeleteObject"
      ],
      "Resource": [
        "arn:aws:s3:::ai-trader-backups-*/*",
        "arn:aws:s3:::ai-trader-backups-*"
      ]
    },
    {
      "Effect": "Allow",
      "Action": [
        "cloudwatch:PutMetricData",
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents"
      ],
      "Resource": "*"
    }
  ]
}
```

Apply the policy:
```bash
aws iam put-role-policy \
  --role-name ai-trader-ec2-role \
  --policy-name ai-trader-policy \
  --policy-document file://ai-trader-policy.json
```

### 3. Setup Secrets

```bash
# Create secret
aws secretsmanager create-secret \
  --name ai-trader/production \
  --description "Production credentials for AI Trading Bot" \
  --secret-string '{
    "BINANCE_API_KEY": "your-api-key",
    "BINANCE_API_SECRET": "your-api-secret",
    "DATABASE_URL": "sqlite:///data/trading_bot.db",
    "TRADING_MODE": "paper",
    "INITIAL_BALANCE": "1000",
    "SLACK_WEBHOOK": "https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
  }'
```

### 4. Launch Production Instance

```bash
# Create production instance with IAM role
aws ec2 run-instances \
  --image-id ami-0c02fb55956c7d316 \
  --instance-type t3.medium \
  --key-name your-key-pair \
  --security-groups ai-trader-sg \
  --iam-instance-profile Name=ai-trader-profile \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":30,"VolumeType":"gp3"}}]' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=ai-trader-prod},{Key=Environment,Value=production}]' \
  --user-data file://user-data.sh
```

### 5. Deploy Production Code

```bash
# Connect to production instance
ssh ubuntu@PROD_INSTANCE_IP

# Copy files and run production setup
cd /tmp/ai-trader
chmod +x deploy/aws_setup_production.sh
./deploy/aws_setup_production.sh
```

## Security Best Practices

### 1. Network Security
- Use VPC with private subnets
- Implement Security Groups as firewall
- Enable VPC Flow Logs
- Use AWS Systems Manager Session Manager instead of SSH

### 2. Access Control
- Never store credentials in code
- Use IAM roles, not access keys
- Enable MFA for AWS console access
- Rotate API keys regularly

### 3. Data Protection
- Encrypt EBS volumes
- Use SSL/TLS for all external communications
- Backup data to encrypted S3 buckets
- Enable S3 versioning for backup buckets

### 4. Monitoring
```bash
# Create CloudWatch alarm for unauthorized API calls
aws cloudwatch put-metric-alarm \
  --alarm-name ai-trader-unauthorized-api \
  --alarm-description "Alert on unauthorized API calls" \
  --metric-name UnauthorizedAPICalls \
  --namespace CloudTrailMetrics \
  --statistic Sum \
  --period 300 \
  --threshold 1 \
  --comparison-operator GreaterThanOrEqualToThreshold \
  --evaluation-periods 1
```

## Cost Optimization

### 1. Instance Selection
- **Development**: t3.micro (free tier eligible)
- **Testing**: t3.small ($15/month)
- **Production**: t3.medium ($30/month)
- Consider Spot instances for non-critical workloads

### 2. Storage Optimization
- Use GP3 volumes (cheaper than GP2)
- Set up lifecycle policies for logs
- Compress old backups

### 3. Data Transfer
- Use CloudFront for static assets
- Minimize cross-region transfers
- Use VPC endpoints for AWS services

### 4. Cost Monitoring
```bash
# Set up billing alarm
aws cloudwatch put-metric-alarm \
  --alarm-name ai-trader-billing-alarm \
  --alarm-description "Alert when estimated charges exceed $50" \
  --metric-name EstimatedCharges \
  --namespace AWS/Billing \
  --statistic Maximum \
  --period 86400 \
  --threshold 50 \
  --comparison-operator GreaterThanThreshold \
  --evaluation-periods 1 \
  --dimensions Name=Currency,Value=USD
```

## Monitoring & Alerts

### 1. CloudWatch Dashboard
```bash
# Create dashboard
python scripts/create_cloudwatch_dashboard.py
```

### 2. SNS Alerts
```bash
# Create SNS topic
aws sns create-topic --name ai-trader-alerts

# Subscribe email
aws sns subscribe \
  --topic-arn arn:aws:sns:region:account-id:ai-trader-alerts \
  --protocol email \
  --notification-endpoint your-email@example.com
```

### 3. Key Metrics to Monitor
- CPU utilization (alert > 80%)
- Memory usage (alert > 90%)
- Disk space (alert > 85%)
- API rate limits
- Trading errors
- Profit/Loss thresholds

## Troubleshooting

### Common Issues

#### 1. Service Won't Start
```bash
# Check logs
sudo journalctl -u ai-trader -n 100

# Check permissions
ls -la /opt/ai-trader/

# Verify secrets
aws secretsmanager get-secret-value --secret-id ai-trader/production
```

#### 2. API Connection Issues
```bash
# Test network connectivity
curl -I https://api.binance.com

# Check DNS resolution
nslookup api.binance.com

# Verify credentials
python -c "import os; print(os.environ.get('BINANCE_API_KEY', 'NOT SET'))"
```

#### 3. High Memory Usage
```bash
# Check process memory
ps aux | grep python | grep trading

# Monitor in real-time
htop

# Check for memory leaks
sudo -u ai-trader /opt/ai-trader/venv/bin/python -m memory_profiler run_live_trading.py
```

### Emergency Procedures

#### Stop Trading Immediately
```bash
sudo systemctl stop ai-trader
```

#### Rollback Deployment
```bash
# Restore from backup
aws s3 cp s3://ai-trader-backups/latest-backup.tar.gz .
tar -xzf latest-backup.tar.gz -C /opt/ai-trader/
sudo systemctl restart ai-trader
```

#### Disable Live Trading
```bash
# Switch to paper trading
aws secretsmanager update-secret \
  --secret-id ai-trader/production \
  --secret-string '{"TRADING_MODE": "paper"}'
  
# Restart service
sudo systemctl restart ai-trader
```

## Maintenance Schedule

### Daily
- Monitor CloudWatch dashboard
- Check trading performance
- Review error logs

### Weekly
- Review AWS costs
- Update price data cache
- Check for security updates

### Monthly
- Rotate API keys
- Review and optimize strategies
- Update ML models
- Full system backup

### Quarterly
- Security audit
- Performance optimization
- Disaster recovery test

## Support Resources

- AWS Support: https://aws.amazon.com/support
- Binance API Docs: https://binance-docs.github.io/apidocs
- Project Issues: https://github.com/yourusername/ai-trader/issues
- CloudWatch Logs: Check `/aws/ec2/ai-trader` log group

---

**Remember**: Always test changes in a development environment before deploying to production! 