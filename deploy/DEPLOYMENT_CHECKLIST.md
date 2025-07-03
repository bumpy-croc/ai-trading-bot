# AWS Deployment Checklist

## Pre-Deployment
- [ ] AWS account with billing enabled
- [ ] Binance API keys (with appropriate permissions)
- [ ] Local AWS CLI configured
- [ ] SSH key pair created in AWS EC2

## Basic Deployment (Development/Testing)

### 1. Prepare Code
- [ ] Commit all changes to git
- [ ] Run tests locally: `python test/run_tests.py`
- [ ] Update requirements.txt if needed

### 2. Launch Instance
- [ ] Create security group with SSH access
- [ ] Launch t3.small Ubuntu 22.04 instance
- [ ] Note instance public IP

### 3. Deploy Code
- [ ] SCP files to instance: `scp -r * ubuntu@IP:/tmp/ai-trading-bot/`
- [ ] SSH to instance: `ssh ubuntu@IP`
- [ ] Run setup script: `./deploy/aws_setup.sh`

### 4. Configure
- [ ] Edit `/opt/ai-trading-bot/.env` with API credentials
- [ ] Download initial data: `python scripts/download_binance_data.py`
- [ ] Test with paper trading first

### 5. Start Trading
- [ ] Start service: `sudo systemctl start ai-trading-bot`
- [ ] Monitor logs: `sudo journalctl -u ai-trading-bot -f`
- [ ] Verify trades in database

## Production Deployment

### 1. AWS Infrastructure
- [ ] Create IAM role with required policies
- [ ] Create Secrets Manager secret
- [ ] Create S3 bucket for backups
- [ ] Set up CloudWatch log groups

### 2. Security Setup
- [ ] Enable MFA on AWS account
- [ ] Create separate IAM user for deployment
- [ ] Configure VPC and security groups
- [ ] Set up fail2ban on instance

### 3. Launch Production Instance
- [ ] Use t3.medium or larger
- [ ] Attach IAM instance profile
- [ ] Use user-data script for initialization
- [ ] Enable detailed monitoring

### 4. Deploy Application
- [ ] Copy code securely (via git or encrypted transfer)
- [ ] Run production setup script: `./deploy/aws_setup_production.sh`
- [ ] Verify secrets are loaded correctly

### 5. Configure Monitoring
- [ ] Set up CloudWatch dashboard
- [ ] Configure billing alerts
- [ ] Create SNS topic for alerts
- [ ] Set up health check alarms

### 6. Initialize and Test
- [ ] Download historical data
- [ ] Run in paper mode for 24 hours
- [ ] Verify all cron jobs are working
- [ ] Test backup and restore process

### 7. Go Live
- [ ] Update Secrets Manager to enable live trading
- [ ] Start with minimal capital
- [ ] Monitor closely for first week
- [ ] Document any issues

## Post-Deployment

### Daily Tasks
- [ ] Check CloudWatch dashboard
- [ ] Review trading logs
- [ ] Monitor P&L

### Weekly Tasks
- [ ] Review AWS costs
- [ ] Check for security updates
- [ ] Analyze trading performance

### Monthly Tasks
- [ ] Rotate API keys
- [ ] Update ML models if needed
- [ ] Full backup verification
- [ ] Security audit

## Emergency Procedures

### If Something Goes Wrong
1. [ ] Stop service immediately: `sudo systemctl stop ai-trading-bot`
2. [ ] Check logs: `sudo journalctl -u ai-trading-bot -n 500`
3. [ ] Switch to paper trading in Secrets Manager
4. [ ] Restore from backup if needed

### Rollback Steps
1. [ ] Download latest backup from S3
2. [ ] Stop current service
3. [ ] Restore files and database
4. [ ] Restart with previous version

## Important Commands

```bash
# Service management
sudo systemctl status ai-trading-bot
sudo systemctl start/stop/restart ai-trading-bot

# Logs
sudo journalctl -u ai-trading-bot -f
tail -f /opt/ai-trading-bot/logs/trading.log

# Database
sqlite3 /opt/ai-trading-bot/data/trading_bot.db

# Manual backup
/usr/local/bin/backup-ai-trading-bot.sh

# Health check
/opt/ai-trading-bot/venv/bin/python /opt/ai-trading-bot/scripts/health_check.py
```

## Contact Information

- AWS Support: [Case ID]
- Binance Support: [Ticket ID]
- On-call Engineer: [Phone/Email]
- Escalation: [Manager Contact]

---

**Remember**: Always have a rollback plan! 