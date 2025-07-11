# AWS Secrets Manager Setup Guide for AI Trading Bot

## Overview

AWS Secrets Manager provides secure storage for your trading bot's sensitive configuration. This guide walks you through setting up secrets for different environments (staging, production).

## Why Use Secrets Manager?

- **Security**: Credentials are never stored in files or code
- **Rotation**: Easy to update credentials without redeploying
- **Audit Trail**: CloudTrail logs all secret access
- **Encryption**: Secrets are encrypted at rest using KMS
- **Access Control**: Fine-grained IAM permissions

## Required Environment Variables

### Essential Variables (Required)

| Variable | Description | Example | Notes |
|----------|-------------|---------|-------|
| `BINANCE_API_KEY` | Your Binance API key | `abc123...` | Get from Binance API Management |
| `BINANCE_API_SECRET` | Your Binance API secret | `xyz789...` | Keep this absolutely secure |
| `DATABASE_URL` | Database connection string | `postgresql://user:pass@db.amazonaws.com:5432/ai_trading_bot` | PostgreSQL connection |
| `TRADING_MODE` | Trading mode | `paper` or `live` | Start with `paper` |
| `INITIAL_BALANCE` | Starting balance for paper trading | `1000` | Only used in paper mode |

### Optional Variables

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `LOG_LEVEL` | Logging verbosity | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `TRADING_PAIR` | Trading pair | `BTCUSDT` | Any valid Binance pair |
| `SLACK_WEBHOOK` | Slack notifications | None | `https://hooks.slack.com/...` |
| `MAX_POSITION_SIZE` | Max position size | `0.1` | Percentage of balance (0.1 = 10%) |
| `STOP_LOSS_PERCENTAGE` | Stop loss | `0.02` | 0.02 = 2% stop loss |
| `TAKE_PROFIT_PERCENTAGE` | Take profit | `0.05` | 0.05 = 5% take profit |

## Step 1: Create Binance API Keys

1. Log into [Binance](https://www.binance.com)
2. Go to **API Management** (Profile → API Management)
3. Create a new API key:
   - Label: `ai-trading-bot-staging` (or `ai-trading-bot-production`)
   - Enable: **Reading** and **Spot Trading**
   - IP Restrictions: Add your EC2 Elastic IP (recommended)
4. Save the API Key and Secret securely

⚠️ **Important**: 
- Never enable withdrawal permissions
- Use different API keys for staging and production
- Consider IP whitelisting for production

## Step 2: Create Secret in AWS Console

### Method A: AWS Console (Visual)

1. Go to **AWS Secrets Manager** in the console
2. Click **"Store a new secret"**
3. Choose **"Other type of secret"**
4. Enter key/value pairs:

```
Key                     Value
---                     -----
BINANCE_API_KEY         your-api-key-here
BINANCE_API_SECRET      your-api-secret-here
DATABASE_URL            sqlite:///data/trading_bot.db
TRADING_MODE            paper
INITIAL_BALANCE         1000
LOG_LEVEL               INFO
```

5. Click **Next**
6. Secret name: `ai-trading-bot/staging`
7. Description: `Staging environment secrets for AI Trading Bot`
8. Configure automatic rotation: **Disable** (we'll rotate manually)
9. Review and store

### Method B: AWS CLI (Command Line)

```bash
# For staging environment
aws secretsmanager create-secret \
  --name ai-trading-bot/staging \
  --description "Staging environment secrets for AI Trading Bot" \
  --secret-string '{
    "BINANCE_API_KEY": "your-staging-api-key",
    "BINANCE_API_SECRET": "your-staging-api-secret",
    "DATABASE_URL": "sqlite:///data/trading_bot.db",
    "TRADING_MODE": "paper",
    "INITIAL_BALANCE": "1000",
    "LOG_LEVEL": "INFO",
    "TRADING_PAIR": "BTCUSDT",
    "MAX_POSITION_SIZE": "0.1",
    "STOP_LOSS_PERCENTAGE": "0.02",
    "TAKE_PROFIT_PERCENTAGE": "0.05"
  }'

# For production environment
aws secretsmanager create-secret \
  --name ai-trading-bot/production \
  --description "Production environment secrets for AI Trading Bot" \
  --secret-string '{
    "BINANCE_API_KEY": "your-production-api-key",
    "BINANCE_API_SECRET": "your-production-api-secret",
    "DATABASE_URL": "postgresql://aitradingbot:password@localhost/aitradingbot",
    "TRADING_MODE": "live",
    "INITIAL_BALANCE": "10000",
    "LOG_LEVEL": "WARNING",
    "TRADING_PAIR": "BTCUSDT",
    "MAX_POSITION_SIZE": "0.05",
    "STOP_LOSS_PERCENTAGE": "0.01",
    "TAKE_PROFIT_PERCENTAGE": "0.03",
    "SLACK_WEBHOOK": "https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
  }'
```

## Step 3: Create IAM Policy for EC2 Instance

Create a policy that allows your EC2 instance to read the secret:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "secretsmanager:GetSecretValue",
        "secretsmanager:DescribeSecret"
      ],
      "Resource": [
        "arn:aws:secretsmanager:us-east-1:YOUR-ACCOUNT-ID:secret:ai-trading-bot/*"
      ]
    },
    {
      "Effect": "Allow",
      "Action": [
        "kms:Decrypt"
      ],
      "Resource": "*",
      "Condition": {
        "StringEquals": {
          "kms:ViaService": "secretsmanager.us-east-1.amazonaws.com"
        }
      }
    }
  ]
}
```

## Step 4: Attach IAM Role to EC2 Instance

1. Create an IAM role with the above policy
2. Attach the role to your EC2 instance:

```bash
# Create the policy
aws iam create-policy \
  --policy-name AITradingBotSecretsAccess \
  --policy-document file://secrets-policy.json

# Attach to existing role
aws iam attach-role-policy \
  --role-name ai-trading-bot-ec2-role \
  --policy-arn arn:aws:iam::YOUR-ACCOUNT-ID:policy/AITradingBotSecretsAccess
```

## Step 5: Test Secret Access

On your EC2 instance, test that you can fetch secrets:

```bash
# Test with AWS CLI
aws secretsmanager get-secret-value --secret-id ai-trading-bot/staging

# Test with the Python script
cd /opt/ai-trading-bot
ENVIRONMENT=staging ./venv/bin/python scripts/fetch_secrets.py
```

## Managing Secrets

### Update a Secret

```bash
# Update via CLI
aws secretsmanager update-secret \
  --secret-id ai-trading-bot/staging \
  --secret-string '{
    "BINANCE_API_KEY": "new-api-key",
    "BINANCE_API_SECRET": "new-api-secret",
    ...
  }'

# Or update specific values
aws secretsmanager put-secret-value \
  --secret-id ai-trading-bot/staging \
  --secret-string '{"TRADING_MODE": "live"}'
```

### Rotate API Keys

1. Create new API keys in Binance
2. Update the secret:
```bash
aws secretsmanager update-secret \
  --secret-id ai-trading-bot/production \
  --secret-string '{"BINANCE_API_KEY": "new-key", "BINANCE_API_SECRET": "new-secret"}'
```
3. Restart the trading bot:
```bash
sudo systemctl restart ai-trading-bot
```
4. Delete old API keys in Binance

### View Secret Versions

```bash
# List all versions
aws secretsmanager list-secret-version-ids --secret-id ai-trading-bot/staging

# Get specific version
aws secretsmanager get-secret-value \
  --secret-id ai-trading-bot/staging \
  --version-id "EXAMPLE-VERSION-ID"
```

## Environment-Specific Configurations

### Staging Environment
- Use paper trading mode
- Higher log levels (INFO/DEBUG)
- Smaller position sizes
- Test API keys with limited permissions

### Production Environment
- Live trading mode
- Lower log levels (WARNING/ERROR)
- Conservative position sizes
- Production API keys with IP restrictions
- Slack/Email notifications enabled

## Security Best Practices

1. **Least Privilege**: Only grant read access to specific secrets
2. **Separate Environments**: Use different secrets for staging/production
3. **Audit Trails**: Enable CloudTrail for secret access logging
4. **Rotation**: Rotate API keys monthly
5. **Monitoring**: Set up CloudWatch alarms for unauthorized access

### Example CloudWatch Alarm

```bash
# Alert on failed secret access attempts
aws cloudwatch put-metric-alarm \
  --alarm-name ai-trading-bot-secret-access-failures \
  --alarm-description "Alert on failed secret access" \
  --metric-name UserErrors \
  --namespace AWS/SecretsManager \
  --statistic Sum \
  --period 300 \
  --threshold 5 \
  --comparison-operator GreaterThanThreshold \
  --evaluation-periods 1
```

## Troubleshooting

### Common Issues

1. **"AccessDeniedException"**
   - Check IAM role is attached to EC2 instance
   - Verify policy allows access to the secret ARN
   - Check KMS key permissions

2. **"ResourceNotFoundException"**
   - Verify secret name matches exactly
   - Check region is correct
   - Ensure secret exists

3. **"InvalidRequestException"**
   - Check JSON format is valid
   - Verify all required fields are present

### Debug Commands

```bash
# Check IAM role
aws sts get-caller-identity

# Test secret access
aws secretsmanager describe-secret --secret-id ai-trading-bot/staging

# Check instance profile
curl -s http://169.254.169.254/latest/meta-data/iam/security-credentials/
```

## Integration with Trading Bot

The trading bot automatically fetches secrets:

1. **On Service Start**: `ExecStartPre` in systemd fetches latest secrets
2. **Scheduled Updates**: Cron job refreshes secrets every 12 hours
3. **Manual Refresh**: Run `fetch_secrets.py` anytime

The secrets are written to `/opt/ai-trading-bot/.env` with secure permissions (600).

## Cost Considerations

- **Storage**: $0.40 per secret per month
- **API Calls**: $0.05 per 10,000 API calls
- **Typical Cost**: ~$0.50/month for one secret with normal usage

## Summary

Using AWS Secrets Manager provides:
- ✅ Secure credential storage
- ✅ Easy rotation without code changes
- ✅ Audit trail of all access
- ✅ Environment separation
- ✅ No hardcoded credentials

Remember to:
- Never commit credentials to git
- Use different API keys for each environment
- Rotate credentials regularly
- Monitor secret access with CloudWatch 