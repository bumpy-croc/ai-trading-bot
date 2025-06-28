# Configuration System Migration Guide

## Overview

The AI Trading Bot now uses a flexible configuration system that supports multiple sources (AWS Secrets Manager, environment variables, and .env files) with automatic fallback. This guide helps you migrate from the old system.

## What Changed?

### Old System
- Required `python-dotenv` and explicit `load_dotenv()` calls
- Only supported `.env` files
- Used `os.environ.get()` directly

### New System
- Unified `ConfigManager` with multiple providers
- Automatic source detection and fallback
- Type-safe configuration access
- No secrets written to disk on AWS

## Migration Steps

### 1. Update Your Code (if you have custom scripts)

**Old way:**
```python
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv('BINANCE_API_KEY')
```

**New way:**
```python
from core.config import get_config

config = get_config()
api_key = config.get('BINANCE_API_KEY')
# or for required values:
api_key = config.get_required('BINANCE_API_KEY')
```

### 2. Your .env File Still Works!

No changes needed to your existing `.env` file. The new system automatically detects and uses it for local development.

### 3. AWS Deployment Changes

The staging and production deployment scripts no longer create `.env` files. Instead:

- Secrets are read directly from AWS Secrets Manager
- More secure (no secrets on disk)
- Automatic refresh when secrets are updated
- No need to restart the service after updating secrets

### 4. New Features You Can Use

#### Typed Configuration Access
```python
config = get_config()

# Get with type conversion
balance = config.get_int('INITIAL_BALANCE', 1000)
max_position = config.get_float('MAX_POSITION_SIZE', 0.1)
paper_mode = config.get_bool('PAPER_TRADING', True)

# Get list values
symbols = config.get_list('TRADING_SYMBOLS', delimiter=',')
```

#### Environment-Specific Configuration
```python
# Automatically uses different secrets based on ENVIRONMENT variable
# development -> ai-trader/development
# staging -> ai-trader/staging
# production -> ai-trader/production
```

#### Configuration Priority

The system checks in this order:
1. AWS Secrets Manager (if available)
2. Environment variables
3. .env file

First found value wins!

## Testing Your Configuration

Run the test script to verify everything works:

```bash
python scripts/test_config_system.py
```

## Common Issues

### "Required configuration 'X' not found"

The new system is stricter about required values. Make sure all required configurations are set:

- `BINANCE_API_KEY`
- `BINANCE_API_SECRET`
- `DATABASE_URL`
- `TRADING_MODE`
- `INITIAL_BALANCE`

### AWS Secrets Manager Not Working

1. Check IAM role has permissions:
```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Action": ["secretsmanager:GetSecretValue"],
    "Resource": "arn:aws:secretsmanager:*:*:secret:ai-trader/*"
  }]
}
```

2. Verify secret exists:
```bash
aws secretsmanager describe-secret --secret-id ai-trader/staging
```

### Different Behavior Between Local and AWS

This is expected! Local uses `.env`, AWS uses Secrets Manager. To test AWS behavior locally:

```bash
# Set environment variable to simulate AWS
export ENVIRONMENT=staging

# Run your script
python your_script.py
```

## Benefits of the New System

1. **Security**: No secrets in files on production servers
2. **Flexibility**: Easy to add new config sources (Azure Key Vault, etc.)
3. **Type Safety**: Built-in type conversion and validation
4. **Hot Reload**: AWS Secrets refresh without restart (with cache TTL)
5. **Debugging**: Clear error messages showing which sources were checked

## Need Help?

- Run `python scripts/test_config_system.py` to debug
- Check logs for which configuration source is being used
- Ensure your IAM role has correct permissions for AWS deployment

The new system is designed to be backward compatible while providing better security and flexibility for production deployments! 