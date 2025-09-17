# Railway Deployment Guide

Comprehensive guide for deploying the AI Trading Bot to Railway platform in production.

## Overview

This guide covers full production deployment with monitoring, scaling, security, and maintenance considerations.

## Production Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Trading Bot   │    │   PostgreSQL    │    │   Monitoring    │
│   (Main App)    │───▶│   Database      │◀───│   Dashboard     │
│                 │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Environment Setup

### 1. Production Project

```bash
# Create new Railway project for production
railway init

# Or use existing project
railway link [project-id]

# Set environment variables for production
railway variables set ENVIRONMENT=production
railway variables set LOG_LEVEL=INFO
railway variables set LOG_JSON=true
```

### 2. Database Configuration

```bash
# Add PostgreSQL database
railway add postgresql

# Set up database connection pool
railway variables set DB_POOL_SIZE=10
railway variables set DB_MAX_OVERFLOW=20
railway variables set DB_POOL_TIMEOUT=30
```

### 3. Trading Configuration

```bash
# Production trading settings
railway variables set TRADING_MODE=live  # or paper for safe production
railway variables set INITIAL_BALANCE=10000
railway variables set MAX_POSITION_SIZE=0.1
railway variables set RISK_PER_TRADE=0.02

# API credentials (use Railway secrets)
railway variables set BINANCE_API_KEY=your_production_key
railway variables set BINANCE_API_SECRET=your_production_secret

# Optional: Restrict API to trading only
railway variables set BINANCE_TESTNET=false
```

## Application Configuration

### 1. Service Configuration

Create `railway.json` in project root:

```json
{
  "$schema": "https://railway.app/railway.schema.json",
  "build": {
    "builder": "nixpacks",
    "buildCommand": "pip install -r requirements-server.txt"
  },
  "deploy": {
    "startCommand": "python -m cli.main live ml_basic --symbol BTCUSDT --paper-trading",
    "healthcheckPath": "/health",
    "healthcheckTimeout": 100,
    "restartPolicyType": "always"
  }
}
```

### 2. Health Monitoring

For production with health endpoint:

```json
{
  "deploy": {
    "startCommand": "python -m cli.main live-health --port $PORT -- ml_basic --symbol BTCUSDT --paper-trading",
    "healthcheckPath": "/health",
    "healthcheckTimeout": 100,
    "restartPolicyType": "always"
  }
}
```

### 3. Dashboard Service (Optional)

Deploy monitoring dashboard as separate service:

```json
{
  "deploy": {
    "startCommand": "python -m cli.main dashboards run monitoring --port $PORT",
    "healthcheckPath": "/",
    "healthcheckTimeout": 30
  }
}
```

## Database Management

### 1. Initial Setup

```bash
# Run database setup script
railway run python scripts/railway_database_setup.py

# Or use CLI command
railway run atb db setup-railway --env production
```

### 2. Migrations

```bash
# Run migrations manually
railway run alembic upgrade head

# Or via CLI
railway run atb db migrate
```

### 3. Backup Strategy

```bash
# Create backup
railway run atb db railway backup --env production

# Or use script
railway run python scripts/railway_database_backup.py
```

## Monitoring and Alerts

### 1. Application Monitoring

Set up monitoring dashboard:

```bash
# Access monitoring at Railway URL
railway url

# Or deploy dedicated monitoring service
railway add --template monitoring-dashboard
```

### 2. Railway Monitoring

Configure in Railway dashboard:
- **CPU Usage Alerts**: > 80% for 5 minutes
- **Memory Usage Alerts**: > 90% for 3 minutes
- **Disk Usage Alerts**: > 85%
- **Health Check Failures**: > 3 consecutive failures

### 3. Trading Alerts

Set up trading-specific monitoring:

```bash
# Environment variables for alerts
railway variables set ALERT_EMAIL=your-email@domain.com
railway variables set ALERT_WEBHOOK=your-webhook-url
railway variables set ALERT_BALANCE_THRESHOLD=500
```

## Security Configuration

### 1. API Security

```bash
# Restrict Binance API permissions
# In Binance: Enable only "Spot & Margin Trading" 
# Disable "Withdrawals" and "Futures"

# Set IP restrictions
railway variables set BINANCE_API_IP_RESTRICT=true
```

### 2. Database Security

```bash
# Enable SSL for database connections
railway variables set DATABASE_SSL_MODE=require

# Set connection limits
railway variables set DATABASE_MAX_CONNECTIONS=20
```

### 3. Network Security

Configure in Railway dashboard:
- **Private Networking**: Enable for database
- **Custom Domains**: Use HTTPS-only
- **Environment Isolation**: Separate prod/staging

## Scaling Configuration

### 1. Vertical Scaling

Resource limits in Railway:

```bash
# Increase memory for data-intensive operations
# Railway Dashboard: Settings → Resources
# Memory: 1GB → 2GB
# CPU: 1vCPU → 2vCPU
```

### 2. Horizontal Scaling

For high-frequency trading:

```bash
# Deploy multiple instances
railway scale --replicas 2

# Use shared database for coordination
railway variables set ENABLE_DISTRIBUTED_LOCK=true
```

## Deployment Automation

### 1. GitHub Actions

Create `.github/workflows/deploy.yml`:

```yaml
name: Deploy to Railway

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: railway/action@v1
        with:
          token: ${{ secrets.RAILWAY_TOKEN }}
          command: deploy
```

### 2. Railway Auto-Deploy

```bash
# Connect GitHub repository
railway connect github:your-username/ai-trading-bot

# Configure auto-deploy on main branch
railway config auto-deploy --branch main
```

## Maintenance

### 1. Log Management

```bash
# View live logs
railway logs --follow

# Download logs for analysis
railway logs --download --from=2024-01-01 --to=2024-01-31
```

### 2. Database Maintenance

```bash
# Regular backup (schedule via cron or GitHub Actions)
railway run atb db railway backup --env production

# Database health check
railway run atb db verify

# Clean old data (if configured)
railway run atb data cleanup --days 90
```

### 3. Performance Monitoring

```bash
# Check resource usage
railway usage

# Monitor trading performance
railway run atb db query --sql "SELECT * FROM trades ORDER BY timestamp DESC LIMIT 10"
```

## Troubleshooting

### Common Production Issues

1. **Memory Leaks**
   ```bash
   railway logs | grep -i "memory\|oom"
   railway restart
   ```

2. **Database Connection Issues**
   ```bash
   railway run atb db verify
   railway variables get DATABASE_URL
   ```

3. **API Rate Limits**
   ```bash
   railway logs | grep -i "rate limit\|429"
   # Increase delay between API calls
   railway variables set API_RATE_LIMIT_DELAY=1000
   ```

4. **Trading Execution Problems**
   ```bash
   railway run atb live-health --port 8000 -- ml_basic --symbol BTCUSDT --paper-trading
   # Check logs for trading errors
   ```

### Emergency Procedures

1. **Stop Trading Immediately**
   ```bash
   railway restart  # Stops current trading session
   railway variables set EMERGENCY_STOP=true
   ```

2. **Database Recovery**
   ```bash
   railway run atb db railway reset --env production  # DANGEROUS!
   # Requires double confirmation
   ```

3. **Rollback Deployment**
   ```bash
   railway rollback  # Rollback to previous deployment
   ```

## Cost Optimization

### 1. Resource Efficiency

```bash
# Use lighter requirements for production
# requirements-server.txt vs requirements.txt

# Optimize database queries
railway variables set DB_QUERY_CACHE=true

# Enable compression
railway variables set ENABLE_GZIP=true
```

### 2. Usage Monitoring

```bash
# Monitor Railway usage
railway usage --month current

# Set billing alerts in Railway dashboard
```

## Related Documentation

- [Railway Quickstart](RAILWAY_QUICKSTART.md) - Quick setup guide
- [Railway Database Centralization Guide](RAILWAY_DATABASE_CENTRALIZATION_GUIDE.md) - Database setup
- [Live Trading Guide](LIVE_TRADING_GUIDE.md) - Trading operations
- [Monitoring Summary](MONITORING_SUMMARY.md) - Monitoring setup