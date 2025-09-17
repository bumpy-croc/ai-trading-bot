# Railway Quickstart Guide

Quick deployment of the AI Trading Bot to Railway platform.

## Prerequisites

- Railway CLI installed: `npm install -g @railway/cli`
- Railway account: [railway.app](https://railway.app)
- Git repository pushed to GitHub/GitLab

## Quick Deploy

### 1. Initialize Railway Project

```bash
# Clone and navigate to your repo
cd ai-trading-bot

# Login to Railway
railway login

# Initialize project
railway init

# Or link to existing project
railway link [your-project-id]
```

### 2. Add PostgreSQL Database

```bash
# Add PostgreSQL service
railway add postgresql

# This automatically provides DATABASE_URL and other environment variables
```

### 3. Set Environment Variables

Set these in Railway dashboard or via CLI:

```bash
railway variables set BINANCE_API_KEY=your_api_key
railway variables set BINANCE_API_SECRET=your_secret
railway variables set TRADING_MODE=paper
railway variables set INITIAL_BALANCE=1000
railway variables set LOG_LEVEL=INFO
```

### 4. Deploy

```bash
# Deploy the application
railway up

# Or setup automatic deployments from Git
railway connect [github-repo-url]
```

### 5. Verify Deployment

```bash
# Check service status
railway status

# View logs
railway logs

# Get service URL
railway url
```

## Service Configuration

Railway automatically detects Python applications. The deployment uses:

- **Start Command**: `python -m cli.main live ml_basic --symbol BTCUSDT --paper-trading`
- **Build Command**: `pip install -r requirements-server.txt`
- **Environment**: Python 3.11+

## Database Setup

Railway PostgreSQL provides these environment variables automatically:
- `DATABASE_URL` - Complete connection string
- `PGHOST`, `PGPORT`, `PGUSER`, `PGPASSWORD`, `PGDATABASE` - Individual components

The application will automatically run migrations on startup.

## Health Monitoring

Once deployed, your bot will be accessible at the Railway URL. The monitoring dashboard will be available at:
```
https://your-app.railway.app:8000
```

## Troubleshooting

### Common Issues

1. **Build Timeout**: Use `requirements-server.txt` instead of full `requirements.txt`
2. **Database Connection**: Ensure PostgreSQL service is running and DATABASE_URL is set
3. **API Credentials**: Verify Binance API keys are set correctly

### Useful Commands

```bash
# View environment variables
railway variables

# Access railway database directly
railway connect postgres

# Restart services
railway restart

# View resource usage
railway usage
```

## Next Steps

1. **Monitor Logs**: Set up log monitoring in Railway dashboard
2. **Set Alerts**: Configure deployment and health alerts
3. **Scale Resources**: Adjust memory/CPU as needed
4. **Backup Strategy**: Set up regular database backups

For detailed deployment configuration, see [Railway Deployment Guide](RAILWAY_DEPLOYMENT_GUIDE.md).

For database-specific setup, see [Railway Database Centralization Guide](RAILWAY_DATABASE_CENTRALIZATION_GUIDE.md).