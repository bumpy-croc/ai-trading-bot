# Railway Deployment Quickstart

Quick guide for deploying the AI Trading Bot to Railway.

## Prerequisites

- [Railway CLI](https://docs.railway.app/develop/cli) installed: `npm install -g @railway/cli`
- GitHub account
- Railway account (connected to GitHub)

## Quick Deployment

### 1. Install Railway CLI

```bash
npm install -g @railway/cli
```

### 2. Login to Railway

```bash
railway login
```

### 3. Initialize Project

```bash
# In your project directory
railway init

# Link to existing project (optional)
railway link
```

### 4. Add PostgreSQL Database

```bash
# Add PostgreSQL service to your project
railway add postgresql
```

Railway automatically creates and configures:
- PostgreSQL database instance
- `DATABASE_URL` environment variable
- Private network connection between services

### 5. Configure Environment Variables

Set required environment variables in Railway dashboard or via CLI:

```bash
railway variables set BINANCE_API_KEY=your_api_key
railway variables set BINANCE_API_SECRET=your_api_secret
railway variables set TRADING_MODE=paper
railway variables set INITIAL_BALANCE=10000
```

Required variables:
- `BINANCE_API_KEY` - Your Binance API key
- `BINANCE_API_SECRET` - Your Binance API secret
- `DATABASE_URL` - Auto-configured by Railway PostgreSQL service
- `TRADING_MODE` - `paper` or `live`
- `INITIAL_BALANCE` - Starting balance for paper trading

Optional variables:
- `LOG_LEVEL` - Logging level (default: INFO)
- `LOG_JSON` - Enable JSON logging (default: true in production)
- `ENVIRONMENT` - Environment name (default: production on Railway)

### 6. Deploy

```bash
# Deploy current branch
railway up

# Or push to trigger deploy
git push railway main
```

## Project Structure on Railway

Typical Railway project setup:

```
Railway Project
├── Trading Bot Service (main app)
│   ├── Environment: DATABASE_URL, BINANCE_API_KEY, etc.
│   ├── Build: pip install -r requirements-server.txt
│   └── Start: atb live ml_basic --paper-trading
│
├── PostgreSQL Service
│   ├── Provides: DATABASE_URL
│   └── Private network connection
│
└── Dashboard Service (optional)
    ├── Environment: DATABASE_URL
    └── Start: atb dashboards run monitoring
```

## Deployment Configuration

### Build Command

Railway automatically detects Python and runs:
```bash
pip install -r requirements-server.txt
```

Or customize in `railway.json`:
```json
{
  "build": {
    "builder": "NIXPACKS"
  },
  "deploy": {
    "startCommand": "atb live ml_basic --paper-trading",
    "restartPolicyType": "ON_FAILURE",
    "restartPolicyMaxRetries": 10
  }
}
```

### Start Command

For live trading (paper mode):
```bash
atb live ml_basic --symbol BTCUSDT --paper-trading
```

For live trading with health endpoint:
```bash
atb live-health --port $PORT -- ml_basic --paper-trading
```

For monitoring dashboard:
```bash
atb dashboards run monitoring --port $PORT
```

## Database Setup

Railway automatically provides PostgreSQL connection:

1. **Add PostgreSQL service** via Railway dashboard or CLI
2. **DATABASE_URL is auto-configured** and injected into your app
3. **Tables are auto-created** on first application start
4. **Migrations run automatically** via Alembic if needed

Connection details are available in Railway dashboard under the PostgreSQL service.

## Monitoring and Logs

### View Logs

```bash
# Stream logs from Railway
railway logs

# Filter by service
railway logs --service trading-bot
```

### Access Dashboard

If you deployed the monitoring dashboard:
1. Check Railway dashboard for the service URL
2. Access at: `https://your-service.railway.app`

## Troubleshooting

### Connection Issues

```bash
# Check environment variables
railway variables

# Test database connection
railway run atb db verify
```

### Build Failures

If build times out or fails:
1. Use `requirements-server.txt` instead of `requirements.txt` (lighter dependencies)
2. Increase build timeout in Railway settings
3. Check logs: `railway logs --deployment`

### Common Issues

**"DATABASE_URL environment variable is required"**
- Ensure PostgreSQL service is added to project
- Check that DATABASE_URL is set: `railway variables`

**"Binance API connection failed"**
- Verify BINANCE_API_KEY and BINANCE_API_SECRET are set correctly
- Check API key permissions on Binance

**"Module not found" errors**
- Ensure requirements-server.txt includes all dependencies
- Try: `railway run pip install <missing-package>`

## CLI Commands Reference

```bash
# Initialize and link
railway init              # Create new project
railway link              # Link to existing project

# Deployment
railway up                # Deploy current directory
railway status            # Check deployment status

# Database
railway add postgresql    # Add PostgreSQL database
railway connect postgres  # Connect to database shell

# Variables
railway variables         # List all variables
railway variables set KEY=value
railway variables delete KEY

# Logs and monitoring
railway logs              # Stream logs
railway logs --deployment # View deployment logs

# Open services
railway open              # Open service in browser
railway domain            # Manage custom domains
```

## Advanced Configuration

### Multiple Environments

```bash
# Create staging environment
railway environment create staging

# Switch environments
railway environment staging
railway variables set TRADING_MODE=paper

# Deploy to specific environment
railway up --environment staging
```

### Custom Domains

```bash
# Add custom domain
railway domain

# Follow instructions in Railway dashboard
```

### Scaling and Resources

Configure in Railway dashboard:
- **Memory**: Adjust based on workload (512MB - 8GB)
- **CPU**: Scales automatically with memory
- **Replicas**: Horizontal scaling (Pro plan)

## Security Best Practices

1. **Never commit secrets** to repository
2. **Use Railway environment variables** for all sensitive data
3. **Enable 2FA** on Railway account
4. **Rotate API keys** regularly
5. **Use paper trading** for testing
6. **Monitor logs** for suspicious activity

## Cost Optimization

1. **Start small**: Use minimum required resources
2. **Monitor usage**: Check Railway dashboard for metrics
3. **Use sleep mode**: For non-production environments
4. **Optimize dependencies**: Use requirements-server.txt
5. **Connection pooling**: Enabled by default for database

## Next Steps

- Read [Railway Database Centralization Guide](RAILWAY_DATABASE_CENTRALIZATION_GUIDE.md) for detailed database setup
- See [Live Trading Guide](LIVE_TRADING_GUIDE.md) for trading configuration
- Check [Monitoring Summary](MONITORING_SUMMARY.md) for observability setup
- Review [Configuration System](CONFIGURATION_SYSTEM_SUMMARY.md) for environment management

## Support

- Railway Documentation: https://docs.railway.app
- Railway Discord: https://discord.gg/railway
- Project Issues: https://github.com/bumpy-croc/ai-trading-bot/issues

---

**Ready to Deploy?** Run `railway init` and follow the steps above!
