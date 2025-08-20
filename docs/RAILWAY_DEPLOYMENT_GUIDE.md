# Railway Deployment Guide

This guide walks you through deploying the AI Trading Bot to [Railway](https://railway.app), a modern platform-as-a-service that makes deployment simple and cost-effective.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Start](#quick-start)
3. [Manual Setup](#manual-setup)
4. [Environment Variables](#environment-variables)
5. [Database Setup](#database-setup)
6. [Deployment](#deployment)
7. [Monitoring and Logs](#monitoring-and-logs)
8. [Troubleshooting](#troubleshooting)
9. [Cost Optimization](#cost-optimization)

## Prerequisites

- Railway account (sign up at [railway.app](https://railway.app))
- Railway CLI installed
- Git repository with your trading bot code
- API keys for trading and data sources

## Quick Start

The fastest way to get started is using our setup script:

```bash
# 1. Run the setup script
./bin/railway-setup.sh

# 2. Set your API keys in Railway dashboard
# (Visit the provided URL and update environment variables)

# 3. Deploy your application
./bin/railway-deploy.sh -p your-project-name -e staging
```

## Manual Setup

### 1. Install Railway CLI

**Linux/macOS:**
```bash
curl -fsSL https://railway.app/install.sh | sh
```

**macOS with Homebrew:**
```bash
brew install railway
```

**Windows:**
Download from [Railway CLI releases](https://github.com/railwayapp/cli/releases)

### 2. Login to Railway

```bash
railway login
```

### 3. Create a New Project

```bash
# Create new project
railway project new ai-trading-bot

# Or link to existing project
railway link your-existing-project
```

### 4. Set Up Environment Variables

Railway automatically injects several environment variables:
- `RAILWAY_ENVIRONMENT_NAME` (staging/production)
- `RAILWAY_PROJECT_NAME`
- `RAILWAY_SERVICE_NAME`

Set your custom variables:

```bash
# Core environment variables
railway variables set ENVIRONMENT=staging
railway variables set PYTHONPATH=/app
railway variables set PYTHONUNBUFFERED=1

# Trading API keys (required)
railway variables set BINANCE_API_KEY=your_binance_api_key
railway variables set BINANCE_API_SECRET=your_binance_secret_key

# Optional sentiment API keys
railway variables set SENTICRYPT_API_KEY=your_senticrypt_key

railway variables set AUGMENTO_API_KEY=your_augmento_key
```

## Environment Variables

### Required Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `ENVIRONMENT` | Deployment environment | `staging` or `production` |
| `BINANCE_API_KEY` | Binance API key for trading | `your_api_key` |
| `BINANCE_API_SECRET` | Binance API secret | `your_secret_key` |

### Optional Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `SENTICRYPT_API_KEY` | SentiCrypt sentiment API key | None |

| `AUGMENTO_API_KEY` | Augmento sentiment API key | None |
| `HEALTH_CHECK_PORT` | Health check server port | `8000` |
| `DATABASE_URL` | PostgreSQL connection string | Auto-generated |

### Railway-Specific Variables

Railway automatically provides these variables:

| Variable | Description |
|----------|-------------|
| `RAILWAY_ENVIRONMENT_NAME` | Current environment name |
| `RAILWAY_PROJECT_NAME` | Project name |
| `RAILWAY_SERVICE_NAME` | Service name |
| `RAILWAY_DEPLOYMENT_ID` | Unique deployment ID |
| `RAILWAY_REPLICA_ID` | Replica identifier |
| `RAILWAY_REGION` | Deployment region |

## Database Setup

### Add PostgreSQL Database

```bash
# Add PostgreSQL database to your project
railway add --database postgresql
```

This automatically:
- Creates a PostgreSQL instance
- Sets the `DATABASE_URL` environment variable
- Links the database to your service

### Database Configuration

The trading bot automatically:
- Creates necessary tables on startup
- Handles migrations
- Sets up connection pooling

No manual database setup required!

## Deployment

### Using the Deploy Script

```bash
# Deploy to staging (default)
./bin/railway-deploy.sh

# Deploy to production (requires confirmation)
./bin/railway-deploy.sh -e production
```

### Manual Deployment

```bash
# Make sure you're linked to a project
railway status

# Deploy directly
railway up
```

### Deployment Process

1. **Build Phase**: Railway builds your Docker container
2. **Health Check**: Railway waits for health check to pass
3. **Traffic Routing**: Traffic is routed to the new deployment
4. **Old Instance**: Previous instance is terminated

## Monitoring and Logs

### View Logs

```bash
# View live logs
railway logs

# View logs with filtering
railway logs --filter "ERROR"

# View logs from specific time
railway logs --since "1h"
```

### Health Checks

The bot includes built-in health check endpoints:

- **Basic Health**: `GET /health`
  - Returns 200 if service is responding
  - Used by Railway for health monitoring

- **Detailed Status**: `GET /status`
  - Returns component health (database, APIs, config)
  - Useful for debugging issues

### Monitoring Dashboard

Railway provides:
- CPU and memory usage graphs
- Network traffic monitoring
- Deployment history
- Error rate tracking

Access at: `https://railway.app/project/your-project-name`

## Troubleshooting

### Common Issues

#### 1. Build Failures

**Symptom**: Deployment fails during build phase

**Solutions**:
- Check `requirements.txt` for invalid packages
- Verify Dockerfile syntax
- Check build logs: `railway logs --deployment-id <id>`

#### 2. Health Check Failures

**Symptom**: Deployment shows as unhealthy

**Solutions**:
```bash
# Check health endpoint manually
curl https://your-app.railway.app/health

# Check detailed status
curl https://your-app.railway.app/status

# Review application logs
railway logs --filter "health"
```

#### 3. API Connection Issues

**Symptom**: Trading API calls fail

**Solutions**:
- Verify API keys are set correctly: `railway variables`
- Check API key permissions (spot trading enabled)
- Verify network connectivity in logs

#### 4. Database Connection Issues

**Symptom**: Database errors in logs

**Solutions**:
```bash
# Check database status
railway database

# Verify DATABASE_URL is set
railway variables | grep DATABASE_URL

# Connect to database directly
railway database shell
```

### Debug Commands

```bash
# Check service status
railway status

# View environment variables
railway variables

# Check recent deployments
railway deployment list

# Get service URL
railway domain

# Access service shell (if enabled)
railway shell
```

### Log Analysis

Common log patterns to look for:

```bash
# Successful startup
railway logs | grep "Trading session started"

# API errors
railway logs | grep "API call failed"

# Database issues
railway logs | grep "Database"

# Strategy execution
railway logs | grep "Signal generated"
```

## Cost Optimization

### Resource Management

Railway pricing is based on:
- **CPU usage** (per vCPU-hour)
- **Memory usage** (per GB-hour)
- **Network egress** (per GB)
- **Storage** (per GB-month)

### Optimization Tips

1. **Right-size your deployment**:
   ```json
   // In railway.json
   {
     "deploy": {
       "resources": {
         "cpu": "0.5",
         "memory": "1Gi"
       }
     }
   }
   ```

2. **Use staging for development**:
   - Deploy to staging for testing
   - Only use production for live trading

3. **Monitor resource usage**:
   - Check Railway dashboard regularly
   - Set up usage alerts
   - Scale down during inactive periods

4. **Database optimization**:
   - Regular cleanup of old data
   - Use connection pooling
   - Monitor query performance

### Cost Comparison

| Resource | AWS (estimate) | Railway | Savings |
|----------|----------------|---------|---------|
| 0.5 vCPU, 1GB RAM | $25-40/month | $15-25/month | 20-40% |
| PostgreSQL | $15-30/month | $5-15/month | 50-70% |
| Load Balancer | $18/month | Included | $18/month |
| **Total** | **$58-88/month** | **$20-40/month** | **$38-48/month** |

*Estimates based on 24/7 operation*

## Advanced Configuration

### Custom Domains

```bash
# Add custom domain
railway domain add yourdomain.com

# Configure SSL (automatic)
# Railway handles SSL certificates automatically
```

### Horizontal Scaling

```bash
# Scale to multiple replicas
railway variables set RAILWAY_REPLICA_COUNT=2
railway up
```

### Database Backups

Railway automatically creates daily backups. To create manual backups:

```bash
# Create database backup
railway database backup create

# List backups
railway database backup list

# Restore from backup
railway database backup restore <backup-id>
```

### CI/CD Integration

Connect Railway to your Git repository for automatic deployments:

1. Go to Railway project settings
2. Connect to GitHub/GitLab repository
3. Set up automatic deployments on push
4. Configure branch-based environments

## Migration from AWS

If migrating from AWS:

1. **Export data**: Use existing AWS scripts to backup data
2. **Set up Railway**: Follow this guide to set up Railway infrastructure
3. **Migrate environment variables**: Transfer secrets from AWS Secrets Manager to Railway
4. **Database migration**: Export from AWS RDS and import to Railway PostgreSQL
5. **Update DNS**: Point your domain to Railway (if using custom domain)
6. **Monitor**: Test thoroughly in staging before switching production traffic

## Support

- **Railway Documentation**: [docs.railway.app](https://docs.railway.app)
- **Railway Discord**: [railway.app/discord](https://railway.app/discord)
- **Railway Support**: [help.railway.app](https://help.railway.app)

For issues specific to this trading bot, check the main README or create an issue in the repository.

---

**Security Note**: Always use environment variables for sensitive data. Never commit API keys or secrets to your repository. Railway encrypts all environment variables at rest and in transit.
