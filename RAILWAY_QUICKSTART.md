# Railway Quick Start Guide

Deploy your AI Trading Bot to Railway in 5 minutes! 🚄

## What is Railway?

[Railway](https://railway.com) is a modern deployment platform that simplifies infrastructure. It's a great alternative to AWS with:

- **Simpler setup** - No complex EC2, VPC, or IAM configuration
- **Built-in databases** - PostgreSQL with automatic backups
- **One-click deploys** - Deploy from Git with zero configuration
- **Cost-effective** - Pay only for what you use, typically 40-60% cheaper than AWS
- **Auto-scaling** - Automatically scales based on demand

## Quick Start (5 Minutes)

### 1. Prerequisites ✅

- [Railway account](https://railway.app) (free tier available)
- Your trading bot code
- Binance API keys

### 2. Setup Railway 🚄

```bash
# Install Railway CLI (choose your platform)
# Linux/macOS:
curl -fsSL https://railway.app/install.sh | sh

# macOS with Homebrew:
brew install railway

# Login to Railway
railway login
```

### 3. Deploy Your Bot 🚀

```bash
# Clone your repository (if not already local)
git clone <your-repo-url>
cd ai-trading-bot

# Run the setup script
./bin/railway-setup.sh

# Follow the prompts:
# - Enter project name (or create new)
# - Choose environment (staging recommended first)
# - Add PostgreSQL database (recommended: yes)
```

### 4. Configure API Keys 🔑

The setup script provides a URL to your Railway dashboard. Set these variables:

**Required:**
- `BINANCE_API_KEY` - Your Binance API key
- `BINANCE_SECRET_KEY` - Your Binance secret key

**Optional:**
- `SENTICRYPT_API_KEY` - For sentiment analysis
- `CRYPTOCOMPARE_API_KEY` - Additional sentiment data
- `AUGMENTO_API_KEY` - Alternative sentiment provider

### 5. Deploy 🎯

```bash
# Deploy to staging environment
./bin/railway-deploy.sh -p your-project-name -e staging

# Monitor the deployment
railway logs

# Check health
railway status
```

## That's It! 🎉

Your trading bot is now running on Railway. You can:

- **View logs**: `railway logs`
- **Check status**: `railway status`
- **Open dashboard**: Visit the URL provided during setup
- **Scale up**: Upgrade your plan in Railway dashboard

## Next Steps

### Test Your Deployment

```bash
# Get your app URL
railway domain

# Test health endpoint
curl https://your-app.railway.app/health

# Check detailed status
curl https://your-app.railway.app/status
```

### Deploy to Production

```bash
# When ready for live trading
./bin/railway-deploy.sh -p your-project-name -e production
```

### Monitor Performance

1. **Railway Dashboard**: Real-time metrics and logs
2. **Health Endpoints**: Built-in monitoring
3. **Database Metrics**: Query performance and storage usage

## Railway vs AWS Comparison

| Feature | AWS (Manual Setup) | Railway | Winner |
|---------|-------------------|---------|---------|
| **Setup Time** | 2-4 hours | 5 minutes | 🚄 Railway |
| **Monthly Cost** | $60-100 | $20-40 | 🚄 Railway |
| **Complexity** | High (EC2, RDS, IAM, etc.) | Low (one config file) | 🚄 Railway |
| **Databases** | Manual setup, RDS | One-click PostgreSQL | 🚄 Railway |
| **SSL/Domains** | Manual (Load Balancer, ACM) | Automatic | 🚄 Railway |
| **Scaling** | Manual (Auto Scaling Groups) | Automatic | 🚄 Railway |
| **Logs** | CloudWatch (complex) | Built-in | 🚄 Railway |
| **Control** | Full infrastructure control | Platform abstraction | ⚡ AWS |
| **Enterprise** | Advanced features | Simplified | ⚡ AWS |

## Troubleshooting

### Common Issues

**Build failing?**
```bash
# Check build logs
railway logs --deployment-id <deployment-id>

# Common fixes:
# - Verify requirements.txt is correct
# - Check Dockerfile syntax
# - Ensure all files are committed to git
```

**Health check failing?**
```bash
# Check health endpoint
curl https://your-app.railway.app/health

# Review application logs
railway logs --filter "health"

# Verify environment variables
railway variables
```

**API connections failing?**
```bash
# Verify API keys are set
railway variables | grep BINANCE

# Check permissions on Binance account
# - Spot trading enabled
# - IP restrictions (Railway uses dynamic IPs)
```

### Get Help

- **Railway Docs**: [docs.railway.app](https://docs.railway.app)
- **Railway Discord**: [railway.app/discord](https://railway.app/discord)
- **Full Guide**: [RAILWAY_DEPLOYMENT_GUIDE.md](docs/RAILWAY_DEPLOYMENT_GUIDE.md)

## Migration from AWS

Already using AWS? You can run both in parallel:

1. **Keep AWS running** (no downtime)
2. **Set up Railway** with staging environment
3. **Test thoroughly** on Railway staging
4. **Switch production traffic** when confident
5. **Decommission AWS** resources

Your existing AWS scripts remain unchanged - Railway is a completely separate deployment option.

---

**Ready to deploy?** Run `./bin/railway-setup.sh` to get started! 🚀