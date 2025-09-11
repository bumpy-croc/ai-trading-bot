# Railway Log Analysis - Quick Start

## 🚀 Setup (5 minutes)

1. **Run setup script**:
   ```bash
   python scripts/setup_railway_log_analysis.py
   ```

2. **Set environment variables** in `.env`:
   ```bash
   RAILWAY_PROJECT_ID=your-railway-project-id
   RAILWAY_SERVICE_ID=your-railway-service-id  # optional
   GITHUB_TOKEN=your-github-token              # for PR creation
   ```

3. **Test the system**:
   ```bash
   atb logs daily --environment staging --hours 1 --dry-run
   ```

## 📊 Daily Usage

### Automated (Recommended)

The system runs automatically via GitHub Actions at 06:00 UTC daily:

- ✅ Fetches last 24h of production logs
- ✅ Analyzes error patterns and performance issues  
- ✅ Generates fixes for common problems
- ✅ Creates PRs for review
- ✅ Creates issues for manual review items

### Manual Execution

```bash
# * Quick analysis of last 24 hours
atb logs daily --environment production

# * Analyze specific time period
atb logs daily --environment production --hours 48 --dry-run

# * Fetch logs only
atb logs fetch --environment production --hours 24 --save

# * Analyze existing log file
atb logs analyze --log-file logs/railway/production/railway_logs_20250109.log
```

## 🔧 What Gets Fixed Automatically

- **API Rate Limiting**: Adds exponential backoff and retry logic
- **Timeout Errors**: Adds timeout configuration constants
- **JSON Parsing**: Creates robust JSON parsing utilities
- **Error Handling**: Improves exception handling patterns

## 🔍 What Requires Manual Review

- **Database Connection Issues**: Infrastructure/config changes needed
- **Memory Problems**: Resource allocation adjustments required
- **Authentication Errors**: API key/permission issues
- **Complex Logic Errors**: Business logic review needed

## 📈 Monitoring

View analysis results in:

- **Monitoring Dashboard**: Real-time error rate tracking
- **Database**: Historical analysis data in `system_events` table
- **GitHub**: PRs and issues for all detected problems
- **Log Files**: Saved in `logs/analysis_reports/`

## 🚨 Alerts

The system creates:

- **Pull Requests**: For automatically fixable issues
- **GitHub Issues**: For manual review items  
- **Database Events**: For monitoring dashboard integration
- **Workflow Notifications**: For system failures

## ⚡ Quick Commands

```bash
# * Check system health
atb logs daily --dry-run --hours 1

# * Emergency analysis
atb logs fetch --environment production --filters ERROR CRITICAL --save
atb logs analyze --log-file logs/railway/production/error/railway_logs_latest.log

# * View recent analysis
ls -la logs/analysis_reports/
```

## 🔗 Related Documentation

- **Full Documentation**: `docs/RAILWAY_LOG_ANALYSIS_SYSTEM.md`
- **Railway Deployment**: `docs/RAILWAY_DEPLOYMENT_GUIDE.md`
- **Monitoring Setup**: `docs/MONITORING_SUMMARY.md`
- **Database Logging**: `docs/DATABASE_LOGGING_GUIDE.md`

---

**Ready to start?** Run `python scripts/setup_railway_log_analysis.py` now! 🎯