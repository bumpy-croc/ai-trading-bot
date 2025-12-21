# Operations Runbook

**Last Updated:** 2025-11-21
**Version:** 1.0

This runbook provides operational procedures, troubleshooting guides, and recovery strategies for the AI Trading Bot in production environments.

---

## Table of Contents

1. [System Architecture Overview](#system-architecture-overview)
2. [Health Monitoring](#health-monitoring)
3. [Common Failure Modes](#common-failure-modes)
4. [Troubleshooting Procedures](#troubleshooting-procedures)
5. [Recovery Procedures](#recovery-procedures)
6. [Performance Optimization](#performance-optimization)
7. [Emergency Procedures](#emergency-procedures)
8. [Maintenance Tasks](#maintenance-tasks)
9. [Monitoring & Alerts](#monitoring--alerts)
10. [Database Operations](#database-operations)

---

## System Architecture Overview

### Core Components

1. **Live Trading Engine** (`src/live/trading_engine.py`)
   - Executes trades in real-time
   - Manages positions and risk
   - Critical: 95%+ uptime required

2. **Backtesting Engine** (`src/backtesting/engine.py`)
   - Validates strategies historically
   - Generates performance metrics
   - Non-critical: Can run asynchronously

3. **Data Providers** (`src/data_providers/`)
   - Binance API integration
   - Coinbase API integration
   - Caching layer for performance

4. **Database** (PostgreSQL)
   - Stores trades, positions, sessions
   - Required for all live trading
   - Backup strategy: Daily full + hourly incrementals

5. **Risk Management** (`src/risk/`)
   - Position sizing
   - Stop loss management
   - Drawdown protection

### System Requirements

- **Python:** 3.11+
- **PostgreSQL:** 15+
- **RAM:** 4GB minimum, 8GB recommended
- **CPU:** 2+ cores
- **Storage:** 10GB minimum (20GB+ for historical data)
- **Network:** Stable internet connection (<100ms latency to exchanges)

---

## Health Monitoring

### System Health Checks

#### 1. Live Engine Health

**Check if engine is running:**
```bash
# Check process
ps aux | grep "atb live"

# Check via health endpoint (if enabled)
curl http://localhost:8000/health

# Expected response:
# {
#   "status": "healthy",
#   "uptime_seconds": 3600,
#   "active_positions": 2,
#   "balance": 10500.45,
#   "last_data_update": "2025-11-21T15:30:00Z"
# }
```

#### 2. Database Connectivity

**Verify database connection:**
```bash
atb db verify

# Expected output:
# ✅ Database connection successful
# ✅ All tables exist
# ✅ Schema version: v1.2.3
```

#### 3. API Connectivity

**Test exchange connectivity:**
```bash
# Test Binance
atb data test-provider binance BTCUSDT 1h

# Test Coinbase
atb data test-provider coinbase BTC-USD 1h

# Expected: Recent OHLCV data without errors
```

#### 4. Data Freshness

**Check last data update:**
```bash
atb data cache-manager info

# Look for:
# - Last update timestamp (should be < 5 minutes old for live trading)
# - Cache hit rate (should be > 70%)
# - Number of cached symbols
```

### Key Metrics to Monitor

| Metric | Healthy Range | Action Threshold | Critical Threshold |
|--------|---------------|------------------|-------------------|
| API Response Time | < 500ms | > 1s | > 3s |
| Database Query Time | < 100ms | > 500ms | > 1s |
| Memory Usage | < 2GB | > 3GB | > 3.5GB |
| CPU Usage | < 50% | > 75% | > 90% |
| Error Rate | < 0.1% | > 1% | > 5% |
| Position Count | 0-5 | > 8 | > 10 |
| Drawdown | < 10% | > 15% | > 20% |

---

## Common Failure Modes

### 1. API Rate Limiting

**Symptoms:**
- `429 Too Many Requests` errors in logs
- Delayed data updates
- Failed order placements

**Root Causes:**
- Check interval too aggressive (< 5 seconds)
- Multiple instances running
- Insufficient API rate limits

**Quick Fix:**
```bash
# Increase check interval
# Edit config or restart with:
atb live ml_basic --symbol BTCUSDT --check-interval 30
```

**Prevention:**
- Use recommended check intervals (15-30 seconds for live trading)
- Implement exponential backoff (already built-in)
- Cache aggressively

---

### 2. Database Connection Loss

**Symptoms:**
- `DatabaseError: connection lost` in logs
- Failed trade logging
- Engine crashes

**Root Causes:**
- PostgreSQL service down
- Network interruption
- Connection pool exhausted
- Database disk full

**Diagnosis:**
```bash
# Check database status
systemctl status postgresql
# Or for Docker:
docker ps | grep postgres

# Check connections
psql -h localhost -U trading_bot -c "SELECT count(*) FROM pg_stat_activity;"

# Check disk space
df -h | grep postgres
```

**Recovery:**
```bash
# Restart PostgreSQL
systemctl restart postgresql
# Or for Docker:
docker restart postgres

# Verify connection
atb db verify

# Resume trading engine
atb live ml_basic --symbol BTCUSDT --resume-from-last-balance
```

**Prevention:**
- Monitor database health continuously
- Set up connection pooling (default: 10 connections)
- Regular database maintenance (see [Maintenance Tasks](#maintenance-tasks))

---

### 3. Exchange API Downtime

**Symptoms:**
- `ConnectionError` or `Timeout` errors
- No new candle data
- Failed order submissions

**Root Causes:**
- Exchange maintenance
- Network issues
- API endpoint changes
- IP ban (rare)

**Diagnosis:**
```bash
# Test connectivity
curl -I https://api.binance.com/api/v3/ping
curl -I https://api.coinbase.com/v2/time

# Check exchange status pages:
# Binance: https://www.binancestatus.com/
# Coinbase: https://status.coinbase.com/
```

**Recovery:**
- **If exchange is down:** Wait for recovery (check status page)
- **If network issue:** Check firewall/routing
- **If timeout:** Increase timeout settings

**Prevention:**
- Use health endpoint to monitor API availability
- Implement graceful degradation (already built-in)
- Have backup exchange configured

---

### 4. Memory Leaks

**Symptoms:**
- Gradually increasing RAM usage
- Slow performance over time
- OOM (Out of Memory) kills

**Root Causes:**
- Large DataFrames not released
- Unclosed database connections
- Growing cache without eviction

**Diagnosis:**
```bash
# Monitor memory usage
top -p $(pgrep -f "atb live")

# Or use memory profiler
pip install memory_profiler
python -m memory_profiler your_script.py
```

**Quick Fix:**
```bash
# Restart trading engine
# Positions and balance are preserved in database
atb live ml_basic --symbol BTCUSDT --resume-from-last-balance
```

**Long-term Fix:**
- Review caching TTL settings
- Ensure DataFrame cleanup after processing
- Regular restarts (e.g., daily at low-volume hours)

**Prevention:**
- Set cache size limits
- Monitor memory usage
- Scheduled restarts (cron job)

---

### 5. Stop Loss Not Triggered

**Symptoms:**
- Position loss exceeds expected stop loss
- Stop orders not filled
- Logs show stop price reached but no exit

**Root Causes:**
- Stop loss not set correctly
- Extreme volatility/slippage
- Engine processing delay
- Exchange order rejection

**Diagnosis:**
```bash
# Check active positions
atb db query "SELECT * FROM positions WHERE status = 'OPEN';"

# Check stop loss values
atb db query "SELECT symbol, entry_price, stop_loss, current_price FROM positions;"

# Review execution logs
grep "stop_loss" logs/trading.log | tail -50
```

**Immediate Action:**
1. **Manual exit:** Close position manually if risk is critical
2. **Verify:** Check if stop loss order exists on exchange
3. **Adjust:** Set tighter stops if needed

**Prevention:**
- Always validate stop loss on position entry
- Use market orders for critical exits (enabled by default)
- Monitor position unrealized PnL continuously
- Enable trailing stops for better protection

---

### 6. Unexpected Balance Decrease

**Symptoms:**
- Balance lower than expected
- No corresponding trades logged
- Discrepancy with exchange account

**Root Causes:**
- Untracked fees
- Database sync issues
- Manual trades on exchange
- Calculation errors

**Diagnosis:**
```bash
# Check account history
atb db query "SELECT * FROM account_balances ORDER BY timestamp DESC LIMIT 20;"

# Check trade history
atb db query "SELECT * FROM trades ORDER BY exit_time DESC LIMIT 20;"

# Calculate expected balance
atb db query "SELECT initial_balance,
    (SELECT SUM(pnl) FROM trades WHERE session_id = trading_sessions.id) as total_pnl
FROM trading_sessions ORDER BY start_time DESC LIMIT 1;"

# Sync with exchange
atb live sync-balance --provider binance
```

**Recovery:**
- If database out of sync: Use `--resume-from-last-balance`
- If calculation error: Review PnL calculation code
- If manual trades: Import them into database

**Prevention:**
- Regular balance reconciliation (hourly)
- Automated alerts on large balance changes
- Avoid manual trading during bot operation

---

## Troubleshooting Procedures

### Debugging Checklist

When issues occur, follow this systematic approach:

#### Step 1: Identify the Symptom
- [ ] What is the observable problem?
- [ ] When did it start?
- [ ] Is it intermittent or persistent?
- [ ] Which component is affected?

#### Step 2: Check Logs
```bash
# View recent logs
tail -100 logs/trading.log

# Search for errors
grep -i error logs/trading.log | tail -50

# Check specific component
grep "LiveTradingEngine" logs/trading.log | tail -50
```

#### Step 3: Verify System Health
- [ ] Database connectivity (`atb db verify`)
- [ ] API connectivity (test providers)
- [ ] Disk space (`df -h`)
- [ ] Memory usage (`free -h`)
- [ ] Network connectivity (`ping api.binance.com`)

#### Step 4: Review Recent Changes
- [ ] Recent code deployments?
- [ ] Configuration changes?
- [ ] Strategy modifications?
- [ ] Infrastructure updates?

#### Step 5: Isolate the Issue
- [ ] Can you reproduce it?
- [ ] Does it happen in backtest mode?
- [ ] Does it happen with different symbols?
- [ ] Does it happen at specific times?

### Common Error Messages

#### "Maximum drawdown exceeded"
```
ERROR: Maximum drawdown exceeded (25.3%). Stopping backtest.
```

**Meaning:** Strategy lost more than configured max drawdown threshold.

**Action:**
1. Review strategy performance
2. Adjust risk parameters if needed
3. Consider reducing position sizes
4. Analyze what caused the drawdown

#### "No module named 'onnxruntime'"
```
ModuleNotFoundError: No module named 'onnxruntime'
```

**Meaning:** Missing dependency for ML model inference.

**Action:**
```bash
.venv/bin/pip install onnxruntime --timeout 1000
```

#### "Database connection refused"
```
psycopg2.OperationalError: could not connect to server: Connection refused
```

**Meaning:** PostgreSQL is not running or not accepting connections.

**Action:**
```bash
# Start PostgreSQL
docker compose up -d postgres
# Or:
systemctl start postgresql

# Verify
atb db verify
```

#### "Rate limit exceeded (429)"
```
requests.exceptions.HTTPError: 429 Client Error: Too Many Requests
```

**Meaning:** API rate limit hit.

**Action:**
1. Wait 60 seconds
2. Increase check interval
3. Review API usage patterns
4. Implement request throttling

---

## Recovery Procedures

### Recovering from Crash

#### Live Engine Crash

1. **Assess damage:**
```bash
# Check last known state
atb db query "SELECT * FROM trading_sessions ORDER BY start_time DESC LIMIT 1;"

# Check open positions
atb db query "SELECT * FROM positions WHERE status = 'OPEN';"

# Get current balance from exchange
atb live sync-balance --provider binance
```

2. **Close at-risk positions (if needed):**
```bash
# If positions are at high risk, close manually on exchange
# Then record in database:
atb db close-position --symbol BTCUSDT --exit-price 50000 --exit-reason "manual_close"
```

3. **Restart engine:**
```bash
# Resume with last known balance
atb live ml_basic --symbol BTCUSDT --resume-from-last-balance --paper-trading

# Verify everything looks correct before enabling live trading
# Then restart without --paper-trading
```

4. **Verify recovery:**
- Check positions match exchange
- Verify balance is correct
- Monitor for 1-2 hours

#### Database Corruption

1. **Stop all services:**
```bash
# Stop trading engine
pkill -f "atb live"

# Stop database
docker compose stop postgres
```

2. **Restore from backup:**
```bash
# Find latest backup
ls -lth backups/

# Restore (adjust date accordingly)
psql -h localhost -U trading_bot -d ai_trading_bot < backups/backup_2025-11-21.sql
```

3. **Verify integrity:**
```bash
atb db verify
atb db query "SELECT COUNT(*) FROM trades;"
```

4. **Restart services:**
```bash
docker compose up -d postgres
atb live ml_basic --symbol BTCUSDT --resume-from-last-balance
```

### Emergency Shutdown

If you need to emergency stop all trading:

```bash
# Stop all trading processes
pkill -9 -f "atb live"

# Manually close all positions on exchange
# (Log into exchange web interface)

# Mark positions as closed in database
atb db emergency-close-all --session-id $(atb db query "SELECT id FROM trading_sessions ORDER BY start_time DESC LIMIT 1;")

# Verify no open positions
atb db query "SELECT * FROM positions WHERE status = 'OPEN';"
```

---

## Performance Optimization

### Database Optimization

```bash
# Vacuum database (reclaim space, update stats)
psql -h localhost -U trading_bot -d ai_trading_bot -c "VACUUM ANALYZE;"

# Reindex tables
psql -h localhost -U trading_bot -d ai_trading_bot -c "REINDEX DATABASE ai_trading_bot;"

# Check slow queries
atb db query "SELECT query, mean_exec_time FROM pg_stat_statements ORDER BY mean_exec_time DESC LIMIT 10;"
```

### Cache Optimization

```bash
# Clear old cache entries
atb data cache-manager clear-old --hours 48

# Prefill cache for faster backtests
atb data prefill-cache --symbols BTCUSDT ETHUSDT --timeframes 1h 4h --years 2
```

### Log Rotation

```bash
# Rotate logs manually
mv logs/trading.log logs/trading.log.$(date +%Y%m%d)
gzip logs/trading.log.$(date +%Y%m%d)

# Keep only last 30 days
find logs/ -name "*.gz" -mtime +30 -delete
```

---

## Emergency Procedures

### Critical Scenarios

#### Scenario 1: Exchange Account Compromised

1. **Immediately:** Change API keys on exchange
2. **Revoke:** Old API keys
3. **Update:** New keys in configuration
4. **Verify:** No unauthorized trades
5. **Review:** All recent activity

#### Scenario 2: Runaway Trading (Too Many Trades)

1. **Stop engine immediately:**
```bash
pkill -9 -f "atb live"
```

2. **Check positions:**
```bash
atb db query "SELECT * FROM positions WHERE status = 'OPEN';"
```

3. **Close all positions on exchange manually**

4. **Investigate root cause:**
- Review strategy code
- Check for infinite loops
- Verify signal logic

#### Scenario 3: Massive Drawdown (>20% in 1 hour)

1. **Stop trading immediately**
2. **Close all positions**
3. **Analyze what happened:**
   - Market event?
   - Strategy bug?
   - Data issue?
4. **Don't restart until issue identified**

---

## Maintenance Tasks

### Daily Tasks

- [ ] Check system health (`atb health check`)
- [ ] Review overnight trade activity
- [ ] Verify database backups completed
- [ ] Check disk space (`df -h`)
- [ ] Review error logs

### Weekly Tasks

- [ ] Analyze strategy performance
- [ ] Database vacuum (`VACUUM ANALYZE`)
- [ ] Clear old cache entries (>7 days)
- [ ] Review and rotate logs
- [ ] Update dependencies (if needed)
- [ ] Test backup restoration

### Monthly Tasks

- [ ] Full system audit
- [ ] Performance review
- [ ] Update ML models if applicable
- [ ] Review and optimize risk parameters
- [ ] Security review (API keys rotation)
- [ ] Test disaster recovery procedures

---

## Monitoring & Alerts

### Recommended Alerts

Configure these alerts for production:

| Alert | Condition | Severity | Action |
|-------|-----------|----------|--------|
| Engine Down | Process not running > 5min | Critical | Restart immediately |
| Database Down | Connection failed > 2min | Critical | Investigate & restart |
| High Drawdown | Drawdown > 15% | High | Review positions |
| API Errors | Error rate > 5% | Medium | Check API status |
| Memory High | Usage > 90% | Medium | Restart engine |
| Disk Full | Space < 10% | High | Clear logs/cache |
| No Trades 24h | No activity in 24h | Low | Check strategy |

### Setting Up Alerts

Example using a simple monitoring script:

```bash
#!/bin/bash
# monitor.sh - Add to crontab to run every 5 minutes

# Check if engine is running
if ! pgrep -f "atb live" > /dev/null; then
    echo "CRITICAL: Trading engine is down!" | mail -s "Trading Alert" admin@example.com
fi

# Check database
if ! atb db verify > /dev/null 2>&1; then
    echo "CRITICAL: Database is down!" | mail -s "Trading Alert" admin@example.com
fi

# Check drawdown
DRAWDOWN=$(atb db query "SELECT (peak_balance - current_balance) / peak_balance FROM trading_sessions ORDER BY start_time DESC LIMIT 1;")
if (( $(echo "$DRAWDOWN > 0.15" | bc -l) )); then
    echo "WARNING: High drawdown detected: $DRAWDOWN" | mail -s "Trading Alert" admin@example.com
fi
```

Add to crontab:
```bash
*/5 * * * * /path/to/monitor.sh
```

---

## Database Operations

### Backup & Restore

#### Creating Backups

```bash
# Full backup
pg_dump -h localhost -U trading_bot ai_trading_bot > backups/backup_$(date +%Y%m%d_%H%M%S).sql

# Compressed backup
pg_dump -h localhost -U trading_bot ai_trading_bot | gzip > backups/backup_$(date +%Y%m%d_%H%M%S).sql.gz
```

#### Automated Backups

Create a backup script (`backup.sh`):
```bash
#!/bin/bash
BACKUP_DIR=/path/to/backups
DATE=$(date +%Y%m%d_%H%M%S)

# Create backup
pg_dump -h localhost -U trading_bot ai_trading_bot | gzip > "$BACKUP_DIR/backup_$DATE.sql.gz"

# Keep only last 30 days
find "$BACKUP_DIR" -name "*.sql.gz" -mtime +30 -delete

# Log
echo "$DATE: Backup completed" >> "$BACKUP_DIR/backup.log"
```

Add to crontab (daily at 2 AM):
```bash
0 2 * * * /path/to/backup.sh
```

#### Restoring Backups

```bash
# Stop trading engine first
pkill -f "atb live"

# Restore from compressed backup
gunzip -c backups/backup_20251121_020000.sql.gz | psql -h localhost -U trading_bot -d ai_trading_bot

# Verify restoration
atb db verify

# Restart engine
atb live ml_basic --symbol BTCUSDT --resume-from-last-balance
```

### Database Queries

#### Useful Queries

**Recent trades:**
```sql
SELECT symbol, side, entry_price, exit_price, pnl, exit_reason
FROM trades
ORDER BY exit_time DESC
LIMIT 10;
```

**Current positions:**
```sql
SELECT symbol, side, entry_price, size, unrealized_pnl
FROM positions
WHERE status = 'OPEN';
```

**Performance summary:**
```sql
SELECT
    COUNT(*) as total_trades,
    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as win_rate,
    SUM(pnl) as total_pnl,
    AVG(pnl) as avg_pnl
FROM trades
WHERE exit_time > NOW() - INTERVAL '30 days';
```

**Daily PnL:**
```sql
SELECT
    DATE(exit_time) as trade_date,
    COUNT(*) as trades,
    SUM(pnl) as daily_pnl
FROM trades
WHERE exit_time > NOW() - INTERVAL '7 days'
GROUP BY DATE(exit_time)
ORDER BY trade_date DESC;
```

---

## Contact & Escalation

### Support Tiers

**Tier 1 - Self-Service:**
- This runbook
- Check logs
- Run diagnostics
- Review documentation

**Tier 2 - Community:**
- GitHub Issues: https://github.com/your-repo/ai-trading-bot/issues
- Discord/Slack community

**Tier 3 - Emergency:**
- For critical production issues
- Contact: [Your escalation procedure]

---

## Appendix

### Log Levels

- `DEBUG`: Detailed diagnostic info
- `INFO`: General operational messages
- `WARNING`: Warning messages (non-critical)
- `ERROR`: Error messages (operation failed)
- `CRITICAL`: Critical errors (system failure)

### Configuration Files

- `.env` - Environment variables (DO NOT COMMIT)
- `src/config/constants.py` - Default constants
- Strategy configs - In strategy files

### Important Directories

- `logs/` - Application logs
- `backups/` - Database backups
- `src/ml/models/` - Trained ML models
- `.cache/` - Data cache
- `docs/` - Documentation

---

**Remember:** When in doubt, stop trading, secure positions, and investigate thoroughly before resuming.
