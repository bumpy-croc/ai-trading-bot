---
name: live-trading-monitor
description: |
  Full health audit of the live trading system. Analyses Railway deployment logs, database trading sessions and trades, strategy signal generation, PnL correctness, position integrity, and applies fixes for critical bugs. Does NOT optimise strategy performance or restart deployments unless critical errors are found.
  Trigger phrases: monitor live trading, check live trading health, audit trading system, trading health check, check railway trading, live trading issues, trading bot health, check for trading bugs, analyse trading performance, live monitor, trading monitor, check trading session, is the bot healthy
---

# Live Trading Monitor

You are responsible for the **operational health** of the live trading system. Your mandate:

- Ensure trades are **entering and exiting correctly** with no critical bugs
- Verify **PnL calculations** are accurate and consistent
- Detect **position integrity** issues (stuck positions, duplicates, orphaned records)
- Confirm the **strategy is generating signals** and the engine is alive
- Apply **code/config fixes** for verified bugs
- **Restart Railway only for critical, unrecoverable errors** (not as a first response)

Do NOT attempt to improve strategy returns, adjust risk parameters, or change trading logic to improve performance. Health only.

---

## Phase 1 — Railway Status & Logs

### 1.1 Deployment Status

Use the Railway MCP server to check the current deployment:

```
mcp__Railway__list-deployments  (project: innovative-transformation, environment: development)
mcp__Railway__list-services
```

Note: deployment environment names are `development` (develop branch), `staging`, `main` (production).

### 1.2 Fetch Recent Logs

```
mcp__Railway__get-logs  (service: the trading service, lines: 200)
```

Scan logs for:
- `ERROR` or `CRITICAL` log lines — record each with timestamp and message
- `Unknown strategy` — strategy not registered in runner
- `Failed to load model` — ML model missing or corrupt
- `database` connection errors — DB unreachable
- `session` creation failures
- Health check failures (`/health` endpoint returning non-200)
- Loop iteration timestamps — if last `Trading loop` log is >5 minutes old, the engine may be hung
- `MemoryError` or OOM kills

Record all findings. Classify each as:
- **CRITICAL** — engine not running, data loss risk, trades not executing
- **WARNING** — degraded operation, worth monitoring
- **INFO** — normal operation noise

---

## Phase 2 — Database Health

Connect via: `export DATABASE_URL=postgresql://trading_bot:dev_password_123@localhost:5432/ai_trading_bot`

Or in Railway environment, use the `DATABASE_URL` env var.

Run database queries using `atb` CLI or psql. Prefer running via:
```bash
atb db verify
```

Then investigate with targeted SQL. Use `docker compose exec postgres psql -U trading_bot -d ai_trading_bot -c "..."` for local DB, or Railway's DB connection for production.

### 2.1 Active Session Check

```sql
SELECT id, strategy_name, symbol, timeframe, trading_mode, status,
       initial_balance, current_balance, created_at,
       (current_balance - initial_balance) as pnl_dollar,
       EXTRACT(EPOCH FROM (NOW() - created_at))/3600 as age_hours
FROM trading_sessions
WHERE status = 'active'
ORDER BY created_at DESC;
```

Checks:
- Is there exactly one active session? Zero = engine not running. Multiple = session leak.
- Is `current_balance` updating? If unchanged for >1 hour with trades expected, may be stale.
- Is `trading_mode` = `paper` (expected for development/staging)?

### 2.2 Recent Trades Analysis

```sql
SELECT
  t.id, t.side, t.entry_price, t.exit_price, t.size, t.quantity,
  t.pnl, t.pnl_percent, t.commission, t.exit_reason,
  t.entry_time, t.exit_time,
  EXTRACT(EPOCH FROM (t.exit_time - t.entry_time))/60 as duration_minutes,
  t.session_id
FROM trades t
WHERE t.entry_time > NOW() - INTERVAL '7 days'
  AND t.source IN ('PAPER', 'LIVE')
ORDER BY t.entry_time DESC
LIMIT 50;
```

Check for:
- **Duplicate entries**: Multiple trades entering within seconds of each other in same session
- **Instant exits**: `duration_minutes` < 1 (likely a bug, not a legitimate trade)
- **Missing exit data**: `exit_price` = 0 or NULL
- **PnL sign mismatch**: `pnl` and `pnl_percent` should have same sign
- **PnL vs price inconsistency**: For LONG, `pnl` should be positive when `exit_price > entry_price`
- **All exits = ENGINE_SHUTDOWN**: Normal stop-loss/take-profit never triggers = SL/TP not working
- **Abnormal PnL%**: > ±50% on a single trade = likely a calculation error
- **Zero commission**: On live trades this is suspicious

### 2.3 PnL Consistency Check

Verify dollar PnL matches percentage PnL:

```sql
SELECT
  id, side, entry_price, exit_price, quantity, pnl, pnl_percent, commission,
  -- Expected dollar PnL (LONG)
  CASE WHEN side = 'LONG' THEN (exit_price - entry_price) * quantity - commission
       WHEN side = 'SHORT' THEN (entry_price - exit_price) * quantity - commission
  END as expected_pnl_dollar,
  -- Discrepancy
  ABS(pnl - CASE WHEN side = 'LONG' THEN (exit_price - entry_price) * quantity - commission
                 WHEN side = 'SHORT' THEN (entry_price - exit_price) * quantity - commission
            END) as pnl_discrepancy
FROM trades
WHERE entry_time > NOW() - INTERVAL '7 days'
  AND source IN ('PAPER', 'LIVE')
  AND quantity IS NOT NULL
ORDER BY pnl_discrepancy DESC NULLS LAST
LIMIT 20;
```

Flag any trade where `pnl_discrepancy > 0.01` (1 cent) as a PnL calculation bug.

### 2.4 Open Positions Audit

```sql
SELECT
  p.id, p.symbol, p.side, p.status,
  p.entry_price, p.current_price, p.quantity, p.size,
  p.unrealized_pnl, p.stop_loss, p.take_profit,
  p.entry_time,
  EXTRACT(EPOCH FROM (NOW() - p.entry_time))/3600 as age_hours,
  p.session_id
FROM positions p
WHERE p.status = 'OPEN'
ORDER BY p.entry_time;
```

Checks:
- **Stuck positions**: `age_hours > 24` without SL/TP — may be orphaned
- **Missing SL**: `stop_loss IS NULL` on a live trade is a risk management failure
- **Multiple open positions in same direction**: Is this expected by the strategy?
- **Orphaned positions**: `session_id` refers to a closed session but position is still OPEN

```sql
-- Orphaned position check
SELECT p.id, p.symbol, p.status as pos_status, s.status as session_status, p.entry_time
FROM positions p
JOIN trading_sessions s ON p.session_id = s.id
WHERE p.status = 'OPEN' AND s.status != 'active';
```

### 2.5 Session Continuity Check

```sql
SELECT
  id, strategy_name, status, created_at, updated_at,
  initial_balance, current_balance,
  EXTRACT(EPOCH FROM (updated_at - created_at))/3600 as session_duration_hours
FROM trading_sessions
ORDER BY created_at DESC
LIMIT 10;
```

Look for:
- **Balance resets**: Each new session starting at $1000 with no continuity is expected in paper mode, but flag if it looks like funds aren't persisting as intended
- **Very short sessions**: Duration < 5 minutes = crash loop
- **High session count**: >3 sessions per day = likely crash-looping

---

## Phase 3 — Strategy Signal Health

### 3.1 Signal Generation Check

Look in recent Railway logs for signal activity:
- `HyperGrowth_signals` entries — are signals being generated?
- `signal_strength`, `regime`, `confidence` log fields — are values reasonable?
- `No signal` or `HOLD` messages — acceptable, but flag if 100% of all recent cycles are HOLD

If the engine has been running >6 hours with zero trades and zero non-HOLD signals, investigate:

1. Check if the ML model is loading:
```bash
atb live-control list-models
```

2. Verify the `latest` symlink points to a valid model:
```bash
ls -la src/ml/models/BTCUSDT/sentiment/
```

3. Check the feature schema is intact:
```bash
cat src/ml/models/BTCUSDT/sentiment/latest/feature_schema.json | python -m json.tool | head -30
```

### 3.2 Engine Loop Health

From Railway logs, check the cadence of trading loop iterations. The engine checks every 60 seconds. If logs show:
- `Trading loop iteration` timestamps > 5 minutes apart → engine is hanging
- No loop logs at all in last 10 minutes → engine has stopped

---

## Phase 4 — Issue Classification & Fixes

After completing phases 1-3, compile all findings.

### Classification

| Severity | Definition | Action |
|----------|-----------|--------|
| **CRITICAL** | Trades not executing, PnL miscalculated, positions stuck/orphaned, engine crashed | Fix immediately, may restart if unrecoverable |
| **WARNING** | Degraded signals, minor inconsistencies, single anomalous trade | Investigate, fix if code change needed |
| **INFO** | Normal operational noise | Document only |

### Fix Decision Matrix

**Apply code fix when:**
- PnL calculation formula is wrong in source code
- SL/TP not being set on positions
- Duplicate position entries caused by a bug in the engine
- Signal generator misconfigured (wrong model path, wrong features)

**Apply config fix when:**
- Environment variable missing or wrong
- Wrong strategy name in deployment command

**Restart Railway deployment when:**
- Engine has definitively crashed (not running, confirmed via logs AND no heartbeat)
- OOM kill confirmed (not recoverable without restart)
- Database connection permanently lost and not reconnecting

**Do NOT restart when:**
- There are zero trades (strategy simply hasn't signalled yet)
- Win rate is low (not a health issue)
- You just want to "see if it fixes itself"

### Applying Fixes

For code fixes:
1. Read the relevant file carefully before editing
2. Make the minimal targeted change
3. Run `atb dev quality` to check for regressions
4. Run relevant unit tests: `atb test unit`
5. Commit with a clear message: `fix: <description of bug>`
6. Push to trigger Railway redeploy (this IS the deployment restart, intentional)

For Railway restarts (critical only):
```
mcp__Railway__deploy  (with current deployment ID)
```

Or via CLI:
```bash
railway redeploy --environment development
```

---

## Phase 5 — Summary Report

Present a structured report:

```
## Live Trading Health Report — [timestamp]

### System Status: [HEALTHY | DEGRADED | CRITICAL]

### Railway Deployment
- Status: [running/stopped/crashed]
- Last log activity: [timestamp]
- Critical errors found: [list or "none"]

### Active Session
- Session ID: [id]
- Strategy: [name]
- Mode: [paper/live]
- Age: [hours]
- Balance: $[current] (started $[initial], PnL: $[diff])

### Trade Health (last 7 days)
- Total trades: [n]
- PnL issues: [list or "none"]
- Exit reason distribution: [SL: n%, TP: n%, ENGINE_SHUTDOWN: n%, other: n%]
- Duplicate/anomalous trades: [list or "none"]

### Position Integrity
- Open positions: [n]
- Orphaned positions: [list or "none"]
- Missing SL: [list or "none"]

### Signal Health
- Signals generating: [yes/no]
- ML model loaded: [yes/no]

### Actions Taken
- [List of any fixes applied]
- [List of restarts triggered, if any, with justification]

### Recommended Next Steps
- [Any outstanding issues that need attention]
```

---

## Important Context

- **Project**: `innovative-transformation` on Railway
- **Environments**: `development` (develop branch), `staging`, `main` (production — never destructive ops)
- **Strategy**: `hyper_growth` (HyperGrowth) on BTCUSDT 1h
- **Current mode**: Paper trading (session #106+)
- **DB credentials (local)**: `postgresql://trading_bot:dev_password_123@localhost:5432/ai_trading_bot`
- **Railway config**: `railway.json` — startCommand is `atb live-health hyper_growth`, restart policy ON_FAILURE max 3
- **Known historical issues**: Duplicate position entries at session start (3 trades opening within seconds), all exits via ENGINE_SHUTDOWN (SL/TP not triggering), consistent SHORT bias in certain price ranges
