# Reconciliation Chaos Test Strategy

## Context

PR #576 adds a comprehensive Binance reconciliation system (order journal, startup/periodic reconciliation, close-only mode, audit events). Before deploying to live trading, we need to validate in a controlled paper-trading environment that generates enough trade volume to exercise every reconciliation path.

Existing strategies (HyperGrowth, MlBasic) trade too infrequently — one position held for hours/days. We need a **chaos test strategy** that trades every few candles, paired with a **smoke test script** that simulates failure scenarios.

**Branch:** `feat/reconciliation-chaos-tests` (from `develop`)
**Depends on:** PR #576 (reconciliation system) being merged first

---

## Part 1: `chaos_test` Strategy

### New file: `src/strategies/chaos_test.py`

Factory function `create_chaos_test_strategy()` that composes:

**ChaosSignalGenerator** — Generates frequent alternating signals:
- Uses RSI as base indicator (warmup_period=20)
- RSI < 35 → BUY, RSI > 65 → SELL
- **Forced alternation**: after `max_hold_candles` (default 3), generates the opposite signal regardless of RSI — guarantees constant trade flow
- Tracks `_candles_in_position` and `_last_direction` internally
- High confidence (0.9) to ensure trades execute past any confidence threshold

**Risk overrides** (via `get_risk_overrides()`):
- Tight stop-loss: 1%
- Partial exits at 0.5%, 1% (exercise partial exit path quickly)
- Scale-in at 0.3% dip (exercise scale-in)
- Trailing stop activation at 0.5%
- Small position size: 2% of balance (~$20 per trade on $1000)

**Reuse existing components:**
- `FlatRiskManager` from `src/strategies/hyper_growth.py` — fixed fraction, no ML dependency
- `FixedFractionSizer` from `src/strategies/components/position_sizer.py` — 0.02 base fraction
- `EnhancedRegimeDetector` from `src/strategies/components/regime_context.py` — for regime metadata

### Modify: `src/engines/live/runner.py`

Register `chaos_test` in the strategies dict:
```python
"chaos_test": create_chaos_test_strategy,
```

### New file: `tests/unit/strategies/test_chaos_test.py`

Unit tests:
- Signal alternation after max_hold_candles
- RSI-based signals at extremes
- Warmup period returns HOLD
- Risk overrides contain expected partial exit/scale-in config
- High confidence output

---

## Part 2: Smoke Test Script

### New file: `scripts/chaos_smoke_test.py`

Standalone script that orchestrates failure scenario testing. Uses subprocess to start/kill the bot and SQLAlchemy to query the DB directly.

**Phase 1 — Journal Validation** (paper mode, automated):
1. Start bot: `atb live chaos_test --symbol BTCUSDT --timeframe 1m --paper-trading --check-interval 30`
2. Poll DB until N trades complete (default 5, configurable)
3. Query `orders` table: verify all orders have `client_order_id`, status lifecycle is valid (PENDING→FILLED or PENDING→CANCELLED)
4. Query `reconciliation_audit_events`: verify empty (no corrections needed in clean run)
5. Kill bot, print pass/fail report

**Phase 2 — Crash Recovery** (paper mode, automated):
1. Start bot, poll DB until open position exists
2. `os.kill(pid, signal.SIGKILL)` — hard kill, no cleanup
3. Wait 2s, restart bot with same args
4. Poll DB: verify position recovered (same entry_price, same session or new session with balance carry-over)
5. Wait for bot to complete 1 more trade cycle
6. Kill bot, print pass/fail report

**Phase 3 — Balance Integrity** (paper mode, automated):
1. Start bot, run until 10+ trades complete
2. Query all completed trades from DB
3. Calculate expected balance: `initial + sum(pnl) - sum(fees)`
4. Compare against `account_balances` table latest value
5. Report drift (should be < $0.01)

**Phase 4 — SL Reconciliation** (live testnet only, manual guidance):
- Print instructions for manual testing on Binance testnet
- Start bot with `--testnet --live-trading --i-understand-the-risks`
- User cancels SL on Binance UI
- Script polls DB for audit events showing SL re-placement or close-only activation

### CLI interface:
```bash
# Local mode (spawns bot as subprocess)
python scripts/chaos_smoke_test.py --phase journal --trades 5 --timeout 300
python scripts/chaos_smoke_test.py --phase crash --timeout 120
python scripts/chaos_smoke_test.py --phase balance --trades 10 --timeout 600
python scripts/chaos_smoke_test.py --phase all --timeout 900

# Railway mode (bot running as separate service, DB-only validation)
python scripts/chaos_smoke_test.py --phase journal --railway --timeout 300
python scripts/chaos_smoke_test.py --phase balance --railway --timeout 600
python scripts/chaos_smoke_test.py --phase crash --railway --timeout 120
```

### Two operating modes:
- **Local** (default): Spawns `atb live chaos_test ...` as subprocess. Manages full lifecycle — start, wait, kill, restart, verify. Good for dev machines.
- **Railway** (`--railway`): Assumes bot is running as a separate Railway service sharing the same PostgreSQL. Connects directly to DB via `DATABASE_URL`. For crash tests, restarts the bot service via `railway service restart`. Good for production-like validation.

---

## Part 3: Running on Railway

The smoke test needs to run **inside Railway** alongside the trading bot since that's where the production DB and deployment environment live. There are two approaches depending on the test phase:

### Approach A: Swap the start command (for strategy testing)

To run the chaos_test strategy instead of hyper_growth on Railway:

1. **Use the development environment** (synced to `develop` branch) — never test on production
2. Override the start command via Railway variables or a temporary `railway.json` change:
   ```json
   {
     "build": {"builder": "DOCKERFILE"},
     "deploy": {
       "startCommand": "atb live-health chaos_test --symbol BTCUSDT --timeframe 1m --check-interval 30 --max-position 0.02",
       "healthcheckPath": "/health"
     }
   }
   ```
3. Push to `develop` → Railway auto-deploys with chaos_test strategy
4. Monitor via `railway logs` — should see trades every ~3 candles (3 minutes on 1m timeframe)
5. Query the Railway-connected PostgreSQL DB to verify journal entries
6. Revert `railway.json` to hyper_growth when done

### Approach B: One-off smoke test command (for crash/journal/balance testing)

Railway supports one-off commands via `railway run`:

```bash
# Run the smoke test script directly on Railway infrastructure
# This connects to the Railway PostgreSQL database automatically
railway run --environment development python scripts/chaos_smoke_test.py --phase journal --trades 5 --timeout 300

# Crash recovery test (starts bot, kills it, restarts, verifies)
railway run --environment development python scripts/chaos_smoke_test.py --phase crash --timeout 120

# Balance integrity test
railway run --environment development python scripts/chaos_smoke_test.py --phase balance --trades 10 --timeout 600
```

**Important:** `railway run` executes commands in the Railway environment with access to all env vars (DATABASE_URL, BINANCE_API_KEY, etc.) but as a one-off process, not a persistent service.

### Approach C: Dedicated smoke test service (for continuous validation)

For ongoing validation, add a second Railway service in the same project:

1. **Create new service** in Railway dashboard: `chaos-smoke-test`
2. **Link to same repo** but with different start command
3. **Configure variables:**
   - Same DATABASE_URL (shared PostgreSQL)
   - `SMOKE_TEST_PHASE=journal` (or `all`)
   - `SMOKE_TEST_TRADES=10`
   - `SMOKE_TEST_TIMEOUT=600`
4. **Start command:** `python scripts/chaos_smoke_test.py --phase $SMOKE_TEST_PHASE --trades $SMOKE_TEST_TRADES --timeout $SMOKE_TEST_TIMEOUT`
5. Service runs the test, logs results, then exits (Railway shows it as completed)

### Railway-Specific Requirements for the Smoke Test Script

The script needs to handle Railway's environment:

1. **No subprocess spawning for the bot** — on Railway, the bot runs as a separate service. The smoke test should connect to the **same database** and validate state, not start the bot itself.
2. **Two operating modes:**
   - **Local mode** (default): Spawns bot as subprocess, manages lifecycle, kills/restarts
   - **Railway mode** (`--railway`): Assumes bot is already running as a separate service. Only does DB validation — polls DB for trades, checks journal integrity, verifies balance. Crash recovery is tested by redeploying the bot service via `railway` CLI or MCP.
3. **DATABASE_URL from environment**: On Railway, `DATABASE_URL` is auto-injected. The script connects directly.
4. **Crash recovery on Railway**: Instead of SIGKILL, use `railway` CLI to restart the service:
   ```bash
   # From the smoke test service or local machine:
   railway service restart --environment development
   ```
   Then poll DB to verify position recovery after restart.

### Modified smoke test CLI for Railway:

```bash
# Local testing (spawns bot subprocess)
python scripts/chaos_smoke_test.py --phase all --timeout 900

# Railway testing (bot already running as separate service)
python scripts/chaos_smoke_test.py --phase journal --railway --timeout 300
python scripts/chaos_smoke_test.py --phase balance --railway --timeout 600

# Railway crash test (restarts bot service, then validates)
python scripts/chaos_smoke_test.py --phase crash --railway --timeout 120
```

### Railway Setup Steps (one-time)

1. Merge PR #576 (reconciliation) into `develop`
2. Merge this PR (chaos tests) into `develop`
3. In Railway dashboard for `development` environment:
   - Change start command to use `chaos_test` strategy temporarily
   - Or create a second service for the smoke test script
4. Deploy and monitor via `railway logs`
5. Run smoke tests via `railway run` or dedicated service
6. Revert to `hyper_growth` strategy when validation complete

---

## Files Summary

| File | Action | Purpose |
|------|--------|---------|
| `src/strategies/chaos_test.py` | CREATE | ChaosSignalGenerator + factory |
| `src/engines/live/runner.py` | MODIFY | Register chaos_test strategy |
| `scripts/chaos_smoke_test.py` | CREATE | Automated smoke test orchestrator |
| `tests/unit/strategies/test_chaos_test.py` | CREATE | Unit tests for signal generator |

---

## Verification

1. `pytest tests/unit/strategies/test_chaos_test.py -v` — unit tests pass
2. `atb live chaos_test --symbol BTCUSDT --timeframe 1m --paper-trading --check-interval 30` — runs, generates trades every ~3 minutes
3. `python scripts/chaos_smoke_test.py --phase journal --trades 3 --timeout 180` — journal validation passes
4. `python scripts/chaos_smoke_test.py --phase crash --timeout 120` — crash recovery works
5. `python scripts/chaos_smoke_test.py --phase balance --trades 5 --timeout 300` — no balance drift
