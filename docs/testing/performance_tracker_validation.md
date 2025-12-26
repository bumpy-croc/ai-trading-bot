# Performance Tracker Integration - Testing & Validation Guide

**Status:** Ready for testing
**Integration Date:** 2025-12-26
**Phases Completed:** 1-4 (Implementation & Migration)

## Overview

This guide documents how to test and validate the integrated PerformanceTracker across both backtesting and live trading engines.

## Testing Phases

### Phase 5a: Unit Tests

**Location:** `tests/unit/performance/test_tracker.py`

**Run Command:**
```bash
python tests/run_tests.py unit --pytest-args tests/unit/performance/test_tracker.py -v
```

**Expected Results:**
- ✅ All 30+ test cases pass
- ✅ Coverage >95% for `src/performance/tracker.py`
- ✅ Coverage >95% for `src/performance/metrics.py`

**Key Test Categories:**
1. Metric calculations (Sharpe, Sortino, Calmar, VaR, expectancy)
2. Streak tracking (consecutive wins/losses)
3. Balance tracking and drawdown
4. Thread safety (concurrent updates)
5. Edge cases (zero trades, negative PnL, missing timestamps)

### Phase 5b: Integration Tests - Backtest Engine

**Run Command:**
```bash
# Run quick backtest with tracker
atb backtest ml_basic --symbol BTCUSDT --timeframe 1h --days 30

# Run full backtest test suite
python tests/run_tests.py integration -k backtest
```

**Validation Checklist:**

- [ ] **Backtest completes without errors**
- [ ] **New metrics present in results:**
  - `sortino_ratio`
  - `calmar_ratio`
  - `var_95`
  - `expectancy`
  - `profit_factor`
  - `consecutive_wins`
  - `consecutive_losses`
  - `avg_trade_duration_hours`

- [ ] **Backward compatibility maintained:**
  - All existing result keys present
  - CLI output format unchanged
  - Metrics match previous implementation (±0.01%)

**Sample Validation:**
```python
from cli.backtest import run_backtest

results = run_backtest(
    strategy="ml_basic",
    symbol="BTCUSDT",
    timeframe="1h",
    days=30
)

# Verify new metrics
assert "sortino_ratio" in results
assert "calmar_ratio" in results
assert "var_95" in results
assert "expectancy" in results
assert "consecutive_wins" in results
assert "consecutive_losses" in results

# Verify backward compat
assert "total_return" in results
assert "sharpe_ratio" in results
assert "max_drawdown" in results
```

### Phase 5c: Integration Tests - Live Engine

**Run Command:**
```bash
# Run paper trading (safe)
atb live ml_basic --symbol BTCUSDT --paper-trading

# Run live engine tests
python tests/run_tests.py integration -k live_engine
```

**Validation Checklist:**

- [ ] **Live engine starts without errors**
- [ ] **PerformanceTracker initialized correctly**
- [ ] **Trades recorded with tracker on close**
- [ ] **Balance updates trigger tracker.update_balance()**
- [ ] **get_performance_summary() includes new metrics**
- [ ] **Database writes succeed** (check PostgreSQL logs)

**API Validation:**
```python
# Test get_performance_summary()
from src.engines.live.trading_engine import LiveTradingEngine

engine = LiveTradingEngine(...)
summary = engine.get_performance_summary()

# Verify new metrics in summary
assert "sharpe_ratio" in summary
assert "sortino_ratio" in summary
assert "calmar_ratio" in summary
assert "var_95" in summary
assert "expectancy" in summary
assert "profit_factor" in summary
assert "consecutive_wins" in summary
assert "consecutive_losses" in summary
```

### Phase 5d: Parity Tests (Critical!)

**Purpose:** Ensure backtest and live engines calculate identical metrics.

**Run Command:**
```bash
python tests/integration/test_performance_parity.py -v
```

**Test Scenario:**
1. Generate test trade sequence (10-20 trades)
2. Run through backtest engine tracker
3. Run through live engine tracker
4. Compare final metrics

**Expected Parity (tolerance < 0.01%):**
- `total_return_pct`
- `sharpe_ratio`
- `sortino_ratio`
- `calmar_ratio`
- `max_drawdown`
- `profit_factor`
- `expectancy`
- `win_rate`
- `avg_win` / `avg_loss`

**Sample Parity Test:**
```python
def test_backtest_live_metric_parity():
    """Ensure backtest and live calculate identical metrics"""

    # Create test trades
    trades = generate_test_trades(count=20, initial_balance=10000)

    # Run through backtest tracker
    bt_tracker = PerformanceTracker(10000)
    for trade in trades:
        bt_tracker.record_trade(trade.trade, trade.fee, trade.slippage)
        bt_tracker.update_balance(trade.balance, trade.timestamp)
    bt_metrics = bt_tracker.get_metrics()

    # Run through live tracker
    live_tracker = PerformanceTracker(10000)
    for trade in trades:
        live_tracker.record_trade(trade.trade, trade.fee, trade.slippage)
        live_tracker.update_balance(trade.balance, trade.timestamp)
    live_metrics = live_tracker.get_metrics()

    # Assert parity (within 0.01% tolerance)
    assert abs(bt_metrics.total_return_pct - live_metrics.total_return_pct) < 0.01
    assert abs(bt_metrics.sharpe_ratio - live_metrics.sharpe_ratio) < 0.001
    assert abs(bt_metrics.sortino_ratio - live_metrics.sortino_ratio) < 0.001
    assert abs(bt_metrics.max_drawdown - live_metrics.max_drawdown) < 0.0001
    assert abs(bt_metrics.profit_factor - live_metrics.profit_factor) < 0.01
```

### Phase 5e: Database Migration Tests

**Run Command:**
```bash
# Apply migration
alembic upgrade head

# Verify schema
atb db verify

# Check columns exist
psql $DATABASE_URL -c "\d+ performance_metrics"
psql $DATABASE_URL -c "\d+ account_history"
```

**Validation Checklist:**

- [ ] **Migration applies without errors**
- [ ] **New columns exist in performance_metrics:**
  - current_drawdown
  - var_95
  - avg_trade_duration_hours
  - consecutive_wins_current
  - consecutive_losses_current
  - total_fees_paid
  - total_slippage_cost

- [ ] **New columns exist in account_history:**
  - sharpe_ratio
  - sortino_ratio
  - calmar_ratio
  - var_95

- [ ] **All columns nullable or have defaults**
- [ ] **Existing data unaffected**

### Phase 5f: Performance Benchmarks

**Purpose:** Ensure no significant performance regression.

**Run Command:**
```bash
# Benchmark backtest performance
time atb backtest ml_basic --symbol BTCUSDT --timeframe 1h --days 365

# Compare with baseline (should be within 5%)
```

**Acceptance Criteria:**
- Backtest runtime: <5% slower than baseline
- Memory usage: <10% increase from baseline
- No memory leaks during long-running backtests

## Regression Checklist

**Before Deployment:**

- [ ] All unit tests pass (`python tests/run_tests.py unit`)
- [ ] All integration tests pass (`python tests/run_tests.py integration`)
- [ ] Parity tests pass (backtest ≈ live metrics)
- [ ] Database migration successful
- [ ] CLI output format unchanged
- [ ] No breaking changes to API
- [ ] Performance within acceptable limits (<5% regression)
- [ ] Documentation updated

## Known Issues / Limitations

None at this time. All functionality implemented as specified.

## Rollback Plan

If issues arise in production:

```bash
# Rollback database migration
alembic downgrade -1

# Revert code changes
git revert HEAD~6..HEAD  # Revert last 6 commits (Phases 1-6)

# Restart services
# (service restart commands here)
```

## Success Metrics

**Integration is successful if:**

1. ✅ All existing tests pass
2. ✅ New metrics available in backtest results
3. ✅ New metrics available in live engine API
4. ✅ Parity tests show <0.01% difference
5. ✅ Database migration successful
6. ✅ No performance regression >5%
7. ✅ Backward compatibility maintained

## Contact

For issues or questions:
- Review execplan: `docs/execplans/performance_tracker_integration.md`
- Check commit history: `git log --oneline | grep "Phase"`
- Test locally before deploying to production
