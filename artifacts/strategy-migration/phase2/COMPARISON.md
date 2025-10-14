# Baseline vs Phase 2: Complete Metrics Comparison

## ml_basic Strategy - Backtest Mode

| Metric | Baseline (Sep 10) | Phase 2 (Oct 11) | Δ Absolute | Δ % | Pass? |
|--------|------------------|------------------|-----------|------|-------|
| **Total Trades** | 9 | 300 | +291 | +3,233% | ❌ |
| **Win Rate %** | 44.44% | 50.00% | +5.56% | +12.5% | ⚠️ |
| **Final Balance** | $10,104.13 | $10,147.26 | +$43.13 | +0.43% | ⚠️ |
| **Total Return %** | 1.04% | 1.47% | +0.43% | +41.3% | ⚠️ |
| **Max Drawdown %** | 4.04% | 3.26% | -0.78% | -19.3% | ⚠️ |
| **Sharpe Ratio** | 0.97 | 1.62 | +0.65% | +67.0% | ⚠️ |
| **Ann. Return %** | 13.43% | 19.47% | +6.04% | +45.0% | ⚠️ |
| **Hold Return %** | -1.48% | -1.48% | 0.00% | 0.0% | ✅ |
| **Wall Time (s)** | 4.87s | 3.00s | -1.87s | -38.4% | ✅ |
| **CPU Time (s)** | 7.58s | 2.98s | -4.60s | -60.7% | ✅ |
| **Dataset Rows** | 721 | 721 | 0 | 0.0% | ✅ |

**Legend**:
- ❌ = Critical failure (behavioral change)
- ⚠️ = Suspicious (better but due to behavioral change)
- ✅ = Expected/acceptable

## ml_basic Strategy - Live Paper Trading Mode

| Metric | Baseline (Sep 10) | Phase 2 (Oct 11) | Δ Absolute | Δ % | Pass? |
|--------|------------------|------------------|-----------|------|-------|
| **Steps Executed** | 20 | 50 | +30 | +150% | ℹ️ |
| **Total Trades** | 0 | 50 | +50 | N/A | ❌ |
| **Winning Trades** | 0 | 30 | +30 | N/A | N/A |
| **Win Rate %** | 0% | 60.00% | +60% | N/A | N/A |
| **Final Balance** | $10,000.00 | $10,031.46 | +$31.46 | +0.31% | ⚠️ |
| **Total Return %** | 0.00% | 0.31% | +0.31% | N/A | ⚠️ |
| **Total PnL** | $0.00 | $31.46 | +$31.46 | N/A | N/A |
| **Max Drawdown %** | 0.00% | 0.19% | +0.19% | N/A | N/A |
| **Cur. Drawdown %** | 0.00% | 0.00% | 0.00% | 0.0% | ✅ |
| **Wall Time (s)** | 43.11s | 67.34s | +24.23s | +56.2% | ❌ |
| **CPU Time (s)** | 2.38s | 16.03s | +13.65s | +573.5% | ❌ |

**Legend**:
- ❌ = Critical failure
- ⚠️ = Changed behavior (not necessarily bad)
- ℹ️ = Different test parameters
- N/A = No baseline for comparison

## ml_adaptive Strategy - Backtest Mode

| Metric | Baseline | Phase 2 | Status |
|--------|----------|---------|---------|
| **Total Trades** | N/A | 300 | ℹ️ No baseline |
| **Win Rate %** | N/A | 50.00% | ℹ️ No baseline |
| **Total Return %** | N/A | 1.47% | ℹ️ No baseline |
| **Sharpe Ratio** | N/A | 1.62 | ℹ️ No baseline |
| **Wall Time (s)** | N/A | 1.44s | ℹ️ No baseline |
| **CPU Time (s)** | N/A | 1.43s | ℹ️ No baseline |

**Notable**: Identical returns to ml_basic (1.47%) but 52% faster execution

## ml_adaptive Strategy - Live Paper Trading Mode

| Metric | Baseline | Phase 2 | Status |
|--------|----------|---------|---------|
| **Total Trades** | N/A | 50 | ℹ️ No baseline |
| **Win Rate %** | N/A | 58.00% | ℹ️ No baseline |
| **Total Return %** | N/A | 0.29% | ℹ️ No baseline |
| **Wall Time (s)** | N/A | 52.18s | ℹ️ No baseline |
| **CPU Time (s)** | N/A | 1.79s | ℹ️ No baseline |

**Notable**: Much better CPU efficiency than ml_basic (1.79s vs 16.03s)

## Cross-Strategy Comparison (Phase 2 Only)

### Backtest Performance

| Strategy | Trades | Return % | Sharpe | Wall Time | CPU Time | Efficiency Rank |
|----------|--------|----------|--------|-----------|----------|----------------|
| ml_basic | 300 | 1.47% | 1.62 | 3.00s | 2.98s | 2nd |
| ml_adaptive | 300 | 1.47% | 1.62 | 1.44s | 1.43s | 🥇 1st (52% faster) |

**Observation**: Identical trading results, but ml_adaptive much faster

### Live Performance

| Strategy | Trades | Return % | Win Rate | Wall Time | CPU Time | Efficiency Rank |
|----------|--------|----------|----------|-----------|----------|----------------|
| ml_basic | 50 | 0.31% | 60% | 67.34s | 16.03s | 2nd (slow) |
| ml_adaptive | 50 | 0.29% | 58% | 52.18s | 1.79s | 🥇 1st (89% faster CPU) |

**Observation**: ml_adaptive dramatically more efficient in live mode

## Performance Summary

### What Improved ✅

| Aspect | Improvement | Measure |
|--------|-------------|---------|
| Backtest CPU time | -60.7% | 7.58s → 2.98s |
| Backtest wall time | -38.4% | 4.87s → 3.00s |
| Sharpe ratio | +67.0% | 0.97 → 1.62 |
| Max drawdown | -19.3% | 4.04% → 3.26% |
| Win rate | +12.5% | 44.44% → 50.00% |

### What Regressed ❌

| Aspect | Regression | Measure |
|--------|-----------|---------|
| Trade count | +3,233% | 9 → 300 |
| Live CPU time | +573.5% | 2.38s → 16.03s |
| Live wall time | +56.2% | 43.11s → 67.34s |
| Live activation rate | N/A | 0% → 100% |
| Behavioral consistency | N/A | Completely changed |

## Trade Pattern Analysis

### Baseline Pattern (9 Trades)
```
Selective entries based on strong signals
Average ~80 bars between trades
Mix of stop-loss and take-profit exits
```

### Phase 2 Pattern (300 Trades)
```
Frequent entries on every signal
Average ~2.4 bars between trades
All exits via stop-loss (no take-profit)
Continuous churn pattern
```

### Live Mode Pattern (50/50 Activation)
```
Entry → Stop Loss → Entry → Stop Loss → ...
100% trade activation rate (1 trade per step)
Immediate re-entry after exit
No cooldown period observed
```

## Risk Analysis

### Production Deployment Risk

| Component | Risk Level | Can Deploy? | Reason |
|-----------|-----------|-------------|---------|
| **Backtest Engine** | 🟡 MEDIUM | ⚠️ MAYBE | Different behavior but profitable; needs validation across seeds |
| **Live Engine** | 🔴 HIGH | ❌ NO | 573% CPU increase + 100% activation rate unacceptable |

### Validation Required

- [ ] Run on 10+ different seeds to validate consistency
- [ ] Compare trade-by-trade with legacy  
- [ ] Profile live engine CPU usage
- [ ] Fix RegimeDetector errors
- [ ] Implement automated parity tests
- [ ] Test on out-of-sample data

## Overall Assessment

| Criterion | Target | Actual | Pass? |
|-----------|--------|--------|-------|
| **Behavioral Parity** | Identical trades | 9 vs 300 | ❌ |
| **Trade Sequences** | Same order | Different | ❌ |
| **PnL Equivalence** | Within 1% | +41% | ❌ |
| **Risk Metrics** | Within 5% | -19% to +67% | ❌ |
| **Backtest Performance** | Maintain or improve | Improved | ✅ |
| **Live Performance** | Maintain | +573% CPU | ❌ |

**Overall Grade**: ❌ **FAILED** (0/6 criteria met)

## Conclusion

Phase 2 shows **identical behavioral issues** to previous analysis:
- Same 33x trade count increase
- Same 100% live activation pattern  
- Same RegimeDetector errors
- Same adapter translation bugs

**Status**: Phase 2 remains **BLOCKED** pending fixes to adapter layer.

---

**Generated**: 2025-10-11  
**Baseline Date**: 2025-09-10  
**Phase 2 Date**: 2025-10-11  
**Comparison**: 31 days apart

