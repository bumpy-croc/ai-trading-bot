# Baseline vs Phase 2: Metrics Comparison

**Purpose**: Side-by-side comparison of all key metrics to identify regressions  
**Date**: 2025-10-11

## ML Basic Strategy - Backtest Mode

| Metric | Baseline (Phase 0) | Phase 2 | Δ Absolute | Δ % | Status |
|--------|-------------------|---------|-----------|------|---------|
| **Total Trades** | 9 | 300 | +291 | +3,233% | 🔴 REGRESSION |
| **Win Rate** | 44.44% | 50.00% | +5.56% | +12.5% | ⚠️ SUSPICIOUS |
| **Final Balance** | $10,104.13 | $10,147.26 | +$43.13 | +0.43% | ✅ OK |
| **Total Return** | 1.04% | 1.47% | +0.43% | +41.3% | ⚠️ SUSPICIOUS |
| **Max Drawdown** | 4.04% | 3.01% | -1.03% | -25.5% | ⚠️ SUSPICIOUS |
| **Sharpe Ratio** | 0.97 | 1.67 | +0.70 | +72.2% | ⚠️ SUSPICIOUS |
| **Annualized Return** | 13.43% | 19.47% | +6.04% | +45.0% | ⚠️ SUSPICIOUS |
| **Hold Return** | -1.48% | -1.48% | 0.00% | 0.0% | ✅ STABLE |
| **Wall Time** | 4.87s | 3.42s | -1.45s | -29.8% | ✅ IMPROVED |
| **CPU Time** | 7.58s | 3.33s | -4.25s | -56.1% | ✅ IMPROVED |
| **Dataset Size** | 721 rows | 721 rows | 0 | 0.0% | ✅ SAME |

### Analysis

**Behavioral Changes**:
- 🔴 Trade count increased 33x (9→300) - **CRITICAL REGRESSION**
- ⚠️ All performance metrics improved suspiciously
- ⚠️ Better Sharpe with more trades is unusual

**Performance Changes**:
- ✅ 56% faster CPU time - beneficial
- ✅ 30% faster wall time - beneficial
- ⚠️ Speed gains alongside behavioral changes concerning

**Verdict**: **BEHAVIORAL REGRESSION** - Trade generation logic fundamentally changed

---

## ML Basic Strategy - Live Paper Trading Mode

| Metric | Baseline (Phase 0) | Phase 2 | Δ Absolute | Δ % | Status |
|--------|-------------------|---------|-----------|------|---------|
| **Steps Executed** | 20 | 50 | +30 | +150% | ⚠️ DIFFERENT |
| **Total Trades** | 0 | 50 | +50 | N/A | 🔴 REGRESSION |
| **Winning Trades** | 0 | 28 | +28 | N/A | N/A |
| **Win Rate** | 0% | 56.00% | +56% | N/A | N/A |
| **Final Balance** | $10,000.00 | $10,034.66 | +$34.66 | +0.35% | ⚠️ CHANGED |
| **Total Return** | 0.00% | 0.35% | +0.35% | N/A | ⚠️ CHANGED |
| **Total PnL** | $0.00 | $34.66 | +$34.66 | N/A | N/A |
| **Max Drawdown** | 0.00% | 0.16% | +0.16% | N/A | N/A |
| **Current Drawdown** | 0.00% | 0.00% | 0.00% | 0.0% | ✅ OK |
| **Wall Time** | 43.11s | 68.41s | +25.30s | +58.7% | 🔴 REGRESSION |
| **CPU Time** | 2.38s | 16.28s | +13.90s | +583.8% | 🔴 CRITICAL |

### Analysis

**Behavioral Changes**:
- 🔴 Went from 0 trades to 50 trades - **CRITICAL REGRESSION**
- 🔴 100% activation rate (1 trade per step) - **SUSPICIOUS**
- ⚠️ All exits via "Stop loss" - no strategy signals

**Performance Changes**:
- 🔴 CPU time increased 584% - **UNACCEPTABLE FOR PRODUCTION**
- 🔴 Wall time increased 59% - concerning
- ⚠️ 150% more steps (20→50) explains some increase

**Trade Pattern** (Phase 2):
```
Every step: Entry → Stop Loss Exit → Immediate Re-Entry
```

**Verdict**: **CRITICAL REGRESSION** - Live trading behavior completely broken

---

## ML Adaptive Strategy - Backtest Mode

| Metric | Baseline (Phase 0) | Phase 2 | Δ | Status |
|--------|-------------------|---------|---|---------|
| **Total Trades** | N/A | 300 | N/A | ℹ️ NEW |
| **Win Rate** | N/A | 50.00% | N/A | ℹ️ NEW |
| **Final Balance** | N/A | $10,147.26 | N/A | ℹ️ NEW |
| **Total Return** | N/A | 1.47% | N/A | ℹ️ NEW |
| **Max Drawdown** | N/A | 3.01% | N/A | ℹ️ NEW |
| **Sharpe Ratio** | N/A | 1.67 | N/A | ℹ️ NEW |
| **Wall Time** | N/A | 1.33s | N/A | ℹ️ NEW |
| **CPU Time** | N/A | 1.32s | N/A | ℹ️ NEW |

### Observations

**No baseline** - Cannot assess regressions

**Notable findings**:
- ✅ Identical results to ml_basic (1.47% return, 300 trades)
- ✅ **61% faster** than ml_basic (1.33s vs 3.42s)
- ⚠️ Suspiciously identical results suggest shared code path

---

## ML Adaptive Strategy - Live Paper Trading Mode

| Metric | Baseline (Phase 0) | Phase 2 | Δ | Status |
|--------|-------------------|---------|---|---------|
| **Steps Executed** | N/A | 50 | N/A | ℹ️ NEW |
| **Total Trades** | N/A | 50 | N/A | ℹ️ NEW |
| **Win Rate** | N/A | 48.00% | N/A | ℹ️ NEW |
| **Final Balance** | N/A | $9,990.69 | N/A | ℹ️ NEW |
| **Total Return** | N/A | -0.09% | N/A | ℹ️ NEW |
| **Wall Time** | N/A | 52.20s | N/A | ℹ️ NEW |
| **CPU Time** | N/A | 1.79s | N/A | ℹ️ NEW |

### Observations

**No baseline** - Cannot assess regressions

**Notable findings**:
- ⚠️ Slight loss (-0.09%) vs ml_basic gain (+0.35%)
- ✅ **91% faster CPU** than ml_basic (1.79s vs 16.28s)
- ⚠️ Same 1:1 trade-to-step ratio as ml_basic

---

## Cross-Strategy Comparison (Phase 2 Only)

### Backtest Performance

| Strategy | Trades | Return | Sharpe | CPU Time | Speed Rank |
|----------|--------|--------|--------|----------|------------|
| ml_basic | 300 | 1.47% | 1.67 | 3.33s | 2nd |
| ml_adaptive | 300 | 1.47% | 1.67 | 1.32s | 🥇 1st |

**Findings**:
- Identical trading results (suspicious)
- ml_adaptive is 2.5x faster
- Both show same behavioral patterns

### Live Performance

| Strategy | Trades | Return | Win Rate | CPU Time | Efficiency |
|----------|--------|--------|----------|----------|------------|
| ml_basic | 50 | +0.35% | 56% | 16.28s | Poor |
| ml_adaptive | 50 | -0.09% | 48% | 1.79s | 🥇 Good |

**Findings**:
- ml_adaptive 9x faster in live mode
- Both show 100% trade activation rate
- ml_basic has excessive CPU usage

---

## Performance vs Behavior Trade-offs

### What Improved ✅

| Aspect | Improvement | Value |
|--------|-------------|-------|
| Backtest CPU time | -56% | From 7.58s to 3.33s |
| Backtest wall time | -30% | From 4.87s to 3.42s |
| Sharpe ratio | +72% | From 0.97 to 1.67 |
| Max drawdown | -25% | From 4.04% to 3.01% |
| Win rate | +12.5% | From 44% to 50% |

### What Regressed 🔴

| Aspect | Regression | Impact |
|--------|------------|--------|
| Trade count | +3,233% | 9→300 trades |
| Live CPU time | +584% | 2.38s→16.28s |
| Live activation | N/A | 0→50 trades |
| Behavioral consistency | N/A | Completely changed |

### The Problem ⚠️

The improvements and regressions are **correlated**, suggesting they may have a common cause:

1. **More trades** → More stop-loss hits → Lower drawdown
2. **Faster backtest** → Possibly skipping validations
3. **Better metrics** → Possibly overfitting to seed=42

**Question**: Are improvements real or artifacts of broken logic?

---

## Risk Assessment

### Production Deployment Risks

| Engine | Risk Level | Can Deploy? | Reasoning |
|--------|------------|-------------|-----------|
| **Backtest** | 🟡 MEDIUM | ⚠️ MAYBE | Different behavior but profitable; need validation across multiple seeds |
| **Live Trading** | 🔴 HIGH | ❌ NO | 584% CPU increase unacceptable; 100% trade activation suspicious |

### Validation Required Before Deployment

- [ ] Run backtest with 10+ different seeds
- [ ] Compare trade-by-trade with legacy implementation
- [ ] Profile and optimize live engine CPU usage
- [ ] Fix RegimeDetector errors
- [ ] Add automated parity tests
- [ ] Validate no look-ahead bias
- [ ] Test on out-of-sample data

---

## Summary Statistics

### Regression Severity Distribution

```
🔴 CRITICAL:  3 issues (Trade count, Live activation, Live CPU)
⚠️ WARNING:   4 issues (Metric improvements, Behavioral drift)
🟡 MINOR:     1 issue (RegimeDetector errors)
✅ IMPROVED:  2 aspects (Backtest speed, ml_adaptive efficiency)
```

### Behavioral Parity Assessment

```
Backtest:  ❌ FAILED  (9 vs 300 trades)
Live:      ❌ FAILED  (0 vs 50 trades)
Metrics:   ❌ FAILED  (Different risk profile)
Performance: ⚠️ MIXED (Backtest better, live worse)
```

### Overall Verdict

**Phase 2 Status**: 🚨 **BLOCKED**

**Reason**: Critical regressions in both backtest and live engines

**Next Steps**:
1. Capture true baseline (pre-refactor code)
2. Root cause analysis of trade count explosion
3. Fix live engine CPU performance
4. Implement automated parity testing

---

**Generated**: 2025-10-11  
**Full Analysis**: `REGRESSION_ANALYSIS.md`  
**Quick Summary**: `REGRESSION_SUMMARY.md`  
**Raw Data**: `baseline_summary.json`

