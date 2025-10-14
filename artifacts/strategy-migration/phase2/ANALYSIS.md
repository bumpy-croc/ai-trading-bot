# Phase 2 Benchmark Analysis

**Date**: 2025-10-11  
**Branch**: New Phase 2 Implementation  
**Status**: 🚨 **BEHAVIORAL PARITY NOT ACHIEVED**

## Executive Summary

This Phase 2 implementation shows **identical behavioral issues** to the previous implementation:
- **Trade count mismatch**: 9 trades (baseline) vs 300 trades (current)
- **Live trading activation**: 0 trades (baseline) vs 50 trades (current)
- **Same underlying issue**: Component-based adapter not maintaining parity with legacy behavior

## Quick Comparison

| Metric | Baseline (Legacy) | Phase 2 (Current) | Status |
|--------|------------------|-------------------|--------|
| **ml_basic backtest trades** | 9 | 300 | ❌ 3,233% increase |
| **ml_basic live trades** | 0 (20 steps) | 50 (50 steps) | ❌ 100% activation |
| **ml_adaptive backtest** | Not captured | 300 | ℹ️ No baseline |
| **ml_adaptive live** | Not captured | 50 | ℹ️ No baseline |

## Detailed Metrics: ml_basic Backtest

| Metric | Baseline | Phase 2 | Δ | % Change |
|--------|----------|---------|---|----------|
| Total Trades | 9 | 300 | +291 | +3,233% |
| Win Rate | 44.44% | 50.00% | +5.56% | +12.5% |
| Final Balance | $10,104.13 | $10,147.26 | +$43.13 | +0.43% |
| Total Return | 1.04% | 1.47% | +0.43% | +41.3% |
| Max Drawdown | 4.04% | 3.26% | -0.78% | -19.3% |
| Sharpe Ratio | 0.97 | 1.62 | +0.65 | +67.0% |
| Wall Time | 4.87s | 3.00s | -1.87s | -38.4% |
| CPU Time | 7.58s | 2.98s | -4.60s | -60.7% |

### Analysis
- **Same issue**: 33x more trades than baseline
- **Better metrics**: Return, Sharpe, and drawdown improved, but suspect due to behavioral changes
- **Faster execution**: Significant performance improvement (60% CPU reduction)

## Detailed Metrics: ml_basic Live Paper Trading

| Metric | Baseline | Phase 2 | Δ | % Change |
|--------|----------|---------|---|----------|
| Steps | 20 | 50 | +30 | +150% |
| Total Trades | 0 | 50 | +50 | N/A |
| Final Balance | $10,000.00 | $10,031.46 | +$31.46 | +0.31% |
| Total Return | 0.00% | 0.31% | +0.31% | N/A |
| Win Rate | 0% | 60.00% | +60% | N/A |
| Max Drawdown | 0.00% | 0.19% | +0.19% | N/A |
| Wall Time | 43.11s | 67.34s | +24.23s | +56.2% |
| CPU Time | 2.38s | 16.03s | +13.65s | +573.5% |

### Analysis
- **Critical issue**: Baseline had 0 trades, now 50 trades (1:1 with steps)
- **100% activation rate**: Every step generates entry → exit cycle
- **CPU regression**: 573% increase in CPU time unacceptable for production
- **Suspicious behavior**: All exits via stop-loss, immediate re-entry

## ml_adaptive Results (No Baseline for Comparison)

**Backtest**:
- 300 trades (identical to ml_basic)
- 1.47% return (identical to ml_basic)
- 1.44s wall time (52% faster than ml_basic)
- Suspiciously identical results suggest shared code path

**Live Paper**:
- 50 trades (1:1 with steps)
- 0.29% return (slightly worse than ml_basic)
- 52.18s wall time, much better CPU efficiency than ml_basic

## Key Findings

### 1. Behavioral Parity Failure ❌

**Expected**: Comp

onent-based strategies should produce identical results to legacy  
**Actual**: Completely different behavior (9 vs 300 trades)

**Root Cause**: `LegacyStrategyAdapter` translation layer has bugs:
- Not tracking position state correctly
- Missing trade cooldown logic
- Entry signals generated too frequently

### 2. Live Trading Issues 🚨

**Pattern observed**:
```
Step 1: Entry → Stop Loss → 
Step 2: Entry → Stop Loss →
Step 3: Entry → Stop Loss →
...continuous churn
```

**Problems**:
- No cooldown between trades
- Immediate re-entry after stop-loss exit
- 100% trade activation rate suspicious
- 573% CPU increase unacceptable

### 3. Performance Changes ⚠️

**Positive**:
- Backtest 60% faster (CPU)
- ml_adaptive very efficient (1.44s)

**Negative**:
- Live engine 573% slower (CPU)
- Wall time increased 56%
- RegimeDetector errors spam logs

### 4. RegimeDetector Errors 🟡

Hundreds of errors throughout execution:
```
Error detecting regime: 'RegimeDetector' object has no attribute 'detect_regime'
Error calculating indicators: 'RegimeDetector' object has no attribute 'base_detector'
```

Impact: Non-fatal but wastes CPU and clutters logs

## Comparison with Previous Phase 2 Run

This branch shows **identical behavior** to the previous Phase 2 analysis:
- Same trade count (300)
- Same live activation (50/50)
- Same RegimeDetector errors
- Similar performance profile

**Conclusion**: The behavioral issues persist across both Phase 2 implementations.

## Root Cause Hypothesis

### Why 300 Trades Instead of 9?

**Theory 1**: Position state not tracked
```python
# Adapter may not be checking:
if already_have_position:
    return False  # Don't enter again
```

**Theory 2**: No trade cooldown
```python
# Missing logic:
if bars_since_last_trade < COOLDOWN:
    return False
```

**Theory 3**: Signal threshold misalignment
```python
# Legacy: threshold = 0.6
# Component: threshold = 0.5  # Too low, more signals
```

### Why 100% Trade Activation in Live?

**Theory**: Entry conditions always return True
- Adapter may bypass position checks in live mode
- Different code path for live vs backtest
- MockDataProvider generates data that always satisfies entry

## Recommendations

### Immediate (Priority: CRITICAL)

1. **Create Diagnostic Script** to compare signals step-by-step:
   ```bash
   python scripts/debug_adapter_parity.py
   # Should show why 300 signals generated vs expected 9
   ```

2. **Fix Adapter Position Tracking**:
   - Add `_current_position` tracking
   - Don't allow entry when position exists
   - Enforce minimum cooldown between trades

3. **Fix RegimeDetector**:
   - Implement missing `detect_regime()` method
   - Add `base_detector` attribute
   - Suppress duplicate error logging

4. **Optimize Live Engine**:
   - Profile to find 573% CPU bottleneck
   - Cache regime detection results
   - Reduce adapter overhead

### Short-term (Next 2 Weeks)

1. **Implement Automated Parity Tests**:
   ```python
   def test_adapter_matches_baseline():
       assert signal_count < 20  # Not 300
       assert live_activation_rate < 0.3  # Not 1.0
   ```

2. **Fix and Validate**:
   - Re-run benchmarks after fixes
   - Verify ~9 trades achieved
   - Validate across multiple seeds

3. **Document Changes**:
   - Record all fixes applied
   - Update migration status
   - Mark Phase 2 complete only after parity achieved

## Success Criteria

Phase 2 is **not complete** until:

- ✅ Trade count matches baseline (9 ±2 trades)
- ✅ Live mode doesn't have 100% activation
- ✅ CPU time acceptable for production (< 5s for 50 steps)
- ✅ No RegimeDetector errors
- ✅ Automated tests validate parity

## Migration Proposal Compliance

Per `artifacts/strategy-migration/strategy_migration_proposal.md`:

> "Results produced by the engines must remain equivalent to the current legacy path"

**Status**: ❌ **NOT COMPLIANT**

**Violations**:
- Trade sequences not equivalent (9 vs 300)
- Risk metrics different (drawdown profile changed)
- Live behavior completely different (0 vs 50 trades)

## Conclusion

This Phase 2 implementation has **identical behavioral issues** as the previous version. The component-based adapter is not maintaining parity with legacy strategies. The same debugging and fixes from the previous analysis apply here.

**Next Steps**:
1. Follow the action plan in previous analysis
2. Debug adapter translation layer
3. Fix position tracking and cooldowns
4. Re-run benchmarks to validate
5. Mark Phase 2 complete only after parity achieved

---

**Files Generated**:
- `baseline_summary.json` - Aggregate results
- `baseline_summary.md` - Human-readable summary
- `baseline_backtest_*.json` - Detailed backtest results
- `baseline_live_*.json` - Detailed live trading results
- `*_trades.csv` - Trade logs
- `*.log` - Execution logs

**Previous Analysis**: See git history for detailed action plan and debugging steps

**Related Docs**:
- `artifacts/strategy-migration/strategy_migration_proposal.md` - Migration requirements
- `artifacts/strategy-migration/baseline/` - Ground truth results
