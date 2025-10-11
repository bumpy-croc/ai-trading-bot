# Phase 2 Benchmark Analysis: Migration Impact Assessment

**Date**: 2025-10-11  
**Context**: Strategy Migration Proposal - Phase 2 Engine Integration Analysis  
**Reference**: `docs/strategy_migration_proposal.md`

## Executive Summary

Phase 2 benchmarks reveal **significant behavioral drift** between baseline (Phase 0) and current implementations. This analysis identifies critical issues that must be addressed before proceeding with runtime integration work described in the migration proposal.

### 🚨 Critical Findings

1. **Trade count explosion**: ml_basic strategy generated 300 trades vs baseline 9 trades (3233% increase)
2. **Live trading activation**: Baseline live mode produced 0 trades; Phase 2 produced 50 trades
3. **Performance variance**: Behavioral changes occurred despite using identical market data (seed=42)
4. **RegimeDetector errors**: Non-critical but widespread missing method errors throughout execution

## Detailed Comparison: Baseline vs Phase 2

### ml_basic Strategy - Backtest Mode

| Metric | Baseline (Phase 0) | Phase 2 | Delta | % Change |
|--------|-------------------|---------|-------|----------|
| **Total Trades** | 9 | 300 | +291 | +3233% |
| **Final Balance** | $10,104.13 | $10,147.26 | +$43.13 | +0.43% |
| **Total Return %** | 1.04% | 1.47% | +0.43% | +41.3% |
| **Win Rate %** | 44.44% | 50.00% | +5.56% | +12.5% |
| **Max Drawdown %** | 4.04% | 3.01% | -1.03% | -25.5% |
| **Sharpe Ratio** | 0.97 | 1.67 | +0.70 | +72.2% |
| **Wall Time (s)** | 4.87 | 3.42 | -1.45 | -29.8% |
| **CPU Time (s)** | 7.58 | 3.33 | -4.25 | -56.1% |

**Analysis**: The dramatic increase in trade count suggests a fundamental change in entry/exit logic. Surprisingly, despite 33x more trades, the final return only increased by 41%, indicating much smaller position sizes or tighter stop losses. The performance improvement (both speed and Sharpe ratio) is suspicious and warrants investigation.

### ml_basic Strategy - Live Paper Trading Mode

| Metric | Baseline (Phase 0) | Phase 2 | Delta | % Change |
|--------|-------------------|---------|-------|----------|
| **Steps** | 20 | 50 | +30 | +150% |
| **Total Trades** | 0 | 50 | +50 | N/A |
| **Final Balance** | $10,000.00 | $10,034.66 | +$34.66 | +0.35% |
| **Total Return %** | 0.00% | 0.35% | +0.35% | N/A |
| **Win Rate %** | 0% | 56.00% | +56% | N/A |
| **Max Drawdown %** | 0.00% | 0.16% | +0.16% | N/A |
| **Wall Time (s)** | 43.11 | 68.41 | +25.30 | +58.7% |
| **CPU Time (s)** | 2.38 | 16.28 | +13.90 | +583.8% |

**Analysis**: The baseline produced **zero trades** in live mode, which indicates either:
1. Entry conditions were never met in the baseline run
2. Different random seed for market data generation
3. A bug in the baseline live trading setup

Phase 2 shows 1 trade per step (50 trades in 50 steps), suggesting very aggressive entry/exit behavior. The 584% increase in CPU time is concerning for production deployment.

### ml_adaptive Strategy - Backtest Mode

| Metric | Baseline (Phase 0) | Phase 2 | Delta | % Change |
|--------|-------------------|---------|-------|----------|
| **Total Trades** | N/A | 300 | N/A | N/A |
| **Final Balance** | N/A | $10,147.26 | N/A | N/A |
| **Total Return %** | N/A | 1.47% | N/A | N/A |
| **Win Rate %** | N/A | 50.00% | N/A | N/A |
| **Wall Time (s)** | N/A | 1.33 | N/A | N/A |
| **CPU Time (s)** | N/A | 1.32 | N/A | N/A |

**Analysis**: No baseline exists for ml_adaptive. Notable that ml_adaptive is **61% faster** than ml_basic in backtest mode (1.33s vs 3.42s) despite producing identical results. This suggests ml_adaptive has better optimization or caching.

### ml_adaptive Strategy - Live Paper Trading Mode

| Metric | Baseline (Phase 0) | Phase 2 | Delta | % Change |
|--------|-------------------|---------|-------|----------|
| **Steps** | N/A | 50 | N/A | N/A |
| **Total Trades** | N/A | 50 | N/A | N/A |
| **Final Balance** | N/A | $9,990.69 | N/A | N/A |
| **Total Return %** | N/A | -0.09% | N/A | N/A |
| **Win Rate %** | N/A | 48.00% | N/A | N/A |
| **Wall Time (s)** | N/A | 52.20 | N/A | N/A |
| **CPU Time (s)** | N/A | 1.79 | N/A | N/A |

**Analysis**: ml_adaptive shows a small loss in live mode (-0.09%) compared to ml_basic's gain (+0.35%). However, it's **91% faster** in CPU time (1.79s vs 16.28s), making it significantly more efficient for live deployment.

## Implications for Migration Proposal

### Phase 2: Engine Integration (Current Phase)

The migration proposal states:
> "Side-by-side regression harness that replays datasets through both engines (legacy vs runtime) and asserts identical trade sequences, PnL, and risk metrics."

**Status**: ❌ **BLOCKED** - Current results show non-identical behavior:
- Trade sequences differ dramatically (9 vs 300 trades)
- PnL differs by 41% (though both profitable)
- Risk metrics show different drawdown profiles

**Required Actions Before Proceeding**:

1. **Root Cause Analysis**
   - Investigate what changed between baseline and Phase 2 runs
   - Verify both runs used identical:
     - Market data (same seed, same MockDataProvider parameters)
     - Strategy parameters
     - Risk parameters
     - Entry/exit conditions
   
2. **Behavioral Parity Verification**
   - Re-run baseline with Phase 2 parameters to isolate changes
   - Document intentional vs unintentional differences
   - Establish whether 300 trades or 9 trades is correct behavior
   
3. **Live Mode Investigation**
   - Determine why baseline live mode produced 0 trades
   - Verify Phase 2 live mode 1:1 trade-to-step ratio is expected
   - Profile CPU usage spike (584% increase)

### Feature Preparation and Caching (Proposal Section 2)

The proposal emphasizes:
> "To keep backtesting fast, indicator work must remain vectorised."

**Status**: ✅ **POSITIVE** - Phase 2 shows performance improvements:
- 30% faster wall time for ml_basic backtest
- 56% faster CPU time for ml_basic backtest
- ml_adaptive is exceptionally fast (1.33s for 721 rows)

**Concern**: The speed improvement alongside behavioral changes suggests potential issues with feature computation. The vectorization may be working differently than in the baseline.

### Risk Mitigation (Proposal Section)

The proposal warns:
> "Performance regressions – Benchmark backtests before and after runtime integration using representative datasets"

**Status**: ⚠️ **PARTIALLY MET**
- Performance improved (not regressed)
- But behavioral drift occurred
- Need to verify if speed gains came at cost of correctness

> "Behavioural drift – Use regression tests and audit trail tooling to compare trade sequences"

**Status**: ❌ **FAILED** - Clear behavioral drift detected:
- Trade count mismatch indicates fundamental strategy behavior change
- No audit trail comparison performed yet
- Need detailed trade-by-trade analysis

## Technical Issues Identified

### 1. RegimeDetector Errors

Throughout both Phase 2 runs, numerous errors appear:
```
Error detecting regime at index X: 'RegimeDetector' object has no attribute 'detect_regime'
Error calculating indicators: 'RegimeDetector' object has no attribute 'base_detector'
```

**Impact**: 
- Currently non-fatal (strategies continue executing)
- May impact regime-aware strategies when implemented
- Indicates incomplete refactoring or missing method implementations

**Recommendation**: Fix before runtime integration to avoid compounding issues.

### 2. MockDataProvider Seed Inconsistency

Baseline and Phase 2 used different seeds:
- Baseline: `seed=42` (backtest), `seed=1337` (live)
- Phase 2: `seed=42` (backtest), `seed=1337` (live)

Seeds appear identical, but market data generation may have changed. This needs verification.

### 3. Live Trading Step Count Mismatch

- Baseline: 20 steps
- Phase 2: 50 steps

This alone could explain behavioral differences in live mode, but doesn't explain backtest divergence.

## Recommendations

### Immediate Actions (Before Phase 2 Completion)

1. **Establish Ground Truth**
   - Re-run baseline benchmarks with current codebase
   - Document any known strategy changes since baseline capture
   - Create a "known differences" document if changes were intentional

2. **Implement Regression Testing**
   - Add trade sequence comparison tooling
   - Generate trade-by-trade diff between baseline and Phase 2
   - Flag any differences beyond expected variance

3. **Fix RegimeDetector**
   - Implement missing `detect_regime` method
   - Add `base_detector` attribute
   - Ensure all regime detection code is functional

4. **Profile Performance Changes**
   - Investigate why ml_basic CPU time improved by 56%
   - Verify feature calculation is still correct
   - Ensure vectorization didn't skip required computations

### Phase 2 Completion Criteria

Before marking Phase 2 complete and moving to Phase 3:

- [ ] Root cause of trade count discrepancy identified and documented
- [ ] Behavioral parity established OR differences justified
- [ ] RegimeDetector errors resolved
- [ ] Live mode 0-trade baseline issue explained
- [ ] Performance improvements validated as legitimate
- [ ] Regression test suite implemented and passing
- [ ] Documentation updated with findings

### Modified Phase 2 Timeline

Given these findings, recommend:

1. **Phase 2A**: Investigation & Parity Restoration (current focus)
2. **Phase 2B**: Runtime Integration (as originally planned)
3. **Phase 2C**: Validation & Artifacts (enhanced scope)

## Benchmark Artifacts

### Phase 0 (Baseline)
- Location: `artifacts/strategy-migration/baseline/`
- Captured: 2025-09-10
- Strategies: ml_basic only
- Dataset: 721 rows, 1h timeframe, 30 days

### Phase 2 (Current)
- Location: `artifacts/strategy-migration/phase2/`
- Captured: 2025-10-11
- Strategies: ml_basic, ml_adaptive
- Dataset: 721 rows, 1h timeframe, 30 days

### Files Generated
```
phase2/
├── baseline_backtest_ml_basic.json
├── baseline_backtest_ml_basic.log
├── baseline_backtest_ml_basic_trades.csv
├── baseline_backtest_ml_adaptive.json
├── baseline_backtest_ml_adaptive.log
├── baseline_backtest_ml_adaptive_trades.csv
├── baseline_live_ml_basic.json
├── baseline_live_ml_basic.log
├── baseline_live_ml_basic_trades.csv
├── baseline_live_ml_adaptive.json
├── baseline_live_ml_adaptive.log
├── baseline_live_ml_adaptive_trades.csv
├── baseline_summary.json
├── baseline_summary.md
└── PHASE2_ANALYSIS.md (this file)
```

## Conclusion

Phase 2 benchmarks successfully captured current system performance but revealed critical behavioral drift that must be addressed before proceeding with runtime integration. The dramatic increase in trade frequency (9 → 300) and activation of live trading (0 → 50 trades) indicates fundamental strategy logic changes that were either:

1. **Intentional improvements** that need documentation, or
2. **Unintended regressions** that need fixes

The performance improvements (faster execution, better Sharpe ratio) are encouraging but must be validated as legitimate rather than artifacts of skipped computations or changed logic.

**Next Step**: Create detailed trade-by-trade comparison between baseline and Phase 2 to identify exact points of divergence.

---

**References**:
- Migration Proposal: `docs/strategy_migration_proposal.md`
- Baseline Artifacts: `artifacts/strategy-migration/baseline/`
- Phase 2 Artifacts: `artifacts/strategy-migration/phase2/`
- Benchmark Script: `scripts/benchmark_legacy_baseline.py`

