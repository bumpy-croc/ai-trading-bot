# Regression Summary: Critical Issues Identified

**Status**: 🚨 **PHASE 2 BLOCKED**  
**Date**: 2025-10-11  
**Severity**: CRITICAL

## Quick Answer: Are There Regressions?

**YES - Multiple critical regressions identified:**

| # | Regression | Severity | Impact | Status |
|---|------------|----------|--------|--------|
| 1 | Trade count explosion (9→300) | 🔴 CRITICAL | Behavioral parity violated | OPEN |
| 2 | Live trading activation (0→50) | 🔴 CRITICAL | Live mode behavior changed | OPEN |
| 3 | Suspicious metric improvements | ⚠️ WARNING | May indicate data leakage | OPEN |
| 4 | Live CPU usage +584% | ⚠️ WARNING | Production performance concern | OPEN |
| 5 | RegimeDetector errors | 🟡 MINOR | Incomplete refactoring | OPEN |

## What Changed?

Between baseline (Sep 10) and Phase 2 (Oct 11), the codebase underwent **major architectural refactoring**:

```
Legacy Strategy (Baseline)
└─ MlBasic with direct ONNX calls
   └─ 9 trades, simple logic

        ⬇️ REFACTORED (Sep 19-23)

Component Strategy (Phase 2)
└─ LegacyStrategyAdapter
   └─ Strategy(
      ├─ MLBasicSignalGenerator
      ├─ FixedRiskManager  
      ├─ ConfidenceWeightedSizer
      └─ EnhancedRegimeDetector)
   └─ 300 trades, complex logic
```

## Critical Finding: Comparing Apples to Oranges

The baseline was captured on **legacy code** (pre-refactor), but Phase 2 runs on **component-based code** (post-refactor). We're not measuring migration drift—we're comparing two completely different implementations.

## Migration Proposal Violations

The proposal requires:
> "Results produced by the engines must remain equivalent to the current legacy path"

**Actual results**:
- Trade sequences: **NOT EQUIVALENT** (9 vs 300 trades)
- Risk metrics: **NOT EQUIVALENT** (different drawdown)
- Performance: **MIXED** (backtest faster, live slower)

## Root Causes Identified

### 1. Trade Count Explosion (9 → 300)
**Likely causes**:
- Adapter layer calls entry/exit logic more frequently
- Missing cooldown periods between trades
- Stop-loss triggering logic changed
- Signal filtering not working properly

### 2. Live Trading Activation (0 → 50)
**Likely causes**:
- Entry conditions always returning True in new code
- Position state tracking broken
- Different MockDataProvider behavior (despite same seed)
- Immediate re-entry after exit (no cooldown)

### 3. CPU Performance Regression (+584% in live)
**Causes identified**:
- RegimeDetector called on every candle with errors
- Adapter translation overhead
- Component orchestration overhead
- Hundreds of failed method calls consuming CPU

## Immediate Actions Required

### Before Phase 2 Can Continue:

1. **Establish True Baseline** ⚠️ URGENT
   - Re-run baseline on pre-refactor code (commit a0c5d53)
   - This gives us proper comparison point

2. **Fix Critical Bugs** 🔴 CRITICAL
   - Debug why 300 trades generated vs 9
   - Fix live trading behavior (0→50 trades)
   - Implement missing RegimeDetector methods

3. **Build Parity Tests** 🔴 CRITICAL
   - Automated trade sequence comparison
   - CI checks for behavioral drift
   - Golden datasets for regression testing

## Timeline Impact

**Best case**: 2 weeks to resolve
- 3 days: Establish baseline + root cause analysis
- 5 days: Fix critical bugs
- 2 days: Implement parity tests

**Worst case**: 4 weeks + potential rollback
- If parity can't be achieved with current architecture
- May need to rollback component refactoring
- Restart migration following proposal process

## Recommendations

### Option A: Fix Forward (Recommended)
1. Keep component architecture
2. Fix adapter logic to match legacy behavior  
3. Add automated parity testing
4. Validate before proceeding to Phase 3

**Pros**: Keep architectural improvements  
**Cons**: 2-3 weeks additional work  

### Option B: Rollback and Restart
1. Revert component refactoring
2. Follow migration proposal properly
3. Capture baseline, then migrate incrementally
4. Validate parity at each step

**Pros**: Follow proven migration path  
**Cons**: Lose 3 weeks of refactoring work  

## Can We Deploy Current Code?

### Backtest Engine: ⚠️ MAYBE
- **Pros**: Faster performance, better metrics
- **Cons**: Different behavior, unvalidated
- **Risk**: Medium - results not comparable to history

### Live Trading Engine: ❌ NO
- **Cons**: 584% CPU increase unacceptable
- **Cons**: Behavioral changes unvalidated  
- **Cons**: Constant trade churning suspicious
- **Risk**: HIGH - could lose money in production

## Bottom Line

**Yes, there are critical regressions.** The component-based refactoring introduced behavioral changes that weren't caught because:

1. Baseline captured before refactoring began
2. No parity validation during refactoring
3. No automated regression tests
4. Migration process not followed

**Phase 2 must be paused** until regressions are understood and resolved.

---

**Full Analysis**: See `REGRESSION_ANALYSIS.md` for detailed technical investigation  
**Proposal Reference**: `docs/strategy_migration_proposal.md`  
**Benchmark Data**: `artifacts/strategy-migration/phase2/`

