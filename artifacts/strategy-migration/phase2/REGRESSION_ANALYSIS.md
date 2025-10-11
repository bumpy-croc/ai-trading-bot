# Regression Analysis: Strategy Migration Phase 2

**Date**: 2025-10-11  
**Analysis Type**: Behavioral Consistency Review  
**Status**: 🚨 **CRITICAL REGRESSIONS IDENTIFIED**

## Executive Summary

The Phase 2 benchmarks reveal **critical behavioral regressions** introduced by the component-based architecture migration. Between baseline (Phase 0, September 10) and Phase 2 (October 11), the codebase underwent a major refactoring from legacy strategy architecture to component-based strategies. This refactoring has introduced significant behavioral changes that violate the migration proposal's core constraint:

> **"Maintain behavioural parity – Results produced by the engines must remain equivalent to the current legacy path for each supported strategy during the migration."**  
> — `docs/strategy_migration_proposal.md`, Section "Goals and Constraints"

## Identified Regressions

### 🔴 CRITICAL: Trade Count Explosion (Regression #1)

**Symptom**: ml_basic backtest generates 300 trades instead of 9 trades (3233% increase)

**Impact**: 
- Violates behavioral parity requirement
- Makes historical validation impossible
- Suggests fundamental logic changes in entry/exit conditions

**Root Cause Analysis**:

Between baseline and Phase 2, commit `8235739` ("feat: Add calculate_indicators to MlBasic strategy") refactored MlBasic to use:
- `LegacyStrategyAdapter` wrapping component-based implementation
- `MLBasicSignalGenerator` for entry signals
- `FixedRiskManager` for risk management
- `ConfidenceWeightedSizer` for position sizing

**Evidence from git history**:
```bash
8235739 feat: Add calculate_indicators to MlBasic strategy (Sep 23)
7c15747 fix: respect risk manager sizing and stage hot swap (Sep 23)
d06a98c refactor: Transition to component-based architecture in ML strategies (Sep 20)
```

**Likely cause**: The adapter layer may be:
1. Calling entry/exit logic more frequently than legacy implementation
2. Not properly filtering signals before generating trades
3. Missing cooldown periods between trades
4. Converting signals to trades at different thresholds

**Required Investigation**:
- [ ] Compare `check_entry_conditions` logic between legacy and adapter
- [ ] Verify signal filtering in `LegacyStrategyAdapter`
- [ ] Check if adapter respects strategy cooldown periods
- [ ] Validate stop-loss/take-profit triggering logic

### 🔴 CRITICAL: Live Trading Activation (Regression #2)

**Symptom**: Baseline live mode generated 0 trades; Phase 2 generates 50 trades (1:1 with steps)

**Impact**:
- Live trading behavior completely changed
- 100% trade activation rate (1 trade per step) is suspicious
- Suggests entry conditions are always satisfied

**Root Cause Analysis**:

The 1:1 trade-to-step ratio indicates one of:
1. **Always-true entry condition**: Entry logic may have a bug that always returns True
2. **Disabled entry filtering**: Adapter may bypass entry condition checks
3. **Different data generation**: MockDataProvider may generate different patterns despite same seed
4. **Missing position checks**: May not be checking if position already exists before entering

**Evidence**:
- Baseline used 20 steps with seed 1337, got 0 trades
- Phase 2 used 50 steps with seed 1337, got 50 trades
- Every single step generated exactly 1 trade and 1 exit

**Trade Pattern Analysis** (Phase 2):
```
Entry → Exit (Stop loss) → Entry → Exit (Stop loss) → ...
```

This pattern suggests:
- Entry immediately after exit (no cooldown)
- All exits via stop loss (not take profit or strategy signal)
- Possible "churn" behavior where strategy constantly enters/exits

**Required Investigation**:
- [ ] Debug entry condition evaluation in live engine
- [ ] Verify MockDataProvider seed determinism
- [ ] Check if position state tracking works correctly
- [ ] Validate stop-loss distance calculations

### ⚠️ WARNING: Performance Metrics Improvement (Regression #3)

**Symptom**: Better metrics despite behavioral changes

| Metric | Baseline | Phase 2 | Change |
|--------|----------|---------|--------|
| Sharpe Ratio | 0.97 | 1.67 | +72% |
| Max Drawdown | 4.04% | 3.01% | -25% (better) |
| Win Rate | 44.44% | 50.00% | +12.5% |

**Impact**:
- Suspicious improvements alongside behavioral changes
- May indicate overfitting or data leakage
- Could be masking underlying logic errors

**Root Cause Analysis**:

The combination of more trades (300 vs 9) with better risk metrics is unusual because:
1. **More trades typically increase drawdown** due to more exposure
2. **Higher win rate with stop-loss-only exits** suggests stops are placed optimally (suspicious)
3. **Better Sharpe with 33x more trades** suggests each trade has lower variance (unusual)

**Possible explanations**:
1. **Overfitting**: New implementation may be using future information
2. **Risk limits**: Smaller position sizes reducing drawdown
3. **Lucky seed**: MockDataProvider with seed 42 may favor new logic
4. **Tighter stops**: Stop-loss distance may be calculated differently

**Required Investigation**:
- [ ] Profile performance across multiple seeds (not just seed=42)
- [ ] Verify no look-ahead bias in signal generation
- [ ] Compare position sizes between baseline and Phase 2
- [ ] Check stop-loss calculation methods

### ⚠️ WARNING: CPU Time Improvement (Regression #4)

**Symptom**: CPU time improved despite more complex architecture

| Engine | Baseline CPU | Phase 2 CPU | Change |
|--------|--------------|-------------|--------|
| Backtest ml_basic | 7.58s | 3.33s | -56% |
| Live ml_basic | 2.38s | 16.28s | +584% |

**Impact**:
- Backtest faster but live slower
- Inconsistent performance profile suggests different code paths
- 584% CPU increase in live mode is unacceptable for production

**Root Cause Analysis**:

**Backtest speedup** likely due to:
- Vectorized operations in component architecture
- Better caching in prediction engine
- Removal of redundant calculations

**Live slowdown** likely due to:
- Component overhead per candle
- Excessive regime detection calls (errors indicate heavy usage)
- Prediction engine overhead
- Adapter translation layer overhead

**Evidence from errors**:
```
Error calculating indicators: 'RegimeDetector' object has no attribute 'base_detector'
Error detecting regime at index X: 'RegimeDetector' object has no attribute 'detect_regime'
```
These errors appear hundreds of times, suggesting:
- RegimeDetector called on every candle in live mode
- Failed calls still consume CPU time
- Errors not properly cached/suppressed

**Required Investigation**:
- [ ] Profile live engine to identify bottlenecks
- [ ] Fix RegimeDetector missing methods
- [ ] Optimize adapter overhead
- [ ] Cache failed regime detection results

### 🟡 MINOR: RegimeDetector Errors (Regression #5)

**Symptom**: Widespread missing method errors throughout execution

```
Error detecting regime at index X: 'RegimeDetector' object has no attribute 'detect_regime'
Error calculating indicators: 'RegimeDetector' object has no attribute 'base_detector'
```

**Impact**:
- Currently non-fatal but indicates incomplete refactoring
- May cause issues when regime-aware strategies are deployed
- Error spam makes debugging difficult

**Root Cause**:

The component-based `EnhancedRegimeDetector` is missing methods expected by:
- Legacy adapter layer
- Backtesting engine
- Strategy indicator calculation

**Required Fix**:
- [ ] Implement missing `detect_regime` method
- [ ] Add `base_detector` attribute
- [ ] Ensure backward compatibility with legacy interface
- [ ] Add tests to prevent regression

## Git History Analysis

### Key Commits Between Baseline and Phase 2

1. **5e6c23b** - "Gate runtime short entries behind metadata flag" (Oct 11)
   - Added short-selling safety gates
   - May affect entry logic

2. **242b07d** - "Add strategy runtime orchestration and feature interfaces" (Sep 28)
   - Major architectural change
   - Introduced `StrategyRuntime` layer

3. **8235739** - "feat: Add calculate_indicators to MlBasic strategy" (Sep 23)
   - **KEY COMMIT**: Refactored MlBasic to component-based
   - This is when behavioral changes likely began

4. **d06a98c** - "refactor: Transition to component-based architecture in ML strategies" (Sep 20)
   - Broad refactoring of ML strategies
   - Introduced adapter pattern

5. **7634131** - "refactor: Implement component-based architecture and strategy versioning system" (Sep 19)
   - Foundation of new architecture
   - Created `LegacyStrategyAdapter`

### Timeline

```
Sep 10: Baseline captured (Phase 0)
Sep 19-23: Major component-based refactoring
Sep 28: Runtime orchestration added (Phase 1)
Oct 11: Phase 2 benchmarks run (this analysis)
```

**Conclusion**: The behavioral drift was introduced during the Sep 19-23 refactoring sprint, not during Phase 2 work.

## Comparison with Migration Proposal

### Phase 0: Baseline Benchmarking ✅

**Status**: Complete

**Deliverables met**:
- ✅ Benchmark scripts created (`scripts/benchmark_legacy_baseline.py`)
- ✅ Baseline artifacts stored (`artifacts/strategy-migration/baseline/`)
- ✅ Summary reports generated

**Issue**: Baseline captured before component refactoring began, making it incomparable with current code.

### Phase 1: Runtime Foundations ⚠️

**Status**: Partially complete with regressions

**Deliverables met**:
- ✅ `StrategyRuntime` class implemented
- ✅ Feature generator interface created
- ✅ Component strategies updated
- ❌ Unit tests incomplete (RegimeDetector failures)

**Issues**:
- LegacyStrategyAdapter introduces behavioral changes
- Runtime layer not properly tested against legacy
- Feature caching may be skipping computations

### Phase 2: Engine Integration ❌

**Status**: Blocked by Phase 1 regressions

**Deliverables required**:
- ❌ "Identical trade sequences, PnL, and risk metrics"
- ❌ "Backtest throughput remains within agreed bounds"
- ⚠️ "Documentation and guardrails" (partial - short entry gates added)

**Blockers**:
- Trade count mismatch (9 vs 300)
- Live mode behavioral change (0 vs 50 trades)
- No regression harness comparing legacy vs runtime
- CPU performance regression in live mode

## Root Cause: When Did Behavioral Drift Occur?

The proposal states:
> "Before modifying runtime code, run representative backtests (including CPU and wall-clock timing) through the existing legacy pathway."

**What actually happened**:

1. **Sep 10**: Baseline captured using **pre-refactor** code
2. **Sep 19-23**: Major refactoring to component-based architecture
3. **Sep 28**: Runtime orchestration added
4. **Oct 11**: Phase 2 benchmarks run on **post-refactor** code

**The problem**: The baseline was captured on legacy code, but Phase 2 runs on component-based code. We're comparing **two different implementations**, not measuring migration drift.

**The proposal assumed**: Baseline would be captured, **then** migration would begin. But migration (component refactoring) happened **before** Phase 2 benchmarks.

## Behavioral Parity Violations

The migration proposal explicitly requires:

> "Maintain behavioural parity – Results produced by the engines must remain equivalent to the current legacy path for each supported strategy during the migration."

### Violations Identified:

| Requirement | Status | Evidence |
|-------------|--------|----------|
| Identical trade sequences | ❌ FAILED | 9 vs 300 trades |
| Equivalent PnL | ⚠️ PARTIAL | Similar return but different path |
| Same risk metrics | ❌ FAILED | Different drawdown profile |
| Preserved throughput | ⚠️ MIXED | Backtest faster, live slower |

### Why Parity Failed:

The component-based refactoring (Sep 19-23) was **not incremental**. Instead of:
1. Capture baseline
2. Add runtime alongside legacy
3. Validate parity
4. Switch default to runtime

The actual sequence was:
1. Capture baseline (legacy code)
2. **Refactor everything to components immediately**
3. Run new benchmarks (comparing apples to oranges)

This violates the "incremental rollout" principle:

> "Facilitate incremental rollout – Legacy strategies should continue to function (via adapters) until they are ported, but new code paths must be default for component strategies."

## Recommendations

### Immediate Actions (Unblock Phase 2)

1. **Establish True Baseline** (Priority: CRITICAL)
   ```bash
   # Check out pre-refactor code
   git checkout a0c5d53  # Just before component refactoring
   
   # Run benchmarks
   python scripts/benchmark_legacy_baseline.py \
     --strategies ml_basic ml_adaptive \
     --output-dir artifacts/strategy-migration/baseline_pre_refactor
   ```

2. **Create Parity Comparison** (Priority: CRITICAL)
   - Write script to diff trade sequences line-by-line
   - Generate visual comparison of equity curves
   - Document every behavioral difference

3. **Fix RegimeDetector** (Priority: HIGH)
   - Implement missing `detect_regime` method
   - Add `base_detector` attribute
   - Suppress duplicate error logging

4. **Profile Live Engine** (Priority: HIGH)
   - Identify source of 584% CPU increase
   - Optimize component overhead
   - Cache regime detection results

### Corrective Actions (Fix Regressions)

#### Option A: Fix Component Implementation (Recommended)

Modify component-based implementation to match legacy behavior:

1. **Debug trade generation logic**
   - Add detailed logging to `LegacyStrategyAdapter`
   - Compare signal generation between legacy and adapter
   - Fix discrepancies in entry/exit translation

2. **Validate stop-loss logic**
   - Ensure stop distances calculated identically
   - Verify triggering conditions match legacy
   - Test with multiple seeds

3. **Add cooldown periods**
   - Implement minimum bars between trades
   - Prevent immediate re-entry after exit
   - Match legacy behavior

#### Option B: Rollback and Restart (Conservative)

Roll back component refactoring and follow proposal properly:

1. **Revert to baseline code** (commit a0c5d53)
2. **Capture comprehensive baseline** with all strategies
3. **Implement runtime alongside legacy** (not replacing it)
4. **Add parity tests** before switching default
5. **Gradually migrate** one strategy at a time

### Long-term Fixes (Prevent Future Regressions)

1. **Automated Parity Testing**
   ```python
   # Add to CI pipeline
   def test_strategy_parity():
       legacy_result = run_backtest(LegacyMlBasic())
       runtime_result = run_backtest(ComponentMlBasic())
       
       assert_trade_sequences_match(legacy_result, runtime_result)
       assert_pnl_within_tolerance(legacy_result, runtime_result, 0.01)
       assert_risk_metrics_match(legacy_result, runtime_result)
   ```

2. **Regression Harness**
   - Store golden datasets with expected trade sequences
   - Run against every PR
   - Block merges that change behavior without justification

3. **Performance Benchmarks**
   - Set acceptable ranges for CPU/wall time
   - Alert on performance regressions
   - Profile before and after changes

4. **Documentation**
   - Document all intentional behavioral changes
   - Maintain changelog of strategy modifications
   - Version strategies explicitly

## Migration Status Assessment

### Can Phase 2 Proceed? ❌ NO

**Blockers**:
1. No true baseline for comparison
2. Behavioral parity not established
3. Critical regressions unresolved
4. Live engine performance unacceptable

### Recommended Path Forward

**Week 1: Establish Ground Truth**
- Capture true legacy baseline (pre-refactor)
- Document all changes made during Sep 19-23 refactoring
- Classify changes as intentional vs bugs

**Week 2: Fix Critical Regressions**
- Fix RegimeDetector errors
- Debug trade count explosion
- Optimize live engine CPU usage

**Week 3: Implement Parity Testing**
- Build automated comparison tools
- Add regression tests to CI
- Validate current implementation against legacy

**Week 4: Decision Point**
- If parity achievable: Continue with Phase 2
- If parity impossible: Rollback and restart migration

## Conclusion

The Phase 2 benchmarks have revealed that the component-based refactoring (Sep 19-23) introduced **significant behavioral regressions** that violate the migration proposal's core requirements. The issues are **severe enough to block Phase 2 completion** and require immediate investigation and remediation.

### Key Findings:

1. **Behavioral drift is real**: 3233% increase in trades is not explained by performance optimization
2. **Live mode broken**: 0 → 50 trades indicates fundamental logic change
3. **Comparison invalid**: Baseline captured pre-refactor, Phase 2 post-refactor
4. **Migration process violated**: Changes made without parity validation

### Critical Next Steps:

1. ✅ Establish true baseline with pre-refactor code
2. 🔴 Fix trade generation logic to match legacy behavior
3. 🔴 Resolve RegimeDetector errors
4. 🟡 Optimize live engine performance
5. ⚠️ Implement automated parity testing

### Risk Assessment:

**If regressions not fixed**:
- Production deployment will behave unpredictably
- Historical backtests incomparable
- Strategy optimization impossible
- Validation against paper trading invalid

**Timeline impact**: +2-4 weeks to resolve issues before Phase 2 can complete.

---

**References**:
- Migration Proposal: `docs/strategy_migration_proposal.md`
- Phase 2 Analysis: `artifacts/strategy-migration/phase2/PHASE2_ANALYSIS.md`
- Baseline Artifacts: `artifacts/strategy-migration/baseline/`
- Phase 2 Artifacts: `artifacts/strategy-migration/phase2/`

