# Action Plan: Fix Adapter Behavioral Parity Issues

**Priority**: CRITICAL  
**Goal**: Make component-based strategies produce identical results to legacy strategies  
**Context**: Phase 2 benchmarks show 9 trades (legacy) vs 300 trades (component-based adapter)

---

## Overview

The `LegacyStrategyAdapter` is not maintaining behavioral parity with legacy strategies. This action plan will:
1. Identify the exact points where behavior diverges
2. Fix the adapter translation logic
3. Validate parity is achieved
4. Re-run benchmarks to confirm

---

## Task 1: Create Diagnostic Tooling

**Objective**: Build tools to compare legacy vs adapter signal generation step-by-step

### 1.1 Create Signal Comparison Script

Create `scripts/debug_adapter_parity.py`:

```python
#!/usr/bin/env python3
"""
Debug adapter parity by comparing legacy vs component-based strategy signals.

This script runs both implementations side-by-side on identical data and reports
any divergence in signals, positions, or calculations.
"""

from pathlib import Path
import sys
import pandas as pd
from datetime import datetime, timedelta

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.backtesting.engine import Backtester
from src.data_providers.mock_data_provider import MockDataProvider
from src.risk import RiskParameters
from src.strategies.ml_basic import MlBasic

def compare_strategies_step_by_step():
    """Run both strategies and compare at each step."""
    
    # Setup identical data
    provider = MockDataProvider(interval_seconds=3600, num_candles=721, seed=42)
    end = datetime.utcnow()
    start = end - timedelta(days=30)
    
    symbol = "BTCUSDT"
    timeframe = "1h"
    
    # Get data once
    df = provider.get_historical_data(symbol, timeframe, start, end)
    
    # Create strategy (component-based via adapter)
    strategy = MlBasic()
    
    # Prepare strategy
    df_prepared = strategy.calculate_indicators(df.copy(), timeframe)
    
    # Track divergences
    divergences = []
    signals = []
    
    print(f"Analyzing {len(df_prepared)} candles...")
    print("=" * 80)
    
    for index in range(len(df_prepared)):
        if index < 120:  # Skip warmup period
            continue
            
        # Check entry conditions
        entry_signal = strategy.check_entry_conditions(df_prepared, index)
        
        # Check exit conditions (if we have a position)
        exit_signal = strategy.check_exit_conditions(df_prepared, index, has_position=True)
        
        signals.append({
            'index': index,
            'timestamp': df_prepared.iloc[index]['timestamp'],
            'close': df_prepared.iloc[index]['close'],
            'entry_signal': entry_signal,
            'exit_signal': exit_signal,
        })
        
        if entry_signal:
            print(f"Index {index}: ENTRY SIGNAL - Close: {df_prepared.iloc[index]['close']:.2f}")
            
    print("\n" + "=" * 80)
    print(f"Total entry signals: {sum(1 for s in signals if s['entry_signal'])}")
    print(f"Total candles analyzed: {len(signals)}")
    print(f"Signal frequency: {sum(1 for s in signals if s['entry_signal']) / len(signals) * 100:.2f}%")
    
    return signals

if __name__ == "__main__":
    signals = compare_strategies_step_by_step()
```

**Expected output**: Should show how many entry/exit signals are generated

**Run**:
```bash
python scripts/debug_adapter_parity.py
```

### 1.2 Add Detailed Logging to Adapter

Modify `src/strategies/adapters/legacy_adapter.py` to log signal translations:

```python
# Add at the top
import logging
logger = logging.getLogger(__name__)

# In check_entry_conditions method, add:
def check_entry_conditions(self, df: pd.DataFrame, index: int) -> bool:
    decision = self._get_trading_decision(df, index)
    
    entry_signal = (
        decision.action == TradingAction.ENTER_LONG
        or (decision.action == TradingAction.ENTER_SHORT and self._allow_short_entries(decision))
    )
    
    # ADD THIS LOGGING
    if entry_signal or index % 100 == 0:  # Log entries and every 100 candles
        logger.debug(
            f"Index {index}: action={decision.action}, "
            f"confidence={decision.confidence:.4f}, "
            f"entry_signal={entry_signal}"
        )
    
    return entry_signal
```

---

## Task 2: Root Cause Analysis

**Objective**: Identify why adapter generates different signals than expected

### 2.1 Compare with Baseline Trade Count

From baseline, we know:
- **Expected**: 9 trades over 721 candles (~1.2% entry frequency)
- **Actual**: 300 trades over 721 candles (~41.6% entry frequency)

**Hypothesis 1**: Adapter is generating entry signals too frequently (33x more often)

### 2.2 Inspect Component Signal Generation

Read `src/strategies/components/signal_generators/ml_basic.py` and check:

```python
# Expected behavior for MlBasic:
# - Load ONNX model
# - Predict price direction
# - Generate signal when confidence > threshold
# - Should NOT generate signal every candle
```

**Check**:
1. Is model prediction being called every candle?
2. What's the confidence threshold for entry?
3. Is there a cooldown between signals?

### 2.3 Inspect Adapter Translation Logic

Read `src/strategies/adapters/legacy_adapter.py` and verify:

```python
def check_entry_conditions(self, df: pd.DataFrame, index: int) -> bool:
    # Question: Does this respect:
    # 1. Position state (don't enter if already in position)?
    # 2. Cooldown periods?
    # 3. Confidence thresholds?
```

### 2.4 Run Diagnostic Script

```bash
# Enable debug logging
export PYTHONPATH=/Users/alex/Sites/ai-trading-bot
python scripts/debug_adapter_parity.py 2>&1 | tee artifacts/strategy-migration/phase2/debug_signals.log

# Analyze output
echo "Expected: ~9 entry signals"
echo "Actual: (see output above)"
```

**Document findings** in `artifacts/strategy-migration/phase2/root_cause_findings.md`

---

## Task 3: Investigate Specific Issues

Based on the analysis, investigate these likely causes:

### 3.1 Check if Position State is Tracked

**Issue**: Adapter may not be checking if position already exists before generating entry signal

**Investigation**:
```python
# In legacy_adapter.py, check_entry_conditions should verify:
# - Do we already have a position?
# - If yes, return False (don't enter again)
```

**Test**:
```python
# Add to debug script
def test_position_tracking():
    strategy = MlBasic()
    df = get_test_data()
    
    # First call - should allow entry
    entry1 = strategy.check_entry_conditions(df, 150)
    
    # Simulate we now have position
    # Second call - should NOT allow entry
    entry2 = strategy.check_entry_conditions(df, 151)
    
    print(f"First entry allowed: {entry1}")
    print(f"Second entry allowed (should be False): {entry2}")
```

### 3.2 Check Confidence Thresholds

**Issue**: Component may use different confidence threshold than legacy

**Investigation**:
```python
# In MlBasic (legacy), check what threshold was used:
# OLD: if prediction > 0.5: enter
# NEW: if confidence > ???: enter

# Compare thresholds
```

**Check these files**:
- `src/strategies/ml_basic.py` (component version)
- Look for `SHORT_ENTRY_THRESHOLD` and confidence calculations
- Verify they match legacy behavior

### 3.3 Check Trade Cooldown Logic

**Issue**: No cooldown between trades, causing immediate re-entry after exit

**Investigation**:
```python
# Legacy strategies typically have:
# - Minimum bars between trades (e.g., 5 candles)
# - Or: don't enter within X% of last exit price

# Check if adapter respects this
```

**Test pattern from Phase 2**:
```
Entry → Stop Loss Exit → Immediate Entry → Stop Loss Exit → ...
```
This suggests NO cooldown is being enforced.

### 3.4 Check RegimeDetector Integration

**Issue**: Hundreds of errors about missing `detect_regime` method

**Investigation**:
```bash
# Search for regime detection calls
grep -r "detect_regime" src/strategies/
grep -r "base_detector" src/strategies/

# Check what's calling these missing methods
```

**Fix**: Implement missing methods or remove regime detection calls

---

## Task 4: Implement Fixes

Based on findings from Task 3, implement fixes:

### 4.1 Fix Position State Tracking (if broken)

```python
# In legacy_adapter.py
class LegacyStrategyAdapter:
    def __init__(self, ...):
        self._has_position = False
        self._last_exit_index = -1
        self._cooldown_periods = 5  # Minimum bars between trades
    
    def check_entry_conditions(self, df: pd.DataFrame, index: int) -> bool:
        # Don't enter if we already have position
        if self._has_position:
            return False
        
        # Enforce cooldown
        if index - self._last_exit_index < self._cooldown_periods:
            return False
        
        # Then check actual signal
        decision = self._get_trading_decision(df, index)
        return decision.action in [TradingAction.ENTER_LONG, TradingAction.ENTER_SHORT]
    
    def check_exit_conditions(self, df: pd.DataFrame, index: int, position) -> tuple:
        result = super().check_exit_conditions(df, index, position)
        if result[0]:  # If exiting
            self._has_position = False
            self._last_exit_index = index
        return result
    
    def update_position(self, action: str, price: float, index: int) -> None:
        if action in ["buy", "sell"]:
            self._has_position = True
        elif action == "close":
            self._has_position = False
            self._last_exit_index = index
```

### 4.2 Fix RegimeDetector Errors

```python
# In src/strategies/components/regime/enhanced_detector.py
class EnhancedRegimeDetector:
    
    def __init__(self):
        self.base_detector = None  # Add missing attribute
    
    def detect_regime(self, df: pd.DataFrame, index: int) -> str:
        """Detect regime at given index."""
        # Implement method or return default
        return "neutral"  # or implement actual logic
```

### 4.3 Align Confidence Thresholds

Ensure component-based signal generator uses same thresholds:

```python
# In MLBasicSignalGenerator
class MLBasicSignalGenerator:
    # Verify these match legacy values
    LONG_ENTRY_THRESHOLD = 0.0005   # Must match legacy
    SHORT_ENTRY_THRESHOLD = -0.0005  # Must match legacy
    CONFIDENCE_MULTIPLIER = 12       # Must match legacy
```

### 4.4 Add Trade Cooldown

```python
# In signal generator or adapter
MIN_BARS_BETWEEN_TRADES = 1  # Or whatever legacy used

def should_allow_entry(self, index: int) -> bool:
    if self.last_trade_index is not None:
        bars_since_last = index - self.last_trade_index
        if bars_since_last < MIN_BARS_BETWEEN_TRADES:
            return False
    return True
```

---

## Task 5: Validate Fixes

### 5.1 Re-run Diagnostic Script

```bash
# After implementing fixes
python scripts/debug_adapter_parity.py

# Should now show ~9 entry signals instead of ~300
```

### 5.2 Run Unit Tests

Create `tests/integration/test_adapter_parity.py`:

```python
"""Test that adapter maintains behavioral parity."""

import pytest
from datetime import datetime, timedelta
from src.strategies.ml_basic import MlBasic
from src.data_providers.mock_data_provider import MockDataProvider

def test_adapter_trade_frequency():
    """Verify adapter doesn't generate excessive trades."""
    provider = MockDataProvider(interval_seconds=3600, num_candles=721, seed=42)
    end = datetime.utcnow()
    start = end - timedelta(days=30)
    
    df = provider.get_historical_data("BTCUSDT", "1h", start, end)
    strategy = MlBasic()
    df = strategy.calculate_indicators(df, "1h")
    
    entry_signals = 0
    for index in range(120, len(df)):  # Skip warmup
        if strategy.check_entry_conditions(df, index):
            entry_signals += 1
    
    # Should be close to 9, not 300
    assert entry_signals < 20, f"Too many entry signals: {entry_signals}"
    assert entry_signals >= 5, f"Too few entry signals: {entry_signals}"

def test_adapter_respects_position_state():
    """Verify adapter doesn't enter when already in position."""
    strategy = MlBasic()
    provider = MockDataProvider(interval_seconds=3600, num_candles=200, seed=42)
    end = datetime.utcnow()
    start = end - timedelta(days=10)
    
    df = provider.get_historical_data("BTCUSDT", "1h", start, end)
    df = strategy.calculate_indicators(df, "1h")
    
    # Find an entry signal
    entry_index = None
    for index in range(120, len(df)):
        if strategy.check_entry_conditions(df, index):
            entry_index = index
            break
    
    assert entry_index is not None, "Should find at least one entry signal"
    
    # Simulate taking position
    strategy.update_position("buy", df.iloc[entry_index]['close'], entry_index)
    
    # Next candle should NOT allow entry
    next_entry = strategy.check_entry_conditions(df, entry_index + 1)
    assert not next_entry, "Should not allow entry when already in position"

def test_adapter_cooldown_period():
    """Verify adapter enforces cooldown between trades."""
    strategy = MlBasic()
    # Similar test for cooldown logic
    pass
```

**Run tests**:
```bash
pytest tests/integration/test_adapter_parity.py -v
```

### 5.3 Re-run Full Benchmarks

```bash
# After fixes validated
python scripts/benchmark_legacy_baseline.py \
  --output-dir artifacts/strategy-migration/phase2_fixed \
  --strategies ml_basic ml_adaptive \
  --timeframe 1h \
  --backtest-days 30 \
  --live-steps 50
```

**Expected results**:
- ml_basic backtest: ~9 trades (matching baseline)
- ml_basic live: Similar low trade count (not 50/50)
- Same final PnL and risk metrics as baseline

### 5.4 Compare Results

```bash
# Compare trade counts
echo "Baseline:"
jq '.[] | select(.mode=="backtest" and .strategy=="ml_basic") | .results.total_trades' \
  artifacts/strategy-migration/baseline/baseline_summary.json

echo "Phase 2 Fixed:"
jq '.[] | select(.mode=="backtest" and .strategy=="ml_basic") | .results.total_trades' \
  artifacts/strategy-migration/phase2_fixed/baseline_summary.json

# Should be identical or very close (within 1-2 trades)
```

---

## Task 6: Fix Live Engine CPU Performance

**Issue**: Live engine CPU time increased from 2.38s to 16.28s (+584%)

### 6.1 Profile Live Engine

Create profiling script `scripts/profile_live_engine.py`:

```python
#!/usr/bin/env python3
"""Profile live engine to find performance bottlenecks."""

import cProfile
import pstats
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.live.trading_engine import LiveTradingEngine
from src.data_providers.mock_data_provider import MockDataProvider
from src.risk import RiskParameters
from src.strategies.ml_basic import MlBasic

def profile_live_run():
    strategy = MlBasic()
    provider = MockDataProvider(interval_seconds=3600, num_candles=250, seed=1337)
    risk_params = RiskParameters(base_risk_per_trade=0.01, max_risk_per_trade=0.02)
    
    engine = LiveTradingEngine(
        strategy=strategy,
        data_provider=provider,
        sentiment_provider=None,
        risk_parameters=risk_params,
        check_interval=1,
        initial_balance=10000.0,
        max_position_size=0.2,
        enable_live_trading=False,
        log_trades=False,
        alert_webhook_url=None,
        enable_hot_swapping=False,
        resume_from_last_balance=False,
        database_url="sqlite:///:memory:",
        max_consecutive_errors=3,
        account_snapshot_interval=0,
        enable_dynamic_risk=False,
        enable_partial_operations=False,
    )
    
    engine.start(symbol="BTCUSDT", timeframe="1h", max_steps=50)

if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    
    profile_live_run()
    
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(30)  # Top 30 functions
```

**Run**:
```bash
python scripts/profile_live_engine.py > artifacts/strategy-migration/phase2/live_engine_profile.txt
```

### 6.2 Analyze Bottlenecks

Look for in profile output:
1. Functions called hundreds/thousands of times
2. RegimeDetector calls with errors (wasted CPU)
3. Redundant model predictions
4. Inefficient DataFrame operations

### 6.3 Implement Optimizations

Based on profile, likely optimizations:

```python
# 1. Cache regime detection results
class EnhancedRegimeDetector:
    def __init__(self):
        self._cache = {}
    
    def detect_regime(self, df, index):
        if index in self._cache:
            return self._cache[index]
        result = self._compute_regime(df, index)
        self._cache[index] = result
        return result

# 2. Suppress duplicate error logging
def detect_regime(self, df, index):
    try:
        return self._actual_detection(df, index)
    except AttributeError as e:
        if not hasattr(self, '_error_logged'):
            logger.error(f"Regime detection error: {e}")
            self._error_logged = True
        return "neutral"  # Default fallback

# 3. Batch model predictions if possible
# Instead of: predict(candle_1), predict(candle_2), ...
# Do: predict_batch([candle_1, candle_2, ...])
```

---

## Task 7: Documentation and Validation

### 7.1 Document Findings

Create `artifacts/strategy-migration/phase2/FIX_SUMMARY.md`:

```markdown
# Adapter Parity Fixes - Summary

## Issues Found
1. [Issue description]
2. [Issue description]

## Root Causes
1. [Root cause analysis]
2. [Root cause analysis]

## Fixes Implemented
1. [Fix description + file/line references]
2. [Fix description + file/line references]

## Validation Results
- Trade count: 300 → 9 (matches baseline) ✅
- CPU time: 16.28s → X.XXs (acceptable) ✅
- Behavioral parity: Achieved ✅

## Remaining Issues
- [Any outstanding issues]
```

### 7.2 Update Phase 2 Analysis

Update `artifacts/strategy-migration/phase2/REGRESSION_SUMMARY.md`:

```markdown
# UPDATED: 2025-10-XX

## Status: ✅ REGRESSIONS RESOLVED

All critical issues have been fixed:
- Trade count parity achieved (9 trades in both baseline and phase 2)
- Live CPU performance optimized (XX% improvement)
- Adapter translation logic corrected
- RegimeDetector errors fixed

Phase 2 can now proceed to Phase 3.
```

---

## Task 8: Final Validation

### 8.1 Run Complete Test Suite

```bash
# Unit tests
pytest tests/unit/strategies/ -v

# Integration tests
pytest tests/integration/test_adapter_parity.py -v

# Strategy-specific tests
pytest tests/integration/test_ml_basic.py -v
```

### 8.2 Run Benchmarks on Multiple Seeds

Validate fixes work across different random seeds:

```bash
for seed in 42 1337 2024 9999 12345; do
  echo "Testing with seed=$seed"
  # Modify benchmark script to use this seed
  python scripts/benchmark_legacy_baseline.py \
    --output-dir artifacts/strategy-migration/phase2_seed_${seed} \
    --strategies ml_basic \
    --backtest-days 30
done

# Analyze variance
python scripts/analyze_seed_variance.py
```

### 8.3 Mark Phase 2 Complete

Only after:
- ✅ Trade count matches baseline (9 ±2 trades)
- ✅ Live CPU time < 5s (acceptable for production)
- ✅ All regression tests passing
- ✅ Works across multiple seeds
- ✅ No RegimeDetector errors
- ✅ Documentation updated

---

## Success Criteria

Phase 2 is complete when:

1. **Behavioral Parity**: ✅
   - Component-based strategies produce identical trades to baseline
   - Trade count: 9 (±10% variance acceptable)
   - Final PnL: Within 1% of baseline
   - Risk metrics: Within 5% of baseline

2. **Performance**: ✅
   - Backtest maintains or improves speed
   - Live engine CPU time < 5s for 50 steps
   - No performance regressions vs baseline

3. **Quality**: ✅
   - All tests passing
   - No RegimeDetector errors
   - Code linted (ruff, black)
   - Documentation updated

4. **Validation**: ✅
   - Tested across multiple seeds
   - Automated parity tests in CI
   - Regression harness in place

---

## Deliverables

At completion, ensure these exist:

1. `scripts/debug_adapter_parity.py` - Diagnostic tool
2. `tests/integration/test_adapter_parity.py` - Automated tests
3. `artifacts/strategy-migration/phase2/FIX_SUMMARY.md` - Fix documentation
4. `artifacts/strategy-migration/phase2_fixed/` - Final benchmark results
5. Updated regression analysis showing issues resolved

---

## Estimated Timeline

- **Task 1-2**: 4-6 hours (diagnostic + root cause)
- **Task 3**: 4-6 hours (investigation)
- **Task 4**: 6-8 hours (implement fixes)
- **Task 5**: 2-3 hours (validation)
- **Task 6**: 3-4 hours (performance optimization)
- **Task 7-8**: 2-3 hours (documentation + final validation)

**Total**: 21-30 hours (~3-4 days)

---

## Notes for AI Agent

### Context to Preserve
- Baseline results are correct (ground truth)
- Goal is to make Phase 2 match baseline
- LegacyStrategyAdapter is the likely bug source

### Files to Focus On
1. `src/strategies/adapters/legacy_adapter.py` - Adapter implementation
2. `src/strategies/ml_basic.py` - Component-based strategy
3. `src/strategies/components/signal_generators/ml_basic.py` - Signal generation
4. `src/strategies/components/regime/enhanced_detector.py` - Regime detection

### Key Questions to Answer
1. Why does adapter generate 33x more trades?
2. Is position state being tracked correctly?
3. Are confidence thresholds aligned?
4. Is there a trade cooldown mechanism?
5. Why are RegimeDetector methods missing?

### Red Flags to Watch For
- Always-true conditions in entry logic
- Missing position state checks
- Disabled cooldown periods
- Different thresholds than legacy
- Uncaught exceptions in regime detection

---

**Start with Task 1** and work sequentially. Document findings at each step before proceeding.

