# AI Agent Brief: Fix Adapter Behavioral Parity

## Mission

Fix `LegacyStrategyAdapter` to make component-based strategies produce **identical results** to legacy strategies.

## Problem

**Baseline (legacy code)**: 9 trades, 1.04% return  
**Phase 2 (adapter code)**: 300 trades, 1.47% return  
**Expected**: Identical behavior

## Root Cause

The `LegacyStrategyAdapter` in `src/strategies/adapters/legacy_adapter.py` is not correctly translating component-based strategy signals into legacy-compatible behavior.

## Your Tasks

### 1. Diagnose the Issue

Create `scripts/debug_adapter_parity.py` to compare signal generation:

```python
#!/usr/bin/env python3
"""Compare legacy vs adapter signals step-by-step."""
from pathlib import Path
import sys
from datetime import datetime, timedelta

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_providers.mock_data_provider import MockDataProvider
from src.strategies.ml_basic import MlBasic

def main():
    # Setup test data (same as baseline)
    provider = MockDataProvider(interval_seconds=3600, num_candles=721, seed=42)
    end = datetime.utcnow()
    start = end - timedelta(days=30)
    df = provider.get_historical_data("BTCUSDT", "1h", start, end)
    
    # Test component-based strategy
    strategy = MlBasic()
    df = strategy.calculate_indicators(df, "1h")
    
    # Count signals
    entry_count = 0
    for index in range(120, len(df)):  # Skip warmup
        if strategy.check_entry_conditions(df, index):
            entry_count += 1
            print(f"Index {index}: Entry signal at close={df.iloc[index]['close']:.2f}")
    
    print(f"\nTotal entries: {entry_count} (expected: ~9)")
    return entry_count

if __name__ == "__main__":
    main()
```

**Run it**: `python scripts/debug_adapter_parity.py`

**Question to answer**: Why 300 signals instead of 9?

### 2. Investigate Likely Causes (4 hours)

Check these in `src/strategies/adapters/legacy_adapter.py`:

#### A. Position State Tracking
```python
# Does check_entry_conditions verify we don't already have a position?
# Look for: if self._has_position: return False
```

#### B. Trade Cooldown
```python
# Is there a minimum bars between trades?
# Look for: if index - self._last_trade_index < COOLDOWN: return False
```

#### C. Signal Translation
```python
# In check_entry_conditions, verify:
decision = self._get_trading_decision(df, index)
# Is decision.action being translated correctly?
# Are confidence thresholds aligned with legacy?
```

#### D. RegimeDetector Errors
```bash
# Search for errors
grep -n "detect_regime" src/strategies/components/regime/
# These methods are missing - need to implement or stub
```

### 3. Implement Fixes (6 hours)

Based on findings, fix the adapter. Example fixes:

#### Fix A: Add Position Tracking
```python
# In LegacyStrategyAdapter.__init__
self._current_position = None
self._last_exit_index = -1
self._cooldown_bars = 1  # Minimum bars between trades

# In check_entry_conditions
def check_entry_conditions(self, df: pd.DataFrame, index: int) -> bool:
    # Don't enter if we have position
    if self._current_position is not None:
        return False
    
    # Enforce cooldown
    if index - self._last_exit_index < self._cooldown_bars:
        return False
    
    # Then check actual signal
    decision = self._get_trading_decision(df, index)
    return decision.action in [TradingAction.ENTER_LONG, TradingAction.ENTER_SHORT]

# In update_position
def update_position(self, action: str, price: float, index: int) -> None:
    if action in ["buy", "sell"]:
        self._current_position = {"action": action, "price": price, "index": index}
    elif action == "close":
        self._current_position = None
        self._last_exit_index = index
    super().update_position(action, price, index)
```

#### Fix B: Implement Missing RegimeDetector Methods
```python
# In src/strategies/components/regime/enhanced_detector.py
class EnhancedRegimeDetector:
    def __init__(self):
        self.base_detector = None  # Add missing attribute
    
    def detect_regime(self, df: pd.DataFrame, index: int) -> str:
        """Detect current market regime."""
        try:
            # Implement actual logic or return default
            return "neutral"
        except Exception as e:
            # Suppress repeated errors
            if not hasattr(self, '_error_logged'):
                import logging
                logging.getLogger(__name__).warning(f"Regime detection disabled: {e}")
                self._error_logged = True
            return "neutral"
```

### 4. Validate Fixes (3 hours)

#### A. Re-run Diagnostic
```bash
python scripts/debug_adapter_parity.py
# Should now show ~9 entries, not 300
```

#### B. Create Automated Test
Create `tests/integration/test_adapter_parity.py`:

```python
"""Test adapter maintains parity with legacy."""
import pytest
from datetime import datetime, timedelta
from src.strategies.ml_basic import MlBasic
from src.data_providers.mock_data_provider import MockDataProvider

def test_adapter_signal_frequency():
    """Adapter should generate ~9 signals, not 300."""
    provider = MockDataProvider(interval_seconds=3600, num_candles=721, seed=42)
    end = datetime.utcnow()
    start = end - timedelta(days=30)
    df = provider.get_historical_data("BTCUSDT", "1h", start, end)
    
    strategy = MlBasic()
    df = strategy.calculate_indicators(df, "1h")
    
    signals = sum(1 for i in range(120, len(df)) if strategy.check_entry_conditions(df, i))
    
    assert signals < 20, f"Too many signals: {signals} (expected ~9)"
    assert signals >= 5, f"Too few signals: {signals} (expected ~9)"
    
def test_adapter_position_tracking():
    """Adapter should not enter when already in position."""
    strategy = MlBasic()
    provider = MockDataProvider(interval_seconds=3600, num_candles=200, seed=42)
    end = datetime.utcnow()
    start = end - timedelta(days=10)
    df = provider.get_historical_data("BTCUSDT", "1h", start, end)
    df = strategy.calculate_indicators(df, "1h")
    
    # Find entry signal
    entry_index = next((i for i in range(120, len(df)) if strategy.check_entry_conditions(df, i)), None)
    assert entry_index is not None
    
    # Simulate position
    strategy.update_position("buy", df.iloc[entry_index]['close'], entry_index)
    
    # Should not allow second entry
    assert not strategy.check_entry_conditions(df, entry_index + 1)
```

**Run**: `pytest tests/integration/test_adapter_parity.py -v`

#### C. Re-run Full Benchmarks
```bash
python scripts/benchmark_legacy_baseline.py \
  --output-dir artifacts/strategy-migration/phase2_fixed \
  --strategies ml_basic \
  --timeframe 1h \
  --backtest-days 30 \
  --live-steps 50
```

**Compare**:
```bash
echo "Baseline trades:"
jq '.[] | select(.mode=="backtest") | .results.total_trades' \
  artifacts/strategy-migration/baseline/baseline_summary.json

echo "Fixed trades:"
jq '.[] | select(.mode=="backtest") | .results.total_trades' \
  artifacts/strategy-migration/phase2_fixed/baseline_summary.json
```

Should be identical (9 trades).

### 5. Optimize Live Engine (3 hours)

**Issue**: Live CPU time 2.38s → 16.28s (+584%)

Profile it:
```python
# Create scripts/profile_live_engine.py
import cProfile, pstats
# ... run live engine with 50 steps
# Output: artifacts/strategy-migration/phase2/profile.txt
```

**Look for**:
- RegimeDetector called hundreds of times with errors (cache it)
- Redundant model predictions (batch them)
- Inefficient DataFrame operations (optimize)

### 6. Document and Close (2 hours)

Update `artifacts/strategy-migration/phase2/FIX_SUMMARY.md`:

```markdown
# Fixes Applied

## Issues Found
1. Adapter not tracking position state - entered on every signal
2. No cooldown between trades - immediate re-entry after exit
3. RegimeDetector missing methods - errors on every call

## Fixes
1. Added position tracking in LegacyStrategyAdapter
2. Added 1-bar cooldown between trades
3. Implemented stub methods in EnhancedRegimeDetector

## Results
- Trade count: 300 → 9 ✅
- Behavioral parity: Achieved ✅
- Live CPU: 16.28s → Xs ✅

## Phase 2 Status
✅ COMPLETE - Can proceed to Phase 3
```

## Success Criteria

✅ You're done when:
1. Diagnostic script shows ~9 signals (not 300)
2. Automated tests pass
3. Full benchmark matches baseline
4. Live CPU time < 5s
5. No RegimeDetector errors in logs

## Key Files to Modify

1. `src/strategies/adapters/legacy_adapter.py` - Main adapter logic
2. `src/strategies/components/regime/enhanced_detector.py` - Fix missing methods
3. Create `scripts/debug_adapter_parity.py` - Diagnostic tool
4. Create `tests/integration/test_adapter_parity.py` - Automated tests

## Context

- **Baseline is correct** - 9 trades is the expected behavior
- **Goal**: Make Phase 2 match baseline exactly
- **Don't change strategy logic** - fix the adapter translation
- **Priority**: Behavioral parity over performance

## Quick Start

```bash
# 1. Run diagnostic
python scripts/debug_adapter_parity.py

# 2. Fix issues in legacy_adapter.py

# 3. Re-run diagnostic (should show ~9 signals)

# 4. Run tests
pytest tests/integration/test_adapter_parity.py -v

# 5. Full validation
python scripts/benchmark_legacy_baseline.py \
  --output-dir artifacts/strategy-migration/phase2_fixed \
  --strategies ml_basic
```

## Estimated Time

- Total: 21-30 hours (3-4 days)
- Critical path: Tasks 1-4 (diagnosis + fixes)
- Can parallelize: Performance optimization while fixes are tested

---

**Full details**: See `ACTION_PLAN.md` in this directory  
**Project root**: `/Users/alex/Sites/ai-trading-bot`

