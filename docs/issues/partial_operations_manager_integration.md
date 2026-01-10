# GitHub Issue: Integrate shared PartialOperationsManager into both engines

**Title:** Integrate shared PartialOperationsManager into both engines

**Labels:** `enhancement`, `refactoring`, `engines`

---

## Summary

The shared `PartialOperationsManager` module was created in `src/engines/shared/partial_operations_manager.py` as part of Issue #454, but integration into the backtest and live engines was deferred due to interface mismatches between existing implementations.

## Background

During the engine consolidation work (#454), we extracted 8 shared modules. Most were integrated successfully:
- ✅ `DynamicRiskHandler` - integrated into both entry handlers
- ✅ `TrailingStopManager` - integrated into both exit handlers
- ⏳ `PartialOperationsManager` - **blocked by interface mismatch**

## Problem: Interface Mismatch

The existing handlers use **different APIs** for partial operations:

### Backtest ExitHandler (`src/engines/backtest/execution/exit_handler.py`)
```python
# Uses a manager with list-returning methods
actions = self.partial_manager.check_partial_exits(state, current_price)  # Returns list
scale = self.partial_manager.check_scale_in_opportunity(state, current_price, indicators)
```

### Live ExitHandler (`src/engines/live/execution/exit_handler.py`)
```python
# Uses PartialExitPolicy with single-action methods
exit_action = self.partial_manager.check_partial_exit(pnl_pct, state)  # Returns single or None
scale_action = self.partial_manager.check_scale_in(pnl_pct, state)
```

### Shared PartialOperationsManager (`src/engines/shared/partial_operations_manager.py`)
```python
# Has a third interface design
check_partial_exit(position, current_price, current_pnl_pct)  # Different signature
check_scale_in(position, current_price, balance, current_pnl_pct)  # Different signature
```

## Proposed Solutions

### Option A: Adapter Pattern (Recommended)
Add adapter methods to the shared manager that support both existing interfaces:

```python
class PartialOperationsManager:
    # Existing methods...

    # Adapter for backtest interface
    def check_partial_exits_list(self, state, current_price) -> list[dict]:
        """Backtest-compatible interface returning list of actions."""
        ...

    def check_scale_in_opportunity(self, state, current_price, indicators) -> dict | None:
        """Backtest-compatible scale-in check."""
        ...

    # Adapter for live interface
    def check_partial_exit_single(self, pnl_pct, state) -> PartialExitAction | None:
        """Live-compatible interface returning single action."""
        ...

    def check_scale_in_single(self, pnl_pct, state) -> ScaleInAction | None:
        """Live-compatible scale-in check."""
        ...
```

### Option B: Unify Handler Interfaces
Modify both handlers to use a single consistent interface. More invasive but cleaner long-term.

### Option C: Wrapper Classes
Create `BacktestPartialOpsAdapter` and `LivePartialOpsAdapter` that wrap the shared manager.

## Files to Modify

1. `src/engines/shared/partial_operations_manager.py` - Add adapter methods
2. `src/engines/backtest/execution/exit_handler.py` - Update `check_partial_operations()` method (lines 152-281)
3. `src/engines/live/execution/exit_handler.py` - Update `check_partial_operations()` method (lines 488-546)

## Current Implementation Details

### Backtest check_partial_operations (lines 152-281)
- Builds `PositionState` from trade
- Calls `self.partial_manager.check_partial_exits(state, current_price)` expecting a list
- Iterates over actions, executes via `self.position_tracker.apply_partial_exit()`
- Calls `self.partial_manager.check_scale_in_opportunity(state, current_price, indicators)`
- Updates risk manager after each operation

### Live check_partial_operations (lines 488-546)
- Loops over all positions
- Builds `PositionState` using `_build_position_state()`
- Calculates PnL percentage
- Calls `self.partial_manager.check_partial_exit(pnl_pct, state)` expecting single action or None
- Calls `self.partial_manager.check_scale_in(pnl_pct, state)`
- Uses `_execute_partial_exit()` and `_execute_scale_in()` helper methods

## Testing Requirements

1. Run existing partial exit tests:
   ```bash
   pytest tests/unit/position_management/test_partial*.py -v
   ```

2. Run full backtest to verify partial exits work:
   ```bash
   atb backtest ml_basic --symbol BTCUSDT --timeframe 1h --days 30
   ```

3. Verify live engine (paper mode):
   ```bash
   atb live ml_basic --symbol BTCUSDT --paper-trading
   ```

4. Run integration tests:
   ```bash
   pytest tests/integration/ -v -k "partial"
   ```

## Acceptance Criteria

- [ ] Both exit handlers use the shared `PartialOperationsManager`
- [ ] All existing partial exit/scale-in tests pass
- [ ] Backtest produces same results before/after integration
- [ ] Live engine partial operations work correctly
- [ ] No duplicate partial exit logic remains in handlers
- [ ] Code coverage maintained or improved

## Related

- Parent Issue: #454 (Extract shared logic between engines)
- Commit with TrailingStopManager integration: `13f3a1c`
- Branch: `claude/extract-shared-engine-logic-vh4lQ`

## Notes

- This is a financial system handling real money - extra care required
- Consider adding comparison tests that verify identical behavior before/after
- The adapter pattern (Option A) is recommended as it's least invasive
