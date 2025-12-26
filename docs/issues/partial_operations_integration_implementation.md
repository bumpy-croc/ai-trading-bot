# PartialOperationsManager Integration - Implementation Report

**Status:** ✅ COMPLETED
**Approach:** Option B - Unified Handler Interfaces
**Date:** 2025-12-25

## Summary

Successfully integrated the shared `PartialOperationsManager` into both backtest and live trading engines using **Option B: Unify Handler Interfaces**. This provides a single, clean architecture with minimal coupling and maximum code reuse.

## Architecture

### Clean Separation of Concerns

```
┌─────────────────────────────────┐
│   PartialExitPolicy             │  ← Configuration (pure data)
│   - exit_targets: list[float]   │
│   - exit_sizes: list[float]     │
│   - scale_in_thresholds: list[float] │
└─────────────────────────────────┘
           ↓ wrapped in
┌─────────────────────────────────┐
│   PartialOperationsManager      │  ← Logic (checking only)
│   - check_partial_exit()        │
│   - check_scale_in()            │
│   Returns: Single next action   │
└─────────────────────────────────┘
           ↓ used by
┌─────────────────────────────────┐
│   ExitHandler (Backtest/Live)   │  ← Coordination
│   - Calls manager in loop        │
│   - Delegates to position tracker│
└─────────────────────────────────┘
           ↓ delegates to
┌─────────────────────────────────┐
│   PositionTracker              │  ← State mutation
│   - apply_partial_exit()       │
│   - apply_scale_in()           │
└─────────────────────────────────┘
```

### Key Design Decisions

1. **Single Action Pattern**: `check_partial_exit()` returns the **next single action** to execute, not a list
   - Cleaner control flow
   - Matches live trading reality (execute, reassess, execute)
   - Both engines loop if multiple exits can trigger simultaneously

2. **Separation of Checking and Application**:
   - `PartialOperationsManager`: ONLY checks and returns results
   - `PositionTracker`: ONLY applies and mutates state
   - No coupling between layers

3. **Configuration Wrapping**:
   - Engines wrap `PartialExitPolicy` in `PartialOperationsManager` at initialization
   - Handlers receive the manager, not the raw policy
   - Clean dependency injection

## Changes Made

### 1. Updated `src/engines/shared/partial_operations_manager.py`

**Before:**
- Expected wrong policy structure (`profit_targets` instead of `exit_targets`)
- Had `apply_*` methods that mutated state (coupling violation)
- Returned dictionaries in inconsistent formats

**After:**
- Works with actual `PartialExitPolicy` structure (`exit_targets`, `exit_sizes`)
- Removed `apply_*` methods (delegated to position trackers)
- Returns typed dataclasses (`PartialExitResult`, `ScaleInResult`)
- Single-action pattern for clean control flow

**Key Methods:**
```python
check_partial_exit(position, current_price, current_pnl_pct=None) -> PartialExitResult
check_scale_in(position, current_price, balance, current_pnl_pct=None) -> ScaleInResult
```

### 2. Updated `src/engines/backtest/execution/exit_handler.py`

**Changes:**
- Import `PartialOperationsManager`
- Constructor accepts `PartialOperationsManager | None`
- `check_partial_operations()` uses unified manager with loop for multiple exits
- Removed dependency on old `PartialExitPolicy.check_partial_exits()` list-returning method

**Conversion Logic:**
```python
# exit_fraction is of ORIGINAL size, need to convert to CURRENT size
exit_size_of_original = result.exit_fraction
exit_size_of_current = exit_size_of_original * trade.original_size / trade.current_size
```

### 3. Updated `src/engines/live/execution/exit_handler.py`

**Changes:**
- Import `PartialOperationsManager`
- Constructor accepts `PartialOperationsManager | None` (was `PartialExitPolicy | None`)
- `check_partial_operations()` uses unified manager (no loop needed - processes per position)
- Removed `_build_position_state()` helper (manager works with position objects directly)

**Conversion Logic:**
```python
# exit_fraction is of ORIGINAL size, need to convert to CURRENT size
exit_size_of_original = exit_result.exit_fraction
current_size_fraction = position.current_size / position.original_size
exit_size_of_current = exit_size_of_original / current_size_fraction
```

### 4. Updated `src/engines/backtest/engine.py`

**Changes:**
- Import `PartialOperationsManager`
- Wrap `partial_manager` (PartialExitPolicy) in `PartialOperationsManager` before passing to handler

**Code:**
```python
partial_ops_manager = (
    PartialOperationsManager(policy=partial_manager) if partial_manager is not None else None
)
```

### 5. Updated `src/engines/live/trading_engine.py`

**Changes:**
- Import `PartialOperationsManager`
- Wrap `self.partial_manager` (PartialExitPolicy) in `PartialOperationsManager` before passing to handler

**Code:**
```python
partial_ops_manager = (
    PartialOperationsManager(policy=self.partial_manager)
    if self.partial_manager is not None
    else None
)
```

## Benefits

### Code Reuse ✅
- Both engines now use identical logic for partial operations
- No duplicate code between backtest and live handlers
- Future changes only need to be made once

### Clean Architecture ✅
- Single interface for both engines
- Clear separation: config → logic → execution
- No cross-layer coupling

### Maintainability ✅
- Changes to partial operations logic isolated to one file
- Type-safe with dataclasses
- Self-documenting code

### Testability ✅
- Manager can be unit tested independently
- Position trackers can be tested independently
- No mocking required for cross-layer interactions

## Risks & Mitigations

### Risk 1: Behavioral Changes
**Risk:** Logic changes might alter backtest/live behavior
**Mitigation:**
- Used same PnL calculation logic as before
- Same target matching logic (check next target based on `partial_exits_taken`)
- Same size calculation (fraction of original position)
- ✅ **Action Required:** Run comparison backtests before/after

### Risk 2: Size Conversion Errors
**Risk:** Converting between "fraction of original" vs "fraction of current" is error-prone
**Mitigation:**
- Documented conversion formulas in code comments
- Same conversion logic both engines already used
- ✅ **Action Required:** Add unit tests for edge cases (e.g., multiple partial exits)

### Risk 3: Loop Termination
**Risk:** Backtest handler uses `while True` loop - could infinite loop
**Mitigation:**
- Loop breaks when `should_exit=False` (no more targets)
- Loop breaks if calculated size <= 0 or > 1.0
- Manager only checks next uncompleted target (not all)

### Risk 4: Test Failures
**Risk:** Existing tests might expect old interface
**Mitigation:**
- Tests mostly interact at engine level, not handler level
- Only one test mentions `partial_manager` (just checks it exists)
- ✅ **Action Required:** Run full test suite

## Validation Checklist

- [x] Both handlers use `PartialOperationsManager`
- [x] No duplicate partial exit logic remains
- [x] Clean architecture with single interface
- [x] Code compiles (type hints correct)
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Backtest produces identical results before/after
- [ ] Live engine (paper mode) works correctly

## Testing Plan

1. **Unit Tests:**
   ```bash
   pytest tests/unit/position_management/test_partial*.py -v
   ```

2. **Integration Tests:**
   ```bash
   pytest tests/integration/ -v -k "partial"
   ```

3. **Backtest Comparison:**
   ```bash
   # With partial exits enabled
   atb backtest ml_basic --symbol BTCUSDT --timeframe 1h --days 30
   ```

4. **Live Engine (Paper):**
   ```bash
   atb live ml_basic --symbol BTCUSDT --paper-trading
   # Monitor for partial exits in logs
   ```

## Backward Compatibility

**Breaking Changes:** None for end users
- Public API unchanged (still configure via `PartialExitPolicy`)
- Engines wrap policy automatically
- CLI commands unchanged

**Internal Changes:** Handler constructors now expect different type
- `ExitHandler(partial_manager=PartialOperationsManager | None)`
- `LiveExitHandler(partial_manager=PartialOperationsManager | None)`
- Only affects direct handler instantiation (not common)

## Future Improvements

1. **Add unit tests** for `PartialOperationsManager` with various policy configurations
2. **Add integration tests** comparing backtest vs live partial operation behavior
3. **Consider** extracting PnL calculation to shared utility (DRY)
4. **Consider** stricter type hints (`position: Trade | LivePosition`)

## Conclusion

Successfully implemented **Option B: Unify Handler Interfaces** with:
- ✅ Single, clean interface
- ✅ No duplicate logic
- ✅ Clear separation of concerns
- ✅ Both engines using shared manager

**Next Steps:**
1. Run full test suite
2. Run comparison backtests
3. Verify live engine (paper mode)
4. Commit and push changes
