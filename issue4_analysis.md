# Issue 4 Analysis: Partial Exit Loop Divergence

## Executive Summary

**Critical Finding**: The backtest and live engines have fundamentally different behaviors for partial exit execution when price gaps significantly between cycles.

- **Backtest**: Executes **ALL** triggered partial exits in a single cycle (via while loop)
- **Live/Shared**: Executes **ONE** partial exit per cycle (returns on first match)

This creates **incorrect financial calculations** and **undermines backtesting accuracy** for a real money system.

---

## The Problem

### Scenario
Position entered at $100 with partial exit targets:
- Target 1: +10% ($110)
- Target 2: +20% ($120)
- Target 3: +30% ($130)

Price jumps from $100 to $140 between cycles.

### Current Behavior

**Backtest Engine** (`src/engines/backtest/execution/exit_handler.py:192-228`):
```python
# Uses PartialExitPolicy.check_partial_exits() -> returns LIST
actions = self.partial_manager.check_partial_exits(state, current_price)
for act in actions:  # Executes ALL triggered exits
    pnl = self.position_tracker.apply_partial_exit(...)
    realized_pnl += pnl
    state.current_size -= exec_frac
    state.partial_exits_taken += 1
```
✅ **Result**: All 3 exits execute in one cycle at $140

**PartialOperationsManager** (`src/engines/shared/partial_operations_manager.py:86-141`):
```python
def check_partial_exit(self, position, current_price, current_pnl_pct=None) -> PartialExitResult:
    for i, target in enumerate(targets):
        if i < partial_exits_taken:
            continue

        if current_pnl_pct >= target_pct and exit_fraction > 0:
            return PartialExitResult(...)  # ❌ RETURNS IMMEDIATELY - only first!

    return PartialExitResult()
```
❌ **Result**: Only exit 1 executes. Exits 2 and 3 require additional cycles.

### Financial Impact

| Metric | Backtest (all exits) | Live (one exit) | Impact |
|--------|---------------------|-----------------|---------|
| **Realized PnL** | 3 exits worth | 1 exit worth | -66% immediate realization |
| **Position risk** | Reduced by all targets | Reduced by one target | 3x more exposure |
| **Cash available** | More for next trade | Less available | Liquidity mismatch |
| **Fee calculation** | 3 exits in one candle | Spread across cycles | Timing differences |
| **Slippage modeling** | All at $140 | Potentially different prices | Execution price risk |

**This makes backtesting results UNRELIABLE for production deployment.**

---

## Root Cause

### Code Structure

There are **THREE** different partial exit interfaces in the codebase:

1. **`PartialExitPolicy`** (`src/position_management/partial_manager.py:62-87`)
   - Method: `check_partial_exits()` (plural)
   - Returns: `list[dict]` via **while loop**
   - Used by: Backtest engine

2. **`PartialOperationsManager`** (`src/engines/shared/partial_operations_manager.py:86-141`)
   - Method: `check_partial_exit()` (singular)
   - Returns: `PartialExitResult` (single)
   - Used by: Intended for both engines (but broken interface)

3. **Live engine** (`src/engines/live/execution/exit_handler.py:524`)
   - Calls non-existent methods on PartialExitPolicy
   - Code appears broken/incomplete

### Key Code Comparison

**PartialExitPolicy.check_partial_exits()** (CORRECT - returns all):
```python
def check_partial_exits(self, position: PositionState, current_price: float) -> list[dict]:
    actions: list[dict] = []
    pnl = self._pnl_pct(position, current_price)

    next_idx = position.partial_exits_taken
    while next_idx < len(self.exit_targets):  # ← WHILE LOOP
        target = self.exit_targets[next_idx]
        if pnl >= target:
            actions.append({...})  # ← APPEND to list
            next_idx += 1
        else:
            break

    return actions  # ← Returns ALL triggered exits
```

**PartialOperationsManager.check_partial_exit()** (BROKEN - returns one):
```python
def check_partial_exit(self, position, current_price, current_pnl_pct=None) -> PartialExitResult:
    for i, target in enumerate(targets):
        if i < partial_exits_taken:
            continue

        if current_pnl_pct >= target_pct and exit_fraction > 0:
            return PartialExitResult(...)  # ← EARLY RETURN - only first match!

    return PartialExitResult()
```

---

## Options Analysis

### Option 1: Change Backtest to Execute One Exit Per Cycle

**Implementation**: Modify backtest to only execute the first triggered exit per cycle.

**Pros**:
- ✅ Simple change to backtest code
- ✅ More conservative (potentially safer)

**Cons**:
- ❌ **Less realistic**: With limit orders, all would execute
- ❌ **Changes historical results**: All past backtests become invalid
- ❌ **Wrong direction**: Live should match reality, not backtest's limitation
- ❌ **Slower risk reduction**: Keeps more capital exposed longer than necessary
- ❌ **Not how exchanges work**: Limit orders at $110, $120, $130 ALL fill when price hits $140

**Verdict**: ❌ **NOT RECOMMENDED** - Makes backtesting less realistic

---

### Option 2: Change Live/Shared to Execute All Exits Per Cycle ⭐ RECOMMENDED

**Implementation**: Modify `PartialOperationsManager.check_partial_exit()` to return ALL triggered exits.

**Pros**:
- ✅ **Realistic modeling**: Matches how limit orders actually work on exchanges
- ✅ **Preserves backtest results**: No need to invalidate historical testing
- ✅ **Parity achieved**: Both engines behave identically
- ✅ **Financial correctness**: Proper PnL realization timing
- ✅ **Better risk management**: Exits trigger when they should
- ✅ **Shared code works correctly**: The `shared/` module provides identical behavior
- ✅ **Standard practice**: Professional trading bots use limit orders for take-profit levels

**Cons**:
- ⚠️ Requires careful implementation of list-returning interface
- ⚠️ Need to update both engine handlers to process lists
- ⚠️ More testing required

**Verdict**: ✅ **STRONGLY RECOMMENDED** - This is the robust, correct solution

---

### Option 3: Document Difference and Add Warnings

**Implementation**: Keep divergent behavior, add documentation/warnings.

**Pros**:
- ✅ No code changes

**Cons**:
- ❌ **Doesn't solve the problem**: Parity still broken
- ❌ **Unacceptable for real money**: Users get different live vs backtest results
- ❌ **Professional trading no-go**: Industry standard is parity
- ❌ **Liability risk**: Known inaccurate backtests with real money

**Verdict**: ❌ **UNACCEPTABLE** - Not a solution

---

## Recommended Solution: Option 2

### Why This Is More Robust Long-Term

#### 1. **Realistic Execution Model**

In real automated trading:
- Traders place **limit orders** at all take-profit levels when entering positions
- When price gaps up (e.g., overnight move, high volatility), **all orders that were hit execute**
- This is exactly what the backtest currently models
- It's standard exchange behavior

#### 2. **Correct Financial Calculations**

```
Scenario: $10,000 position, targets at +10%, +20%, +30% (each 25% of position)
Price gaps from entry to +35%

Option 1 (one per cycle):
Cycle 1: Realize $250 (+10% on $2,500)
Cycle 2: Realize $500 (+20% on $2,500)
Cycle 3: Realize $750 (+30% on $2,500)
Total: $1,500 realized across 3 cycles

Option 2 (all at once) - CORRECT:
Cycle 1: Realize $1,500 (all targets hit)
- $250 from first 25% at +10%
- $500 from second 25% at +20%
- $750 from third 25% at +30%
Total: $1,500 realized in 1 cycle
```

The PnL is the same eventually, but:
- **Risk exposure** is different (held longer in Option 1)
- **Available capital** timing differs (reinvestment opportunities)
- **Fee structure** may differ (exchange discounts for batched orders)
- **Slippage modeling** changes (price might move between cycles)

#### 3. **Industry Standards**

Professional trading systems:
- Use limit orders for take-profit levels (industry standard)
- Expect all limit orders to fill when price crosses them
- Backtest engines simulate this accurately
- Live engines must match this behavior

#### 4. **Parity Is Critical**

For a system handling real money:
- Backtest results must reliably predict live performance
- Traders rely on backtest metrics (Sharpe, max drawdown, win rate)
- Divergent behavior undermines confidence
- Could lead to unexpected losses in production

#### 5. **Code Architecture**

The `PartialOperationsManager` is in `src/engines/shared/`:
- Intended to provide **identical behavior** for both engines
- Currently broken (returns single instead of list)
- Fixing it achieves the architectural goal

---

## Implementation Plan

### Changes Required

**File**: `src/engines/shared/partial_operations_manager.py`

**Modify** `check_partial_exit()` method:

```python
def check_partial_exits(  # Rename to plural
    self,
    position: Any,
    current_price: float,
    current_pnl_pct: float | None = None,
) -> list[PartialExitResult]:  # Return list instead of single
    """Check for ALL triggered partial exits.

    Returns list of all exits that should execute this cycle.
    Matches limit order execution model.
    """
    if self.policy is None:
        return []

    results = []  # Collect all triggered exits

    # ... (calculate pnl_pct)

    targets = getattr(self.policy, "profit_targets", [])
    partial_exits_taken = getattr(position, "partial_exits_taken", 0)

    # Check ALL targets, not just first
    for i, target in enumerate(targets):
        if i < partial_exits_taken:
            continue

        target_pct = target.get("profit_pct", 0)
        exit_fraction = target.get("exit_fraction", 0)

        if current_pnl_pct >= target_pct and exit_fraction > 0:
            results.append(PartialExitResult(
                should_exit=True,
                exit_fraction=exit_fraction,
                target_index=i,
                reason=f"Partial exit target {i+1}: {current_pnl_pct:.2%} >= {target_pct:.2%}",
            ))
        else:
            break  # Ordered targets - stop on first miss

    return results
```

**Update** both engine handlers to:
1. Call `check_partial_exits()` (plural)
2. Iterate over returned list
3. Execute all exits in sequence
4. Update position state after each

### Testing Required

1. **Unit tests**: Verify all targets trigger when price gaps
2. **Integration tests**: Confirm parity between engines
3. **Regression tests**: Ensure historical backtest behavior preserved
4. **Financial tests**: Validate PnL calculations match expected values

---

## Conclusion

**Option 2 is the robust, correct solution** because:

1. ✅ **Realistic**: Matches how exchanges and limit orders actually work
2. ✅ **Accurate**: Provides correct financial calculations
3. ✅ **Parity**: Achieves engine consistency (critical for real money)
4. ✅ **Standard**: Aligns with professional trading practices
5. ✅ **Architectural**: Fulfills the purpose of shared code modules
6. ✅ **Safe**: Preserves existing backtest results and behavior

The system handles **real money**. Backtesting accuracy is not optional—it's a requirement for responsible trading system design. Option 2 ensures users can trust their backtest results when deploying strategies with real capital.
