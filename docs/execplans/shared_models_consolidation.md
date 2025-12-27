# ExecPlan: Shared Models Consolidation

**Created:** 2025-12-26
**Status:** Planning
**Branch:** `claude/plan-shared-models-hVzfd`

## Purpose

Consolidate position and trade models into a unified architecture shared by both backtesting and live trading engines. This ensures:
- **Parity**: Identical behavior and data structures between engines
- **Accuracy**: Single source of truth for financial calculations
- **Maintainability**: No duplicate code or diverging implementations
- **Type Safety**: Consistent type system with proper enum usage
- **Fault Tolerance**: Validated, normalized data structures

## Current State Analysis

### Existing Model Implementations

#### 1. Shared Models (`src/engines/shared/models.py`) ✅
**Status:** Well-designed, comprehensive, but NOT integrated

- `BasePosition`:
  - Uses `PositionSide` enum (LONG/SHORT)
  - Has `metadata` dict for extensibility
  - Validates and normalizes in `__post_init__` (clamps size, converts strings)
  - Includes all partial operations fields
  - Has convenience methods: `is_long()`, `is_short()`, `side_str`

- `BaseTrade`:
  - Complete trade record with MFE/MAE tracking
  - Uses `PositionSide` enum
  - Has `metadata` dict for extensibility
  - Includes duration calculation helpers

- Type aliases: `Position = BasePosition`, `Trade = BaseTrade`

#### 2. Backtest Models (`src/engines/backtest/models.py`) ⚠️
**Status:** Simpler, uses strings, integrated throughout backtest engine

- `ActiveTrade`:
  - Uses string side ("long"/"short")
  - Simpler `__post_init__` (just size clamping)
  - Has `component_notional` field (backtest-specific)
  - Missing `metadata` field

- `Trade`:
  - Basic completed trade record
  - Uses string side
  - Has all MFE/MAE fields
  - No helper methods

**Used in:** 9+ files across backtest engine

#### 3. Live Models (`src/engines/live/execution/position_tracker.py`) ⚠️
**Status:** Live-specific, integrated throughout live engine

- `LivePosition`:
  - **DUPLICATE** `PositionSide` enum definition (conflicts with shared)
  - Live-specific fields: `unrealized_pnl`, `unrealized_pnl_percent`, `order_id`
  - Has partial operations fields
  - No validation in constructor

- Result classes:
  - `PositionCloseResult`: Different from backtest version
  - `PartialExitResult`: **CONFLICTS** with shared version (different structure)
  - `ScaleInResult`: **CONFLICTS** with shared version (different structure)

**Used in:** 6+ files across live engine

#### 4. Shared Result Classes (`src/engines/shared/partial_operations_manager.py`) ⚠️
**Status:** Well-designed but conflicts with live versions

- `PartialExitResult`: Has `should_exit`, `exit_fraction`, `target_index`, `reason`
- `ScaleInResult`: Has `should_scale`, `scale_fraction`, `target_index`, `reason`

**Conflict:** Live's result classes have `realized_pnl`, `new_current_size`, `new_size`

### Key Conflicts & Issues

| Issue | Impact | Risk |
|-------|--------|------|
| **Duplicate `PositionSide` enums** | Type inconsistency, import confusion | Medium |
| **Conflicting result classes** | Cannot use both, breaks type system | High |
| **String vs Enum side** | Type safety, validation issues | Medium |
| **Different field semantics** | Behavior divergence between engines | High |
| **No shared result types** | Cannot reuse close/exit logic | Medium |

### Files Affected (17+ files)

**Backtest Engine:**
- `src/engines/backtest/engine.py`
- `src/engines/backtest/execution/entry_handler.py`
- `src/engines/backtest/execution/execution_engine.py`
- `src/engines/backtest/execution/exit_handler.py`
- `src/engines/backtest/execution/position_tracker.py`
- `src/engines/backtest/logging/event_logger.py`
- `src/engines/backtest/__init__.py`

**Live Engine:**
- `src/engines/live/trading_engine.py`
- `src/engines/live/execution/execution_engine.py`
- `src/engines/live/execution/entry_handler.py`
- `src/engines/live/execution/exit_handler.py`
- `src/engines/live/execution/position_tracker.py`
- `src/engines/live/health/health_monitor.py`
- `src/engines/live/logging/event_logger.py`
- `src/engines/live/__init__.py`

**Shared:**
- `src/engines/shared/models.py`
- `src/engines/shared/partial_operations_manager.py`

**Tests:**
- Multiple unit and integration tests

## Architectural Decision: Composition Over Inheritance

### Proposed Architecture

```
src/engines/shared/models.py
├── PositionSide (enum)          # Single source, no duplicates
├── OrderStatus (enum)           # Shared order states
│
├── BasePosition                 # Core position data
│   ├── Common fields (symbol, side, entry_price, etc.)
│   ├── Partial operations state
│   ├── Trailing stop state
│   └── metadata dict (engine-specific extensions)
│
├── BaseTrade                    # Core trade record
│   ├── Entry/exit data
│   ├── MFE/MAE metrics
│   └── metadata dict
│
├── PositionCloseResult          # Unified close result
│   ├── realized_pnl             # Financial data
│   ├── realized_pnl_percent
│   ├── exit_price
│   ├── exit_time
│   └── mfe_mae_metrics
│
└── PositionOperationResult      # NEW: Unified partial ops result
    ├── operation_type           # "partial_exit" | "scale_in"
    ├── realized_pnl (optional)  # Only for exits
    ├── new_size
    ├── new_current_size
    ├── operations_count
    ├── target_index
    └── reason
```

### Engine-Specific Extensions

**Backtest:** Use `metadata` dict for `component_notional`
```python
position.metadata["component_notional"] = 1000.0
```

**Live:** Use `metadata` dict for `unrealized_pnl`, `order_id`
```python
position.metadata["order_id"] = "abc123"
position.metadata["unrealized_pnl"] = 150.0
position.metadata["unrealized_pnl_percent"] = 15.0
```

**Alternative:** Create lightweight engine-specific wrappers that inherit from base:
```python
@dataclass
class LivePosition(BasePosition):
    """Live trading position with real-time tracking."""
    order_id: str | None = None
    unrealized_pnl: float = 0.0
    unrealized_pnl_percent: float = 0.0
```

## Migration Strategy

### Phase 1: Resolve Enum Conflicts ✅ LOW RISK
**Goal:** Single `PositionSide` enum, remove duplicates

1. **Remove duplicate** `PositionSide` from `src/engines/live/execution/position_tracker.py`
2. **Update imports** in live engine to use `from src.engines.shared.models import PositionSide`
3. **Update `__init__.py`** exports to re-export from shared
4. **Verify:** No import errors, all tests pass

**Risk:** Low - simple import change
**Rollback:** Restore duplicate enum
**Testing:** `pytest tests/unit/live/ -v`

### Phase 2: Unify Result Classes 🔄 MEDIUM RISK
**Goal:** Single set of result classes for position operations

**Option A: Merge into shared (RECOMMENDED)**
- Enhance shared `PartialExitResult` to include `realized_pnl`, `new_current_size`
- Enhance shared `ScaleInResult` to include `new_size`, `new_current_size`
- Keep decision-making fields (`should_exit`, `target_index`, `reason`)
- Remove duplicates from `live/execution/position_tracker.py`

**Option B: Create new unified class**
- New `PositionOperationResult` with operation_type discriminator
- Single class handles both partial exits and scale-ins
- Cleaner but requires more changes

**Recommended: Option A** - Less disruptive, maintains existing patterns

**Changes:**
```python
# src/engines/shared/partial_operations_manager.py
@dataclass
class PartialExitResult:
    should_exit: bool = False
    exit_fraction: float | None = None
    target_index: int | None = None
    reason: str | None = None
    # NEW: Add live engine fields
    realized_pnl: float | None = None
    new_current_size: float | None = None
    operations_count: int | None = None

@dataclass
class ScaleInResult:
    should_scale: bool = False
    scale_fraction: float | None = None
    target_index: int | None = None
    reason: str | None = None
    # NEW: Add live engine fields
    new_size: float | None = None
    new_current_size: float | None = None
    operations_count: int | None = None
```

**Migration:**
1. Enhance shared result classes with additional fields (nullable)
2. Update `PartialOperationsManager` to NOT populate these fields (stays decision-only)
3. Update `LivePositionTracker.apply_partial_exit()` to populate all fields
4. Update `BacktestPositionTracker.apply_partial_exit()` to populate all fields
5. Remove duplicate classes from live position_tracker
6. Update all imports and usages

**Risk:** Medium - changes method signatures
**Rollback:** Restore duplicate classes, revert imports
**Testing:** Full integration tests for both engines

### Phase 3: Consolidate Position Models 🔄 HIGH RISK
**Goal:** Both engines use `BasePosition` or engine-specific subclasses

**Option A: Direct Replacement (AGGRESSIVE)**
- Replace `ActiveTrade` with `BasePosition` in backtest
- Replace `LivePosition` with `BasePosition` in live
- Use `metadata` dict for engine-specific fields
- **Pros:** Maximum consolidation, single source of truth
- **Cons:** High risk, many changes, metadata less type-safe

**Option B: Inheritance (BALANCED - RECOMMENDED)**
- Create `BacktestPosition(BasePosition)` with `component_notional` field
- Create `LivePosition(BasePosition)` with `order_id`, `unrealized_pnl` fields
- Engines use their specific types, but inherit all common logic
- **Pros:** Type-safe, gradual migration, clear ownership
- **Cons:** Still have multiple classes (but shared base)

**Option C: Protocol/Interface (CONSERVATIVE)**
- Define `PositionProtocol` that both implement
- Keep existing classes, ensure they conform
- **Pros:** Minimal changes, very safe
- **Cons:** Still have duplicated code, no real consolidation

**Recommended: Option B (Inheritance)**

**Implementation:**
```python
# src/engines/shared/models.py (enhanced)
@dataclass
class BasePosition:
    """Core position fields shared by all engines."""
    # All existing fields
    ...

# src/engines/backtest/models.py
from src.engines.shared.models import BasePosition

@dataclass
class BacktestPosition(BasePosition):
    """Backtest-specific position with simulation fields."""
    component_notional: float | None = None

    # Alias for backward compatibility
    exit_price: float | None = None
    exit_time: datetime | None = None
    exit_reason: str | None = None

# Deprecation alias
ActiveTrade = BacktestPosition

# src/engines/live/execution/position_tracker.py
from src.engines.shared.models import BasePosition

@dataclass
class LivePosition(BasePosition):
    """Live trading position with real-time tracking."""
    order_id: str | None = None
    unrealized_pnl: float = 0.0
    unrealized_pnl_percent: float = 0.0
```

**Migration Steps:**
1. **Enhance `BasePosition`** with any missing fields from both engines
2. **Create `BacktestPosition(BasePosition)`** in backtest/models.py
3. **Move `LivePosition` to inherit** from `BasePosition`
4. **Add deprecation aliases** (`ActiveTrade = BacktestPosition`)
5. **Update position trackers** to use new types
6. **Update entry/exit handlers** incrementally
7. **Update imports** across all files
8. **Remove deprecated aliases** after verification period

**Risk:** High - touches core data structures
**Rollback:** Revert to original models, restore imports
**Testing:**
- Unit tests for both engines
- Integration tests for both engines
- Backtest comparison (before/after must match exactly)
- Live paper trading test

### Phase 4: Consolidate Trade Models 🔄 MEDIUM RISK
**Goal:** Both engines use `BaseTrade`

**Changes:**
1. Backtest's `Trade` already matches `BaseTrade` structure
2. Simply replace with shared version
3. Update imports
4. Verify database logging uses correct fields

**Risk:** Medium - affects completed trade records
**Rollback:** Restore backtest Trade class
**Testing:** Database integration tests, backtest result comparison

### Phase 5: Update Exports & Cleanup 🔄 LOW RISK
**Goal:** Clean public API, proper re-exports

**Changes:**
1. **Update `src/engines/backtest/__init__.py`:**
   ```python
   from src.engines.shared.models import PositionSide, BaseTrade as Trade
   from .models import BacktestPosition as ActiveTrade
   ```

2. **Update `src/engines/live/__init__.py`:**
   ```python
   from src.engines.shared.models import PositionSide
   from .execution.position_tracker import LivePosition
   # Remove duplicate PositionSide export
   ```

3. **Update `src/engines/shared/__init__.py`:**
   ```python
   # Already exports Position, Trade, PositionSide
   # Add unified result classes
   from .models import PositionCloseResult
   ```

**Risk:** Low - API changes but with aliases
**Rollback:** Restore original exports
**Testing:** Import tests, verify no circular dependencies

## Validation & Testing Strategy

### Financial Accuracy Validation ✅ CRITICAL

**Requirement:** All financial calculations must produce IDENTICAL results before/after

**Test Plan:**
1. **Backtest Baseline:**
   ```bash
   # Before changes
   atb backtest ml_basic --symbol BTCUSDT --timeframe 1h --days 90 --seed 42
   # Save results: trades.csv, metrics.json
   ```

2. **After each phase:**
   ```bash
   # Run same backtest
   atb backtest ml_basic --symbol BTCUSDT --timeframe 1h --days 90 --seed 42
   # Compare: trades must match exactly (entry/exit prices, PnL, sizes)
   ```

3. **Comparison script:**
   ```python
   # tests/integration/test_model_consolidation_parity.py
   def test_backtest_results_unchanged():
       """Verify consolidation doesn't change backtest outcomes."""
       baseline = load_baseline("tests/data/backtest_baseline.csv")
       current = run_backtest(...)
       assert_trades_equal(baseline, current, tolerance=1e-6)
   ```

### Unit Test Coverage 🧪

**New tests required:**
```python
# tests/unit/shared/test_unified_models.py
def test_base_position_validation():
    """Verify __post_init__ validation (size clamping, enum conversion)."""

def test_base_position_partial_operations():
    """Verify partial operations state tracking."""

def test_position_side_enum_conversion():
    """Verify string to enum conversion."""

def test_backtest_position_extends_base():
    """Verify BacktestPosition has all BasePosition fields."""

def test_live_position_extends_base():
    """Verify LivePosition has all BasePosition fields."""

# tests/unit/shared/test_unified_result_classes.py
def test_partial_exit_result_compatibility():
    """Verify result class works for both engines."""

def test_scale_in_result_compatibility():
    """Verify result class works for both engines."""
```

**Existing tests to update:**
- `tests/unit/backtest/test_position_tracker.py` - Update types
- `tests/unit/live/test_position_tracker.py` - Update types
- `tests/unit/live/test_close_position_parity.py` - Verify still passes
- All execution engine tests

### Integration Test Coverage 🧪

**Critical tests:**
```bash
# Backtest engine
pytest tests/integration/test_backtesting.py -v

# Live engine (paper mode)
pytest tests/integration/test_live_trading.py -v

# Cross-engine parity
pytest tests/integration/test_engine_parity.py -v

# Database logging
pytest tests/integration/test_database_logging.py -v
```

### Regression Prevention 🛡️

**Before merge:**
1. ✅ All unit tests pass
2. ✅ All integration tests pass
3. ✅ Backtest results match baseline exactly
4. ✅ Type checking passes (`python bin/run_mypy.py`)
5. ✅ Code quality passes (`atb dev quality`)
6. ✅ Manual live paper trading test (30 min minimum)

## Progress Tracking

### Phase 1: Enum Consolidation ⏳
- [ ] Remove duplicate `PositionSide` from live/position_tracker.py
- [ ] Update all live imports to use shared enum
- [ ] Update live `__init__.py` exports
- [ ] Run live unit tests
- [ ] Run live integration tests

### Phase 2: Result Classes ⏳
- [ ] Enhance shared `PartialExitResult` with live fields
- [ ] Enhance shared `ScaleInResult` with live fields
- [ ] Update `PartialOperationsManager` (keep decision-only)
- [ ] Update `LivePositionTracker.apply_partial_exit()`
- [ ] Update `BacktestPositionTracker.apply_partial_exit()`
- [ ] Remove duplicate result classes from live
- [ ] Update all imports
- [ ] Run both engine unit tests
- [ ] Run both engine integration tests

### Phase 3: Position Models ⏳
- [ ] Review BasePosition for completeness
- [ ] Create `BacktestPosition(BasePosition)` with component_notional
- [ ] Update `LivePosition` to inherit from `BasePosition`
- [ ] Add deprecation alias `ActiveTrade = BacktestPosition`
- [ ] Update backtest position_tracker to use BacktestPosition
- [ ] Update live position_tracker (already uses LivePosition)
- [ ] Update backtest entry_handler
- [ ] Update backtest exit_handler
- [ ] Update backtest execution_engine
- [ ] Update backtest event_logger
- [ ] Update live entry_handler
- [ ] Update live exit_handler
- [ ] Update live execution_engine
- [ ] Update live event_logger
- [ ] Update live health_monitor
- [ ] Update all imports
- [ ] Run full test suite
- [ ] **CRITICAL:** Compare backtest results with baseline

### Phase 4: Trade Models ⏳
- [ ] Replace backtest Trade with shared BaseTrade
- [ ] Update backtest imports
- [ ] Update database logging code
- [ ] Verify trade records in database
- [ ] Run backtest integration tests
- [ ] **CRITICAL:** Compare trade outputs with baseline

### Phase 5: Cleanup ⏳
- [ ] Update backtest `__init__.py` exports
- [ ] Update live `__init__.py` exports
- [ ] Update shared `__init__.py` exports
- [ ] Remove deprecated code (after verification period)
- [ ] Update documentation
- [ ] Final test suite run

## Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| **PnL calculation divergence** | Medium | Critical | Baseline comparison, exact match requirement |
| **Type errors in existing code** | Medium | High | Incremental migration, mypy at each step |
| **Database schema mismatch** | Low | High | Integration tests, field mapping validation |
| **Import circular dependencies** | Low | Medium | Careful import ordering, TYPE_CHECKING guards |
| **Breaking external API** | Low | Low | Deprecation aliases, version compatibility |
| **Test failures** | High | Low | Fix tests incrementally, expect some breakage |

## Rollback Plan

**If critical issues found at any phase:**
1. **Stop immediately** - Don't proceed to next phase
2. **Document issue** in "Surprises & Discoveries" below
3. **Assess severity:**
   - **P0 (Financial error):** Immediate rollback
   - **P1 (Type error):** Fix in place or rollback
   - **P2 (Test failure):** Fix test or code as appropriate
4. **Rollback procedure:**
   ```bash
   git stash  # Save current work
   git reset --hard <last-good-commit>
   git stash pop  # Review what broke
   ```
5. **Create issue** documenting the blocker
6. **Revise plan** before continuing

## Surprises & Discoveries

### 2025-12-26: Initial Analysis
- Discovered conflicting result classes between shared and live
- Found duplicate PositionSide enum definitions
- Noted that backtest has `component_notional` field not in shared
- Live has `unrealized_pnl` and `order_id` not in shared
- Shared models are well-designed but completely unused

## Decision Log

### 2025-12-26: Inheritance over Metadata
**Decision:** Use inheritance (BacktestPosition, LivePosition) instead of metadata dict
**Rationale:**
- Type safety - editor autocomplete, type checker catches errors
- Clarity - explicit fields better than dict lookups
- Performance - direct attribute access faster than dict
- Compatibility - easier migration path from existing code

**Trade-off:** More classes vs pure consolidation
**Accepted:** The type safety and clarity outweigh the cost of a few extra classes

### 2025-12-26: Enhance Result Classes over New Unified Class
**Decision:** Enhance shared PartialExitResult/ScaleInResult instead of creating new unified class
**Rationale:**
- Less disruptive - maintains existing patterns
- Backward compatible - can add nullable fields
- Simpler migration - fewer changes to calling code

**Trade-off:** Slightly larger dataclasses vs new architecture
**Accepted:** Pragmatic choice for lower risk migration

## Open Questions

1. **Should BacktestPosition.component_notional be optional or required?**
   - Current: Optional (None default)
   - Consider: When is it actually set? Is it always needed?
   - Impact: If required, need to update all instantiation sites

2. **Should we keep `exit_price`, `exit_time`, `exit_reason` on ActiveTrade?**
   - Current: Not in BasePosition, but ActiveTrade has them
   - Consider: Are these used during active trade lifecycle?
   - Impact: If removed, may break some backtest code

3. **Metadata field: keep or remove?**
   - Current: BasePosition has metadata dict
   - Consider: With inheritance, is it still needed?
   - Impact: Could simplify base class if not needed

4. **Should PositionCloseResult be in shared/models.py or separate file?**
   - Current: Different versions in each position_tracker
   - Consider: Logical grouping, import clarity
   - Impact: Import path changes

## Next Steps

**Immediate Actions:**
1. ✅ Get user approval on architectural approach (inheritance)
2. ✅ Get user approval on phase-by-phase migration
3. ⏳ Begin Phase 1: Enum consolidation
4. ⏳ Create baseline backtest for comparison

**Before Starting Implementation:**
1. Answer open questions (1-4 above)
2. Create test data fixtures for baseline comparison
3. Set up continuous comparison script
4. Notify team of upcoming changes (if applicable)

## Success Criteria

**Definition of Done:**
- ✅ Single `PositionSide` enum used by both engines
- ✅ Single set of result classes used by both engines
- ✅ Both engines use BasePosition-derived classes
- ✅ Both engines use shared BaseTrade
- ✅ All unit tests pass (100% pass rate)
- ✅ All integration tests pass (100% pass rate)
- ✅ Backtest results match baseline exactly (< 1e-6 tolerance)
- ✅ Type checking passes with no errors
- ✅ Code quality passes (black, ruff, mypy, bandit)
- ✅ Live paper trading test successful (30+ min runtime, no crashes)
- ✅ Documentation updated
- ✅ No duplicate code between engines

**Quality Gates:**
- Each phase must pass its tests before proceeding
- Financial calculations verified at each phase
- No degradation in test coverage (maintain 85%+ overall)
- No new type: ignore or noqa comments added

## Outcomes & Retrospective

_To be completed after implementation_

### What Went Well
_TBD_

### What Went Wrong
_TBD_

### Lessons Learned
_TBD_

### Metrics
- Files changed: _TBD_
- Lines added/removed: _TBD_
- Test coverage: _TBD_
- Implementation time: _TBD_

---

**Plan Version:** 1.0
**Last Updated:** 2025-12-26
**Next Review:** After Phase 1 completion
