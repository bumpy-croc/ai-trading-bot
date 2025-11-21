# Technical Debt and Code Quality Improvements

**Generated**: 2025-11-21
**Status**: Active Refactoring in Progress

This document tracks code quality issues, technical debt, and improvement opportunities identified during comprehensive codebase analysis.

---

## Executive Summary

- **Total Python files**: 164 in `src/`, 17 in `cli/`
- **Lines of code**: ~51,000 in `src/`
- **Current state**: Good foundation but significant refactoring opportunities
- **Priority**: Address critical type safety and architectural issues first

---

## 1. Type Safety Issues (HIGH PRIORITY)

### Mypy Compliance

**Status**: Many type errors preventing clean mypy runs

**Critical Issues**:
- Module path confusion: Files found under multiple module names
  - `performance/metrics.py` found as both `src.performance.metrics` and `performance.metrics`
  - `dashboards/backtesting/__init__.py` similar issue
- Missing type stubs: `requests` library (types-requests installed but not recognized in temp mypy environment)
- SQLAlchemy Base class validation errors
- Optional parameter issues throughout (implicit Optional deprecated)

**Files with Critical Type Issues**:
- `src/data_providers/coinbase_provider.py`: 13 type errors (incompatible defaults, wrong types)
- `src/data_providers/binance_provider.py`: 5 attr-defined errors (None has no attribute)
- `src/database/models.py`: Base class validation issues

**Action Items**:
1. Fix mypy runner (`bin/run_mypy.py`) to properly handle module paths
2. Add explicit `Optional[]` types where parameters can be None
3. Fix SQLAlchemy Base class type annotations
4. Add proper type hints to all functions missing return type annotations

**Impact**: Medium-High (doesn't prevent runtime but makes development harder)

---

## 2. Architectural Issues (HIGH PRIORITY)

### God Objects and Single Responsibility Violations

#### LiveTradingEngine (src/live/trading_engine.py)
- **Lines**: 3,373
- **Methods**: 68
- **Issue**: Massive class doing too much
- **Top offenders**:
  - `__init__`: 329 lines
  - `_trading_loop`: 285 lines
  - `_check_entry_conditions`: 240 lines
  - `_check_exit_conditions`: 179 lines
  - `start`: 165 lines

**Recommended Split**:
```
LiveTradingEngine/
├── core/
│   ├── engine.py (orchestration only, ~200 lines)
│   ├── trading_loop.py
│   └── config_handler.py
├── execution/
│   ├── order_executor.py
│   ├── position_manager.py
│   └── entry_exit_logic.py
├── monitoring/
│   ├── performance_tracker.py
│   ├── pnl_updater.py
│   └── mfe_mae_tracker.py
└── policies/
    ├── partial_exits.py
    ├── trailing_stops.py
    └── time_restrictions.py
```

#### DatabaseManager (src/database/manager.py)
- **Lines**: 2,492
- **Issue**: Database operations, business logic, and analytics mixed
- **Top offenders**:
  - `get_dynamic_risk_performance_metrics`: 186 lines
  - `log_position`: 117 lines
  - `get_performance_metrics`: 111 lines

**Recommended Split**:
- Separate data access (CRUD) from analytics
- Extract performance metrics to `src/performance/analytics.py`
- Use repository pattern for cleaner separation

#### BacktestingEngine (src/backtesting/engine.py)
- **Lines**: 2,280
- **Issue**: Similar to LiveTradingEngine
- **Top offender**: `run`: 1,248 lines (!!!)

**Critical**: The `run` method is essentially one giant function. This needs immediate refactoring.

---

## 3. Code Quality Issues

### Remaining Ruff Violations (10 total)

✅ **Fixed (14)**: Unused variables, bare except blocks
❌ **Remaining**:

1. **UP031 (2 occurrences)**: Use f-strings instead of % formatting
   - `cli/commands/train_commands.py:93,97`
   - Easy fix, low priority

2. **B039 (1 occurrence)**: Mutable default for ContextVar
   - `src/infrastructure/logging/context.py:11`
   - **Impact**: Medium (potential state bugs)
   - **Fix**: Change `default={}` to `default=None`, initialize with `.set()`

3. **UP007 (1 occurrence)**: Use `X | Y` instead of `Union`
   - `src/performance/metrics.py:16`
   - **Fix**: `Number = Union[int, float]` → `Number = int | float`

4. **B905 (2 occurrences)**: `zip()` without explicit `strict=` parameter
   - `src/prediction/utils/caching.py:86`
   - `src/strategies/components/testing/component_performance_tester.py:783`
   - **Impact**: Low (silent bugs if lengths mismatch)

5. **B904 (1 occurrence)**: Raise without `from err`
   - `src/strategies/components/strategy_registry.py:655`
   - **Fix**: `raise ... from e` or `raise ... from None`

6. **B028 (3 occurrences)**: `warnings.warn` without `stacklevel`
   - `src/strategies/components/testing/test_datasets.py:295,298,451`
   - **Impact**: Low (warnings point to wrong location)
   - **Fix**: Add `stacklevel=2`

### Broad Exception Handling

**Status**: 28 files with `except Exception:`

These should use more specific exception types where possible:
- `src/live/trading_engine.py`
- `src/database/manager.py`
- `src/prediction/engine.py`
- `src/backtesting/engine.py`
- And 24 others...

**Action**: Audit each and replace with specific exceptions (ValueError, KeyError, etc.)

---

## 4. Function Complexity (MEDIUM PRIORITY)

### Functions Over 50 Lines

**Target**: All functions should be under 50 lines with few justified exceptions

**Top Offenders by File**:

#### src/live/trading_engine.py
- `__init__`: 329 lines → Split configuration loading
- `_trading_loop`: 285 lines → Extract order execution, monitoring
- `_check_entry_conditions`: 240 lines → Split signal validation, regime checks
- `_check_exit_conditions`: 179 lines → Extract stop loss, take profit logic
- `start`: 165 lines → Extract initialization steps
- 5+ more over 50 lines

#### src/database/manager.py
- `get_dynamic_risk_performance_metrics`: 186 lines → Extract calculation logic
- `log_position`: 117 lines → Split validation, persistence
- `get_performance_metrics`: 111 lines → Extract metric calculations
- `update_position`: 101 lines → Split update types
- 6+ more over 50 lines

#### src/backtesting/engine.py
- `run`: 1,248 lines → **CRITICAL**: Break into ~20 smaller functions
- `__init__`: 190 lines → Extract configuration steps
- `_apply_correlation_control`: 142 lines → Split risk calculations
- 3+ more over 50 lines

**Recommended Pattern**:
```python
# Instead of one 240-line function:
def _check_entry_conditions(self, ...):
    # 240 lines of mixed logic

# Split into:
def _check_entry_conditions(self, ...):
    """Orchestrates entry condition checking."""
    if not self._validate_entry_preconditions(...):
        return None
    signal = self._get_trading_signal(...)
    if not signal:
        return None
    if not self._check_regime_compatibility(signal, ...):
        return None
    return self._build_entry_decision(signal, ...)

def _validate_entry_preconditions(self, ...) -> bool:
    """Checks if entry conditions can be evaluated."""
    # 15 lines

def _get_trading_signal(self, ...) -> Signal | None:
    """Gets trading signal from strategy."""
    # 20 lines

# etc...
```

---

## 5. Missing Type Hints (MEDIUM PRIORITY)

Many methods lack return type annotations:

**Examples**:
- `trading_engine.py`: 20+ methods missing `-> ReturnType`
- `database/manager.py`: 15+ methods missing `-> ReturnType`
- `backtesting/engine.py`: 10+ methods missing `-> ReturnType`

**Rule**: All functions except `__init__` should have explicit return types.

**Quick wins**: Start with public API methods, work down to private helpers.

---

## 6. Duplicate Code (MEDIUM PRIORITY)

### Identified Patterns

1. **Risk Parameter Merging**
   - Similar logic in `live/trading_engine.py` and `backtesting/engine.py`
   - Extract to `src/risk/parameter_merger.py`

2. **Policy Application**
   - `_apply_policies_from_decision` duplicated across engines
   - Extract to `src/position_management/policy_applicator.py`

3. **DataFrame Validation**
   - Context readiness checks repeated
   - Extract to `src/data_providers/validation.py`

4. **Regime Detection**
   - Parsing logic duplicated
   - Centralize in `src/regime/parser.py`

---

## 7. Documentation Issues (LOW-MEDIUM PRIORITY)

### Docstring Quality

**Status**: Most functions have docstrings, but quality varies

**Issues**:
- Some use imperative ("Get") vs. third-person present ("Gets")
- Missing examples for complex functions
- Incomplete Args/Returns/Raises sections
- Some missing docstrings entirely

**Standard** (from CLAUDE.md):
```python
def calculate_position_size(balance: float, risk: float) -> float:
    """
    Calculates position size based on account balance and risk tolerance.

    Uses the fixed fractional position sizing method to determine
    the appropriate position size while respecting risk limits.

    Args:
        balance: Current account balance in base currency
        risk: Risk percentage as decimal (0.02 = 2%)

    Returns:
        Position size in base currency

    Raises:
        ValueError: If balance is negative or risk is outside [0, 1]

    Example:
        >>> calculate_position_size(10000, 0.02)
        200.0
    ```

### Comment Quality

**Found**: 3 TODO/FIXME comments (good!)
- `position_management/dynamic_risk.py:388`: Implement correlation risk management
- `live/trading_engine.py:2842`: Calculate daily P&L

**Action**: Track these in GitHub issues, add issue numbers to TODOs

---

## 8. Test Code Quality (LOW-MEDIUM PRIORITY)

### Current Status
- Good coverage overall (85%+)
- Tests follow AAA pattern mostly
- Some flaky tests in integration suite

### Issues
1. **Test data organization**: Some hardcoded data in tests
2. **Fixture overuse**: Some fixtures too complex
3. **Missing edge cases**: Not all error paths tested
4. **Slow tests**: Some integration tests could be unit tests

### Improvements Needed
- Move test data to `tests/data/` systematically
- Simplify complex fixtures
- Add property-based testing for critical algorithms
- Mark slow tests clearly

---

## 9. Security Considerations (LOW PRIORITY)

### Bandit Scan Results

**To Run**: `bandit -c pyproject.toml -r src`

**Known Issues**:
- Some pickle usage (only for local cache files - acceptable)
- Some `nosec` annotations (need verification)
- Subprocess usage (should verify array form vs shell=True)

**Action Items**:
1. Run full bandit scan
2. Review all nosec annotations for justification
3. Audit subprocess.run calls for shell=True
4. Verify no API keys in logs

---

## 10. Performance Considerations (LOW PRIORITY)

### Potential Bottlenecks

1. **DataFrame operations**
   - Frequent `.copy()` calls might be unnecessary
   - Review vectorization opportunities

2. **Database queries**
   - Check for N+1 query patterns
   - Consider query optimization for large datasets

3. **Caching**
   - Review cache hit rates
   - Consider cache invalidation strategies

**Action**: Profile hot paths before optimizing

---

## 11. Module Organization (LOW PRIORITY)

### Current Structure

Generally good, but some opportunities:

1. **src/strategies/components/** is growing large (18 files + testing/)
   - Consider splitting into:
     - `components/core/` (base classes)
     - `components/generators/` (signal generators)
     - `components/managers/` (risk, position sizing)
     - `components/testing/` (existing)

2. **src/prediction/** has good structure
   - Keep as model

3. **CLI commands** are well-organized
   - No changes needed

---

## Prioritized Action Plan

### Phase 1: Critical (Do First)
1. ✅ Fix bare except blocks (DONE)
2. ✅ Fix mypy config type errors (DONE)
3. ✅ Remove unused variables (DONE)
4. ✅ Apply black formatting (DONE)
5. Fix remaining ruff errors (10 items, ~2 hours)
6. Fix mutable ContextVar default (B039)
7. Add return type hints to all public methods

### Phase 2: High Priority (Next Sprint)
1. Refactor `BacktestingEngine.run()` - break 1248-line function into ~20 functions
2. Refactor `LiveTradingEngine.__init__` and `_trading_loop`
3. Fix SQLAlchemy type annotations
4. Add explicit Optional types

### Phase 3: Medium Priority (Following Sprint)
1. Extract duplicate code to shared utilities
2. Refactor remaining functions over 50 lines
3. Improve exception specificity (replace broad Exception catches)
4. Complete docstring standardization

### Phase 4: Low Priority (Future)
1. Reorganize large modules
2. Performance optimization
3. Test code improvements
4. Security audit completion

---

## Metrics & Goals

### Current State
- **Mypy compliance**: ~60% (many errors)
- **Ruff compliance**: 99.4% (10 errors out of ~1800 checks)
- **Black compliance**: 100% ✅
- **Test coverage**: ~85%
- **Function length**: ~40% over 50 lines

### Target State (3 months)
- **Mypy compliance**: 95% (strict mode)
- **Ruff compliance**: 100%
- **Black compliance**: 100% (maintained)
- **Test coverage**: 90%
- **Function length**: 95% under 50 lines

---

## Notes

- This document should be updated as issues are resolved
- Create GitHub issues for major refactoring work
- Tag issues with `technical-debt`, `refactoring`, `code-quality`
- Reference this doc in PR descriptions for context

**Last Updated**: 2025-11-21
**Updated By**: Claude (automated analysis)
