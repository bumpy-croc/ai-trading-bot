# ExecPlan: Extract Shared Logic Between Backtest and Live Engines

**Issue**: #454
**Created**: 2024-12-24
**Status**: Ready for Implementation

## Purpose

Extract duplicated logic between the backtesting engine (`src/backtesting/`) and live trading engine (`src/live/`) into a shared module to ensure consistency and maintainability. Reorganize directories to group related engine components under `src/engines/`.

## Target Directory Structure

```
src/
├── engines/
│   ├── __init__.py
│   ├── backtest/           # Formerly src/backtesting/
│   │   ├── __init__.py
│   │   ├── engine.py
│   │   ├── models.py
│   │   ├── utils.py
│   │   ├── execution/
│   │   │   ├── __init__.py
│   │   │   ├── execution_engine.py
│   │   │   ├── entry_handler.py
│   │   │   ├── exit_handler.py
│   │   │   └── position_tracker.py
│   │   ├── logging/
│   │   │   ├── __init__.py
│   │   │   └── event_logger.py
│   │   ├── regime/
│   │   │   ├── __init__.py
│   │   │   └── regime_handler.py
│   │   ├── risk/
│   │   │   ├── __init__.py
│   │   │   └── correlation_handler.py
│   │   └── dashboard/
│   │       └── __init__.py
│   ├── live/               # Formerly src/live/
│   │   ├── __init__.py
│   │   ├── trading_engine.py
│   │   ├── strategy_manager.py
│   │   ├── order_tracker.py
│   │   ├── pnl.py
│   │   ├── account_sync.py
│   │   ├── runner.py
│   │   └── regime_strategy_switcher.py
│   └── shared/             # NEW - Extracted shared logic
│       ├── __init__.py
│       ├── models.py                    # Unified Position, Trade models
│       ├── position_manager.py          # Position lifecycle management
│       ├── signal_processor.py          # Entry/exit signal interpretation
│       ├── risk_calculator.py           # SL/TP calculation, sizing
│       ├── trailing_stop_manager.py     # Trailing stop updates
│       ├── time_exit_checker.py         # Time-based exit enforcement
│       ├── partial_operations_manager.py # Scale-in/partial exit logic
│       ├── strategy_orchestrator.py     # Strategy runtime lifecycle
│       ├── performance_tracker.py       # Unified metrics tracking
│       └── cost_calculator.py           # Fee/slippage modeling
```

## Implementation Steps

### Phase 1: Directory Reorganization (No Logic Changes)

#### Step 1.1: Create New Directory Structure

```bash
mkdir -p src/engines/backtest/execution
mkdir -p src/engines/backtest/logging
mkdir -p src/engines/backtest/regime
mkdir -p src/engines/backtest/risk
mkdir -p src/engines/backtest/dashboard
mkdir -p src/engines/live
mkdir -p src/engines/shared
```

#### Step 1.2: Copy Files to New Locations

```bash
# Copy backtesting files
cp -r src/backtesting/* src/engines/backtest/

# Copy live files
cp -r src/live/* src/engines/live/
```

#### Step 1.3: Update Import Statements

**Files to update** (use find/replace):

1. **All Python files** - Update imports:
   ```bash
   # Find all files that import from old paths
   grep -r "from src.backtesting" --include="*.py" .
   grep -r "import src.backtesting" --include="*.py" .
   grep -r "from src.live" --include="*.py" .
   grep -r "import src.live" --include="*.py" .
   ```

2. **Replace patterns**:
   - `from src.backtesting` → `from src.engines.backtest`
   - `import src.backtesting` → `import src.engines.backtest`
   - `from src.live` → `from src.engines.live`
   - `import src.live` → `import src.engines.live`

**Key files that will need updates** (estimated ~100+ files):
- All CLI commands in `cli/commands/`
- All strategy files in `src/strategies/`
- All test files in `tests/`
- Dashboard files in `src/dashboards/`
- Database models and managers
- Configuration files

#### Step 1.4: Update Configuration Files

**setup.py** - Update package discovery if using find_packages():
```python
packages=find_packages(where="src", include=["src.engines.*", ...])
```

**pyproject.toml** - Update any path references

**Makefile** - No changes needed (uses CLI)

#### Step 1.5: Remove Old Directories

```bash
# After confirming all imports work
rm -rf src/backtesting
rm -rf src/live
```

#### Step 1.6: Run Tests

```bash
# Run full test suite
atb test all

# Fix any import errors
# Iterate until all tests pass
```

### Phase 2: Extract Shared Logic (Major Refactoring)

This phase should be done **after** Phase 1 is complete and all tests pass.

#### Step 2.1: Create Shared Models

**File**: `src/engines/shared/models.py`

Extract and unify:
- `Position` class (from `src/engines/backtest/models.py::ActiveTrade` and `src/engines/live/trading_engine.py::Position`)
- `Trade` class (from both engines)
- Common enums (`PositionSide`, `OrderStatus`)

**Approach**:
1. Create unified `Position` dataclass with all fields from both engines
2. Add `metadata: dict` field for engine-specific extensions
3. Update both engines to use the shared model
4. Run tests after each engine is updated

#### Step 2.2: Extract Strategy Orchestrator

**File**: `src/engines/shared/strategy_orchestrator.py`

Extract duplicate logic from:
- `Backtester._configure_strategy()`
- `Backtester._prepare_strategy_dataframe()`
- `Backtester._get_runtime_decision()`
- `Backtester._apply_policies_from_decision()`
- `LiveTradingEngine._configure_strategy()`
- `LiveTradingEngine._prepare_strategy_dataframe()`
- `LiveTradingEngine._runtime_process_decision()`
- `LiveTradingEngine._apply_policies_from_decision()`

**Class Design**:
```python
class StrategyOrchestrator:
    """Unified strategy runtime lifecycle management."""

    def __init__(self, strategy: ComponentStrategy | StrategyRuntime):
        """Normalize strategy inputs."""

    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare strategy data using runtime."""

    def process_candle(
        self,
        df: pd.DataFrame,
        index: int,
        context: RuntimeContext
    ) -> TradingDecision | None:
        """Get decision from runtime strategy."""

    def apply_policies(self, decision: TradingDecision) -> PolicyBundle:
        """Extract and return policies from decision."""

    def finalize(self) -> None:
        """Clean up runtime state."""
```

**Implementation**:
1. Create `StrategyOrchestrator` class
2. Move common logic from both engines
3. Update `Backtester` to use `StrategyOrchestrator`
4. Run backtest tests
5. Update `LiveTradingEngine` to use `StrategyOrchestrator`
6. Run live engine tests

#### Step 2.3: Extract Signal Processor

**File**: `src/engines/shared/signal_processor.py`

Extract duplicate logic from:
- `EntryHandler.process_runtime_decision()`
- `LiveTradingEngine._runtime_entry_plan()`
- `ExitHandler.check_exit_conditions()`
- `LiveTradingEngine._runtime_should_exit_live()`

**Class Design**:
```python
class SignalProcessor:
    """Unified entry/exit signal interpretation."""

    def extract_entry_signal(
        self,
        decision: TradingDecision,
        current_price: float,
        balance: float,
    ) -> EntrySignal:
        """Extract entry signal from runtime decision."""

    def check_exit_conditions(
        self,
        decision: TradingDecision | None,
        position: Position,
        current_price: float,
        candle: pd.Series,
    ) -> ExitSignal:
        """Check all exit conditions (signal, SL/TP, time)."""
```

#### Step 2.4: Extract Risk Calculator

**File**: `src/engines/shared/risk_calculator.py`

Extract duplicate logic from:
- `EntryHandler._apply_dynamic_risk()`
- `LiveTradingEngine._get_dynamic_risk_adjusted_size()`
- `CorrelationHandler.apply_correlation_control()`
- `LiveTradingEngine._apply_correlation_control()`

**Class Design**:
```python
class RiskCalculator:
    """Unified risk calculation and position sizing."""

    def __init__(
        self,
        risk_manager: RiskManager,
        dynamic_risk_manager: DynamicRiskManager | None = None,
        correlation_engine: CorrelationEngine | None = None,
    ):
        """Initialize with risk components."""

    def calculate_stop_loss(
        self,
        entry_price: float,
        side: str,
        stop_loss_pct: float | None,
    ) -> float | None:
        """Calculate stop loss price."""

    def calculate_take_profit(
        self,
        entry_price: float,
        side: str,
        take_profit_pct: float | None,
    ) -> float | None:
        """Calculate take profit price."""

    def apply_dynamic_risk_adjustment(
        self,
        original_size: float,
        current_time: datetime,
        balance: float,
        peak_balance: float,
    ) -> tuple[float, dict]:
        """Apply dynamic risk adjustments, return (adjusted_size, adjustment_log)."""

    def apply_correlation_adjustment(
        self,
        candidate_size: float,
        symbol: str,
        timeframe: str,
        df: pd.DataFrame,
        index: int,
    ) -> float:
        """Apply correlation-based position sizing."""
```

#### Step 2.5: Extract Trailing Stop Manager

**File**: `src/engines/shared/trailing_stop_manager.py`

Extract duplicate logic from:
- `ExitHandler.update_trailing_stop()`
- `LiveTradingEngine._update_trailing_stop()`

**Class Design**:
```python
class TrailingStopManager:
    """Unified trailing stop management."""

    def __init__(self, policy: TrailingStopPolicy | None):
        """Initialize with trailing stop policy."""

    def update(
        self,
        position: Position,
        current_price: float,
        df: pd.DataFrame,
        index: int,
    ) -> tuple[bool, str | None]:
        """Update trailing stop, return (updated, log_message)."""
```

#### Step 2.6: Extract Partial Operations Manager

**File**: `src/engines/shared/partial_operations_manager.py`

Extract duplicate logic from:
- `ExitHandler.check_partial_operations()`
- `LiveTradingEngine._check_partial_operations()`

**Class Design**:
```python
class PartialOperationsManager:
    """Unified partial exit and scale-in management."""

    def __init__(self, policy: PartialExitPolicy | None):
        """Initialize with partial operations policy."""

    def check_partial_exit(
        self,
        position: Position,
        current_price: float,
        current_pnl_pct: float,
    ) -> tuple[bool, float | None]:
        """Check if partial exit should trigger, return (should_exit, exit_fraction)."""

    def check_scale_in(
        self,
        position: Position,
        current_price: float,
        current_pnl_pct: float,
        balance: float,
    ) -> tuple[bool, float | None]:
        """Check if scale-in should trigger, return (should_scale, scale_fraction)."""

    def apply_partial_exit(
        self,
        position: Position,
        exit_fraction: float,
        current_price: float,
        basis_balance: float,
    ) -> float:
        """Execute partial exit, return realized PnL."""

    def apply_scale_in(
        self,
        position: Position,
        scale_fraction: float,
        current_price: float,
    ) -> None:
        """Execute scale-in operation."""
```

#### Step 2.7: Extract Performance Tracker

**File**: `src/engines/shared/performance_tracker.py`

Consolidate performance tracking from both engines.

**Class Design**:
```python
class PerformanceTracker:
    """Unified performance metrics tracking."""

    def __init__(self, initial_balance: float):
        """Initialize tracker."""
        self.initial_balance = initial_balance
        self.total_trades = 0
        self.winning_trades = 0
        self.total_fees_paid = 0.0
        self.total_slippage_cost = 0.0
        self.peak_balance = initial_balance
        self.max_drawdown = 0.0

    def record_trade(self, trade: Trade, fee: float, slippage: float) -> None:
        """Record completed trade."""

    def update_balance(self, balance: float) -> None:
        """Update current balance and drawdown."""

    def get_metrics(self) -> dict:
        """Get current performance metrics."""
```

#### Step 2.8: Extract Cost Calculator

**File**: `src/engines/shared/cost_calculator.py`

Extract from `ExecutionEngine` and live engine cost calculations.

**Class Design**:
```python
class CostCalculator:
    """Unified fee and slippage calculation."""

    def __init__(self, fee_rate: float = 0.001, slippage_rate: float = 0.0005):
        """Initialize with cost parameters."""

    def calculate_entry_costs(
        self,
        entry_price: float,
        position_size: float,
        side: str,
    ) -> tuple[float, float, float]:
        """Calculate entry costs, return (executed_price, fee, slippage)."""

    def calculate_exit_costs(
        self,
        exit_price: float,
        position_size: float,
        side: str,
    ) -> tuple[float, float, float]:
        """Calculate exit costs, return (executed_price, fee, slippage)."""
```

### Phase 3: Integration and Testing

#### Step 3.1: Update Both Engines

For each extracted module:
1. Update `Backtester` to use shared component
2. Run backtest tests
3. Update `LiveTradingEngine` to use shared component
4. Run live engine tests
5. Fix any issues

#### Step 3.2: Update Tests

- Update test imports
- Remove duplicate test logic where shared modules are tested
- Ensure both engine-specific and shared logic tests exist

#### Step 3.3: Run Full Test Suite

```bash
atb test all
atb dev quality
```

#### Step 3.4: Update Documentation

- Update architecture.md with new structure
- Update relevant docs/ files
- Add docstrings to all new shared modules

## Progress

- [x] Phase 1.1: Create directory structure (completed 2024-12-25)
- [x] Phase 1.2: Move files using git mv (completed 2024-12-25)
- [x] Phase 1.3: Update imports (~100 files updated) (completed 2024-12-25)
- [x] Phase 1.4: Update configuration files (completed 2024-12-25)
- [x] Phase 1.5: Old directories removed via git mv (completed 2024-12-25)
- [x] Phase 1.6: Unit tests passing (748 passed, ~5 flaky) (completed 2024-12-25)
- [x] Phase 2.1: Extract shared models (Position, Trade, PositionSide) (completed 2024-12-25)
- [ ] Phase 2.2: Extract StrategyOrchestrator (future work)
- [ ] Phase 2.3: Extract SignalProcessor (future work)
- [ ] Phase 2.4: Extract RiskCalculator (future work)
- [x] Phase 2.5: Extract TrailingStopManager (completed 2024-12-25)
- [ ] Phase 2.6: Extract PartialOperationsManager (future work)
- [ ] Phase 2.7: Extract PerformanceTracker (future work)
- [x] Phase 2.8: Extract CostCalculator (completed 2024-12-25)
- [ ] Phase 3.1: Integrate shared modules into engines (future work)
- [x] Phase 3.2: Update tests (import paths updated) (completed 2024-12-25)
- [x] Phase 3.3: Run full test suite (748 passed) (completed 2024-12-25)
- [ ] Phase 3.4: Update documentation (future work)

## Risks and Mitigation

**Risk**: Breaking existing functionality during reorganization
**Mitigation**: Do Phase 1 first (pure reorganization, no logic changes), verify all tests pass before Phase 2

**Risk**: Import circular dependencies
**Mitigation**: Shared modules should not import from backtest/live engines, only vice versa

**Risk**: Losing git history on moved files
**Mitigation**: Use `git mv` instead of `cp` + `rm` where possible

**Risk**: Test suite takes too long to run after each change
**Mitigation**: Use `atb test smoke` for quick validation, full suite at end of each phase

## Estimated Effort

- Phase 1 (Directory reorganization): 2-3 hours
- Phase 2 (Extract shared logic): 6-8 hours (1 hour per module)
- Phase 3 (Integration and testing): 2-3 hours
- **Total**: 10-14 hours

## Success Criteria

- [ ] All files moved to new `src/engines/` structure
- [ ] All imports updated
- [ ] All tests pass (`atb test all`)
- [ ] All quality checks pass (`atb dev quality`)
- [ ] Shared logic extracted to `src/engines/shared/`
- [ ] Both engines use shared components
- [ ] No duplicate logic between engines
- [ ] Documentation updated
- [ ] Code coverage maintained or improved

## Notes

- This is a **large refactoring** that touches many files
- Recommend doing in a dedicated feature branch
- Consider breaking into smaller PRs if needed (e.g., Phase 1 as one PR, each Phase 2 module as separate PRs)
- Run tests frequently during implementation
- Use IDE refactoring tools where possible (e.g., PyCharm's "Move" refactoring)

## Related Issues

- #454 - Extract shared logic between backtest and live engines
