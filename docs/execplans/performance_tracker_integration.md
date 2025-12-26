# ExecPlan: Performance Tracker Integration

**Created:** 2025-12-26
**Status:** Planning
**Branch:** `claude/plan-tracker-integration-htyvO`

## Purpose

Integrate unified performance tracking into both backtesting and live trading engines while maintaining clean architecture, respecting engine-specific requirements, and consolidating duplicated metric calculation logic.

## Context & Current State

### Existing Implementations

1. **`src/engines/shared/performance_tracker.py`** (Created but not integrated)
   - Basic implementation with threading support
   - Tracks: trades, balance, fees, slippage, PnL, drawdown
   - **Limitations:** Missing advanced metrics (Sharpe, Sortino, Calmar, VaR, streaks, regime tracking)

2. **`src/strategies/components/performance_tracker.py`** (Strategy-level tracker)
   - Comprehensive with regime tracking, period analysis, caching
   - Advanced metrics: Sharpe, Sortino, Calmar, VaR, expectancy, streaks
   - Performance comparison and storage backend support
   - **Issue:** Too complex for engine use, mixes concerns (calculation + storage + comparison)

3. **`src/performance/metrics.py`** (Pure metric functions)
   - Shared across backtest, live, dashboards
   - Functions: `total_return()`, `cagr()`, `sharpe()`, `max_drawdown()`, `pnl_percent()`, `cash_pnl()`
   - Pure, testable, strictly typed
   - **Best practice:** Clean separation of calculation logic

4. **Backtest Engine** (`src/engines/backtest/engine.py`)
   - Manual metric tracking in main loop
   - Uses `compute_performance_metrics()` utility
   - Yearly balance tracking, prediction metrics
   - Early stop on max drawdown

5. **Live Engine** (`src/engines/live/trading_engine.py`)
   - Manual tracking: `total_trades`, `winning_trades`, `total_pnl`, `peak_balance`, `max_drawdown`
   - Method: `_update_performance_metrics()`
   - Database persistence via `AccountHistory`, `PerformanceMetrics`, `DynamicPerformanceMetrics`
   - MFE/MAE tracking (separate module)

### Database Schema Alignment

**Relevant Tables:**
- `PerformanceMetrics`: Aggregated by period (daily/weekly/monthly/all-time)
- `DynamicPerformanceMetrics`: Rolling metrics for adaptive risk management
- `AccountHistory`: Balance snapshots with drawdown tracking
- `StrategyPerformance`: Strategy comparison data

## Architecture Analysis

### Core Principles

**Clean Architecture Layers:**
```
┌─────────────────────────────────────────────────────┐
│  Engine Layer (Backtest/Live)                       │
│  - Orchestration logic                              │
│  - Engine-specific workflows                        │
│  - Persistence decisions                            │
└─────────────────────────────────────────────────────┘
                      ↓ uses
┌─────────────────────────────────────────────────────┐
│  Shared Performance Tracker                          │
│  - State management                                  │
│  - Metric aggregation                                │
│  - Trade/balance recording                           │
└─────────────────────────────────────────────────────┘
                      ↓ uses
┌─────────────────────────────────────────────────────┐
│  Performance Metrics (Pure Functions)                │
│  - Sharpe, Sortino, Calmar, VaR calculations        │
│  - Drawdown, return calculations                    │
│  - No side effects, fully testable                  │
└─────────────────────────────────────────────────────┘
```

### What Should Be Shared?

**Shared Responsibilities (All Engines):**
- Trade recording with metadata (entry/exit price, time, PnL, fees, slippage)
- Balance tracking (current, peak, initial)
- Drawdown calculation (current, max)
- Win/loss statistics (counts, rates, profit factor)
- Trade duration tracking
- Largest win/loss tracking
- Consecutive win/loss streaks
- Fee and slippage aggregation
- Thread-safe state updates (for live engine)

**Metric Calculations (Delegated to `src/performance/metrics.py`):**
- Total return percentage
- CAGR (annualized return)
- Sharpe ratio (from balance series)
- Max drawdown (from balance series)
- Trade-level PnL calculations

**NOT Shared (Engine-Specific):**
- Database persistence logic → Engine decides when/what to persist
- Caching strategies → Different needs (backtest: none, live: short TTL)
- Storage backends → Engines manage their own I/O
- Presentation/reporting → Different formats (CLI summary vs. dashboard)
- Regime-specific tracking → Strategy component concern, not engine concern

### Engine-Specific Requirements

#### Backtesting Engine

**Unique Needs:**
1. **Vectorized/batch analysis** over historical data
2. **Yearly breakdowns** for long-term backtest analysis
3. **Prediction accuracy metrics** (directional accuracy, MAE, Brier score)
4. **Buy-and-hold comparison** baseline
5. **Early stop conditions** based on max drawdown
6. **Fast bulk metric calculation** (minimize overhead in tight loop)
7. **Optional database logging** (often runs without DB)

**Performance Characteristics:**
- Calculate metrics once at the end (bulk processing)
- Minimize per-candle overhead
- Balance history stored in memory (DataFrame)

**Should Metrics Be Different?**
- **No** - Core metrics (win rate, Sharpe, drawdown) should be identical
- **Yes** - Additional metrics specific to backtesting (e.g., yearly returns, hold comparison)

#### Live Trading Engine

**Unique Needs:**
1. **Real-time incremental updates** as trades complete
2. **Database persistence** for audit trail and monitoring
3. **Health monitoring** triggers and alerts
4. **Dynamic risk adjustments** based on rolling metrics
5. **MFE/MAE tracking** for trade quality analysis
6. **Account snapshots** at regular intervals
7. **Rolling window metrics** (e.g., 30-day Sharpe, win rate)

**Performance Characteristics:**
- Update metrics incrementally (single trade at a time)
- Persist to database immediately
- Low-latency requirements for real-time decisions

**Should Metrics Be Different?**
- **No** - Core metrics must match backtest for validation
- **Yes** - Additional real-time metrics (rolling Sharpe, current drawdown %, dynamic risk factors)

### Missing Metrics

#### Currently Missing in Shared Tracker (but needed):

**Risk Metrics:**
- ✅ Max drawdown → Present but needs improvement
- ❌ **Sortino ratio** → Only in strategy tracker
- ❌ **Calmar ratio** → Only in strategy tracker
- ❌ **VaR (Value at Risk)** → Only in strategy tracker
- ❌ **Sharpe ratio** → Partially implemented, needs balance series support

**Trade Quality Metrics:**
- ✅ Profit factor → Present
- ✅ Win rate → Present
- ✅ Avg win/loss → Present
- ❌ **Expectancy** → Only in strategy tracker
- ❌ **Best/worst trade streaks** → Only in strategy tracker (consecutive wins/losses)
- ❌ **Trade duration statistics** → Basic version present

**Efficiency Metrics:**
- ❌ **Trades per day** → Only in strategy tracker
- ❌ **Recovery time from drawdown** → Not implemented anywhere
- ❌ **Max drawdown duration** → Not implemented anywhere

**Advanced Analysis:**
- ❌ **Rolling window metrics** (30-day Sharpe, win rate) → Needed for live engine
- ❌ **Period-based breakdown** (daily/weekly/monthly) → Only in strategy tracker
- ❌ **Regime-specific performance** → Only in strategy tracker (debatable if needed at engine level)

#### Analysis: Should These Be Added to Shared Tracker?

| Metric | Add to Shared? | Rationale |
|--------|----------------|-----------|
| Sortino ratio | ✅ Yes | Standard risk metric, both engines should track |
| Calmar ratio | ✅ Yes | Standard risk metric, useful for both |
| VaR (95%) | ✅ Yes | Risk management metric, useful for both |
| Expectancy | ✅ Yes | Trade quality metric, both engines need |
| Consecutive streaks | ✅ Yes | Already partially present, finish implementation |
| Trades per day | ⚠️ Maybe | Easy to calculate, low overhead |
| Recovery time | ❌ No | Complex calculation, defer to post-analysis |
| Max DD duration | ❌ No | Complex calculation, defer to post-analysis |
| Rolling windows | ❌ No | Live-engine specific (use separate component) |
| Period breakdown | ❌ No | Presentation concern, not core tracking |
| Regime-specific | ❌ No | Strategy component concern |

### Recommended Architecture

#### Layer 1: Pure Metric Functions (`src/performance/metrics.py`)

**Expand with:**
```python
def sortino_ratio(daily_balance: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Sortino ratio using downside deviation."""

def calmar_ratio(annualized_return: float, max_drawdown: float) -> float:
    """Calmar ratio: annualized return / max drawdown."""

def value_at_risk(returns: pd.Series, confidence: float = 0.95) -> float:
    """Value at Risk at given confidence level."""

def expectancy(win_rate: float, avg_win: float, avg_loss: float) -> float:
    """Expected value per trade."""
```

**Why:** These are pure calculations, fully testable, reusable across engines and dashboards.

#### Layer 2: Enhanced Shared Performance Tracker (`src/engines/shared/performance_tracker.py`)

**Responsibilities:**
- Record trades with full metadata
- Track balance history (for Sharpe/Sortino/VaR calculation)
- Maintain trade statistics (counts, durations, PnL)
- Calculate aggregate metrics on demand
- Provide thread-safe access

**Enhanced `PerformanceMetrics` dataclass:**
```python
@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics."""

    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float

    # Returns
    total_pnl: float
    total_return_pct: float
    annualized_return: float

    # Risk metrics
    max_drawdown: float
    current_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float  # NEW
    calmar_ratio: float   # NEW
    var_95: float         # NEW

    # Trade quality
    profit_factor: float
    expectancy: float     # NEW
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float

    # Efficiency
    avg_trade_duration_hours: float
    consecutive_wins: int  # NEW
    consecutive_losses: int  # NEW

    # Costs
    total_fees_paid: float
    total_slippage_cost: float

    # Balance tracking
    initial_balance: float
    current_balance: float
    peak_balance: float
```

**Key Methods:**
```python
class PerformanceTracker:
    def record_trade(self, trade: BaseTrade, fee: float, slippage: float) -> None:
        """Record completed trade."""

    def update_balance(self, balance: float, timestamp: datetime) -> None:
        """Update current balance and recalculate drawdown."""

    def get_metrics(self) -> PerformanceMetrics:
        """Calculate and return current performance metrics."""

    def get_balance_history(self) -> pd.DataFrame:
        """Return balance history as DataFrame for advanced calculations."""

    def reset(self) -> None:
        """Reset all tracking."""
```

**What NOT to include:**
- ❌ Storage backend (engines handle persistence)
- ❌ Caching (different needs per engine)
- ❌ Regime tracking (strategy component concern)
- ❌ Period-based filtering (presentation concern)
- ❌ Comparison logic (separate utility)

#### Layer 3: Engine Integration

**Backtest Engine:**
```python
class BacktestEngine:
    def __init__(self, ...):
        self.performance_tracker = PerformanceTracker(initial_balance)

    def _run_main_loop(self, df, symbol, timeframe):
        # ... main loop ...

        # On trade completion
        self.performance_tracker.record_trade(trade, fee, slippage)
        self.performance_tracker.update_balance(self.balance, current_time)

        # Check early stop
        metrics = self.performance_tracker.get_metrics()
        if metrics.current_drawdown > self.max_drawdown_threshold:
            # Early stop

    def _build_final_results(self, ...):
        metrics = self.performance_tracker.get_metrics()

        # Add backtest-specific metrics
        results = {
            **metrics.to_dict(),
            "yearly_returns": self._calculate_yearly_returns(),
            "hold_return": self._calculate_hold_return(df),
            "prediction_metrics": self._calculate_prediction_metrics(df),
        }
        return results
```

**Live Engine:**
```python
class LiveTradingEngine:
    def __init__(self, ...):
        self.performance_tracker = PerformanceTracker(initial_balance)

    def _close_position(self, ...):
        # ... close position logic ...

        # Record trade
        self.performance_tracker.record_trade(trade, fee, slippage)
        self.performance_tracker.update_balance(self.current_balance, datetime.now())

        # Persist to database
        metrics = self.performance_tracker.get_metrics()
        self.db_manager.save_performance_metrics(
            session_id=self.trading_session_id,
            metrics=metrics,
        )

        # Check dynamic risk adjustments
        if metrics.consecutive_losses >= 3:
            self.dynamic_risk_manager.reduce_risk()

    def _update_performance_metrics(self):
        """Update real-time performance tracking."""
        self.performance_tracker.update_balance(
            self.current_balance,
            datetime.now()
        )
```

### Metrics Alignment Strategy

**Goal: Ensure perfect parity between backtest and live for core metrics.**

| Metric | Must Match? | Why? |
|--------|-------------|------|
| Total return | ✅ Yes | Validation requirement |
| Max drawdown | ✅ Yes | Risk management consistency |
| Sharpe ratio | ✅ Yes | Risk-adjusted return comparison |
| Win rate | ✅ Yes | Strategy effectiveness validation |
| Profit factor | ✅ Yes | Risk/reward validation |
| Average trade duration | ✅ Yes | Execution consistency |
| Total fees/slippage | ✅ Yes | Cost model validation |
| Sortino ratio | ✅ Yes | Downside risk comparison |
| Calmar ratio | ✅ Yes | Drawdown-adjusted return |
| VaR (95%) | ✅ Yes | Risk exposure validation |

**Implementation Strategy:**
1. Both engines use **same** `PerformanceTracker` class
2. Both engines call **same** underlying metric functions from `src/performance/metrics.py`
3. **Automated tests** verify metric parity on identical trade sequences
4. **Integration test:** Run backtest, replay trades in mock live engine, compare final metrics

### Future Requirements

#### Near-term (Next 3-6 months):
1. **Rolling window metrics** for live engine (30-day Sharpe, win rate)
   - Implementation: Separate `RollingMetricsTracker` class in live engine
   - Uses underlying `PerformanceTracker` data

2. **Multi-timeframe analysis** (compare 1h vs. 4h strategy performance)
   - Implementation: `PerformanceComparator` utility class
   - Takes multiple `PerformanceMetrics` objects

3. **Regime-aware metrics** (bull/bear/ranging performance breakdown)
   - Implementation: Enhanced strategy tracker (already exists)
   - Not at engine level (separation of concerns)

4. **Real-time performance dashboards** with WebSocket updates
   - Implementation: Dashboard queries live engine's `PerformanceTracker`
   - Polling or event-driven updates

#### Long-term (6-12 months):
1. **Multi-strategy portfolio metrics** (correlation, combined Sharpe)
   - Implementation: `PortfolioPerformanceTracker`
   - Aggregates multiple engine trackers

2. **Benchmark comparison** (vs. BTC buy-and-hold, vs. indices)
   - Implementation: `BenchmarkComparator` utility
   - Calculates alpha, beta, tracking error

3. **Machine learning on performance metrics** (predict strategy degradation)
   - Implementation: Separate ML pipeline
   - Reads from `PerformanceMetrics` database table

4. **Automated performance reporting** (daily/weekly email summaries)
   - Implementation: Reporting service
   - Queries aggregated metrics from database

## Implementation Plan

### Phase 1: Enhance Shared Tracker (Foundation)

**Files to modify:**
- `src/engines/shared/performance_tracker.py`
- `src/performance/metrics.py`

**Tasks:**
1. ✅ Add missing pure metric functions to `src/performance/metrics.py`:
   - `sortino_ratio()`
   - `calmar_ratio()`
   - `value_at_risk()`
   - `expectancy()`

2. ✅ Enhance `PerformanceMetrics` dataclass with:
   - `sortino_ratio`, `calmar_ratio`, `var_95`, `expectancy`
   - `consecutive_wins`, `consecutive_losses`

3. ✅ Update `PerformanceTracker.get_metrics()` to calculate new fields:
   - Calculate Sortino/Calmar/VaR using balance history
   - Track consecutive streaks during `record_trade()`

4. ✅ Add balance history export as DataFrame:
   ```python
   def get_balance_series(self) -> pd.Series:
       """Return balance history as pandas Series for metric calculations."""
   ```

5. ✅ Write comprehensive unit tests for new metrics

**Acceptance Criteria:**
- All new metrics have >95% test coverage
- All metrics calculations use pure functions from `src/performance/metrics.py`
- Thread safety maintained for live engine use
- No database/storage logic in shared tracker

### Phase 2: Integrate into Backtest Engine

**Files to modify:**
- `src/engines/backtest/engine.py`
- `src/engines/backtest/utils.py` (may deprecate)

**Tasks:**
1. ✅ Initialize `PerformanceTracker` in `__init__()`:
   ```python
   self.performance_tracker = PerformanceTracker(initial_balance)
   ```

2. ✅ Replace manual metric tracking in `_run_main_loop()`:
   - Remove: `self.peak_balance`, `max_drawdown_running` updates
   - Add: `self.performance_tracker.record_trade()` on trade completion
   - Add: `self.performance_tracker.update_balance()` each candle

3. ✅ Update early stop logic to use tracker metrics:
   ```python
   metrics = self.performance_tracker.get_metrics()
   if metrics.current_drawdown > self._early_stop_max_drawdown:
       # Early stop
   ```

4. ✅ Replace `compute_performance_metrics()` in `_build_final_results()`:
   ```python
   metrics = self.performance_tracker.get_metrics()
   results = {
       **metrics.to_dict(),
       "yearly_returns": self._calculate_yearly_returns(),
       "hold_return": hold_return,
       "prediction_metrics": pred_metrics,
   }
   ```

5. ✅ Preserve backward compatibility:
   - Keep all existing result keys
   - Ensure CLI output unchanged
   - Database schema unchanged

6. ✅ Add integration tests:
   - Run backtest, verify metrics match expected values
   - Compare with previous implementation (regression test)

**Acceptance Criteria:**
- All existing backtest tests pass
- Metrics match previous implementation (within 0.01% tolerance)
- No performance regression (runtime within 5% of baseline)
- CLI output format unchanged

### Phase 3: Integrate into Live Engine

**Files to modify:**
- `src/engines/live/trading_engine.py`

**Tasks:**
1. ✅ Initialize `PerformanceTracker` in `__init__()`:
   ```python
   self.performance_tracker = PerformanceTracker(initial_balance)
   ```

2. ✅ Replace manual metric tracking:
   - Remove: `self.total_trades`, `self.winning_trades`, `self.total_pnl`, `self.peak_balance`, `self.max_drawdown`
   - Keep: Database persistence logic (separate concern)

3. ✅ Update `_close_position()` to use tracker:
   ```python
   self.performance_tracker.record_trade(trade, fee, slippage)
   self.performance_tracker.update_balance(self.current_balance, datetime.now())

   # Persist to DB
   metrics = self.performance_tracker.get_metrics()
   self._save_performance_to_db(metrics)
   ```

4. ✅ Update `_update_performance_metrics()`:
   ```python
   def _update_performance_metrics(self):
       """Update performance tracking and persist to database."""
       self.performance_tracker.update_balance(self.current_balance, datetime.now())

       # Database persistence
       if self._should_save_snapshot():
           metrics = self.performance_tracker.get_metrics()
           self._save_account_history(metrics)
   ```

5. ✅ Update `get_performance_summary()`:
   ```python
   def get_performance_summary(self) -> dict:
       metrics = self.performance_tracker.get_metrics()
       return {
           **metrics.to_dict(),
           "open_positions": len(self.positions),
           "total_exposure": self._calculate_total_exposure(),
       }
   ```

6. ✅ Ensure database writes include new metrics:
   - Update `AccountHistory` writes to include Sortino, Calmar, VaR
   - Update `PerformanceMetrics` table writes

7. ✅ Add integration tests:
   - Mock live trading session
   - Verify metrics calculated correctly
   - Verify database persistence

**Acceptance Criteria:**
- All existing live engine tests pass
- Database writes succeed with new metrics
- Thread safety verified (concurrent balance updates)
- Health monitoring endpoints return new metrics

### Phase 4: Database Schema & Migration

**Files to modify:**
- `src/database/models.py`
- New migration file

**Tasks:**
1. ✅ Add columns to `PerformanceMetrics` table:
   ```sql
   ALTER TABLE performance_metrics ADD COLUMN sortino_ratio NUMERIC(18, 8);
   ALTER TABLE performance_metrics ADD COLUMN calmar_ratio NUMERIC(18, 8);
   ALTER TABLE performance_metrics ADD COLUMN var_95 NUMERIC(18, 8);
   ALTER TABLE performance_metrics ADD COLUMN expectancy NUMERIC(18, 8);
   ALTER TABLE performance_metrics ADD COLUMN consecutive_wins INTEGER;
   ALTER TABLE performance_metrics ADD COLUMN consecutive_losses INTEGER;
   ```

2. ✅ Add columns to `AccountHistory` table:
   ```sql
   ALTER TABLE account_history ADD COLUMN sortino_ratio NUMERIC(18, 8);
   ALTER TABLE account_history ADD COLUMN calmar_ratio NUMERIC(18, 8);
   ALTER TABLE account_history ADD COLUMN var_95 NUMERIC(18, 8);
   ```

3. ✅ Update `DatabaseManager` write methods:
   - `save_performance_metrics()` → Include new fields
   - `save_account_history()` → Include new fields

4. ✅ Write Alembic migration:
   ```bash
   alembic revision --autogenerate -m "Add advanced performance metrics"
   ```

5. ✅ Test migration:
   - Run on test database
   - Verify backward compatibility (old code works with new schema)
   - Verify forward compatibility (new code writes all fields)

**Acceptance Criteria:**
- Migration runs without errors
- All new columns nullable (for backward compat)
- Database tests pass
- Monitoring dashboard displays new metrics

### Phase 5: Testing & Validation

**Files to create:**
- `tests/unit/engines/shared/test_performance_tracker.py`
- `tests/integration/test_performance_parity.py`

**Tasks:**
1. ✅ Unit tests for enhanced `PerformanceTracker`:
   - Test all metric calculations
   - Test thread safety (concurrent updates)
   - Test edge cases (zero trades, negative PnL, etc.)

2. ✅ Integration test: Backtest vs. Live parity:
   ```python
   def test_backtest_live_metric_parity():
       """Run same trade sequence through backtest and mock live engine.
       Verify final metrics are identical."""

       trades = generate_test_trades()

       # Run through backtest
       backtest_results = run_backtest_with_trades(trades)

       # Run through mock live engine
       live_results = run_mock_live_with_trades(trades)

       # Compare metrics (within 0.01% tolerance)
       assert_metrics_equal(backtest_results, live_results)
   ```

3. ✅ Regression tests:
   - Run existing backtest examples, compare metrics with baseline
   - Verify CLI output format unchanged
   - Verify database schema compatible

4. ✅ Performance benchmarks:
   - Measure overhead of tracker in backtest loop
   - Verify <5% performance impact
   - Profile and optimize if needed

**Acceptance Criteria:**
- All tests pass with >95% coverage
- Parity test shows <0.01% difference in core metrics
- No performance regression >5%
- Backward compatibility verified

### Phase 6: Documentation & Cleanup

**Files to update:**
- `docs/architecture.md`
- `docs/backtesting.md`
- `docs/live_trading.md`
- `docs/issues/remaining_shared_module_integrations.md`

**Tasks:**
1. ✅ Update architecture docs:
   - Document performance tracking architecture
   - Diagram showing layered design
   - Explain shared vs. engine-specific metrics

2. ✅ Update engine documentation:
   - Document new metrics available
   - Explain how to access metrics via API
   - Show dashboard integration

3. ✅ Add usage examples:
   ```python
   # Example: Accessing performance metrics in backtest
   results = backtest_engine.run(...)
   print(f"Sharpe: {results['sharpe_ratio']:.2f}")
   print(f"Sortino: {results['sortino_ratio']:.2f}")
   print(f"Calmar: {results['calmar_ratio']:.2f}")
   ```

4. ✅ Clean up deprecated code:
   - Consider deprecating `compute_performance_metrics()` in `backtest/utils.py`
   - Remove manual metric tracking variables from engines
   - Update comments referencing old implementation

5. ✅ Update `remaining_shared_module_integrations.md`:
   - Mark `PerformanceTracker` as ✅ Completed
   - Document lessons learned
   - Reference this execplan

**Acceptance Criteria:**
- All docs updated and reviewed
- No broken documentation links
- Code comments accurate
- Deprecated code clearly marked

## Risk Assessment & Mitigation

### Risk: Metric Calculation Differences

**Impact:** High - Backtest metrics not matching live metrics breaks validation workflow

**Mitigation:**
- Use same underlying pure functions from `src/performance/metrics.py`
- Automated parity test in CI/CD
- Regression tests comparing with baseline

### Risk: Performance Regression in Backtest

**Impact:** Medium - Slower backtests reduce developer productivity

**Mitigation:**
- Profile before/after integration
- Optimize hot paths (e.g., balance history storage)
- Use efficient data structures (deque instead of list)
- Benchmark with large datasets (1+ years of 1h candles)

### Risk: Database Migration Issues

**Impact:** Medium - Failed migration could break production live trading

**Mitigation:**
- Test migration on staging environment first
- Make all new columns nullable
- Ensure backward compatibility (old code works with new schema)
- Have rollback plan (reverse migration)

### Risk: Thread Safety Issues in Live Engine

**Impact:** High - Race conditions could corrupt metrics

**Mitigation:**
- Use threading.Lock() in PerformanceTracker (already present)
- Integration tests with concurrent updates
- Code review focusing on thread safety

### Risk: Breaking Changes to CLI Output

**Impact:** Medium - Users/scripts relying on output format break

**Mitigation:**
- Preserve all existing result keys
- Add new metrics as additional fields
- Test CLI output format against baseline
- Document any changes in release notes

## Success Criteria

**Must Have:**
1. ✅ Both engines use shared `PerformanceTracker`
2. ✅ Core metrics (Sharpe, drawdown, win rate) match between backtest and live (<0.01% diff)
3. ✅ All existing tests pass
4. ✅ Database schema supports new metrics
5. ✅ Thread safety verified for live engine
6. ✅ No performance regression >5%

**Should Have:**
1. ✅ Advanced metrics (Sortino, Calmar, VaR, expectancy) calculated
2. ✅ Comprehensive test coverage >95%
3. ✅ Documentation updated
4. ✅ Backward compatibility maintained

**Nice to Have:**
1. ⚠️ Performance improvement from optimizations
2. ⚠️ Monitoring dashboard displays new metrics
3. ⚠️ Deprecation of old metric calculation code

## Progress

### Phase 1: Enhance Shared Tracker
- [ ] Add pure metric functions to `src/performance/metrics.py`
- [ ] Enhance `PerformanceMetrics` dataclass
- [ ] Update `PerformanceTracker.get_metrics()`
- [ ] Add balance history export
- [ ] Write unit tests

### Phase 2: Backtest Integration
- [ ] Initialize tracker in backtest engine
- [ ] Replace manual metric tracking
- [ ] Update early stop logic
- [ ] Update result building
- [ ] Add integration tests

### Phase 3: Live Engine Integration
- [ ] Initialize tracker in live engine
- [ ] Replace manual metric tracking
- [ ] Update trade closure logic
- [ ] Update performance summary
- [ ] Update database persistence

### Phase 4: Database Schema
- [ ] Design schema changes
- [ ] Write Alembic migration
- [ ] Update DatabaseManager
- [ ] Test migration

### Phase 5: Testing & Validation
- [ ] Unit tests for tracker
- [ ] Parity tests (backtest vs. live)
- [ ] Regression tests
- [ ] Performance benchmarks

### Phase 6: Documentation
- [ ] Update architecture docs
- [ ] Update engine docs
- [ ] Add usage examples
- [ ] Clean up deprecated code
- [ ] Update issue tracker

## Surprises & Discoveries

_To be filled as implementation progresses_

## Decision Log

### Decision 1: Separate Concerns (Calculation vs. Storage)
**Date:** 2025-12-26
**Decision:** Keep metric calculation in shared tracker, leave persistence to engines
**Rationale:** Backtest often runs without database, live engine has specific persistence requirements. Clean separation allows flexibility.
**Alternatives Considered:** Include storage backend in shared tracker (rejected due to tight coupling)

### Decision 2: Include Advanced Metrics in Shared Tracker
**Date:** 2025-12-26
**Decision:** Add Sortino, Calmar, VaR, expectancy to shared tracker
**Rationale:** These are standard metrics, useful for both engines, low overhead to calculate
**Alternatives Considered:** Leave in strategy tracker (rejected - creates duplication), Calculate only in post-processing (rejected - less flexible)

### Decision 3: No Regime Tracking in Engine Tracker
**Date:** 2025-12-26
**Decision:** Regime-specific metrics stay in strategy component tracker
**Rationale:** Regime detection is strategy concern, not engine concern. Separation of concerns.
**Alternatives Considered:** Include regime tracking (rejected - increases complexity, violates SRP)

### Decision 4: DataFrame Export for Balance History
**Date:** 2025-12-26
**Decision:** Provide `get_balance_series()` method returning pandas Series
**Rationale:** Sharpe/Sortino/VaR calculations need balance series, pandas is standard for time series
**Alternatives Considered:** Store balance as numpy array (rejected - less flexible), Calculate inline (rejected - duplicates logic)

## Outcomes & Retrospective

_To be filled upon completion_

## Related Work

- Parent Issue: #454 (Extract shared logic between engines)
- Related: `docs/issues/remaining_shared_module_integrations.md`
- Previous Integrations: TrailingStopManager, PolicyHydrator, RiskConfiguration, DynamicRiskHandler
- Previous Integrations: CostCalculator (#466), PartialOperationsManager (#461)

## Commands to Run

**Setup:**
```bash
git checkout develop
git pull origin develop
git checkout claude/plan-tracker-integration-htyvO
```

**Development:**
```bash
# Run tests during development
atb test unit
python -m pytest tests/unit/engines/shared/test_performance_tracker.py -v

# Run integration tests
atb test integration

# Check code quality
atb dev quality
```

**Validation:**
```bash
# Run backtest to verify metrics
atb backtest ml_basic --symbol BTCUSDT --timeframe 1h --days 30

# Run live engine in paper mode
atb live ml_basic --symbol BTCUSDT --paper-trading

# Database migration
alembic upgrade head
atb db verify
```

**Expected Outcomes:**
- Backtest should complete with all new metrics displayed
- Live engine should run without errors
- Database should contain new metric columns
- Tests should show 100% pass rate
