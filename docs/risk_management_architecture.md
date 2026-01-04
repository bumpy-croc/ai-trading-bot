# Risk Management Architecture

This document explains the three-layer risk management architecture and how the components work together to provide comprehensive risk control.

> **Class Naming Update**: The portfolio-level risk manager has been renamed from `RiskManager` to `PortfolioRiskManager` to eliminate naming confusion with the strategy-level `RiskManager` abstract base class. The old name is aliased for backward compatibility.

## Table of Contents

1. [Overview](#overview)
2. [The Three Risk Layers](#the-three-risk-layers)
3. [Data Flow](#data-flow)
4. [When to Use Which Layer](#when-to-use-which-layer)
5. [Examples](#examples)
6. [Common Pitfalls](#common-pitfalls)

## Overview

The trading system uses a **three-layer risk management architecture** that separates concerns and provides defense-in-depth:

```
┌─────────────────────────────────────────────────────────────┐
│                    LAYER 1: STRATEGY LEVEL                  │
│  Component-based risk decisions for individual signals      │
│  Location: src/strategies/components/risk_manager.py        │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                   LAYER 2: PORTFOLIO LEVEL                  │
│  Global risk constraints enforced across all positions      │
│  Location: src/risk/risk_manager.py                         │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                   LAYER 3: DYNAMIC ADJUSTMENT               │
│  Performance-based real-time risk adjustments               │
│  Location: src/engines/shared/dynamic_risk_handler.py       │
└─────────────────────────────────────────────────────────────┘
```

Each layer serves a distinct purpose and operates at a different scope.

## The Three Risk Layers

### Layer 1: Strategy-Level Risk (`src/strategies/components/risk_manager.py`)

**Purpose**: Provides the abstract interface for strategy-specific risk decisions.

**Scope**: Per-signal, per-strategy tactical decisions.

**Key Characteristics**:
- Abstract base class: `RiskManager` (component interface)
- Composed into `Strategy` objects using the component pattern
- Makes decisions based on:
  - Individual trading signals
  - Signal strength and confidence
  - Market regime
  - Technical indicators (ATR, volatility)

**Concrete Implementations**:

```python
# Fixed percentage risk
FixedRiskManager(risk_per_trade=0.02, stop_loss_pct=0.05)

# Volatility-adjusted risk (ATR-based)
VolatilityRiskManager(base_risk=0.02, atr_multiplier=2.0)

# Regime-adaptive risk
RegimeAdaptiveRiskManager(
    base_risk=0.02,
    regime_multipliers={
        "bull_low_vol": 1.5,
        "bear_high_vol": 0.3,
    }
)
```

**Abstract Methods** (must be implemented by subclasses):
- `calculate_position_size(signal, balance, regime)` - Calculate size for a signal
- `should_exit(position, current_data, regime)` - Determine if position should be exited
- `get_stop_loss(entry_price, signal, regime)` - Calculate stop loss level

**Optional Override Methods** (have default implementations):
- `get_take_profit(entry_price, signal, regime)` - Calculate take profit level (default: returns entry_price)
- `get_position_policies(signal, balance, regime)` - Return policy descriptors (default: returns None)

**When to Use**:
- When implementing a new trading strategy
- When you need signal-specific risk logic
- When risk decisions depend on regime or indicator values
- When you want to compose strategies with different risk approaches

**Example**:
```python
from src.strategies.components import Strategy, SignalGenerator
from src.strategies.components.risk_manager import VolatilityRiskManager

# Strategy composes a risk manager component
strategy = Strategy(
    name="my_strategy",
    signal_generator=my_signal_generator,
    risk_manager=VolatilityRiskManager(atr_multiplier=2.0),
    position_sizer=my_position_sizer,
)

# Strategy uses risk manager to calculate position size
decision = strategy.process_candle(df, index, balance, positions)
# decision.position_size comes from risk_manager.calculate_position_size()
```

---

### Bridging Layer 1 & 2: CoreRiskAdapter (`src/strategies/components/risk_adapter.py`)

**Purpose**: Bridges the strategy component interface (Layer 1) with the portfolio risk manager (Layer 2).

**Why It's Needed**:
- Layer 1 components use the abstract `RiskManager` interface
- Layer 2 is a concrete `RiskManager` class with different method signatures
- The adapter wraps Layer 2 to provide the Layer 1 interface

**Key Features**:
- Implements the abstract `RiskManager` interface from Layer 1
- Delegates to the portfolio `RiskManager` from Layer 2
- Provides portfolio state hooks for lifecycle events
- Merges strategy-specific overrides with portfolio constraints

**Usage**:
```python
from src.strategies.components.risk_adapter import CoreRiskAdapter
from src.risk.risk_manager import PortfolioRiskManager, RiskParameters

# Create portfolio risk manager (Layer 2)
portfolio_risk = PortfolioRiskManager(parameters=RiskParameters(max_daily_risk=0.06))

# Wrap it in an adapter for component strategies (Layer 1)
adapter = CoreRiskAdapter(core_manager=portfolio_risk)

# Use adapter in strategy composition
strategy = Strategy(
    name="my_strategy",
    signal_generator=my_signal_gen,
    risk_manager=adapter,  # Uses adapter, not direct portfolio manager
    position_sizer=my_sizer,
)

# Strategy calls adapter methods (Layer 1 interface)
# Adapter delegates to portfolio manager (Layer 2 implementation)
```

**How It Works**:
1. Strategy calls `adapter.calculate_position_size(signal, balance, regime)`
2. Adapter converts to Layer 2 call: `core_manager.calculate_position_fraction(...)`
3. Adapter applies strategy-specific overrides
4. Returns result in Layer 1 format

**Portfolio State Hooks**:
```python
from src.strategies.components.risk_adapter import PortfolioStateHooks

# Define hooks for portfolio events
hooks = PortfolioStateHooks(
    on_fill=lambda symbol, size, price: portfolio_risk.update_position(...),
    on_close=lambda symbol: portfolio_risk.close_position(symbol),
)

adapter.set_portfolio_hooks(hooks)
```

**When to Use**:
- When composing strategies with component-based risk managers
- When you need Layer 1 interface but want Layer 2 constraints
- When integrating with engines that use both layers

**Important Note**:
Most users don't need to use `CoreRiskAdapter` directly - the engines handle this integration automatically. Use it only when manually composing strategies with portfolio risk management.

---

### Layer 2: Portfolio-Level Risk (`src/risk/risk_manager.py`)

**Purpose**: Enforces global risk constraints across the entire portfolio.

**Scope**: Multi-symbol, portfolio-wide strategic constraints.

**Key Characteristics**:
- Concrete class: `PortfolioRiskManager` (global portfolio manager)
- Singleton instance shared across all positions
- Thread-safe with locks for concurrent access
- Tracks state:
  - All open positions (`positions` dict)
  - Daily risk used (`daily_risk_used`)
  - Peak balance for drawdown calculations

**Configuration**:
```python
from src.risk.risk_manager import PortfolioRiskManager, RiskParameters

params = RiskParameters(
    base_risk_per_trade=0.02,           # 2% base risk per trade
    max_risk_per_trade=0.03,            # 3% maximum per trade
    max_position_size=0.10,             # 10% max position size
    max_daily_risk=0.06,                # 6% max daily exposure
    max_correlated_risk=0.10,           # 10% max for correlated assets
    max_drawdown=0.20,                  # 20% max drawdown
)

risk_manager = PortfolioRiskManager(parameters=params, max_concurrent_positions=3)
```

**Key Methods**:

**Position Sizing**:
```python
# Calculate allowed position size (returns fraction 0..1)
fraction = risk_manager.calculate_position_fraction(
    df=df,
    index=i,
    balance=10000,
    strategy_overrides={
        'position_sizer': 'atr_risk',  # or 'fixed_fraction', 'confidence_weighted'
        'base_fraction': 0.02,
    },
    correlation_ctx={...},  # Optional correlation constraints
)
```

**Stop Loss / Take Profit**:
```python
# Compute SL/TP levels
sl_price, tp_price = risk_manager.compute_sl_tp(
    df=df,
    index=i,
    entry_price=50000,
    side='long',
    strategy_overrides={'stop_loss_pct': 0.02, 'take_profit_pct': 0.04},
)
```

**Position Tracking**:
```python
# Update position (thread-safe)
risk_manager.update_position(
    symbol='BTCUSDT',
    side='long',
    size=0.05,  # 5% of balance
    entry_price=50000,
)

# Check drawdown
if risk_manager.check_drawdown(current_balance, peak_balance):
    # Drawdown limit exceeded - halt trading

# Close position (frees up daily risk)
risk_manager.close_position('BTCUSDT')
```

**Correlation Control**:
```python
# Check correlated exposure
exposure = risk_manager.get_position_correlation_risk(
    symbols=['BTCUSDT', 'ETHUSDT'],
    corr_matrix=correlation_matrix,
    threshold=0.7,
)
```

**When to Use**:
- In trading engine initialization (backtest or live)
- For enforcing daily risk limits
- For tracking positions across multiple symbols
- For correlation-based position sizing
- For drawdown protection

**Important Notes**:

**Daily Risk Accounting**:
The `daily_risk_used` tracks **exposure** (capital allocation), NOT actual capital at risk:
- 10% position → `daily_risk_used += 0.1` (regardless of stop loss distance)
- This is conservative: prevents over-leveraging
- Traditional risk would be: `risk = position_size × stop_loss_distance`

```python
# Example: 10% position with 1% stop loss
# Traditional risk: 0.10 × 0.01 = 0.001 (0.1% capital at risk)
# Our accounting: 0.10 (10% exposure tracked)
```

**Thread Safety**:
All operations are protected by `_state_lock` for safe concurrent access from multiple threads (e.g., live trading + monitoring).

---

### Layer 3: Dynamic Risk Adjustment (`src/engines/shared/dynamic_risk_handler.py`)

**Purpose**: Applies real-time performance-based risk adjustments.

**Scope**: Dynamic adjustments based on current performance and market conditions.

**Key Characteristics**:
- Wraps `DynamicRiskManager` from `src/position_management/dynamic_risk.py`
- Reduces position sizes during drawdowns or adverse conditions
- Tracks adjustment history for post-trade analysis
- Used by **both** backtest and live engines for parity
- Thread-safe adjustment tracking

**Configuration**:
```python
from src.engines.shared.dynamic_risk_handler import DynamicRiskHandler
from src.position_management.dynamic_risk import DynamicRiskManager, DynamicRiskConfig

config = DynamicRiskConfig(
    enabled=True,
    drawdown_thresholds=[0.05, 0.10, 0.15],      # 5%, 10%, 15% drawdown
    risk_reduction_factors=[0.8, 0.6, 0.4],       # Reduce to 80%, 60%, 40%
    volatility_adjustment_enabled=True,
)

dynamic_mgr = DynamicRiskManager(config=config, db_manager=db_manager)
handler = DynamicRiskHandler(dynamic_risk_manager=dynamic_mgr)
```

**Usage in Entry Handler**:
```python
# Original position size from portfolio risk manager
original_size = 0.05  # 5% of balance

# Apply dynamic adjustments
adjusted_size = handler.apply_dynamic_risk(
    original_size=original_size,
    current_time=datetime.now(UTC),
    balance=10000,
    peak_balance=12000,  # 16.7% drawdown
    trading_session_id=session_id,
)

# If in 15% drawdown, might reduce to: 0.05 × 0.4 = 0.02 (2%)
```

**Adjustment Tracking**:
```python
# Get tracked adjustments
adjustments = handler.get_adjustments(clear=True)
for adj in adjustments:
    print(f"Time: {adj['timestamp']}")
    print(f"Reason: {adj['primary_reason']}")
    print(f"Factor: {adj['position_size_factor']}")
    print(f"Drawdown: {adj['current_drawdown']}")
```

**When to Use**:
- In entry handlers (both backtest and live)
- When you want performance-based risk reduction
- For drawdown protection beyond hard limits
- For volatility-based adjustments

**Adjustment Reasons**:
- `drawdown_reduction` - In drawdown, reduce risk
- `volatility_increase` - High volatility, reduce risk
- `poor_performance` - Recent losses, reduce risk
- `recovery_scaling` - Recovering from drawdown, scale back up

---

## Data Flow

Here's how a trade entry flows through all three risk layers:

```
┌────────────────────────────────────────────────────────────────┐
│                    TRADING ENGINE (Backtest/Live)              │
└────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌────────────────────────────────────────────────────────────────┐
│                         ENTRY HANDLER                          │
│                                                                │
│  Step 1: Get Signal & Initial Size                            │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │ strategy.process_candle(df, index, balance, positions)   │ │
│  │   └─> Uses LAYER 1: Strategy Component Risk Manager      │ │
│  │       - signal_generator.generate_signal(df, index)       │ │
│  │       - risk_manager.calculate_position_size(signal, ...) │ │
│  │       - risk_manager.get_stop_loss(entry_price, ...)      │ │
│  │       Returns: TradingDecision with recommended size      │ │
│  └──────────────────────────────────────────────────────────┘ │
│                                │                               │
│                                ▼                               │
│  Step 2: Apply Portfolio Constraints                          │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │ LAYER 2: Portfolio Risk Manager                          │ │
│  │ risk_manager.calculate_position_fraction(...)             │ │
│  │   - Check daily_risk_used vs max_daily_risk              │ │
│  │   - Check correlation constraints                         │ │
│  │   - Check max_concurrent_positions                        │ │
│  │   - Enforce max_position_size                             │ │
│  │   Returns: Allowed position fraction (may be < strategy) │ │
│  └──────────────────────────────────────────────────────────┘ │
│                                │                               │
│                                ▼                               │
│  Step 3: Apply Dynamic Adjustments                            │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │ LAYER 3: Dynamic Risk Handler                            │ │
│  │ handler.apply_dynamic_risk(...)                           │ │
│  │   - Check current drawdown vs thresholds                  │ │
│  │   - Check recent performance (win rate, profit factor)    │ │
│  │   - Check market volatility                               │ │
│  │   - Apply reduction factor if needed                      │ │
│  │   Returns: Final adjusted position size                   │ │
│  └──────────────────────────────────────────────────────────┘ │
│                                │                               │
│                                ▼                               │
│  Step 4: Execute Entry                                         │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │ execution_engine.execute_entry(...)                       │ │
│  │   - Convert fraction to notional                          │ │
│  │   - Place order                                            │ │
│  │   - Update position tracker                               │ │
│  │   - Update risk_manager.positions and daily_risk_used     │ │
│  └──────────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────┘
```

### Example with Numbers

```python
# LAYER 1: Strategy says "take a position"
strategy_decision = strategy.process_candle(df, i, balance=10000, positions={})
# → signal: BUY, strength: 0.8, confidence: 0.9
# → risk_manager (VolatilityRiskManager) calculates:
#    base_risk=0.02, atr_adjusted=0.03
# → Recommended size: $300 (3% of balance)

# LAYER 2: Portfolio enforces global constraints
allowed_fraction = risk_manager.calculate_position_fraction(
    df, i, balance=10000,
    strategy_overrides={'base_fraction': 0.03},
)
# → Check: daily_risk_used = 0.02 (2% from existing position)
# → Check: max_daily_risk = 0.06 (6%)
# → Remaining: 0.04 (4%)
# → Clamp: min(0.03, 0.04) = 0.03
# → Allowed size: $300 (3% of balance) ✓

# LAYER 3: Dynamic adjustment for drawdown
adjusted_size = handler.apply_dynamic_risk(
    original_size=0.03,
    balance=10000,
    peak_balance=12000,  # 16.7% drawdown
)
# → Drawdown: (12000 - 10000) / 12000 = 16.7%
# → Threshold: 15% drawdown → reduction factor: 0.4
# → Final size: 0.03 × 0.4 = 0.012 (1.2%)
# → Final notional: $120

# RESULT: Entry executed with $120 position instead of $300
```

---

## When to Use Which Layer

### Use Layer 1 (Strategy Component) When:
- ✅ Implementing a new trading strategy
- ✅ Signal-specific risk logic needed
- ✅ Risk depends on regime or technical indicators
- ✅ Different strategies need different risk approaches
- ✅ Testing different risk management approaches

**Example**:
```python
# Strategy A uses fixed risk
strategy_a = Strategy(
    name="conservative",
    risk_manager=FixedRiskManager(risk_per_trade=0.01),
    ...
)

# Strategy B uses volatility-adjusted risk
strategy_b = Strategy(
    name="adaptive",
    risk_manager=VolatilityRiskManager(atr_multiplier=2.5),
    ...
)
```

### Use Layer 2 (Portfolio Manager) When:
- ✅ Initializing trading engines (backtest or live)
- ✅ Enforcing daily risk limits across all trades
- ✅ Tracking positions across multiple symbols
- ✅ Implementing correlation-based sizing
- ✅ Checking drawdown limits
- ✅ Need thread-safe position tracking

**Example**:
```python
# Initialize engine with global risk manager
engine = BacktestEngine(
    strategy=strategy,
    risk_manager=RiskManager(
        parameters=RiskParameters(max_daily_risk=0.06),
        max_concurrent_positions=3,
    ),
    ...
)
```

### Use Layer 3 (Dynamic Handler) When:
- ✅ Want performance-based risk reduction
- ✅ Implementing drawdown protection
- ✅ Need volatility-based adjustments
- ✅ Tracking adjustment history for analysis
- ✅ Ensuring backtest/live parity in risk adjustments

**Example**:
```python
# Add dynamic risk to entry handler
entry_handler = EntryHandler(
    risk_manager=portfolio_risk_manager,
    dynamic_risk_manager=DynamicRiskManager(config),
    ...
)
```

---

## Examples

### Example 1: Basic Strategy Setup

```python
from src.strategies.components import Strategy
from src.strategies.components.risk_manager import VolatilityRiskManager
from src.risk.risk_manager import PortfolioRiskManager, RiskParameters

# LAYER 1: Strategy-level risk component
strategy = Strategy(
    name="my_strategy",
    signal_generator=my_signal_gen,
    risk_manager=VolatilityRiskManager(
        base_risk=0.02,
        atr_multiplier=2.0,
    ),
    position_sizer=my_sizer,
)

# LAYER 2: Portfolio-level risk manager
portfolio_risk = PortfolioRiskManager(
    parameters=RiskParameters(
        max_daily_risk=0.06,
        max_position_size=0.10,
    ),
    max_concurrent_positions=3,
)

# Use in backtest
engine = BacktestEngine(
    strategy=strategy,
    risk_manager=portfolio_risk,
    ...
)
```

### Example 2: Adding Dynamic Risk

```python
from src.position_management.dynamic_risk import DynamicRiskManager, DynamicRiskConfig
from src.engines.shared.dynamic_risk_handler import DynamicRiskHandler

# LAYER 3: Configure dynamic risk
config = DynamicRiskConfig(
    enabled=True,
    drawdown_thresholds=[0.05, 0.10, 0.15],
    risk_reduction_factors=[0.8, 0.6, 0.4],
)

dynamic_mgr = DynamicRiskManager(config=config, db_manager=db)

# Engine automatically uses DynamicRiskHandler internally
engine = BacktestEngine(
    strategy=strategy,
    risk_manager=portfolio_risk,
    dynamic_risk_manager=dynamic_mgr,  # Adds Layer 3
    ...
)
```

### Example 3: Correlation-Based Sizing

```python
from src.engines.shared.correlation_handler import CorrelationHandler

# Set up correlation control
corr_handler = CorrelationHandler(
    risk_manager=portfolio_risk,
    max_correlated_exposure=0.15,  # Max 15% in correlated assets
)

# Engine uses correlation handler to adjust sizes
engine = BacktestEngine(
    strategy=strategy,
    risk_manager=portfolio_risk,
    correlation_handler=corr_handler,
    ...
)

# Correlation matrix is updated periodically and used during sizing
```

---

## Common Pitfalls

### Pitfall 1: Confusing the Two RiskManager Classes

**Problem**: Both files have a class named `RiskManager`:
- `src/strategies/components/risk_manager.py::RiskManager` (abstract base)
- `src/risk/risk_manager.py::RiskManager` (concrete portfolio manager)

**Solution**: Use fully qualified imports:
```python
# Strategy component (abstract)
from src.strategies.components.risk_manager import RiskManager as StrategyRiskComponent

# Portfolio manager (concrete)
from src.risk.risk_manager import PortfolioRiskManager as PortfolioRiskManager
```

### Pitfall 2: Not Understanding daily_risk_used

**Problem**: `daily_risk_used` tracks **exposure**, not actual capital at risk.

**Explanation**:
- 10% position → `daily_risk_used += 0.10` (regardless of stop loss)
- This is **conservative** to prevent over-leveraging
- Traditional risk: `position_size × stop_loss_distance`

**Impact**:
```python
# If you open a 10% position with 1% stop loss:
# Actual capital at risk: 0.10 × 0.01 = 0.001 (0.1%)
# daily_risk_used: 0.10 (10%)

# This means you can't open as many positions as theoretical risk allows
```

### Pitfall 3: Bypassing Portfolio Risk Manager

**Problem**: Directly using strategy component risk manager for position sizing without global constraints.

**Wrong**:
```python
# DON'T DO THIS
size = strategy.risk_manager.calculate_position_size(signal, balance, regime)
# This bypasses daily risk limits, correlation checks, etc.
```

**Correct**:
```python
# Let the engine handle the flow
decision = strategy.process_candle(df, i, balance, positions)
# Engine will apply portfolio constraints via risk_manager.calculate_position_fraction()
```

### Pitfall 4: Thread Safety Violations

**Problem**: Accessing `risk_manager.positions` or `daily_risk_used` without locks.

**Wrong**:
```python
# DON'T DO THIS (not thread-safe)
if symbol in risk_manager.positions:
    # Another thread could modify positions here!
    risk_manager.positions[symbol]['size'] = new_size
```

**Correct**:
```python
# Use the public API (already thread-safe)
risk_manager.update_position(symbol, side, size, entry_price)
risk_manager.close_position(symbol)
```

### Pitfall 5: Not Resetting daily_risk_used

**Problem**: Forgetting to reset daily risk at the start of each trading day.

**Impact**: Daily risk accumulates forever, blocking all future trades.

**Solution**:
```python
# In live trading engine, reset daily at session start
risk_manager.reset_daily_risk()

# Or schedule it:
if is_new_trading_day(current_time, last_reset_time):
    risk_manager.reset_daily_risk()
```

### Pitfall 6: Ignoring Dynamic Adjustments in Backtests

**Problem**: Not using dynamic risk in backtests, then adding it in live → parity breaks.

**Solution**: Always use dynamic risk in both backtest and live:
```python
# Backtest
backtest_engine = BacktestEngine(
    dynamic_risk_manager=dynamic_mgr,  # Include this!
    ...
)

# Live
live_engine = TradingEngine(
    dynamic_risk_manager=dynamic_mgr,  # Same config
    ...
)
```

---

## Summary

The three-layer risk architecture provides **defense-in-depth**:

1. **Strategy Layer**: Tactical signal-specific risk decisions
2. **Portfolio Layer**: Strategic global constraints and tracking
3. **Dynamic Layer**: Adaptive performance-based adjustments

Each layer serves a distinct purpose and operates at a different scope. Understanding when to use each layer is key to implementing robust risk management.

For more details:
- Strategy components: `src/strategies/README.md`
- Risk management: `src/risk/README.md`
- Position management: `src/position_management/README.md`
- Architecture: `docs/architecture.md`
