# Risk and Position Management Layer

This document describes the refactored risk and position management architecture, how it integrates with strategies, and how to configure per-strategy overrides.

## Goals
- Centralize risk logic (position sizing, stop loss, take profit) outside strategies
- Provide safe defaults while allowing per-strategy risk profiles
- Support multiple sizing policies (fixed fraction, confidence-weighted, ATR risk)
- **Add dynamic risk throttling based on performance and drawdown**
- Preserve backward compatibility with existing strategies and tests

## Components
- `src/risk/risk_manager.py`
  - `RiskParameters`: global defaults and limits
  - `RiskManager`:
    - `calculate_position_fraction(...)` returns fraction of balance to allocate
    - `compute_sl_tp(...)` computes stop loss and take profit
    - Legacy methods kept: `calculate_position_size(...)` (quantity-based), `calculate_stop_loss(...)`
- **`src/position_management/dynamic_risk.py`** *(NEW)*
  - `DynamicRiskManager`: adaptive risk management based on performance
  - `DynamicRiskConfig`: configuration for dynamic adjustments
  - `RiskAdjustments`: container for calculated risk adjustments
- `src/strategies/base.py`
  - `BaseStrategy.get_risk_overrides() -> Optional[dict]` (new optional hook)
- Engines
  - `src/live/trading_engine.py` and `src/backtesting/engine.py` now delegate sizing and SL/TP to `RiskManager`, passing strategy overrides
  - **Dynamic risk integration for real-time risk adjustments**

## Dynamic Risk Management

### Overview
The dynamic risk management system automatically adjusts position sizing, stop-loss levels, and risk limits based on:
- **Recent performance metrics** (win rate, profit factor)
- **Current drawdown levels** (5%, 10%, 15% thresholds)
- **Market volatility conditions** (high/low volatility adjustments)

### Configuration
Dynamic risk management is configured through `DynamicRiskConfig`:

```python
from src.position_management.dynamic_risk import DynamicRiskConfig

config = DynamicRiskConfig(
    enabled=True,                           # Enable/disable dynamic risk
    performance_window_days=30,             # Rolling window for performance calculation
    drawdown_thresholds=[0.05, 0.10, 0.15], # Drawdown levels that trigger reduction
    risk_reduction_factors=[0.8, 0.6, 0.4], # Risk reduction at each threshold
    recovery_thresholds=[0.02, 0.05],       # Performance levels for risk increase
    volatility_adjustment_enabled=True,     # Enable volatility-based adjustments
)
```

### Live Trading Integration
```python
from src.live.trading_engine import LiveTradingEngine
from src.position_management.dynamic_risk import DynamicRiskConfig

engine = LiveTradingEngine(
    strategy=my_strategy,
    data_provider=data_provider,
    enable_dynamic_risk=True,               # Enable dynamic risk (default: True)
    dynamic_risk_config=config,             # Optional custom config
    # ... other parameters
)
```

### Backtesting Integration
```python
from src.backtesting.engine import Backtester
from src.position_management.dynamic_risk import DynamicRiskConfig

backtester = Backtester(
    strategy=my_strategy,
    data_provider=data_provider,
    enable_dynamic_risk=True,               # Enable for historical testing
    dynamic_risk_config=config,             # Optional custom config
    # ... other parameters
)
```

### How It Works

**Drawdown-Based Adjustments:**
- 5% drawdown → 0.8x position size, 1.2x stop-loss tightening
- 10% drawdown → 0.6x position size, 1.4x stop-loss tightening  
- 15% drawdown → 0.4x position size, 1.6x stop-loss tightening

**Performance-Based Adjustments:**
- Poor performance (win rate < 30%) → Reduce position sizes
- Good performance (win rate > 70%) → Allow larger positions
- Requires minimum 10 trades for reliable adjustment

**Volatility-Based Adjustments:**
- High volatility (>3% daily) → 0.7x risk multiplier
- Low volatility (<1% daily) → 1.3x risk multiplier

**Combination Logic:**
Multiple adjustment factors are combined conservatively (most restrictive wins):
```python
final_position_factor = min(
    drawdown_adjustment.position_size_factor,
    performance_adjustment.position_size_factor,
    volatility_adjustment.position_size_factor
)
```

### Database Models
Dynamic risk management includes new database models for tracking:

**`dynamic_performance_metrics` table:**
- Rolling performance metrics (win rate, Sharpe ratio, drawdown)
- Volatility measurements and consecutive loss/win tracking
- Risk adjustment factors applied

**`risk_adjustments` table:**
- All risk parameter changes with timestamps and reasons
- Original vs adjusted values with adjustment factors
- Context information (drawdown, performance score, volatility)
- Effectiveness tracking (trades during adjustment, P&L impact)

## Default Behavior
- Position sizing uses `'fixed_fraction'` policy by default with `base_risk_per_trade`
- SL is ATR-based if no explicit `stop_loss_pct` override is provided
- TP uses `RiskParameters.default_take_profit_pct` if set, otherwise falls back to existing engine defaults (e.g., 4%)
- All sizes are clamped by `RiskParameters.max_position_size` and remaining `max_daily_risk`
- **Dynamic risk adjustments are applied automatically when enabled**

## Per-Strategy Overrides
Strategies can override risk behavior by implementing `get_risk_overrides()`:

```python
class MyStrategy(BaseStrategy):
    def get_risk_overrides(self):
        return {
            'position_sizer': 'confidence_weighted',  # 'fixed_fraction' | 'confidence_weighted' | 'atr_risk'
            'base_fraction': 0.02,                    # 2% base allocation
            'min_fraction': 0.005,                    # 0.5% min allocation
            'max_fraction': 0.10,                     # 10% max per position
            'confidence_key': 'prediction_confidence',
            'stop_loss_pct': 0.02,                    # 2% SL; omit to use ATR-based SL
            'take_profit_pct': 0.04,                  # 4% TP; omit to use defaults
            # Dynamic risk overrides (optional)
            'dynamic_risk': {
                'enabled': True,
                'drawdown_thresholds': [0.03, 0.08, 0.15],  # Custom thresholds
                'risk_reduction_factors': [0.9, 0.7, 0.5],   # Custom reduction factors
                'recovery_thresholds': [0.01, 0.03],          # Custom recovery levels
                'volatility_adjustment_enabled': False,       # Disable volatility adjustments
            }
        }
```

### Dynamic Risk Strategy Overrides
Strategies can customize dynamic risk behavior through the `get_risk_overrides()` method:

- **`enabled`**: Enable/disable dynamic risk for this strategy
- **`drawdown_thresholds`**: Custom drawdown levels (e.g., `[0.03, 0.08, 0.15]`)
- **`risk_reduction_factors`**: Custom reduction factors (e.g., `[0.9, 0.7, 0.5]`)
- **`recovery_thresholds`**: Performance levels that allow risk increase
- **`volatility_adjustment_enabled`**: Enable/disable volatility-based adjustments
- **`performance_window_days`**: Rolling window for performance calculations

- If `position_sizer` is `'confidence_weighted'`, the selected `confidence_key` is read from the indicators/columns for the current index; allocation scales with confidence in [0, 1].
- If `'atr_risk'` is chosen, size is derived from legacy ATR risk sizing and converted to a balance fraction.

## Using The Layer
- Live engine (`LiveTradingEngine`) and backtester (`Backtester`) both:
  - Ask strategy for overrides via `get_risk_overrides()`
  - Call `RiskManager.calculate_position_fraction(...)` to get the position size fraction
  - Call `RiskManager.compute_sl_tp(...)` to compute SL/TP
  - Fallback to prior TP defaults if TP not provided by overrides/params

## Multiple Strategies with Different Risk Profiles
Run different strategies simultaneously, each with its own overrides. The engines pass each strategy’s overrides to the shared risk manager, which enforces global limits (`max_position_size`, `max_daily_risk`) while respecting per-strategy preferences.

Example:
- Strategy A (proven): `'fixed_fraction'` at 2%, ATR-based SL, TP 3%
- Strategy B (experimental): `'confidence_weighted'` with base 0.5%, min 0.1%, max 1%, wider SL/TP

## Backward Compatibility
- Strategies that do not implement `get_risk_overrides()` continue to work with defaults.
- Legacy methods `calculate_position_size(...)` and `calculate_stop_loss(...)` remain available and are used by the new layer for `'atr_risk'` sizing or ATR-based SL.
- Existing tests for drawdown and position limits continue to rely on `RiskParameters` and `RiskManager`.

## API Summary
- Risk sizing
  - `RiskManager.calculate_position_fraction(df, index, balance, price=None, indicators=None, strategy_overrides=None, regime='normal') -> float`
- Stops and targets
  - `RiskManager.compute_sl_tp(df, index, entry_price, side='long', strategy_overrides=None) -> Tuple[Optional[float], Optional[float]]`
- Strategy overrides (optional)
  - `BaseStrategy.get_risk_overrides() -> Optional[dict]`

## Notes
- `daily_risk_used` approximates risk as the sum of opened fraction sizes; adjust as needed for your brokerage/exchange semantics.
- For correlated exposure controls, extend `get_position_correlation_risk` to use actual correlation matrices across symbols/timeframes.