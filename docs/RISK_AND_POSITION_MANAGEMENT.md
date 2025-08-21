# Risk and Position Management Documentation

The AI Trading Bot features a comprehensive risk management system with both static and dynamic risk controls to protect capital and optimize performance under varying market conditions.

## Overview

The risk management system consists of two main components:

1. **Static Risk Management** - Traditional fixed-parameter risk controls
2. **Dynamic Risk Management** - Adaptive risk controls that adjust based on performance and market conditions

## Static Risk Management

### RiskParameters

The base risk management is controlled by the `RiskParameters` class:

```python
@dataclass
class RiskParameters:
    base_risk_per_trade: float = 0.02      # 2% risk per trade
    max_risk_per_trade: float = 0.03       # 3% maximum risk per trade
    max_position_size: float = 0.25        # 25% maximum position size
    max_daily_risk: float = 0.06           # 6% maximum daily risk
    max_correlated_risk: float = 0.10      # 10% maximum risk for correlated positions
    max_drawdown: float = 0.20             # 20% maximum drawdown threshold
    position_size_atr_multiplier: float = 1.0
    default_take_profit_pct: Optional[float] = None
    atr_period: int = 14
```

### Position Sizing Methods

The system supports multiple position sizing approaches:

1. **fixed_fraction** - Fixed percentage of account balance
2. **confidence_weighted** - Size based on ML model confidence scores
3. **atr_risk** - ATR-based risk management (legacy approach)

```python
# Example: Fixed fraction sizing
strategy_overrides = {
    'position_sizer': 'fixed_fraction',
    'base_fraction': 0.02,  # 2% of balance per trade
    'max_fraction': 0.05    # 5% maximum
}

# Example: Confidence-weighted sizing
strategy_overrides = {
    'position_sizer': 'confidence_weighted',
    'base_fraction': 0.03,
    'confidence_key': 'prediction_confidence'  # From indicators
}
```

## Dynamic Risk Management

### Overview

Dynamic risk management automatically adjusts position sizes, stop-loss levels, and risk limits based on:

- **Portfolio drawdown levels** (5%, 10%, 15% thresholds)
- **Recent trading performance** (win rate, profit factor)
- **Market volatility conditions** (high/low volatility adjustments)
- **Recovery patterns** (gradual de-throttling)

### Configuration

Dynamic risk is configured via `DynamicRiskConfig`:

```python
@dataclass
class DynamicRiskConfig:
    enabled: bool = True
    performance_window_days: int = 30
    
    # Drawdown thresholds and adjustments
    drawdown_thresholds: List[float] = [0.05, 0.10, 0.15]  # 5%, 10%, 15%
    risk_reduction_factors: List[float] = [0.8, 0.6, 0.4]   # Reduction at each threshold
    
    # Recovery thresholds for de-throttling
    recovery_thresholds: List[float] = [0.02, 0.05]  # 2%, 5% positive returns
    
    # Volatility adjustments
    volatility_adjustment_enabled: bool = True
    volatility_window_days: int = 30
    high_volatility_threshold: float = 0.03  # 3% daily volatility
    low_volatility_threshold: float = 0.01   # 1% daily volatility
    volatility_risk_multipliers: Tuple[float, float] = (0.7, 1.3)  # (high_vol, low_vol)
```

### Adjustment Logic

**Combination Logic:**
Multiple adjustment factors are combined conservatively (most restrictive wins):
```python
final_position_factor = min(
    drawdown_adjustment.position_size_factor,
    performance_adjustment.position_size_factor,
    volatility_adjustment.position_size_factor
)
```

**Example Scenarios:**

1. **15% Drawdown**: Position sizes reduced to 40% of normal (0.4x factor)
2. **High Volatility**: Position sizes reduced to 70% during volatile periods
3. **Poor Performance** (< 30% win rate): Position sizes reduced to 60%
4. **Recovery** (5% positive return): Gradual return to normal sizing

### Live Trading Integration

```python
# Enable dynamic risk in live trading (default: enabled)
engine = LiveTradingEngine(
    strategy=strategy,
    enable_dynamic_risk=True,
    dynamic_risk_config=DynamicRiskConfig(
        drawdown_thresholds=[0.05, 0.10, 0.15],
        risk_reduction_factors=[0.8, 0.6, 0.4]
    )
)
```

### Backtesting Integration

```python
# Enable dynamic risk in backtesting (default: disabled for historical parity)
backtester = Backtester(
    strategy=strategy,
    enable_dynamic_risk=True,
    dynamic_risk_config=config
)

# Results include dynamic risk tracking
results = backtester.run(symbol="BTCUSDT", timeframe="1h", start=start_date)
print(f"Dynamic adjustments: {len(results['dynamic_risk_adjustments'])}")
print(f"Most common reason: {results['dynamic_risk_summary']['most_common_reason']}")
```

### Per-Strategy Overrides

Strategies can override dynamic risk settings via `get_risk_overrides()`:

```python
class MyStrategy(BaseStrategy):
    def get_risk_overrides(self):
        return {
            'dynamic_risk': {
                'enabled': True,
                'drawdown_thresholds': [0.03, 0.08, 0.15],  # More aggressive thresholds
                'risk_reduction_factors': [0.9, 0.7, 0.5],
                'recovery_thresholds': [0.02, 0.05],
                'volatility_adjustment_enabled': True
            }
        }
```

## Database Tables

### dynamic_performance_metrics

Tracks rolling performance metrics for adaptive risk decisions:

| Column | Type | Description |
|--------|------|-------------|
| id | Integer | Primary key |
| timestamp | DateTime | Metric calculation time |
| rolling_win_rate | Numeric(18,8) | Recent win rate (0.0-1.0) |
| rolling_sharpe_ratio | Numeric(18,8) | Recent Sharpe ratio |
| current_drawdown | Numeric(18,8) | Current drawdown percentage |
| volatility_30d | Numeric(18,8) | 30-day rolling volatility |
| consecutive_losses/wins | Integer | Current streak counts |
| risk_adjustment_factor | Numeric(18,8) | Applied adjustment factor |
| profit_factor | Numeric(18,8) | Gross profit / gross loss |
| expectancy | Numeric(18,8) | Expected value per trade |
| session_id | Integer | Foreign key to trading_sessions |

### risk_adjustments

Logs all risk parameter changes for analysis and tracking:

| Column | Type | Description |
|--------|------|-------------|
| id | Integer | Primary key |
| timestamp | DateTime | Adjustment time |
| adjustment_type | String(50) | 'drawdown', 'performance', 'volatility' |
| trigger_reason | String(200) | Detailed reason (e.g., 'drawdown_15.0%') |
| parameter_name | String(100) | 'position_size_factor', 'stop_loss_multiplier' |
| original_value | Numeric(18,8) | Original parameter value |
| adjusted_value | Numeric(18,8) | New adjusted value |
| adjustment_factor | Numeric(18,8) | Adjustment factor applied |
| current_drawdown | Numeric(18,8) | Drawdown at time of adjustment |
| performance_score | Numeric(18,8) | Performance score context |
| volatility_level | Numeric(18,8) | Volatility level context |
| session_id | Integer | Foreign key to trading_sessions |

## Database Migration

To create the dynamic risk tables, run the migration:

```bash
# Apply the migration
alembic upgrade head

# Check migration status
alembic current
alembic history
```

The migration file `0003_dynamic_risk_tables.py` creates both tables with proper indexes:
- `idx_dynamic_perf_timestamp`, `idx_dynamic_perf_session`
- `idx_risk_adj_timestamp`, `idx_risk_adj_type`, `idx_risk_adj_session`

## Monitoring and Alerts

### Dashboard Integration

The monitoring dashboard displays real-time dynamic risk status:

- **Dynamic Risk Factor**: Current position size multiplier (e.g., "0.60x")
- **Risk Reason**: Why adjustment is active (e.g., "drawdown_15.0%")
- **Risk Status Indicator**: Visual status (green=normal, yellow=active, red=critical)

### Alerts

The system automatically generates alerts when:
- Risk factor changes by more than 10%
- Drawdown thresholds are crossed
- Performance deteriorates significantly

Alerts are displayed in the web dashboard and logged to the database.

## Default Behavior

- **Live Trading**: Dynamic risk is **enabled by default** to protect capital
- **Backtesting**: Dynamic risk is **disabled by default** to preserve historical accuracy
- Position sizing uses `'fixed_fraction'` policy by default with `base_risk_per_trade`
- SL is ATR-based if no explicit `stop_loss_pct` override is provided
- TP uses `RiskParameters.default_take_profit_pct` if set, otherwise falls back to existing engine defaults
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
        }
```

- If `position_sizer` is `'confidence_weighted'`, the selected `confidence_key` is read from the indicators/columns for the current index; allocation scales with confidence in [0, 1].
- If `'atr_risk'` is chosen, size is derived from legacy ATR risk sizing and converted to a balance fraction.

## Using The Layer
- Live engine (`LiveTradingEngine`) and backtester (`Backtester`) both:
  - Ask strategy for overrides via `get_risk_overrides()`
  - Call `RiskManager.calculate_position_fraction(...)` to get the position size fraction
  - Call `RiskManager.compute_sl_tp(...)` to compute SL/TP
  - Fallback to prior TP defaults if TP not provided by overrides/params

## Multiple Strategies with Different Risk Profiles
Run different strategies simultaneously, each with its own overrides. The engines pass each strategy's overrides to the shared risk manager, which enforces global limits (`max_position_size`, `max_daily_risk`) while respecting per-strategy preferences.

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
- Correlation control: The system computes rolling correlations across symbols and enforces portfolio-level exposure caps among highly correlated groups. See below.

## Time-Based Exit Policies

Time-based exits provide guardrails to control holding time, overnight/weekend exposure, and align trading with market sessions.

- Core types
  - Maximum holding period: Force-close after N hours.
  - End-of-day flat: Exit positions at market close.
  - Weekend flat / No weekend: Avoid holding over weekends.
  - Trading-hours-only: Exit outside defined session hours.

- Key components
  - `src/position_management/time_exits.py`
    - `TimeExitPolicy`: Implements checks and next-exit scheduling (timezone-aware via zoneinfo).
    - `MarketSessionDef`: In-memory session definition for engines/backtests.
    - `TimeRestrictions`: Flags for overnight/weekend/hours-only.
  - DB schema (migration `0004_time_exits`)
    - `positions`: `max_holding_until`, `end_of_day_exit`, `weekend_exit`, `time_restriction_group`.
    - `trading_sessions`: `time_exit_config`, `market_timezone`.
    - `market_sessions`: session catalog with timezone, open/close times, days of week.

- Configuration defaults (`src/config/constants.py`)
  - `DEFAULT_MAX_HOLDING_HOURS = 24`
  - `DEFAULT_END_OF_DAY_FLAT = False`
  - `DEFAULT_WEEKEND_FLAT = False`
  - `DEFAULT_MARKET_TIMEZONE = 'UTC'`
  - `DEFAULT_TIME_RESTRICTIONS` with `no_overnight`, `no_weekend`, `trading_hours_only`

### Engine Integration

- Live engine (`src/live/trading_engine.py`): accepts optional `time_exit_policy` and replaces hardcoded 24h with policy-based checks.
- Backtester (`src/backtesting/engine.py`): accepts optional `time_exit_policy` and applies time exits in close checks.

### Example Usage

```python
from datetime import time
from src.position_management.time_exits import TimeExitPolicy, MarketSessionDef, TimeRestrictions

us_equities = MarketSessionDef(
    name='US_EQUITIES',
    timezone='America/New_York',
    open_time=time(9, 30),
    close_time=time(16, 0),
    days_of_week=[1,2,3,4,5],
)

policy = TimeExitPolicy(
    max_holding_hours=48,
    end_of_day_flat=True,
    weekend_flat=True,
    market_timezone='America/New_York',
    time_restrictions=TimeRestrictions(no_overnight=True, trading_hours_only=True),
    market_session=us_equities,
)
```

Pass `time_exit_policy=policy` to both `LiveTradingEngine` and `Backtester`.

### Notes and Edge Cases

- All time comparisons are UTC-normalized; `zoneinfo` is used for local market time computations when available.
- DST transitions: end-of-day logic uses local market session close time each day; tests should include DST boundaries.
- Holiday-aware sessions are planned for future work; current implementation supports weekdays and 24h markets.

## Correlation Control

The correlation layer prevents over-exposure to highly correlated assets.

- Engine: `position_management/correlation_engine.py`
- Config defaults (in `config/constants.py`):
  - `DEFAULT_CORRELATION_WINDOW_DAYS = 30`
  - `DEFAULT_CORRELATION_THRESHOLD = 0.7`
  - `DEFAULT_MAX_CORRELATED_EXPOSURE = 0.15`
  - `DEFAULT_CORRELATION_UPDATE_FREQUENCY_HOURS = 1`
  - `DEFAULT_CORRELATION_SAMPLE_MIN_SIZE = 20`
- Risk parameters (in `RiskParameters`):
  - `correlation_window_days`, `correlation_threshold`, `max_correlated_exposure`, `correlation_update_frequency_hours`

How it works:
- Live and Backtest engines build a per-symbol price series window for open symbols plus the candidate entry symbol.
- `CorrelationEngine` computes a returns-based correlation matrix and clusters symbols whose pairwise correlation ≥ threshold.
- The candidate position fraction is reduced proportionally if projected correlated exposure exceeds `max_correlated_exposure`.
- Database tables `correlation_matrix` and `portfolio_exposures` can store correlation snapshots and group exposures.

Strategy overrides can set per-trade limits via:

```python
def get_risk_overrides(self):
    return {
        'position_sizer': 'fixed_fraction',
        'base_fraction': 0.03,
        'correlation_control': {
            'max_correlated_exposure': 0.15,
        }
    }
```

Monitoring:
- REST endpoints expose recent correlation data:
  - `/api/correlation/matrix`
  - `/api/correlation/exposures`

## Performance Considerations

- **Caching**: Performance metrics are cached for 5 minutes to minimize database overhead
- **Overhead**: Dynamic risk calculations add < 1ms per risk check
- **Database Impact**: Metrics are logged only on significant changes (>10% adjustment)

## Troubleshooting

### Common Issues

**1. Dynamic risk not applying adjustments**
- Check `enable_dynamic_risk=True` in engine initialization
- Verify database tables exist: `alembic current`
- Check logs for dynamic risk manager initialization errors

**2. Migration failures**
- Ensure database URL is properly configured
- Check if tables already exist manually
- Verify PostgreSQL/SQLite compatibility

**3. Performance degradation**
- Monitor database query performance on risk_adjustments table
- Check if excessive logging is occurring (should be < 10 logs per hour typically)
- Verify cache TTL settings are appropriate

### Debugging

Enable detailed logging for dynamic risk operations:

```python
import logging
logging.getLogger('src.position_management.dynamic_risk').setLevel(logging.DEBUG)
```

Check recent risk adjustments:

```sql
SELECT * FROM risk_adjustments 
WHERE session_id = YOUR_SESSION_ID 
ORDER BY timestamp DESC 
LIMIT 10;
```

## Best Practices

1. **Strategy-Specific Tuning**: Use `get_risk_overrides()` to customize thresholds per strategy
2. **Gradual Rollout**: Start with conservative thresholds and adjust based on results
3. **Monitoring**: Regularly review risk adjustment logs and effectiveness
4. **Testing**: Always backtest with dynamic risk enabled to understand historical impact
5. **Recovery Planning**: Set appropriate recovery thresholds to avoid being overly conservative during recoveries

## Configuration Constants

Default values are defined in `src/config/constants.py`:

```python
DEFAULT_DYNAMIC_RISK_ENABLED = True
DEFAULT_PERFORMANCE_WINDOW_DAYS = 30
DEFAULT_DRAWDOWN_THRESHOLDS = [0.05, 0.10, 0.15]
DEFAULT_RISK_REDUCTION_FACTORS = [0.8, 0.6, 0.4]
DEFAULT_RECOVERY_THRESHOLDS = [0.02, 0.05]
DEFAULT_VOLATILITY_ADJUSTMENT_ENABLED = True
DEFAULT_HIGH_VOLATILITY_THRESHOLD = 0.03
DEFAULT_LOW_VOLATILITY_THRESHOLD = 0.01
DEFAULT_VOLATILITY_RISK_MULTIPLIERS = (0.7, 1.3)
DEFAULT_MIN_TRADES_FOR_DYNAMIC_ADJUSTMENT = 10
```
