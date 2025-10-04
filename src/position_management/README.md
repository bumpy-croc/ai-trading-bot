# Position Management

Advanced position sizing, portfolio management, and risk control functionality.

## Overview

This module provides comprehensive position management features including dynamic risk adjustment, correlation control, partial exits, scale-ins, trailing stops, and time-based exit policies.

## Modules

- `correlation_engine.py` - Manages exposure to correlated assets
- `dynamic_risk.py` - Adaptive risk management based on performance and market conditions
- `mfe_mae_analyzer.py` - Maximum Favorable/Adverse Excursion analysis
- `mfe_mae_tracker.py` - Tracks MFE/MAE metrics per trade
- `partial_manager.py` - Handles partial exits and scale-in operations
- `time_exits.py` - Time-based exit policies (max holding, end-of-day, weekend)
- `trailing_stops.py` - Trailing stop loss and breakeven management

## Usage

### Dynamic Risk Management
```python
from src.position_management.dynamic_risk import DynamicRiskManager, DynamicRiskConfig

config = DynamicRiskConfig(
    enabled=True,
    drawdown_thresholds=[0.05, 0.10, 0.15],
    risk_reduction_factors=[0.8, 0.6, 0.4]
)
risk_mgr = DynamicRiskManager(config, database_manager)
adjustment = risk_mgr.calculate_risk_adjustment(current_balance, peak_balance)
```

### Correlation Control
```python
from src.position_management.correlation_engine import CorrelationEngine

corr_engine = CorrelationEngine()
is_allowed, adjusted_size = corr_engine.check_entry_allowed(
    candidate_symbol='ETHUSDT',
    candidate_size=0.05,
    active_positions={'BTCUSDT': 0.10}
)
```

### Partial Exits and Scale-Ins
```python
from src.position_management.partial_manager import PartialExitPolicy

policy = PartialExitPolicy(
    exit_targets=[0.03, 0.06, 0.10],
    exit_sizes=[0.25, 0.25, 0.50],
    scale_in_thresholds=[0.02, 0.05],
    scale_in_sizes=[0.25, 0.25]
)
exits = policy.check_partial_exits(position, current_price)
scale_in = policy.check_scale_in_opportunity(position, current_price, market_data)
```

### Time-Based Exits
```python
from datetime import time
from src.position_management.time_exits import TimeExitPolicy, MarketSessionDef, TimeRestrictions

session = MarketSessionDef(
    name='US_EQUITIES',
    timezone='America/New_York',
    open_time=time(9, 30),
    close_time=time(16, 0),
    days_of_week=[1,2,3,4,5]
)

policy = TimeExitPolicy(
    max_holding_hours=24,
    end_of_day_flat=True,
    weekend_flat=True,
    market_session=session
)
```

### Trailing Stops
```python
from src.position_management.trailing_stops import TrailingStopPolicy

policy = TrailingStopPolicy(
    activation_threshold=0.015,  # Start trailing at 1.5% profit
    trailing_distance_pct=0.02,  # Trail 2% behind
    breakeven_threshold=0.03,    # Move to breakeven at 3% profit
    breakeven_buffer=0.001       # 0.1% buffer at breakeven
)
```

## Documentation

See [docs/RISK_AND_POSITION_MANAGEMENT.md](../../docs/RISK_AND_POSITION_MANAGEMENT.md) for comprehensive documentation on all position management features.