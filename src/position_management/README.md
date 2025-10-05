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

# Calculate risk adjustments based on current performance
adjustments = risk_mgr.calculate_dynamic_risk_adjustments(
    current_balance=10000.0,
    peak_balance=12000.0,
    session_id=None,  # Optional: for database queries
    previous_peak_balance=11000.0  # Optional: for recovery calculation
)

# Access adjustment factors
print(f"Position size factor: {adjustments.position_size_factor}")
print(f"Stop loss tightening: {adjustments.stop_loss_tightening}")
print(f"Primary reason: {adjustments.primary_reason}")
```

### Correlation Control
```python
import pandas as pd
from src.position_management.correlation_engine import CorrelationEngine, CorrelationConfig

# Configure correlation engine
config = CorrelationConfig(
    correlation_threshold=0.7,  # High correlation threshold
    max_correlated_exposure=0.15  # Max 15% exposure to correlated assets
)
corr_engine = CorrelationEngine(config)

# Calculate correlations from price series
price_series = {
    'BTCUSDT': pd.Series([50000, 51000, 52000], index=pd.date_range('2024-01-01', periods=3)),
    'ETHUSDT': pd.Series([3000, 3100, 3200], index=pd.date_range('2024-01-01', periods=3))
}

corr_matrix = corr_engine.calculate_position_correlations(price_series)

# Get correlated groups
groups = corr_engine.get_correlation_groups(corr_matrix)
print(f"Correlated groups: {groups}")

# Calculate size reduction factor for new position
active_positions = {'BTCUSDT': 0.10}  # 10% of balance in BTC
reduction_factor = corr_engine.compute_size_reduction_factor(
    candidate_symbol='ETHUSDT',
    candidate_size_fraction=0.05,
    existing_positions=active_positions,
    corr_matrix=corr_matrix
)
print(f"Reduction factor: {reduction_factor}")
adjusted_size = 0.05 * reduction_factor
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