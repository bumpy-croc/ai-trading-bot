# Performance Monitoring and Automatic Strategy Switching System

This document describes the comprehensive performance monitoring and automatic strategy switching system implemented for the strategy system redesign.

## Overview

The performance monitoring system provides sophisticated performance degradation detection, multi-criteria strategy selection, automatic strategy switching with safety controls, and emergency management capabilities.

## Components

### 1. PerformanceMonitor (`performance_monitor.py`)

Implements sophisticated performance monitoring with degradation detection:

- **Multi-timeframe Analysis**: Analyzes performance across short-term (7 days), medium-term (30 days), and long-term (90 days) periods
- **Statistical Significance Testing**: Uses statistical tests to determine if performance degradation is significant
- **Regime-aware Evaluation**: Considers market regime context when evaluating performance
- **Confidence Interval Analysis**: Calculates confidence intervals for performance metrics

**Key Features:**
- Configurable degradation thresholds
- Regime-specific performance baselines
- Statistical significance testing with 95% confidence
- Multi-criteria degradation detection

### 2. StrategySelector (`strategy_selector.py`)

Implements multi-criteria strategy selection algorithm:

- **Multi-criteria Scoring**: Evaluates strategies based on Sharpe ratio, returns, drawdown, win rate, volatility, and regime performance
- **Risk-adjusted Selection**: Applies risk adjustments and penalties for high-risk strategies
- **Correlation Analysis**: Avoids selecting highly correlated strategies
- **Regime-specific Weighting**: Adjusts scoring based on current market regime

**Key Features:**
- Configurable criteria weights
- Correlation penalty system
- Regime-specific performance weighting
- Risk-adjusted scoring

### 3. StrategySwitcher (`strategy_switcher.py`)

Implements safe automatic strategy switching:

- **Validation Gates**: Multiple validation checks before executing switches
- **Cooling-off Periods**: Prevents excessive switching with configurable intervals
- **Audit Trail**: Comprehensive logging of all switch decisions and executions
- **Performance Impact Analysis**: Tracks performance before and after switches

**Key Features:**
- Configurable switch limits (daily/weekly)
- Pre/post switch callbacks
- Performance impact tracking
- Manual override capabilities

### 4. EmergencyControls (`emergency_controls.py`)

Implements emergency controls and manual override system:

- **Emergency Detection**: Monitors for critical conditions (high drawdown, consecutive losses)
- **Conservative Mode**: Automatically reduces position sizes and risk during emergencies
- **Approval Workflows**: Requires approval for high-risk operations
- **Alert System**: Real-time alerting with cooldown and rate limiting

**Key Features:**
- Emergency level escalation (None → Low → Medium → High → Critical)
- Conservative mode with configurable risk reduction
- Approval workflow with expiration
- Alert system with callbacks

### 5. PerformanceMonitoringSystem (`performance_monitoring_system.py`)

Integrates all components into a unified system:

- **Orchestration**: Coordinates all monitoring components
- **Unified Interface**: Single interface for all monitoring operations
- **Callback System**: Extensible callback system for external integration
- **Comprehensive Status**: Provides complete system status and metrics

## Usage Example

```python
from src.strategies.components.performance_monitoring_system import PerformanceMonitoringSystem
from src.strategies.components.performance_tracker import PerformanceTracker

# Initialize the monitoring system
monitoring_system = PerformanceMonitoringSystem()

# Register strategies
strategy_tracker = PerformanceTracker("my_strategy")
monitoring_system.register_strategy("my_strategy", strategy_tracker)

# Set current strategy
monitoring_system.set_current_strategy("my_strategy")

# Set strategy activation callback
def activate_strategy(strategy_id: str) -> bool:
    # Your strategy activation logic here
    print(f"Activating strategy: {strategy_id}")
    return True

monitoring_system.set_strategy_activation_callback(activate_strategy)

# Update monitoring (call this regularly, e.g., every 5 minutes)
results = monitoring_system.update_monitoring(market_data, current_regime)

# Check results
if results['actions_taken']:
    print(f"Actions taken: {results['actions_taken']}")

if results['active_alerts']:
    print(f"Active alerts: {len(results['active_alerts'])}")

# Manual operations
request_id = monitoring_system.request_manual_switch(
    "better_strategy", "Manual optimization", "trader_1"
)

# Emergency controls
monitoring_system.activate_emergency_stop("System malfunction", "admin")
```

## Configuration

Each component has extensive configuration options:

### PerformanceDegradationConfig
- `min_trades_for_evaluation`: Minimum trades required for evaluation (default: 50)
- `min_days_for_evaluation`: Minimum days required for evaluation (default: 30)
- `max_drawdown_threshold`: Maximum acceptable drawdown (default: 20%)
- `sharpe_ratio_threshold`: Minimum acceptable Sharpe ratio (default: 0.5)
- `confidence_level`: Statistical confidence level (default: 95%)

### SelectionConfig
- `sharpe_weight`: Weight for Sharpe ratio in scoring (default: 0.25)
- `return_weight`: Weight for returns in scoring (default: 0.20)
- `correlation_threshold`: High correlation threshold (default: 0.7)
- `min_trades_for_consideration`: Minimum trades for strategy consideration (default: 30)

### SwitchConfig
- `min_switch_interval_hours`: Minimum time between switches (default: 24 hours)
- `max_switches_per_day`: Maximum switches per day (default: 3)
- `min_confidence_for_switch`: Minimum confidence required for switch (default: 0.7)

### EmergencyConfig
- `critical_drawdown_threshold`: Critical drawdown level (default: 25%)
- `consecutive_loss_threshold`: Consecutive losses trigger (default: 5)
- `conservative_position_size_multiplier`: Position size reduction in conservative mode (default: 0.5)

## Safety Features

The system includes multiple safety features:

1. **Cooling-off Periods**: Prevents rapid switching between strategies
2. **Statistical Validation**: Requires statistical significance for performance degradation
3. **Multi-criteria Evaluation**: Uses multiple metrics to avoid false positives
4. **Manual Override**: Allows manual control to override automatic decisions
5. **Emergency Stops**: Immediate halt of all trading in critical situations
6. **Approval Workflows**: Requires human approval for high-risk operations
7. **Conservative Mode**: Automatically reduces risk during emergencies
8. **Audit Trail**: Complete logging of all decisions and actions

## Testing

Comprehensive test suites are provided:

- `test_performance_monitor.py`: Tests performance monitoring and degradation detection
- `test_strategy_selector.py`: Tests multi-criteria strategy selection
- `test_strategy_switcher.py`: Tests automatic switching with safety controls
- `test_emergency_controls.py`: Tests emergency detection and controls
- `test_integration.py`: Integration tests for the complete system

Run tests with:
```bash
python -m pytest src/strategies/components/testing/ -v
```

## Integration with Existing System

The monitoring system is designed to integrate with the existing strategy architecture:

1. **Performance Tracking**: Uses existing `PerformanceTracker` for metrics
2. **Regime Detection**: Integrates with existing `RegimeContext` system
3. **Strategy Management**: Works with existing `StrategyManager` components
4. **Callback System**: Provides hooks for integration with trading engines

## Monitoring and Alerting

The system provides comprehensive monitoring:

- **Real-time Alerts**: Immediate notification of critical conditions
- **Performance Metrics**: Detailed performance tracking and analysis
- **System Status**: Complete system health and status information
- **Historical Data**: Audit trails and historical performance data

## Future Enhancements

Potential future enhancements include:

1. **Machine Learning**: ML-based performance prediction and strategy selection
2. **Advanced Correlation**: More sophisticated correlation analysis
3. **Market Microstructure**: Integration with market microstructure data
4. **Risk Models**: Advanced risk modeling and stress testing
5. **Backtesting Integration**: Automated backtesting of strategy switches

## Requirements Satisfied

This implementation satisfies all requirements from the specification:

- **7.1**: Multi-timeframe performance analysis with statistical significance testing ✓
- **7.2**: Regime-aware performance evaluation with confidence intervals ✓
- **7.3**: Multi-criteria strategy scoring with risk adjustment and correlation analysis ✓
- **7.4**: Manual override, emergency controls, and approval workflows ✓

The system provides a robust, safe, and comprehensive solution for automatic strategy switching with extensive safety controls and monitoring capabilities.