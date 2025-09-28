# Regime Adaptive Strategy Migration Guide

## Overview

This document outlines the migration from the current `regime_adaptive` strategy to a cleaner, more maintainable implementation using the **Strategy Factory Pattern**.

## Problems with Current Implementation

1. **Violates Single Responsibility Principle**: The current `regime_adaptive` strategy is both a strategy selector AND executor
2. **Code Duplication**: Similar logic exists in both `regime_adaptive.py` and `regime_strategy_switcher.py`
3. **Tight Coupling**: Hard-coded strategy mappings and complex state management
4. **Testing Complexity**: Difficult to test regime detection separately from strategy execution
5. **Performance Overhead**: Calculates indicators for ALL strategies on every call

## New Architecture

### 1. Strategy Factory Pattern
- **`RegimeStrategyFactory`**: Handles regime detection and strategy selection
- **`RegimeAwareStrategy`**: Delegates to appropriate underlying strategy
- **`RegimeAdaptiveV2`**: Clean wrapper that maintains API compatibility

### 2. Separation of Concerns
- **Regime Detection**: Isolated in `RegimeDetector`
- **Strategy Selection**: Handled by `RegimeStrategyFactory`
- **Strategy Execution**: Delegated to underlying strategies
- **Configuration**: Centralized in `RegimeStrategyConfig`

### 3. Benefits
- **Single Responsibility**: Each class has one clear purpose
- **No Code Duplication**: Single implementation for regime-based selection
- **Better Testability**: Components can be tested independently
- **Performance**: Only calculates indicators for active strategy
- **Maintainability**: Easier to modify and extend

## Migration Steps

### Step 1: Update CLI Commands

Update `cli/commands/backtest.py` to support the new implementation:

```python
# Add to _load_strategy function
if strategy_name == "regime_adaptive_v2":
    from src.strategies.regime_adaptive_v2 import RegimeAdaptiveV2
    return RegimeAdaptiveV2()
```

### Step 2: Update Live Trading

The new implementation is compatible with the existing `StrategyManager` and `RegimeStrategySwitcher` in live trading.

### Step 3: Configuration

The new implementation provides better configuration options:

```python
# Create strategy with custom configuration
strategy = RegimeAdaptiveV2()

# Configure strategy mapping
strategy.configure_strategy_mapping(
    bull_low_vol="momentum_leverage",
    bull_high_vol="ensemble_weighted",
    bear_low_vol="bear",
    bear_high_vol="bear",
    range_low_vol="ml_basic",
    range_high_vol="ml_basic"
)

# Configure position sizing
strategy.configure_position_sizing(
    bull_low_vol_multiplier=1.0,
    bull_high_vol_multiplier=0.7,
    bear_low_vol_multiplier=0.6,
    bear_high_vol_multiplier=0.4,
    range_low_vol_multiplier=0.6,
    range_high_vol_multiplier=0.3
)

# Configure switching parameters
strategy.configure_switching_parameters(
    min_confidence=0.4,
    min_regime_duration=12,
    switch_cooldown=20
)
```

### Step 4: Testing

The new implementation maintains the same API as the original, so existing tests should work with minimal changes.

## Usage Examples

### Basic Usage
```python
from src.strategies.regime_adaptive_v2 import RegimeAdaptiveV2

# Create strategy (uses default configuration)
strategy = RegimeAdaptiveV2()

# Use in backtesting or live trading
# ... existing code works unchanged
```

### Advanced Configuration
```python
from src.strategies.regime_adaptive_v2 import RegimeAdaptiveV2

# Create strategy
strategy = RegimeAdaptiveV2()

# Customize configuration
strategy.configure_strategy_mapping(
    bull_low_vol="momentum_leverage",
    bull_high_vol="ensemble_weighted"
)

strategy.configure_position_sizing(
    bull_low_vol_multiplier=1.2,  # More aggressive in bull markets
    bear_low_vol_multiplier=0.3   # More conservative in bear markets
)

strategy.configure_switching_parameters(
    min_confidence=0.5,           # Higher confidence threshold
    min_regime_duration=20,       # Longer regime stability requirement
    switch_cooldown=30            # Longer cooldown between switches
)
```

### Custom Strategies
```python
from src.strategies.regime_adaptive_v2 import RegimeAdaptiveV2
from src.strategies.my_custom_strategy import MyCustomStrategy

# Create strategy
strategy = RegimeAdaptiveV2()

# Register custom strategy
strategy.register_custom_strategy("my_custom", MyCustomStrategy)

# Use in configuration
strategy.configure_strategy_mapping(
    bull_low_vol="my_custom",
    range_low_vol="my_custom"
)
```

## Performance Improvements

1. **Reduced Indicator Calculation**: Only calculates indicators for the active strategy
2. **Better Memory Usage**: Doesn't maintain indicators for all strategies simultaneously
3. **Faster Switching**: Cleaner strategy loading and switching logic
4. **Reduced Complexity**: Simpler state management

## Backward Compatibility

The new implementation maintains full backward compatibility with the existing API:

- All methods have the same signatures
- Same return types and formats
- Same configuration options (with additional flexibility)
- Same logging and monitoring capabilities

## Migration Timeline

1. **Phase 1**: Deploy new implementation alongside existing one
2. **Phase 2**: Update CLI to support both versions
3. **Phase 3**: Migrate existing usage to new implementation
4. **Phase 4**: Deprecate old implementation
5. **Phase 5**: Remove old implementation

## Testing Strategy

1. **Unit Tests**: Test each component independently
2. **Integration Tests**: Test the full regime-aware strategy
3. **Backtesting**: Compare performance with original implementation
4. **Live Trading**: Gradual rollout with monitoring

## Rollback Plan

If issues arise, the original `regime_adaptive` strategy remains available and can be used as a fallback by simply changing the strategy name in configuration.