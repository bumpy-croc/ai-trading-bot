# Strategy System Migration - Complete Cutover Design

## Overview

This design document outlines the approach for completing the migration from the legacy `BaseStrategy` system to the component-based `Strategy` system. The focus is on removing all legacy code, adapters, and migration utilities while ensuring the backtesting and live trading engines work seamlessly with the component-based architecture.

## Architecture

### Current State (Hybrid)

```
┌─────────────────────────────────────────────────────────────┐
│                    Backtesting/Live Engine                   │
│                 (Expects BaseStrategy)                       │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              LegacyStrategyAdapter (TO REMOVE)               │
│  - Implements BaseStrategy interface                         │
│  - Delegates to component-based Strategy internally          │
│  - Converts between interfaces                               │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  Component-Based Strategy + Components                       │
│  - SignalGenerator, RiskManager, PositionSizer               │
└─────────────────────────────────────────────────────────────┘
```

### Target State (Component-Based Only)

```
┌─────────────────────────────────────────────────────────────┐
│                    Backtesting/Live Engine                   │
│                 (Uses Strategy.process_candle)               │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  FOR EACH CANDLE (index i):                                 │
│    decision = strategy.process_candle(df, i, balance)       │
│                                                              │
│    Internally:                                               │
│    ┌──────────────────────────────────────────────────────┐ │
│    │ 1. regime = regime_detector.detect_regime(df, i)     │ │
│    │ 2. signal = signal_gen.generate_signal(df, i, regime)│ │
│    │ 3. risk_size = risk_mgr.calculate_size(signal, bal) │ │
│    │ 4. position = pos_sizer.calculate_size(signal, bal) │ │
│    └──────────────────────────────────────────────────────┘ │
│                                                              │
│    Returns: TradingDecision {                               │
│      signal, position_size, regime, risk_metrics, metadata  │
│    }                                                         │
└─────────────────────────────────────────────────────────────┘
```

## Components and Interfaces

### Strategy Interface (Component-Based)

The component-based `Strategy` class is already implemented with the following interface:

```python
class Strategy:
    def __init__(self, name: str, signal_generator: SignalGenerator,
                 risk_manager: RiskManager, position_sizer: PositionSizer,
                 regime_detector: Optional[EnhancedRegimeDetector] = None):
        """Initialize strategy with composed components"""
        
    def process_candle(self, df: pd.DataFrame, index: int, balance: float,
                      current_positions: Optional[list[Position]] = None) -> TradingDecision:
        """Process a single candle and make trading decision"""
        
    def should_exit_position(self, position: Position, current_data: MarketData,
                           regime: Optional[RegimeContext] = None) -> bool:
        """Determine if a position should be exited"""
        
    def get_stop_loss_price(self, entry_price: float, signal: Signal,
                          regime: Optional[RegimeContext] = None) -> float:
        """Get stop loss price for a position"""
```

### TradingDecision Data Model

```python
@dataclass
class TradingDecision:
    timestamp: datetime
    signal: Signal                    # BUY, SELL, or HOLD with confidence
    position_size: float              # Calculated position size
    regime: Optional[RegimeContext]   # Market regime context
    risk_metrics: dict[str, float]    # Risk-related metrics
    execution_time_ms: float          # Time taken for decision
    metadata: dict[str, Any]          # Additional decision metadata
```

## Migration Strategy

### Phase 1: Convert Concrete Strategies

Convert all concrete strategy implementations from `BaseStrategy` to component-based:

1. **ml_basic.py** → Component-based strategy using `MLBasicSignalGenerator`
2. **ml_adaptive.py** → Component-based strategy using `MLSignalGenerator` with regime adaptation
3. **ml_sentiment.py** → Component-based strategy using sentiment-aware signal generator
4. **ensemble_weighted.py** → Component-based strategy using `WeightedVotingSignalGenerator`
5. **momentum_leverage.py** → Component-based strategy using `MomentumSignalGenerator`

Each strategy will be converted to a factory function that returns a configured `Strategy` instance:

```python
def create_ml_basic_strategy(name: str = "ml_basic") -> Strategy:
    """Create ML Basic strategy using component composition"""
    signal_generator = MLBasicSignalGenerator(
        model_registry=get_model_registry(),
        confidence_threshold=0.6
    )
    risk_manager = FixedRiskManager(risk_per_trade=0.02)
    position_sizer = ConfidenceWeightedSizer(base_fraction=0.02)
    
    return Strategy(
        name=name,
        signal_generator=signal_generator,
        risk_manager=risk_manager,
        position_sizer=position_sizer
    )
```

### Phase 2: Update Backtesting Engine

Modify the backtesting engine to use `process_candle()` instead of legacy interface:

**Current Code (Legacy):**
```python
# Upfront indicator calculation
df = self.strategy.calculate_indicators(df)

# Main loop
for i in range(len(df)):
    if strategy.check_entry_conditions(df, i):
        position_size = strategy.calculate_position_size(df, i, balance)
        stop_loss = strategy.calculate_stop_loss(df, i, price)
```

**New Code (Component-Based):**
```python
# No upfront calculation needed

# Main loop
for i in range(len(df)):
    decision = strategy.process_candle(df, i, balance, current_positions)
    
    if decision.signal.direction == SignalDirection.BUY:
        position_size = decision.position_size
        stop_loss = strategy.get_stop_loss_price(entry_price, decision.signal, decision.regime)
        # Execute buy
    elif decision.signal.direction == SignalDirection.SELL:
        # Execute sell/exit
```

### Phase 3: Update Live Trading Engine

Modify the live trading engine similarly:

**Current Code (Legacy):**
```python
df = self.strategy.calculate_indicators(df)

if self.strategy.check_entry_conditions(df, current_index):
    size = self.strategy.calculate_position_size(df, current_index, balance)
```

**New Code (Component-Based):**
```python
decision = self.strategy.process_candle(df, current_index, balance, positions)

if decision.signal.direction != SignalDirection.HOLD:
    # Use decision.position_size, decision.signal, decision.metadata
    # Log decision.to_dict() to database
```

### Phase 4: Update Tests

1. **Strategy Unit Tests**: Test component-based strategies directly
2. **Component Tests**: Test individual components in isolation
3. **Integration Tests**: Test complete workflows with `TradingDecision` objects
4. **Delete Adapter Tests**: Remove all tests for `LegacyStrategyAdapter`

**Example Test Migration:**

```python
# OLD (Legacy)
def test_ml_basic_predictions():
    strategy = MlBasic()
    df = create_test_data()
    df = strategy.calculate_indicators(df)
    assert "ml_prediction" in df.columns

# NEW (Component-Based)
def test_ml_basic_predictions():
    strategy = create_ml_basic_strategy()
    df = create_test_data()
    
    decision = strategy.process_candle(df, index=150, balance=10000)
    assert decision.signal.direction in [SignalDirection.BUY, SignalDirection.SELL, SignalDirection.HOLD]
    assert 0 <= decision.signal.confidence <= 1
```

### Phase 5: Remove Legacy Code

Delete all legacy code and adapters:

1. Delete `src/strategies/adapters/` directory
2. Delete `src/strategies/migration/` directory
3. Delete `src/strategies/base.py` (BaseStrategy)
4. Delete `src/strategies/MIGRATION.md`
5. Update `src/strategies/README.md` to remove legacy references

## Data Flow

### Backtesting Data Flow

```
1. Load OHLCV data → DataFrame
2. For each candle index:
   a. Call strategy.process_candle(df, index, balance, positions)
   b. Receive TradingDecision
   c. If signal.direction == BUY:
      - Execute buy with decision.position_size
      - Set stop loss using strategy.get_stop_loss_price()
   d. If signal.direction == SELL:
      - Execute sell/exit
   e. For existing positions:
      - Check strategy.should_exit_position()
3. Calculate performance metrics
```

### Live Trading Data Flow

```
1. Fetch latest OHLCV data → DataFrame
2. Call strategy.process_candle(df, current_index, balance, positions)
3. Receive TradingDecision
4. Log decision.to_dict() to database
5. If signal.direction != HOLD:
   - Validate decision
   - Execute order with decision.position_size
   - Set stop loss using strategy.get_stop_loss_price()
6. For existing positions:
   - Check strategy.should_exit_position()
   - Execute exits as needed
```

## Error Handling

### Strategy Errors

If `process_candle()` raises an exception:
1. Log error with full context
2. Return safe `TradingDecision` with HOLD signal
3. Continue execution (don't crash)

### Component Errors

If individual components fail:
1. Strategy catches exception
2. Returns safe fallback decision
3. Logs error for debugging

## Testing Strategy

### Unit Tests

1. **Component Tests**: Test each component type in isolation
   - SignalGenerator tests
   - RiskManager tests
   - PositionSizer tests
   - RegimeDetector tests

2. **Strategy Tests**: Test strategy composition
   - Test `process_candle()` returns valid `TradingDecision`
   - Test component coordination
   - Test error handling

### Integration Tests

1. **Backtesting Integration**: Test complete backtest workflows
   - Test single strategy backtests
   - Test regime switching
   - Test performance metrics

2. **Live Trading Integration**: Test live trading workflows
   - Test order execution
   - Test position management
   - Test database logging

### Test Coverage Requirements

- All unit tests must pass
- All integration tests must pass
- No tests should reference legacy interface methods
- Test coverage should not decrease

## Performance Considerations

### Memory Usage

- **Before**: DataFrame polluted with strategy-specific columns (ml_prediction, regime_label, etc.)
- **After**: Clean DataFrame, on-demand computation in components
- **Expected**: 20-30% reduction in memory usage

### Execution Speed

- **Before**: Upfront `calculate_indicators()` for entire DataFrame
- **After**: On-demand computation per candle
- **Expected**: Similar or slightly faster (no wasted computation)

### Scalability

- Component-based system scales better with multiple strategies
- Easier to add new strategies by composing existing components
- Cleaner separation of concerns

## Documentation Updates

### Files to Update

1. **src/strategies/README.md**: Remove legacy references, document component-based approach
2. **docs/**: Update strategy development guides
3. **Code comments**: Remove "legacy", "componentised", "new" terminology
4. **Docstrings**: Update to reflect component-based interface

### New Documentation

1. **Strategy Development Guide**: How to create strategies using components
2. **Component Guide**: How to create custom components
3. **Migration Complete**: Document the final architecture

## Rollback Plan

Since the system is not in production, no rollback plan is needed. If issues are discovered:

1. Fix bugs in component-based system
2. Update tests to catch the issue
3. Continue forward with component-based architecture

The goal is to complete the migration, not maintain backward compatibility.
