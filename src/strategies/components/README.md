# Strategy Components Architecture

## Overview

The Strategy Components module implements a **component-based architecture** for trading strategies, replacing monolithic strategy classes with composable, testable components. This design enables rapid strategy development, comprehensive testing, and easy optimization of individual components.

## Architecture Philosophy

```
Strategy = SignalGenerator + RiskManager + PositionSizer + RegimeDetector
```

Each component has a single responsibility and can be tested, optimized, and replaced independently. This modular approach enables:

- ✅ **Component Reuse**: Share signal generators, risk managers, and position sizers across strategies
- ✅ **Isolated Testing**: Test each component independently with comprehensive test frameworks
- ✅ **Rapid Experimentation**: Quickly test new component combinations
- ✅ **Performance Attribution**: Identify which components drive strategy performance
- ✅ **Regime Adaptation**: Components can adapt behavior based on market conditions
- ✅ **Version Control**: Track strategy evolution and enable rollbacks

## Table of Contents

- [Core Components](#core-components)
- [Management Components](#management-components)
- [Specialized Components](#specialized-components)
- [Testing Framework](#testing-framework)
- [Usage Examples](#usage-examples)
- [Best Practices](#best-practices)
- [Migration Guide](#migration-guide)

---

## Core Components

### 1. Strategy (`strategy.py`)

The main orchestrator that composes all components into a unified trading strategy.

**Purpose**: Coordinates signal generation, risk management, and position sizing to make trading decisions.

**Key Features**:
- Comprehensive decision logging and tracking
- Performance metrics calculation
- Error handling and graceful degradation
- Regime-aware decision making
- Component parameter introspection

**Usage**:
```python
from src.strategies.components import Strategy, MLSignalGenerator, VolatilityRiskManager, ConfidenceWeightedSizer

# Create components
signal_gen = MLSignalGenerator()
risk_mgr = VolatilityRiskManager()
pos_sizer = ConfidenceWeightedSizer()

# Compose strategy
strategy = Strategy(
    name="my_ml_strategy",
    signal_generator=signal_gen,
    risk_manager=risk_mgr,
    position_sizer=pos_sizer
)

# Make trading decision
decision = strategy.process_candle(df, index=100, balance=10000.0)
print(f"Signal: {decision.signal.direction.value}")
print(f"Position Size: {decision.position_size}")
print(f"Confidence: {decision.signal.confidence}")
```

**Key Methods**:
- `process_candle(df, index, balance)` - Main decision-making method
- `get_performance_metrics()` - Get strategy performance statistics
- `get_recent_decisions()` - Get recent trading decisions
- `should_exit_position()` - Determine if position should be closed

### 1b. StrategyRuntime (`runtime.py`)

The `StrategyRuntime` provides the orchestration layer between trading engines
and component strategies.

**Purpose**: Prepare datasets, execute strategies candle-by-candle, and manage
feature caching during a run.

**Key Concepts**:
- `StrategyRuntime.prepare_data(df)` enriches the input DataFrame using
  component-declared `FeatureGeneratorSpec`s and returns a `StrategyDataset`
  containing the augmented data, warmup period, and feature cache metadata.
- `RuntimeContext` carries per-candle execution information such as account
  balance and open positions.
- `StrategyRuntime.process(index, context)` delegates to the strategy's
  `process_candle` method using the prepared dataset.
- Feature generators can optionally provide incremental update callables that
  support efficient live trading updates without recomputing entire columns.

```python
from src.strategies.components import StrategyRuntime, RuntimeContext

runtime = StrategyRuntime(strategy)
dataset = runtime.prepare_data(price_dataframe)

for i in range(dataset.warmup_period, len(dataset.data)):
    decision = runtime.process(i, RuntimeContext(balance=10_000))
    # translate TradingDecision into engine-specific actions

runtime.finalize()
```

### 2. SignalGenerator (`signal_generator.py`)

Abstract base class for generating trading signals based on market data.

**Purpose**: Analyze market data and generate BUY/SELL/HOLD signals with confidence scores.

**Key Features**:
- Regime-aware signal generation
- Confidence scoring
- Multiple signal combination strategies
- Error handling and validation

**Available Implementations**:
- `MLSignalGenerator` - Machine learning-based signals
- `TechnicalSignalGenerator` - Technical indicator-based signals
- `RandomSignalGenerator` - Random signals for testing
- `HoldSignalGenerator` - Always returns HOLD signals
- `WeightedVotingSignalGenerator` - Combines multiple generators
- `HierarchicalSignalGenerator` - Primary/secondary signal confirmation

**Usage**:
```python
from src.strategies.components import MLSignalGenerator, TechnicalSignalGenerator

# ML-based signal generator
# Note: Supports both flat (src/ml/*.onnx) and nested (src/ml/models/{SYMBOL}/{TYPE}/{VERSION}/model.onnx) paths
ml_gen = MLSignalGenerator(
    model_path="src/ml/btcusdt_price.onnx",
    sequence_length=120
)

# Technical indicator-based generator
tech_gen = TechnicalSignalGenerator(
    indicators=['rsi', 'macd', 'bollinger'],
    thresholds={'rsi': {'oversold': 30, 'overbought': 70}}
)

# Generate signal
signal = ml_gen.generate_signal(df, index=100, regime=regime_context)
print(f"Direction: {signal.direction.value}")
print(f"Strength: {signal.strength}")
print(f"Confidence: {signal.confidence}")
```

**Signal Data Structure**:
```python
@dataclass
class Signal:
    direction: SignalDirection  # BUY, SELL, HOLD
    strength: float            # 0.0 to 1.0
    confidence: float          # 0.0 to 1.0
    metadata: dict[str, Any]   # Additional context
```

### 3. RiskManager (`risk_manager.py`)

Abstract base class for managing position sizing and risk controls.

**Purpose**: Calculate position sizes based on risk parameters and market conditions.

**Key Features**:
- Risk-based position sizing
- Stop loss calculation
- Position exit decisions
- Regime-aware risk adjustment

**Available Implementations**:
- `FixedRiskManager` - Fixed percentage risk per trade
- `VolatilityRiskManager` - Risk adjusted for market volatility
- `RegimeAdaptiveRiskManager` - Risk adjusted for market regime
- `KellyRiskManager` - Kelly criterion-based sizing

**Usage**:
```python
from src.strategies.components import VolatilityRiskManager, FixedRiskManager

# Volatility-adjusted risk manager
vol_risk_mgr = VolatilityRiskManager(
    base_risk=0.02,        # 2% base risk
    atr_multiplier=2.0     # ATR multiplier for volatility
)

# Fixed risk manager
fixed_risk_mgr = FixedRiskManager(
    risk_per_trade=0.01,   # 1% risk per trade
    stop_loss_pct=0.03    # 3% stop loss
)

# Calculate position size
position_size = vol_risk_mgr.calculate_position_size(signal, balance=10000, regime=regime)
print(f"Risk-based position size: {position_size}")
```

**Key Methods**:
- `calculate_position_size(signal, balance, regime)` - Calculate position size
- `get_stop_loss(entry_price, signal, regime)` - Calculate stop loss price
- `should_exit(position, market_data, regime)` - Determine if position should be exited

### 4. PositionSizer (`position_sizer.py`)

Abstract base class for final position size calculation and adjustment.

**Purpose**: Apply additional sizing logic on top of risk manager calculations.

**Key Features**:
- Confidence-based sizing
- Regime-aware adjustments
- Kelly criterion implementation
- Bounds checking and validation

**Available Implementations**:
- `FixedFractionSizer` - Fixed fraction of balance
- `ConfidenceWeightedSizer` - Size based on signal confidence
- `KellySizer` - Kelly criterion-based sizing
- `RegimeAdaptiveSizer` - Regime-aware sizing adjustments

**Usage**:
```python
from src.strategies.components import ConfidenceWeightedSizer, KellySizer

# Confidence-weighted position sizer
conf_sizer = ConfidenceWeightedSizer(
    base_fraction=0.04,     # 4% base position size
    min_confidence=0.4      # Minimum confidence threshold
)

# Kelly criterion sizer
kelly_sizer = KellySizer(
    win_rate=0.55,          # Expected win rate
    avg_win=0.025,          # Average win percentage
    avg_loss=0.02,          # Average loss percentage
    kelly_fraction=0.3      # Kelly fraction multiplier
)

# Calculate final position size
final_size = conf_sizer.calculate_size(signal, balance=10000, risk_amount=200, regime=regime)
print(f"Final position size: {final_size}")
```

### 5. RegimeContext (`regime_context.py`)

Market regime detection and context provision for components.

**Purpose**: Detect market regimes and provide context to components for regime-aware behavior.

**Key Features**:
- Trend and volatility detection
- Regime confidence scoring
- Duration tracking
- Risk adjustment multipliers

**Usage**:
```python
from src.strategies.components import EnhancedRegimeDetector, RegimeContext

# Create regime detector
detector = EnhancedRegimeDetector()

# Detect current regime
regime = detector.detect_regime(df, index=100)

if regime:
    print(f"Trend: {regime.trend.value}")
    print(f"Volatility: {regime.volatility.value}")
    print(f"Confidence: {regime.confidence}")
    print(f"Duration: {regime.duration} periods")
    
    # Get risk adjustment
    risk_multiplier = regime.get_risk_multiplier()
    print(f"Risk multiplier: {risk_multiplier}")
```

**Regime Types**:
- **Trend**: `TREND_UP`, `TREND_DOWN`, `RANGE`
- **Volatility**: `HIGH`, `LOW`

---

## Management Components

### 1. StrategyFactory (`strategy_factory.py`)

Factory for creating pre-configured strategies with common component combinations.

**Purpose**: Provide convenient methods for creating standard strategy configurations.

**Available Strategies**:
- `create_conservative_strategy()` - Low-risk strategy
- `create_balanced_strategy()` - Moderate-risk strategy
- `create_aggressive_strategy()` - Higher-risk strategy
- `create_regime_adaptive_strategy()` - Regime-aware strategy
- `create_ensemble_strategy()` - Multi-generator strategy
- `create_hierarchical_strategy()` - Primary/secondary confirmation

**Usage**:
```python
from src.strategies.components import StrategyFactory

# Create pre-configured strategies
conservative = StrategyFactory.create_conservative_strategy("MyConservative")
balanced = StrategyFactory.create_balanced_strategy("MyBalanced")
aggressive = StrategyFactory.create_aggressive_strategy("MyAggressive")

# Custom strategy builder
from src.strategies.components import StrategyBuilder

custom_strategy = (StrategyBuilder("MyCustom")
    .with_signal_generator(MLSignalGenerator())
    .with_risk_manager(VolatilityRiskManager())
    .with_position_sizer(ConfidenceWeightedSizer())
    .with_logging(True)
    .build())
```

### 2. StrategyManager (`strategy_manager.py`)

Strategy orchestration with versioning capabilities for A/B testing and rollbacks.

**Purpose**: Manage strategy execution with version control and performance tracking.

**Key Features**:
- Version management and rollbacks
- Performance tracking
- Execution history
- Component hot-swapping

**Usage**:
```python
    from src.strategies.components import ComponentStrategyManager

# Create strategy manager
manager = StrategyManager(
    name="my_strategy_manager",
    signal_generator=MLSignalGenerator(),
    risk_manager=VolatilityRiskManager(),
    position_sizer=ConfidenceWeightedSizer()
)

# Execute strategy
signal, position_size, metadata = manager.execute_strategy(df, index=100, balance=10000)

# Create new version
version_id = manager.create_version(
    name="v2.0",
    description="Updated ML model",
    signal_generator=new_ml_generator
)

# Activate version
manager.activate_version(version_id)

# Get performance metrics
performance = manager.get_version_performance(version_id)
print(f"Version performance: {performance}")
```

### 3. StrategyRegistry (`strategy_registry.py`)

Centralized registry for strategy management with metadata tracking and serialization.

**Purpose**: Register, version, and manage strategies with comprehensive metadata.

**Key Features**:
- Strategy registration and metadata tracking
- Version control and lineage
- Serialization/deserialization
- Validation and integrity checking

**Usage**:
```python
from src.strategies.components import StrategyRegistry

# Create registry
registry = StrategyRegistry()

# Register strategy
strategy_id = registry.register_strategy(
    strategy=my_strategy,
    metadata={
        'created_by': 'developer',
        'description': 'ML-based strategy with volatility adjustment',
        'tags': ['ml', 'volatility', 'production'],
        'status': 'PRODUCTION'
    }
)

# Update strategy
new_version = registry.update_strategy(
    strategy_id=strategy_id,
    strategy=updated_strategy,
    changes=['Updated ML model', 'Improved risk parameters'],
    is_major=True
)

# Get strategy metadata
metadata = registry.get_strategy_metadata(strategy_id)
print(f"Strategy: {metadata.name}")
print(f"Version: {metadata.version}")
print(f"Status: {metadata.status.value}")

# Serialize strategy for backup
serialized = registry.serialize_strategy(strategy_id)
```

---

## Specialized Components

### 1. MLSignalGenerator (`ml_signal_generator.py`)

Machine learning-based signal generator using ONNX models.

**Purpose**: Generate trading signals using trained ML models with regime-aware adjustments.

**Key Features**:
- ONNX model inference
- Regime-aware threshold adjustment
- Prediction engine integration
- Confidence calculation based on prediction quality

**Usage**:
```python
from src.strategies.components import MLSignalGenerator

# Create ML signal generator
ml_gen = MLSignalGenerator(
    name="btc_ml_generator",
    model_path="src/ml/btcusdt_price.onnx",
    sequence_length=120,
    use_prediction_engine=True
)

# Generate signal
signal = ml_gen.generate_signal(df, index=100, regime=regime_context)

# Get model information
model_info = ml_gen.get_model_info()
print(f"Model: {model_info['model_name']}")
print(f"Input shape: {model_info['input_shape']}")
```

### 2. TechnicalSignalGenerator (`technical_signal_generator.py`)

Technical indicator-based signal generator.

**Purpose**: Generate signals based on technical analysis indicators.

**Available Indicators**:
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- Moving Averages

**Usage**:
```python
from src.strategies.components import TechnicalSignalGenerator, RSISignalGenerator

# RSI-based generator
rsi_gen = RSISignalGenerator(
    period=14,
    oversold_threshold=30,
    overbought_threshold=70
)

# Multi-indicator generator
tech_gen = TechnicalSignalGenerator(
    indicators=['rsi', 'macd', 'bollinger'],
    weights={'rsi': 0.4, 'macd': 0.4, 'bollinger': 0.2}
)
```

### 3. PerformanceTracker (`performance_tracker.py`)

Comprehensive performance tracking and analysis system.

**Purpose**: Track, analyze, and compare strategy performance across different time periods and regimes.

**Key Features**:
- Real-time performance metrics
- Historical data storage
- Regime-specific analysis
- Performance comparison utilities

**Usage**:
```python
from src.strategies.components import PerformanceTracker

# Create performance tracker
tracker = PerformanceTracker(strategy_name="my_strategy")

# Record trade result
tracker.record_trade_result(
    timestamp=datetime.now(),
    symbol="BTCUSDT",
    side="long",
    entry_price=50000.0,
    exit_price=52000.0,
    quantity=0.1,
    pnl=200.0,
    regime="bull_low_vol"
)

# Get performance metrics
metrics = tracker.get_performance_metrics(period="daily")
print(f"Daily return: {metrics['total_return']:.2%}")
print(f"Sharpe ratio: {metrics['sharpe_ratio']:.2f}")
print(f"Max drawdown: {metrics['max_drawdown']:.2%}")
```

### 4. StrategyLineageTracker (`strategy_lineage.py`)

Strategy evolution and lineage tracking system.

**Purpose**: Track strategy evolution, parent-child relationships, and change impact analysis.

**Key Features**:
- Parent-child relationship tracking
- Change impact analysis
- Evolution visualization
- Branching and merging capabilities

**Usage**:
```python
from src.strategies.components import StrategyLineageTracker

# Create lineage tracker
lineage = StrategyLineageTracker()

# Register strategy relationship
lineage.add_relationship(
    parent_id="strategy_v1",
    child_id="strategy_v2",
    relationship_type="EVOLUTION",
    change_description="Updated ML model parameters"
)

# Get evolution path
evolution_path = lineage.get_evolution_path("strategy_v2")
print(f"Evolution: {evolution_path}")

# Analyze change impact
impact = lineage.analyze_change_impact("strategy_v2")
print(f"Impact level: {impact.impact_level.value}")
```

---

## Testing Framework

The component testing framework provides comprehensive testing capabilities for individual components and complete strategies.

### Key Components

1. **TestDatasetGenerator** - Generate synthetic market data for testing
2. **ComponentPerformanceTester** - Test individual components in isolation
3. **RegimeTester** - Test components in specific market regimes
4. **PerformanceAttributionAnalyzer** - Analyze component contributions to performance

### Usage

```python
from src.strategies.components.testing import (
    TestDatasetGenerator,
    ComponentPerformanceTester,
    RegimeTester
)

# Generate test data
generator = TestDatasetGenerator()
test_data = generator.generate_synthetic_dataset("strong_bull_low_vol", seed=42)

# Test signal generator
tester = ComponentPerformanceTester(test_data)
results = tester.test_signal_generator(MLSignalGenerator())

print(f"Accuracy: {results.accuracy:.2%}")
print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
print(f"Max Drawdown: {results.max_drawdown:.2%}")

# Test in specific regime
regime_tester = RegimeTester(test_data)
regime_results = regime_tester.test_component_in_regime(
    MLSignalGenerator(),
    regime_type='trend_up_low_vol'
)
```

For detailed testing framework documentation, see [testing/README.md](testing/README.md).

---

## Usage Examples

### Example 1: Creating a Simple ML Strategy

```python
from src.strategies.components import (
    Strategy, MLSignalGenerator, VolatilityRiskManager, 
    ConfidenceWeightedSizer, EnhancedRegimeDetector
)

# Create components
signal_gen = MLSignalGenerator(
    model_path="src/ml/btcusdt_price.onnx",
    sequence_length=120
)

risk_mgr = VolatilityRiskManager(
    base_risk=0.02,
    atr_multiplier=2.0
)

pos_sizer = ConfidenceWeightedSizer(
    base_fraction=0.04,
    min_confidence=0.4
)

regime_detector = EnhancedRegimeDetector()

# Compose strategy
strategy = Strategy(
    name="ml_volatility_strategy",
    signal_generator=signal_gen,
    risk_manager=risk_mgr,
    position_sizer=pos_sizer,
    regime_detector=regime_detector
)

# Use strategy
decision = strategy.process_candle(df, index=100, balance=10000.0)
print(f"Decision: {decision.signal.direction.value}")
print(f"Position Size: ${decision.position_size:.2f}")
print(f"Confidence: {decision.signal.confidence:.2f}")
```

### Example 2: Testing Component Performance

```python
from src.strategies.components.testing import TestDatasetGenerator, ComponentPerformanceTester

# Generate test data
generator = TestDatasetGenerator()
test_data = generator.generate_synthetic_dataset("moderate_bull_low_vol", seed=42)

# Test different signal generators
signal_generators = [
    MLSignalGenerator(),
    TechnicalSignalGenerator(),
    RandomSignalGenerator(buy_prob=0.3, sell_prob=0.3)
]

tester = ComponentPerformanceTester(test_data)

results = []
for gen in signal_generators:
    result = tester.test_signal_generator(gen)
    results.append({
        'name': gen.name,
        'accuracy': result.accuracy,
        'sharpe': result.sharpe_ratio,
        'max_drawdown': result.max_drawdown
    })

# Sort by Sharpe ratio
results.sort(key=lambda x: x['sharpe'], reverse=True)

print("Signal Generator Rankings:")
for i, result in enumerate(results, 1):
    print(f"{i}. {result['name']}: Sharpe={result['sharpe']:.2f}, "
          f"Accuracy={result['accuracy']:.2%}")
```

### Example 3: Strategy Version Management

```python
    from src.strategies.components import ComponentStrategyManager, StrategyRegistry

# Create strategy manager
manager = StrategyManager(
    name="production_strategy",
    signal_generator=MLSignalGenerator(),
    risk_manager=VolatilityRiskManager(),
    position_sizer=ConfidenceWeightedSizer()
)

# Register in registry
registry = StrategyRegistry()
strategy_id = registry.register_strategy(
    strategy=manager,
    metadata={
        'created_by': 'trading_team',
        'description': 'Production ML strategy',
        'tags': ['ml', 'production', 'btc'],
        'status': 'PRODUCTION'
    }
)

# Create improved version
new_version_id = manager.create_version(
    name="v2.0",
    description="Improved ML model with better feature engineering",
    signal_generator=ImprovedMLSignalGenerator()
)

# Test new version
test_results = manager.get_version_performance(new_version_id)

if test_results.get('sharpe_ratio', 0) > 1.5:
    # Activate new version
    manager.activate_version(new_version_id)
    print("✅ New version activated")
else:
    print("⚠️ New version performance insufficient")
```

### Example 4: Regime-Adaptive Strategy

```python
from src.strategies.components import (
    Strategy, RegimeAdaptiveSignalGenerator, 
    RegimeAdaptiveRiskManager, RegimeAdaptiveSizer
)

# Create regime-adaptive components
regime_gen = RegimeAdaptiveSignalGenerator(
    regime_generators={
        'bull_low_vol': MLSignalGenerator(),
        'bear_low_vol': TechnicalSignalGenerator(),
        'range_low_vol': HoldSignalGenerator()
    }
)

regime_risk = RegimeAdaptiveRiskManager(
    base_risk=0.025,
    regime_multipliers={
        'bull_low_vol': 1.2,
        'bear_low_vol': 0.7,
        'range_low_vol': 0.9
    }
)

regime_sizer = RegimeAdaptiveSizer(
    base_fraction=0.03,
    volatility_adjustment=True
)

# Compose adaptive strategy
adaptive_strategy = Strategy(
    name="regime_adaptive_strategy",
    signal_generator=regime_gen,
    risk_manager=regime_risk,
    position_sizer=regime_sizer
)

# Strategy automatically adapts to market regimes
decision = adaptive_strategy.process_candle(df, index=100, balance=10000.0)
print(f"Regime: {decision.regime.trend.value}/{decision.regime.volatility.value}")
print(f"Decision: {decision.signal.direction.value}")
```

---

## Best Practices

### 1. Component Design

- **Single Responsibility**: Each component should have one clear purpose
- **Interface Compliance**: Implement all required abstract methods
- **Error Handling**: Handle edge cases gracefully with fallback behavior
- **Parameter Validation**: Validate inputs and provide clear error messages

### 2. Strategy Composition

- **Start Simple**: Begin with basic components and add complexity gradually
- **Test Components**: Test individual components before composing strategies
- **Monitor Performance**: Track component performance and identify bottlenecks
- **Version Control**: Use version management for strategy evolution

### 3. Testing

- **Comprehensive Testing**: Test across multiple market scenarios and regimes
- **Edge Case Testing**: Test with missing data, extreme volatility, etc.
- **Performance Testing**: Monitor execution time and resource usage
- **Regression Testing**: Ensure changes don't break existing functionality

### 4. Performance Optimization

- **Component Profiling**: Identify slow components and optimize them
- **Caching**: Cache expensive calculations where appropriate
- **Batch Processing**: Process multiple signals efficiently
- **Resource Management**: Manage memory and CPU usage effectively

### 5. Production Deployment

- **Gradual Rollout**: Deploy new strategies with small position sizes initially
- **Monitoring**: Monitor strategy performance and component health
- **Rollback Plan**: Have rollback procedures ready for failed deployments
- **Documentation**: Document strategy parameters and expected behavior

---

## Migration Guide

### From Monolithic Strategies

If you have existing monolithic strategies, follow these steps:

1. **Identify Components**: Break down your strategy into signal generation, risk management, and position sizing logic

2. **Create Component Classes**: Implement the appropriate abstract base classes

3. **Test Components**: Use the testing framework to validate individual components

4. **Compose Strategy**: Use the Strategy class to combine your components

5. **Migrate Gradually**: Start with one strategy and gradually migrate others

### Example Migration

**Before (Monolithic)**:
```python
class MyStrategy:
    def __init__(self):
        self.ml_model = load_model()
        self.risk_pct = 0.02
        
    def make_decision(self, df, index, balance):
        # Signal generation
        prediction = self.ml_model.predict(df.iloc[index-120:index])
        if prediction > 0.001:
            signal = "BUY"
        elif prediction < -0.001:
            signal = "SELL"
        else:
            signal = "HOLD"
            
        # Risk management
        position_size = balance * self.risk_pct
        
        # Position sizing
        if signal == "BUY":
            return position_size
        else:
            return 0
```

**After (Component-based)**:
```python
class MyMLSignalGenerator(SignalGenerator):
    def __init__(self):
        super().__init__("my_ml_generator")
        self.ml_model = load_model()
        
    def generate_signal(self, df, index, regime=None):
        prediction = self.ml_model.predict(df.iloc[index-120:index])
        if prediction > 0.001:
            return Signal(SignalDirection.BUY, 0.8, 0.7, {})
        elif prediction < -0.001:
            return Signal(SignalDirection.SELL, 0.8, 0.7, {})
        else:
            return Signal(SignalDirection.HOLD, 0.0, 0.5, {})

# Compose strategy
strategy = Strategy(
    name="my_migrated_strategy",
    signal_generator=MyMLSignalGenerator(),
    risk_manager=FixedRiskManager(risk_per_trade=0.02),
    position_sizer=FixedFractionSizer(fraction=0.02)
)
```

---

## API Reference

### Core Classes

- **Strategy**: Main strategy orchestrator
- **SignalGenerator**: Abstract base for signal generation
- **RiskManager**: Abstract base for risk management
- **PositionSizer**: Abstract base for position sizing
- **RegimeContext**: Market regime information

### Management Classes

- **StrategyFactory**: Pre-configured strategy creation
- **StrategyBuilder**: Fluent interface for custom strategies
- **StrategyManager**: Strategy execution with versioning
- **StrategyRegistry**: Centralized strategy management

### Specialized Classes

- **MLSignalGenerator**: ML-based signal generation
- **TechnicalSignalGenerator**: Technical indicator-based signals
- **PerformanceTracker**: Performance tracking and analysis
- **StrategyLineageTracker**: Strategy evolution tracking

### Testing Classes

- **TestDatasetGenerator**: Synthetic data generation
- **ComponentPerformanceTester**: Component testing
- **RegimeTester**: Regime-specific testing
- **PerformanceAttributionAnalyzer**: Performance attribution

---

## Related Documentation

- [Testing Framework](testing/README.md) - Comprehensive testing documentation
- [Strategy System Design](../.kiro/specs/strategy-system-redesign/) - Architecture design documents
- [Component Examples](../examples/) - Usage examples and tutorials

---

## Support

For questions or issues:

1. Check this README and the testing framework documentation
2. Review the design documentation in `.kiro/specs/strategy-system-redesign/`
3. Look at existing tests in `tests/unit/strategies/components/`
4. Check the main strategy system documentation

---

**Last Updated**: 2025-11-02  
**Version**: 1.0.0  
**Status**: Production Ready