# Component Testing Framework

## Overview

The Component Testing Framework provides comprehensive testing capabilities for the modular strategy system, enabling isolated testing and performance analysis of trading strategy components. This framework is a core part of the **Strategy System Redesign** initiative that transitions from monolithic strategies to composable, testable components.

## Table of Contents

- [Why This Framework Exists](#why-this-framework-exists)
- [When to Use This Framework](#when-to-use-this-framework)
- [Quick Start](#quick-start)
- [Core Components](#core-components)
- [Testing Workflows](#testing-workflows)
- [Advanced Usage](#advanced-usage)
- [Best Practices](#best-practices)
- [Integration with Strategy System](#integration-with-strategy-system)

---

## Why This Framework Exists

### The Problem

The original trading system architecture had several issues:

1. **Monolithic Strategies**: Strategies were large, complex classes that mixed signal generation, risk management, and position sizing
2. **Difficult Testing**: Testing individual aspects of a strategy required running the entire strategy
3. **No Component Reuse**: Similar logic was duplicated across multiple strategies
4. **Limited Regime Analysis**: No easy way to test how components perform in specific market conditions
5. **No Performance Attribution**: Couldn't identify which part of a strategy was responsible for performance

### The Solution

The **Strategy System Redesign** introduced a component-based architecture:

```
Strategy = SignalGenerator + RiskManager + PositionSizer + RegimeDetector
```

This testing framework enables:

- ✅ **Isolated Component Testing**: Test signal generators, risk managers, and position sizers independently
- ✅ **Regime-Specific Analysis**: Evaluate component performance in bull/bear markets, high/low volatility
- ✅ **Performance Attribution**: Identify which components contribute most to overall performance
- ✅ **Rapid Experimentation**: Quickly test new component combinations without full strategy deployment
- ✅ **Comprehensive Test Data**: Generate synthetic market scenarios for thorough testing

---

## When to Use This Framework

### Use Cases

#### 1. **Component Development** 🛠️

When building a new SignalGenerator, RiskManager, or PositionSizer:

```python
from src.strategies.components.testing import ComponentPerformanceTester, TestDatasetGenerator

# Generate test data
generator = TestDatasetGenerator()
test_data = generator.generate_synthetic_dataset("strong_bull_low_vol", seed=42)

# Test your new component
tester = ComponentPerformanceTester(test_data)
results = tester.test_signal_generator(MyNewSignalGenerator())

print(f"Signal Accuracy: {results.accuracy:.2%}")
print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
print(f"Win Rate: {results.win_rate:.2%}")
```

#### 2. **Regime-Specific Optimization** 📊

When optimizing strategy behavior for different market conditions:

```python
from src.strategies.components.testing import RegimeTester

# Test component in specific regime
tester = RegimeTester(historical_data)
results = tester.test_component_in_regime(
    my_signal_generator,
    regime_type='bull_high_vol'
)

# Compare across all regimes
comparison = tester.compare_across_regimes(my_signal_generator)
print(f"Best regime: {comparison['best_regime']}")
print(f"Worst regime: {comparison['worst_regime']}")
```

#### 3. **Performance Attribution Analysis** 🔍

When identifying which components drive strategy performance:

```python
from src.strategies.components.testing import PerformanceAttributionAnalyzer

# Analyze component contributions
analyzer = PerformanceAttributionAnalyzer(test_data)
attribution = analyzer.analyze_strategy_attribution(my_strategy)

print(f"Signal Generator Contribution: {attribution.signal_generator_contribution:.1%}")
print(f"Risk Manager Contribution: {attribution.risk_manager_contribution:.1%}")
print(f"Position Sizer Contribution: {attribution.position_sizer_contribution:.1%}")
```

#### 4. **Component Comparison** ⚖️

When choosing between different component implementations:

```python
# Test multiple signal generators
generators = [
    MLSignalGenerator(),
    TechnicalSignalGenerator(),
    SentimentSignalGenerator()
]

for gen in generators:
    results = tester.test_signal_generator(gen)
    print(f"{gen.name}: Accuracy={results.accuracy:.2%}, Sharpe={results.sharpe_ratio:.2f}")
```

#### 5. **Edge Case Validation** 🧪

When stress-testing components against unusual market conditions:

```python
# Test against various edge cases
edge_cases = ['missing_data', 'extreme_volatility', 'zero_volume', 'price_gaps']

for case_type in edge_cases:
    edge_data = generator.generate_edge_case_dataset(case_type, seed=42)
    results = tester.test_signal_generator(my_component, edge_data)
    print(f"{case_type}: {results.error_count} errors, {results.accuracy:.2%} accuracy")
```

---

## Quick Start

### Installation

The framework is part of the main trading bot codebase. Ensure you have all dependencies:

```bash
pip install -r requirements.txt
```

### Basic Example

```python
from src.strategies.components.testing import (
    ComponentPerformanceTester,
    TestDatasetGenerator,
    RegimeTester
)
from src.strategies.components.signal_generator import MLSignalGenerator

# 1. Generate test data
generator = TestDatasetGenerator()
test_data = generator.generate_synthetic_dataset("moderate_bull_low_vol", seed=42)

# 2. Create tester
tester = ComponentPerformanceTester(test_data)

# 3. Test your component
signal_gen = MLSignalGenerator()
results = tester.test_signal_generator(signal_gen)

# 4. Analyze results
print(f"""
Signal Generator Performance:
  Accuracy: {results.accuracy:.2%}
  Precision: {results.precision:.2%}
  Recall: {results.recall:.2%}
  F1 Score: {results.f1_score:.2f}
  Sharpe Ratio: {results.sharpe_ratio:.2f}
  Max Drawdown: {results.max_drawdown:.2%}
  Total Signals: {results.total_signals}
""")
```

---

## Core Components

### 1. TestDatasetGenerator

Generates comprehensive test datasets for component testing.

#### Available Scenarios

```python
generator = TestDatasetGenerator()

# Market scenarios
scenarios = generator.get_all_scenarios()
# ['strong_bull_low_vol', 'moderate_bull_high_vol', 'strong_bear_low_vol',
#  'volatile_bear_crash', 'tight_range_low_vol', 'wide_range_high_vol',
#  'bubble_and_crash', 'gap_heavy_market', 'bull_to_bear_transition', ...]

# Generate specific scenario
test_data = generator.generate_synthetic_dataset("strong_bull_low_vol", seed=42)
```

#### Edge Cases

```python
# Available edge cases
edge_cases = [
    'missing_data',       # Missing OHLCV data points
    'extreme_volatility', # Volatility spikes
    'zero_volume',        # Zero volume periods
    'price_gaps',         # Price gaps between periods
    'flat_prices',        # Flat price periods
    'extreme_outliers'    # Extreme price outliers
]

# Generate edge case dataset
edge_data = generator.generate_edge_case_dataset('missing_data', duration_days=100, seed=42)
```

#### Regime-Labeled Datasets

```python
# Generate data with regime labels
market_data, regime_labels = generator.create_regime_labeled_dataset(
    scenario_name="bull_to_bear_transition",
    regime_detection_method="simple",
    seed=42
)
```

### 2. ComponentPerformanceTester

Tests individual components in isolation with detailed metrics.

#### Testing Signal Generators

```python
tester = ComponentPerformanceTester(test_data)
results = tester.test_signal_generator(my_signal_gen)

# Access detailed metrics
print(f"Accuracy: {results.accuracy:.2%}")          # % of profitable signals
print(f"Precision: {results.precision:.2%}")        # % of buy signals that were profitable
print(f"Recall: {results.recall:.2%}")              # % of opportunities captured
print(f"F1 Score: {results.f1_score:.2f}")          # Harmonic mean of precision/recall
print(f"Confidence Correlation: {results.confidence_accuracy_correlation:.2f}")

# Regime breakdown
for regime, metrics in results.regime_breakdown.items():
    print(f"{regime}: {metrics['accuracy']:.2%} accuracy")
```

#### Testing Risk Managers

```python
results = tester.test_risk_manager(my_risk_manager, test_balance=10000.0)

print(f"Max Drawdown Achieved: {results.max_drawdown_achieved:.2%}")
print(f"Drawdown Control Score: {results.drawdown_control_score:.2f}")
print(f"Average Position Size: ${results.average_position_size:.2f}")
print(f"Stop Loss Effectiveness: {results.stop_loss_effectiveness:.2f}")
print(f"Risk Efficiency Score: {results.risk_efficiency_score:.2f}")
```

#### Testing Position Sizers

```python
results = tester.test_position_sizer(my_position_sizer, test_balance=10000.0)

print(f"Kelly Criterion Adherence: {results.kelly_criterion_adherence:.2f}")
print(f"Regime Adaptation: {results.regime_adaptation_effectiveness:.2f}")
print(f"Volatility Adjustment Quality: {results.volatility_adjustment_quality:.2f}")
print(f"Average Position Size: ${results.average_position_size:.2f}")
print(f"Bounds Adherence Rate: {results.bounds_adherence_rate:.2%}")
```

#### Testing All Components Together

```python
results = tester.test_all_components(
    signal_generator=my_signal_gen,
    risk_manager=my_risk_manager,
    position_sizer=my_position_sizer,
    test_balance=10000.0
)

print(f"Overall Performance Score: {results.overall_performance_score:.2f}")
print(f"Component Synergy Score: {results.component_synergy_score:.2f}")
```

### 3. RegimeTester

Tests components and strategies in specific market regimes.

```python
tester = RegimeTester(test_data)

# Test in specific regime
results = tester.test_strategy_in_regime(
    my_strategy,
    regime_type='trend_up_low_vol',
    initial_balance=10000.0
)

print(f"Regime: {results.regime_type}")
print(f"Total Return: {results.total_return:.2%}")
print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
print(f"Trade Count: {results.trade_count}")

# Compare across regimes
comparison = tester.compare_strategy_across_regimes(my_strategy)
print(f"Best Regime: {comparison.best_regime}")
print(f"Worst Regime: {comparison.worst_regime}")

# Analyze regime transitions
transitions = tester.analyze_regime_transitions(test_data)
print(f"Total Transitions: {transitions.total_transitions}")
print(f"Average Transition Impact: {transitions.avg_impact:.2%}")
```

### 4. PerformanceAttributionAnalyzer

Analyzes how each component contributes to overall performance.

```python
analyzer = PerformanceAttributionAnalyzer(test_data)

# Analyze baseline strategy
attribution = analyzer.analyze_strategy_attribution(my_strategy, initial_balance=10000.0)

print(f"Total Return: {attribution.baseline_performance.total_return:.2%}")
print(f"\nComponent Contributions:")
print(f"  Signal Generator: {attribution.signal_generator_attribution.total_contribution:.2%}")
print(f"  Risk Manager: {attribution.risk_manager_attribution.total_contribution:.2%}")
print(f"  Position Sizer: {attribution.position_sizer_attribution.total_contribution:.2%}")
print(f"\nPrimary Driver: {attribution.primary_driver}")
print(f"Weakest Component: {attribution.weakest_component}")

# Test component replacement impact
impact = analyzer.analyze_component_replacement(
    baseline_strategy=my_strategy,
    new_signal_generator=alternative_signal_gen
)

print(f"Performance Delta: {impact.performance_delta:.2%}")
print(f"Recommendation: {impact.recommendation}")
```

---

## Testing Workflows

### Workflow 1: Developing a New Signal Generator

```python
# Step 1: Generate diverse test datasets
generator = TestDatasetGenerator()
datasets = {
    'bull_market': generator.generate_synthetic_dataset('strong_bull_low_vol', seed=42),
    'bear_market': generator.generate_synthetic_dataset('strong_bear_low_vol', seed=42),
    'sideways': generator.generate_synthetic_dataset('tight_range_low_vol', seed=42),
    'volatile': generator.generate_synthetic_dataset('moderate_bull_high_vol', seed=42)
}

# Step 2: Create your signal generator
class MyNewSignalGenerator(SignalGenerator):
    def __init__(self):
        super().__init__("my_new_generator")
    
    def generate_signal(self, df, index, regime=None):
        # Your signal logic here
        pass
    
    def get_confidence(self, df, index):
        # Your confidence calculation here
        pass

# Step 3: Test across all scenarios
my_gen = MyNewSignalGenerator()

for scenario_name, data in datasets.items():
    tester = ComponentPerformanceTester(data)
    results = tester.test_signal_generator(my_gen)
    print(f"\n{scenario_name}:")
    print(f"  Accuracy: {results.accuracy:.2%}")
    print(f"  Sharpe: {results.sharpe_ratio:.2f}")
    print(f"  Errors: {results.error_count}")

# Step 4: Test edge cases
for edge_case in ['missing_data', 'extreme_volatility', 'price_gaps']:
    edge_data = generator.generate_edge_case_dataset(edge_case, seed=42)
    tester = ComponentPerformanceTester(edge_data)
    results = tester.test_signal_generator(my_gen)
    print(f"{edge_case}: {results.error_count} errors")
```

### Workflow 2: Optimizing for Specific Regimes

```python
# Step 1: Get historical data with regime labels
generator = TestDatasetGenerator()
market_data, regime_labels = generator.create_regime_labeled_dataset(
    "multiple_regime_changes",
    seed=42
)

# Step 2: Test in each regime
tester = RegimeTester(market_data)
regime_types = ['trend_up_low_vol', 'trend_down_high_vol', 'range_medium_vol']

regime_results = {}
for regime in regime_types:
    results = tester.test_component_in_regime(my_signal_gen, regime)
    regime_results[regime] = results

# Step 3: Identify weaknesses
for regime, results in regime_results.items():
    if results.get('accuracy', 0) < 0.5:
        print(f"⚠️ Weak performance in {regime}: {results['accuracy']:.2%}")
    else:
        print(f"✅ Good performance in {regime}: {results['accuracy']:.2%}")

# Step 4: Optimize weak regimes
# Now you can focus optimization efforts on specific regimes
```

### Workflow 3: Comparing Component Alternatives

```python
# Test multiple implementations
signal_generators = [
    MLSignalGenerator(),
    TechnicalSignalGenerator(),
    SentimentSignalGenerator(),
    WeightedVotingSignalGenerator({...})
]

test_data = generator.generate_synthetic_dataset("moderate_bull_low_vol", seed=42)
tester = ComponentPerformanceTester(test_data)

results_comparison = []
for sig_gen in signal_generators:
    results = tester.test_signal_generator(sig_gen)
    results_comparison.append({
        'name': sig_gen.name,
        'accuracy': results.accuracy,
        'sharpe': results.sharpe_ratio,
        'max_drawdown': results.max_drawdown,
        'error_rate': results.error_rate
    })

# Sort by Sharpe ratio
results_comparison.sort(key=lambda x: x['sharpe'], reverse=True)

print("\nSignal Generator Rankings:")
for i, result in enumerate(results_comparison, 1):
    print(f"{i}. {result['name']}: Sharpe={result['sharpe']:.2f}, Accuracy={result['accuracy']:.2%}")
```

### Workflow 4: Strategy Component Attribution

```python
# Understand which components are performing well
analyzer = PerformanceAttributionAnalyzer(test_data)
attribution = analyzer.analyze_strategy_attribution(my_strategy)

# Check if replacing a component would improve performance
replacement_impact = analyzer.analyze_component_replacement(
    baseline_strategy=my_strategy,
    new_signal_generator=better_signal_gen
)

if replacement_impact.performance_delta > 0.05:  # 5% improvement
    print(f"✅ Replacing signal generator would improve performance by {replacement_impact.performance_delta:.2%}")
    print(f"Recommendation: {replacement_impact.recommendation}")
else:
    print("⚠️ Component replacement not recommended")
```

---

## Core Components

### TestDatasetGenerator

**Purpose**: Generate comprehensive, reproducible test datasets

**Key Methods**:
- `generate_synthetic_dataset(scenario_name, seed)` - Generate market scenario data
- `generate_edge_case_dataset(case_type, duration_days, seed)` - Generate edge case data
- `create_regime_labeled_dataset(scenario_name, method, seed)` - Generate regime-labeled data
- `get_comprehensive_test_suite(seed)` - Get all scenarios and edge cases
- `get_all_scenarios()` - List available market scenarios

**Market Scenarios**:
- Bull markets: `strong_bull_low_vol`, `moderate_bull_high_vol`
- Bear markets: `strong_bear_low_vol`, `volatile_bear_crash`
- Sideways: `tight_range_low_vol`, `wide_range_high_vol`
- Special: `bubble_and_crash`, `gap_heavy_market`, `bull_to_bear_transition`

### ComponentPerformanceTester

**Purpose**: Test components in isolation with detailed metrics

**Key Methods**:
- `test_signal_generator(generator, scenarios)` → `SignalTestResults`
- `test_risk_manager(risk_manager, test_balance, scenarios)` → `RiskTestResults`
- `test_position_sizer(position_sizer, test_balance, scenarios)` → `SizingTestResults`
- `test_all_components(signal_generator, risk_manager, position_sizer, test_balance)` → `ComponentTestResults`

**Result Types**:

```python
@dataclass
class SignalTestResults:
    accuracy: float              # % of profitable signals
    precision: float             # % of buy signals that were profitable
    recall: float                # % of opportunities captured
    f1_score: float             # Harmonic mean
    total_return: float         # Simulated return
    sharpe_ratio: float         # Risk-adjusted return
    max_drawdown: float         # Maximum drawdown
    buy_signals: int            # Count of buy signals
    sell_signals: int           # Count of sell signals
    hold_signals: int           # Count of hold signals
    avg_confidence: float       # Average signal confidence
    confidence_accuracy_correlation: float  # Confidence vs accuracy correlation
    regime_breakdown: dict      # Performance by regime
    error_count: int            # Number of errors
```

### RegimeTester

**Purpose**: Test components in specific market regimes

**Key Methods**:
- `test_strategy_in_regime(strategy, regime_type, initial_balance)` → `RegimeTestResults`
- `test_component_in_regime(component, regime_type)` → `dict`
- `compare_strategy_across_regimes(strategy, regime_types, initial_balance)` → `RegimeComparisonResults`
- `analyze_regime_transitions(test_data)` → `RegimeTransitionAnalysis`
- `get_regime_statistics(test_data)` → `dict`

**Regime Types**:
- `trend_up_low_vol` - Bull market, low volatility
- `trend_up_high_vol` - Bull market, high volatility
- `trend_down_low_vol` - Bear market, low volatility
- `trend_down_high_vol` - Bear market, high volatility
- `range_low_vol` - Sideways, low volatility
- `range_high_vol` - Sideways, high volatility

### PerformanceAttributionAnalyzer

**Purpose**: Analyze component contributions to strategy performance

**Key Methods**:
- `analyze_strategy_attribution(strategy, initial_balance)` → `AttributionReport`
- `analyze_component_replacement(baseline_strategy, new_component)` → `ReplacementImpact`
- `compare_component_variants(baseline_strategy, component_variants)` → `ComponentComparison`

---

## Advanced Usage

### Creating Custom Test Scenarios

```python
from src.strategies.components.testing.test_datasets import MarketScenario

# Define custom scenario
custom_scenario = MarketScenario(
    name="my_custom_scenario",
    description="Custom test scenario",
    duration_days=252,
    trend_direction="up",
    volatility_level="medium",
    trend_strength=0.6,
    volatility_value=0.25,
    initial_price=100.0,
    final_price=130.0,
    max_drawdown=0.12
)

# Generate data from custom scenario
generator = TestDatasetGenerator()
custom_data = generator.synthetic_generator.generate_scenario_data(custom_scenario, seed=42)
```

### Comprehensive Component Test Suite

```python
def run_comprehensive_component_tests(component):
    """Run a component through all standard tests"""
    
    generator = TestDatasetGenerator()
    comprehensive_suite = generator.get_comprehensive_test_suite(seed=42)
    
    results_summary = []
    
    for test_name, test_data in comprehensive_suite.items():
        try:
            tester = ComponentPerformanceTester(test_data)
            
            if isinstance(component, SignalGenerator):
                results = tester.test_signal_generator(component)
                metric = results.accuracy
            elif isinstance(component, RiskManager):
                results = tester.test_risk_manager(component)
                metric = results.drawdown_control_score
            elif isinstance(component, PositionSizer):
                results = tester.test_position_sizer(component)
                metric = results.optimal_sizing_score
            else:
                continue
            
            results_summary.append({
                'test': test_name,
                'metric': metric,
                'errors': results.error_count
            })
        except Exception as e:
            results_summary.append({
                'test': test_name,
                'metric': 0.0,
                'errors': 1,
                'error_msg': str(e)
            })
    
    return results_summary
```

### Regime-Specific Component Tuning

```python
# Find optimal parameters for each regime
from src.strategies.components.position_sizer import RegimeAdaptiveSizer

# Test different regime multipliers
multiplier_configs = [
    {'trend_up_low_vol': 1.2, 'trend_down_high_vol': 0.5},
    {'trend_up_low_vol': 1.5, 'trend_down_high_vol': 0.3},
    {'trend_up_low_vol': 1.0, 'trend_down_high_vol': 0.7},
]

best_config = None
best_score = 0

for config in multiplier_configs:
    sizer = RegimeAdaptiveSizer(regime_multipliers=config)
    tester = ComponentPerformanceTester(test_data)
    results = tester.test_position_sizer(sizer)
    
    if results.regime_adaptation_effectiveness > best_score:
        best_score = results.regime_adaptation_effectiveness
        best_config = config

print(f"Best regime multipliers: {best_config}")
print(f"Adaptation effectiveness: {best_score:.2f}")
```

---

## Best Practices

### 1. Always Use Seeds for Reproducibility

```python
# ✅ GOOD: Reproducible tests
test_data = generator.generate_synthetic_dataset("bull_market", seed=42)
edge_data = generator.generate_edge_case_dataset("missing_data", seed=42)

# ❌ BAD: Non-reproducible tests
test_data = generator.generate_synthetic_dataset("bull_market")  # Different each time
```

### 2. Test Across Multiple Scenarios

```python
# ✅ GOOD: Comprehensive testing
scenarios = ['strong_bull_low_vol', 'strong_bear_low_vol', 'tight_range_low_vol']
for scenario in scenarios:
    data = generator.generate_synthetic_dataset(scenario, seed=42)
    results = tester.test_signal_generator(my_component, data)
    # Analyze results

# ❌ BAD: Testing only in favorable conditions
data = generator.generate_synthetic_dataset('strong_bull_low_vol', seed=42)
results = tester.test_signal_generator(my_component, data)
```

### 3. Include Edge Case Testing

```python
# ✅ GOOD: Test robustness
edge_cases = ['missing_data', 'extreme_volatility', 'zero_volume']
for case in edge_cases:
    edge_data = generator.generate_edge_case_dataset(case, seed=42)
    results = tester.test_signal_generator(my_component, edge_data)
    assert results.error_count == 0, f"Component fails on {case}"

# ❌ BAD: Only testing ideal conditions
test_data = generator.generate_synthetic_dataset('strong_bull_low_vol', seed=42)
```

### 4. Use Regime-Specific Testing

```python
# ✅ GOOD: Regime-aware testing
tester = RegimeTester(test_data)
for regime in ['trend_up_low_vol', 'trend_down_low_vol', 'range_low_vol']:
    results = tester.test_component_in_regime(my_component, regime)
    print(f"{regime}: {results['accuracy']:.2%}")

# ❌ BAD: Ignoring regime-specific performance
tester = ComponentPerformanceTester(test_data)
results = tester.test_signal_generator(my_component)
# Missing regime-specific insights
```

### 5. Monitor Error Rates

```python
# ✅ GOOD: Check for errors
results = tester.test_signal_generator(my_component)
if results.error_count > 0:
    print(f"⚠️ Warning: {results.error_count} errors occurred")
    print(f"Error rate: {results.error_rate:.2%}")
    # Investigate and fix

# ❌ BAD: Ignoring errors
results = tester.test_signal_generator(my_component)
print(f"Accuracy: {results.accuracy:.2%}")  # May be misleadingly good if errors are ignored
```

### 6. Compare Before Replacing Components

```python
# ✅ GOOD: Validate improvement before replacing
analyzer = PerformanceAttributionAnalyzer(test_data)
impact = analyzer.analyze_component_replacement(
    baseline_strategy=current_strategy,
    new_signal_generator=new_component
)

if impact.performance_delta > 0.03:  # 3% improvement threshold
    print("✅ Replacement approved")
    # Update strategy
else:
    print("⚠️ Replacement not justified")

# ❌ BAD: Replacing without validation
my_strategy.signal_generator = new_component  # Hope for the best
```

---

## Integration with Strategy System

### How This Fits in the Strategy Redesign

The Component Testing Framework is **Task 9** in the strategy redesign implementation plan:

```
Phase 1: Core Components (Tasks 1-6) ✅ COMPLETE
├── Component interfaces and base classes
├── Legacy adapters for backward compatibility
├── Signal generators, risk managers, position sizers
└── Signal combination strategies

Phase 2: Strategy Orchestration (Tasks 7-8) ✅ COMPLETE
├── Composable Strategy class
├── Strategy manager with versioning
├── Performance tracking
└── Strategy lineage and evolution

Phase 3: Component Testing Framework (Task 9) ✅ COMPLETE ← YOU ARE HERE
├── ComponentPerformanceTester
├── RegimeTester
├── PerformanceAttributionAnalyzer
└── TestDatasetGenerator

Phase 4: Performance Monitoring (Tasks 10-12) 🚧 IN PROGRESS
├── Performance degradation detection
├── Automatic strategy switching
└── Migration utilities
```

### Integration Example

```python
# Complete workflow: Develop → Test → Deploy

# 1. Develop components
signal_gen = MLSignalGenerator()
risk_mgr = VolatilityRiskManager()
pos_sizer = ConfidenceWeightedSizer()

# 2. Test components individually
generator = TestDatasetGenerator()
test_data = generator.generate_synthetic_dataset("moderate_bull_low_vol", seed=42)
tester = ComponentPerformanceTester(test_data)

sig_results = tester.test_signal_generator(signal_gen)
risk_results = tester.test_risk_manager(risk_mgr)
sizing_results = tester.test_position_sizer(pos_sizer)

# 3. Compose into strategy
from src.strategies.components.strategy import Strategy

strategy = Strategy(
    name="my_optimized_strategy",
    signal_generator=signal_gen,
    risk_manager=risk_mgr,
    position_sizer=pos_sizer,
    regime_detector=EnhancedRegimeDetector()
)

# 4. Test complete strategy
strategy_results = tester.test_all_components(
    signal_generator=signal_gen,
    risk_manager=risk_mgr,
    position_sizer=pos_sizer
)

# 5. Analyze attribution
attribution = analyzer.analyze_strategy_attribution(strategy)

# 6. Register in strategy manager (if performance is good)
from src.strategies.components.strategy_registry import StrategyRegistry

if strategy_results.overall_performance_score > 0.7:
    registry = StrategyRegistry()
    strategy_id = registry.register_strategy(
        strategy,
        metadata={
            'created_by': 'developer',
            'description': 'Optimized ML strategy with confidence weighting',
            'tags': ['ml', 'optimized', 'bull-market']
        }
    )
    print(f"✅ Strategy registered with ID: {strategy_id}")
```

---

## API Reference

### TestDatasetGenerator

```python
class TestDatasetGenerator:
    def __init__(self, data_dir: Optional[str] = None, cache_dir: Optional[str] = None)
    
    # Scenario generation
    def generate_synthetic_dataset(self, scenario_name: str, seed: Optional[int] = None) -> pd.DataFrame
    def get_all_scenarios(self) -> List[str]
    def get_scenario_description(self, scenario_name: str) -> str
    
    # Edge case generation
    def generate_edge_case_dataset(self, case_type: str, duration_days: int = 100, seed: Optional[int] = None) -> pd.DataFrame
    
    # Regime-labeled datasets
    def create_regime_labeled_dataset(self, scenario_name: str, regime_detection_method: str = "simple", seed: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame]
    
    # Comprehensive suite
    def get_comprehensive_test_suite(self, seed: Optional[int] = None) -> Dict[str, pd.DataFrame]
```

### ComponentPerformanceTester

```python
class ComponentPerformanceTester:
    def __init__(self, test_data: pd.DataFrame, regime_data: Optional[pd.DataFrame] = None)
    
    def test_signal_generator(self, generator: SignalGenerator, scenarios: Optional[List[str]] = None) -> SignalTestResults
    def test_risk_manager(self, risk_manager: RiskManager, test_balance: float = 10000.0, scenarios: Optional[List[str]] = None) -> RiskTestResults
    def test_position_sizer(self, position_sizer: PositionSizer, test_balance: float = 10000.0, scenarios: Optional[List[str]] = None) -> SizingTestResults
    def test_all_components(self, signal_generator, risk_manager, position_sizer, test_balance: float = 10000.0) -> ComponentTestResults
```

### RegimeTester

```python
class RegimeTester:
    def __init__(self, test_data: pd.DataFrame, regime_detection_params: Optional[Dict[str, Any]] = None)
    
    def test_strategy_in_regime(self, strategy: Strategy, regime_type: str, initial_balance: float = 10000.0) -> RegimeTestResults
    def test_component_in_regime(self, component, regime_type: str) -> Dict[str, Any]
    def compare_strategy_across_regimes(self, strategy: Strategy, regime_types: Optional[List[str]] = None, initial_balance: float = 10000.0) -> RegimeComparisonResults
    def analyze_regime_transitions(self, test_data: pd.DataFrame) -> RegimeTransitionAnalysis
    def get_regime_statistics(self, test_data: pd.DataFrame) -> Dict[str, Any]
```

### PerformanceAttributionAnalyzer

```python
class PerformanceAttributionAnalyzer:
    def __init__(self, test_data: pd.DataFrame, regime_data: Optional[pd.DataFrame] = None)
    
    def analyze_strategy_attribution(self, strategy: Strategy, initial_balance: float = 10000.0) -> AttributionReport
    def analyze_component_replacement(self, baseline_strategy: Strategy, new_signal_generator=None, new_risk_manager=None, new_position_sizer=None) -> ReplacementImpact
```

---

## Examples

### Example 1: Complete Component Development Cycle

See `examples/component_testing_example.py` for a complete example demonstrating:

- Generating test datasets
- Testing signal generators
- Testing risk managers
- Testing position sizers
- Regime-specific testing
- Performance attribution analysis

### Example 2: Optimizing an Existing Strategy

```python
# Load historical data
from src.data_providers.binance_provider import BinanceDataProvider

provider = BinanceDataProvider()
historical_data = provider.get_historical_data("BTCUSDT", "1h", days=90)

# Test current strategy
current_strategy = my_existing_strategy
tester = ComponentPerformanceTester(historical_data)
baseline_results = tester.test_all_components(
    signal_generator=current_strategy.signal_generator,
    risk_manager=current_strategy.risk_manager,
    position_sizer=current_strategy.position_sizer
)

print(f"Baseline Score: {baseline_results.overall_performance_score:.2f}")

# Try alternative components
alternative_sizers = [
    KellySizer(),
    ConfidenceWeightedSizer(),
    RegimeAdaptiveSizer()
]

for sizer in alternative_sizers:
    results = tester.test_position_sizer(sizer)
    print(f"{sizer.name}: {results.optimal_sizing_score:.2f}")
    
# Use best performer
# Update strategy with best component
```

---

## Troubleshooting

### Issue: "Test data missing required columns"

**Solution**: Ensure your test data has all required OHLCV columns:

```python
required_columns = ['open', 'high', 'low', 'close', 'volume']
assert all(col in test_data.columns for col in required_columns)
```

### Issue: "No valid signals generated during testing"

**Solution**: Check that your signal generator doesn't fail on all test data:

```python
# Add error handling to your signal generator
def generate_signal(self, df, index, regime=None):
    try:
        # Your logic
        return Signal(...)
    except Exception as e:
        logger.error(f"Signal generation failed: {e}")
        # Return neutral signal as fallback
        return Signal(SignalDirection.HOLD, 0.0, 0.5, {})
```

### Issue: "High error count in test results"

**Solution**: Check component validation and input handling:

```python
# Ensure your component validates inputs
def generate_signal(self, df, index, regime=None):
    if index < 0 or index >= len(df):
        raise IndexError(f"Index {index} out of bounds")
    
    if index < 50:  # Need minimum history
        return Signal(SignalDirection.HOLD, 0.0, 0.3, {'reason': 'insufficient_history'})
    
    # Your logic here
```

---

## Performance Considerations

### Test Data Size

- **Small tests** (50-100 periods): Quick feedback, good for development
- **Medium tests** (250-500 periods): Standard testing, good coverage
- **Large tests** (1000+ periods): Comprehensive, slower but thorough

```python
# Fast development testing
quick_test = generator.generate_synthetic_dataset("bull_market", duration_days=50, seed=42)

# Standard testing
standard_test = generator.generate_synthetic_dataset("bull_market", duration_days=252, seed=42)

# Comprehensive testing
full_test = generator.generate_synthetic_dataset("bull_market", duration_days=1000, seed=42)
```

### Caching

The framework automatically caches generated datasets:

```python
# First call generates and caches
data1 = generator.generate_synthetic_dataset("bull_market", seed=42)  # ~100ms

# Second call loads from cache
data2 = generator.generate_synthetic_dataset("bull_market", seed=42)  # ~10ms
```

### Parallel Testing

For testing multiple components, consider parallel execution:

```python
from concurrent.futures import ProcessPoolExecutor

def test_component(component):
    generator = TestDatasetGenerator()
    test_data = generator.generate_synthetic_dataset("moderate_bull_low_vol", seed=42)
    tester = ComponentPerformanceTester(test_data)
    return tester.test_signal_generator(component)

components = [gen1, gen2, gen3, gen4]

with ProcessPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(test_component, components))
```

---

## Contributing

When adding new test capabilities:

1. **Add new scenarios** to `MarketScenario` definitions in `TestDatasetGenerator`
2. **Add new edge cases** by creating `_create_*_case` methods
3. **Add new metrics** to the appropriate `*TestResults` dataclass
4. **Update this README** with examples of your new features

---

## Related Documentation

- **Strategy System Design**: `.kiro/specs/strategy-system-redesign/design.md`
- **Component Interfaces**: `src/strategies/components/README.md`
- **Strategy Manager**: `src/strategies/components/strategy_registry.py`
- **Example Usage**: `examples/component_testing_example.py`

---

## Support

For questions or issues with the testing framework:

1. Check this README and the examples
2. Review the design documentation in `.kiro/specs/strategy-system-redesign/`
3. Look at existing tests in `tests/strategies/components/test_component_testing_framework.py`
4. Check the main strategy system documentation

---

**Last Updated**: 2025-10-26  
**Version**: 1.0.0  
**Status**: Production Ready