# Performance Parity Validation System

This module provides comprehensive tools for validating performance parity between legacy and new strategy implementations during migration. It ensures that migrated strategies maintain equivalent performance within acceptable tolerances.

## Overview

The Performance Parity Validation System consists of three main components:

1. **Performance Parity Validator** - Core validation engine for comparing strategy performance
2. **Statistical Tests** - Comprehensive statistical analysis tools for financial time series
3. **Performance Comparison Engine** - High-level orchestration for complete validation workflows

## Key Features

- **Comprehensive Metric Comparison** - Validates return, risk, efficiency, and timing metrics
- **Statistical Significance Testing** - Performs distribution, mean, and variance equality tests
- **Equivalence Testing** - Uses Two One-Sided Test (TOST) for practical equivalence
- **Configurable Tolerances** - Customizable acceptance criteria for different validation scenarios
- **Detailed Reporting** - Generates certification reports with actionable recommendations
- **Export Capabilities** - Exports results to CSV and JSON formats for further analysis

## Quick Start

### Basic Usage

```python
from src.strategies.components.testing import (
    PerformanceParityValidator,
    ToleranceConfig,
    ValidationResult
)

# Create validator with default tolerances
validator = PerformanceParityValidator()

# Validate performance parity
report = validator.validate_performance_parity(
    legacy_results=legacy_backtest_df,
    new_results=new_backtest_df,
    strategy_name="ML Basic Strategy",
    legacy_strategy_id="ml_basic_v1",
    new_strategy_id="ml_basic_v2"
)

# Check results
if report.overall_result == ValidationResult.PASS:
    print("✓ Performance parity validated - migration approved")
else:
    print("✗ Performance parity failed - review required")
```

### Custom Tolerances

```python
# Configure custom tolerances for strict validation
strict_tolerances = ToleranceConfig(
    total_return_tolerance=0.01,      # 1% max difference in total return
    sharpe_ratio_tolerance=0.05,      # 0.05 max difference in Sharpe ratio
    minimum_correlation=0.98,         # 98% minimum equity curve correlation
    max_drawdown_tolerance=0.005      # 0.5% max difference in drawdown
)

validator = PerformanceParityValidator(strict_tolerances)
```

### Convenience Functions

```python
from src.strategies.components.testing import (
    quick_strategy_comparison,
    validate_migration_readiness
)

# Quick comparison with default settings
result = quick_strategy_comparison(
    legacy_strategy, new_strategy, market_data
)

# Migration readiness check
is_ready, issues = validate_migration_readiness(
    legacy_strategy, new_strategy, market_data, strict_validation=True
)
```

## Components

### 1. Performance Parity Validator

The core validation engine that compares performance metrics between strategies.

#### Key Classes

- `PerformanceParityValidator` - Main validation engine
- `ToleranceConfig` - Configuration for acceptable performance differences
- `PerformanceComparisonReport` - Comprehensive validation results
- `MetricComparison` - Individual metric comparison results

#### Validated Metrics

**Return Metrics:**
- Total Return
- Compound Annual Growth Rate (CAGR)

**Risk Metrics:**
- Maximum Drawdown
- Annualized Volatility

**Efficiency Metrics:**
- Sharpe Ratio
- Win Rate (if trade data available)

**Timing Metrics:**
- Trade Count
- Equity Curve Correlation

#### Example

```python
# Create validator
validator = PerformanceParityValidator()

# Run validation
report = validator.validate_performance_parity(
    legacy_results, new_results, "Strategy Name", "legacy_id", "new_id"
)

# Generate certification report
cert_report = validator.generate_certification_report(report)
print(cert_report)
```

### 2. Statistical Tests

Comprehensive statistical analysis tools specifically designed for financial time series.

#### Key Classes

- `FinancialStatisticalTests` - Main statistical testing engine
- `EquivalenceTests` - Specialized equivalence testing
- `StatisticalTestResult` - Individual test results

#### Available Tests

**Distribution Tests:**
- Kolmogorov-Smirnov Two-Sample Test
- Anderson-Darling k-Sample Test

**Mean Equality Tests:**
- Welch's t-test (unequal variances)
- Mann-Whitney U Test (non-parametric)

**Variance Equality Tests:**
- Levene's Test
- Bartlett's Test

**Normality Tests:**
- Shapiro-Wilk Test
- Jarque-Bera Test
- D'Agostino's Test

**Time Series Tests:**
- Ljung-Box Autocorrelation Test (requires statsmodels)
- Augmented Dickey-Fuller Stationarity Test (requires statsmodels)
- KPSS Stationarity Test (requires statsmodels)

**Equivalence Tests:**
- Two One-Sided Test (TOST)

#### Example

```python
from src.strategies.components.testing.statistical_tests import (
    FinancialStatisticalTests,
    EquivalenceTests
)

# Initialize test engines
stat_tests = FinancialStatisticalTests(significance_level=0.05)
equiv_tests = EquivalenceTests(equivalence_margin=0.01)

# Run comprehensive comparison
results = stat_tests.comprehensive_comparison(
    returns1, returns2, "Strategy A", "Strategy B"
)

# Test for practical equivalence
tost_result = equiv_tests.two_one_sided_test(returns1, returns2)
```

### 3. Performance Comparison Engine

High-level orchestration engine for complete validation workflows including backtesting.

#### Key Classes

- `PerformanceComparisonEngine` - Main orchestration engine
- `ComparisonConfig` - Configuration for comparison process
- `StrategyComparisonResult` - Complete comparison results

#### Features

- **Automated Backtesting** - Runs backtests for both strategies
- **Comprehensive Analysis** - Combines parity validation and statistical testing
- **Result Export** - Automatically exports results to files
- **Certification** - Provides migration readiness assessment

#### Example

```python
from src.strategies.components.testing import (
    PerformanceComparisonEngine,
    ComparisonConfig
)

# Configure comparison
config = ComparisonConfig(
    initial_balance=10000.0,
    export_results=True,
    export_directory="validation_results"
)

# Create engine (requires backtest engine)
engine = PerformanceComparisonEngine(config, backtest_engine)

# Run complete comparison
result = engine.compare_strategies(
    legacy_strategy, new_strategy, market_data
)

# Check certification status
print(f"Certification: {result.certification_status}")
```

## Configuration

### Tolerance Configuration

The `ToleranceConfig` class allows customization of validation criteria:

```python
config = ToleranceConfig(
    # Return metrics tolerances (absolute)
    total_return_tolerance=0.02,      # 2% difference allowed
    cagr_tolerance=0.02,              # 2% difference allowed
    
    # Risk metrics tolerances
    max_drawdown_tolerance=0.01,      # 1% difference allowed
    volatility_tolerance=0.05,        # 5% relative difference allowed
    
    # Efficiency metrics tolerances
    sharpe_ratio_tolerance=0.1,       # 0.1 absolute difference allowed
    win_rate_tolerance=0.05,          # 5% difference allowed
    
    # Statistical requirements
    statistical_significance_level=0.05,  # 5% significance level
    minimum_sample_size=30,               # Minimum trades for tests
    minimum_correlation=0.95,             # 95% correlation required
    
    # Trade-level tolerances
    trade_count_tolerance=0.1,            # 10% difference in trade count
    avg_trade_duration_tolerance=0.2      # 20% difference in duration
)
```

### Comparison Configuration

The `ComparisonConfig` class configures the complete comparison process:

```python
config = ComparisonConfig(
    # Validation settings
    tolerance_config=custom_tolerance_config,
    statistical_significance_level=0.05,
    equivalence_margin=0.01,
    
    # Backtesting settings
    initial_balance=10000.0,
    commission_rate=0.001,
    
    # Export settings
    generate_detailed_report=True,
    export_results=True,
    export_directory="validation_results",
    
    # Validation requirements
    require_statistical_equivalence=True,
    require_performance_parity=True,
    minimum_correlation_threshold=0.95
)
```

## Validation Results

### Validation Status

The system provides four validation results:

- `ValidationResult.PASS` - All metrics within tolerance, migration approved
- `ValidationResult.WARNING` - Some metrics outside tolerance but acceptable
- `ValidationResult.FAIL` - Critical metrics failed, migration not recommended
- `ValidationResult.INCONCLUSIVE` - Insufficient data or test failures

### Certification Levels

- **CERTIFIED** - Full performance parity validated
- **CONDITIONAL** - Performance parity with minor concerns
- **NOT CERTIFIED** - Performance parity failed

### Report Contents

Each validation generates a comprehensive report containing:

- Overall validation result and certification status
- Individual metric comparisons with pass/fail status
- Statistical test results and p-values
- Equity curve correlation analysis
- Detailed recommendations for next steps
- Export capabilities for further analysis

## Export Formats

### CSV Export

Exports metric comparisons in tabular format:

```python
from src.strategies.components.testing.performance_parity_validator import (
    PerformanceParityReporter
)

PerformanceParityReporter.export_to_csv(report, "validation_metrics.csv")
```

### JSON Export

Exports complete report with all details:

```python
PerformanceParityReporter.export_to_json(report, "validation_report.json")
```

## Best Practices

### 1. Tolerance Configuration

- **Strict Validation**: Use tight tolerances (1-2%) for critical production migrations
- **Development Validation**: Use moderate tolerances (3-5%) for development iterations
- **Research Validation**: Use loose tolerances (5-10%) for experimental comparisons

### 2. Data Requirements

- **Minimum Sample Size**: Ensure at least 30 trades for statistical significance
- **Temporal Overlap**: Validate strategies on the same time periods
- **Data Quality**: Clean data with consistent timestamps and balance calculations

### 3. Interpretation Guidelines

- **High Correlation (>95%)**: Strategies behave very similarly
- **Medium Correlation (85-95%)**: Strategies have similar patterns with some differences
- **Low Correlation (<85%)**: Strategies have fundamentally different behaviors

### 4. Migration Decisions

- **PASS + High Correlation**: Safe to migrate immediately
- **WARNING + Medium Correlation**: Migrate with monitoring
- **FAIL or Low Correlation**: Investigate differences before migration

## Dependencies

### Required

- `pandas` - Data manipulation and analysis
- `numpy` - Numerical computations
- `scipy` - Statistical functions

### Optional

- `statsmodels` - Advanced time series tests (autocorrelation, stationarity)

## Examples

See `examples/performance_parity_validation_example.py` for comprehensive usage examples including:

- Basic validation workflow
- Custom tolerance configuration
- Statistical analysis capabilities
- Export functionality
- Error handling and edge cases

## Testing

The module includes comprehensive unit tests:

```bash
# Run all tests
python -m pytest tests/unit/strategies/components/test_performance_parity_validator.py
python -m pytest tests/unit/strategies/components/test_statistical_tests.py
python -m pytest tests/unit/strategies/components/test_performance_comparison_engine.py

# Run specific test
python -m pytest tests/unit/strategies/components/test_performance_parity_validator.py::TestPerformanceParityValidator::test_full_validation_workflow -v
```

## Integration

### With Existing Backtesting System

To integrate with the existing `Backtester` class:

```python
class BacktestAdapter:
    """Adapter for existing Backtester class."""
    
    def __init__(self, data_provider, sentiment_provider=None):
        self.data_provider = data_provider
        self.sentiment_provider = sentiment_provider
    
    def run_backtest(self, strategy, data, **kwargs):
        from src.backtesting.engine import Backtester
        
        backtester = Backtester(
            strategy=strategy,
            data_provider=self.data_provider,
            sentiment_provider=self.sentiment_provider,
            initial_balance=kwargs.get('initial_balance', 10000.0)
        )
        
        # Extract parameters from data
        symbol = kwargs.get('symbol', 'BTCUSDT')
        timeframe = kwargs.get('timeframe', '1d')
        start = data['timestamp'].min()
        end = data['timestamp'].max()
        
        # Run backtest
        results = backtester.run(symbol, timeframe, start, end)
        
        # Convert to expected format
        return self._convert_results(results)
    
    def _convert_results(self, results):
        # Convert backtester results to expected DataFrame format
        # Implementation depends on actual Backtester output format
        pass
```

### With Strategy Migration System

```python
from src.strategies.components.testing import validate_migration_readiness

def migrate_strategy(legacy_strategy, new_strategy, market_data):
    """Safe strategy migration with validation."""
    
    # Validate migration readiness
    is_ready, issues = validate_migration_readiness(
        legacy_strategy, new_strategy, market_data, strict_validation=True
    )
    
    if not is_ready:
        print("Migration not ready:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    
    # Proceed with migration
    print("✓ Migration validated - proceeding with deployment")
    return True
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **Insufficient Data**: Provide at least 30 data points for statistical tests
3. **Missing Columns**: Ensure DataFrames have 'timestamp' and 'balance' columns
4. **No Temporal Overlap**: Verify strategies tested on overlapping time periods

### Performance Considerations

- Large datasets may require sampling for statistical tests
- Export operations can be memory-intensive for very large reports
- Statistical tests scale with O(n log n) complexity

## Contributing

When extending the validation system:

1. Add new metrics to appropriate metric type categories
2. Include comprehensive unit tests for new functionality
3. Update tolerance configurations for new metrics
4. Document new features in this README
5. Provide usage examples for complex features