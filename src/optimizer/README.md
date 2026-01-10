# Optimizer

Parameter optimization and strategy tuning tools for systematic strategy improvement.

## Overview

The optimizer module provides tools for systematically finding optimal strategy parameters through backtesting across parameter ranges. It supports grid search, parameter validation, and performance analysis.

## Features

- **Grid search optimization** - Systematically test parameter combinations
- **Parameter range testing** - Define min/max ranges for each parameter
- **Performance metric optimization** - Optimize for Sharpe ratio, return, or drawdown
- **Strategy parameter tuning** - Fine-tune strategy-specific parameters
- **Results analysis** - Compare performance across parameter combinations
- **Validation and schemas** - Type-safe parameter definitions

## Modules

- `runner.py` - Main optimization runner with parallel execution support
- `analyzer.py` - Performance analysis and comparison tools
- `validator.py` - Parameter validation and constraint checking
- `schemas.py` - Parameter schemas and type definitions

## CLI Usage

```bash
# Run parameter optimization with defaults (ml_basic strategy, BTCUSDT, 30 days)
atb optimizer --strategy ml_basic

# Customize optimization parameters
atb optimizer --strategy ml_basic --symbol BTCUSDT --days 365

# Specify provider and caching
atb optimizer --provider binance --no-cache

# Persist results to database
atb optimizer --strategy ml_basic --persist

# Disable validation step
atb optimizer --strategy ml_basic --no-validate
```

## Programmatic Usage

```python
from datetime import datetime, timedelta
from src.optimizer.runner import ExperimentRunner
from src.optimizer.schemas import ExperimentConfig, ParameterSet
from src.optimizer.analyzer import PerformanceAnalyzer

# Configure experiment
end = datetime.now(UTC)
start = end - timedelta(days=365)

config = ExperimentConfig(
    strategy_name='ml_basic',
    symbol='BTCUSDT',
    timeframe='1h',
    start=start,
    end=end,
    initial_balance=10000,
    risk_parameters={},
    feature_flags={},
    use_cache=True,
    provider='binance'
)

# Run experiment
runner = ExperimentRunner()
result = runner.run(config)

print(f"Total Return: {result.total_return:.2f}%")
print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
print(f"Max Drawdown: {result.max_drawdown:.2f}%")

# Analyze results and get suggestions
analyzer = PerformanceAnalyzer()
suggestions = analyzer.analyze([result])

for suggestion in suggestions:
    print(f"Suggestion: {suggestion.rationale}")
    print(f"Changes: {suggestion.change}")
    print(f"Confidence: {suggestion.confidence:.2f}")
```

## Parameter Optimization

The optimizer analyzes strategy performance and suggests parameter adjustments:

```python
# Test with custom parameters
config_with_params = ExperimentConfig(
    strategy_name='ml_basic',
    symbol='BTCUSDT',
    timeframe='1h',
    start=start,
    end=end,
    initial_balance=10000,
    parameters=ParameterSet(
        name='custom',
        values={
            'MlBasic.stop_loss_pct': 0.03,
            'MlBasic.take_profit_pct': 0.06
        }
    ),
    use_cache=True
)

result_custom = runner.run(config_with_params)
```

## Documentation

See [docs/backtesting.md](../../docs/backtesting.md#optimisation-loop) for detailed information on optimizer features and usage
patterns.
