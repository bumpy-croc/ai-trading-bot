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
atb optimizer run --strategy ml_basic

# Customize optimization parameters
atb optimizer run --strategy ml_basic --symbol BTCUSDT --days 365

# Specify parameter ranges
atb optimizer run --strategy ml_basic --param-ranges params.json

# Different data provider
atb optimizer run --provider binance --no-cache

# Analyze optimization results
atb optimizer analyze --results-file optimizer_results.json
```

## Programmatic Usage

```python
from src.optimizer.runner import OptimizerRunner
from src.optimizer.schemas import ParameterRange

# Define parameter ranges to test
param_ranges = {
    'risk_per_trade': ParameterRange(min=0.01, max=0.03, step=0.005),
    'stop_loss_pct': ParameterRange(min=0.02, max=0.05, step=0.01)
}

# Run optimization
optimizer = OptimizerRunner(
    strategy_name='ml_basic',
    symbol='BTCUSDT',
    timeframe='1h',
    days=365
)
results = optimizer.optimize(param_ranges)

# Analyze results
from src.optimizer.analyzer import OptimizerAnalyzer
analyzer = OptimizerAnalyzer(results)
best_params = analyzer.get_best_parameters(metric='sharpe_ratio')
print(f"Best parameters: {best_params}")
```

## Parameter Definition Format

Define parameter ranges in JSON:

```json
{
  "risk_per_trade": {
    "min": 0.01,
    "max": 0.03,
    "step": 0.005,
    "type": "float"
  },
  "stop_loss_pct": {
    "min": 0.02,
    "max": 0.05,
    "step": 0.01,
    "type": "float"
  },
  "position_size": {
    "values": [0.1, 0.15, 0.2, 0.25],
    "type": "discrete"
  }
}
```

## Documentation

See [docs/OPTIMIZER_MVP.md](../../docs/OPTIMIZER_MVP.md) for detailed information on optimizer features and usage patterns.