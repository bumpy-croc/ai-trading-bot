# Optimizer

Parameter optimization and strategy tuning tools.

## Overview

The optimizer module provides tools for systematically finding optimal strategy parameters through backtesting across parameter ranges.

## Features

- Grid search optimization
- Parameter range testing
- Performance metric optimization
- Strategy parameter tuning

## Usage

```bash
# Run parameter optimization with defaults (ml_basic strategy, BTCUSDT, 30 days)
atb optimizer

# Customize optimization parameters
atb optimizer --strategy ml_basic --symbol BTCUSDT --days 365

# Different data provider
atb optimizer --provider binance --no-cache
```

## Programmatic Usage

```python
from src.optimizer import ParameterOptimizer

optimizer = ParameterOptimizer()
best_params = optimizer.optimize_strategy("ml_basic", symbol="BTCUSDT")
```