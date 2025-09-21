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
# Run parameter optimization
atb optimizer --strategy ml_basic --days 365

# Custom parameter ranges
atb optimizer --strategy ml_basic --param-file custom_ranges.json
```

## Programmatic Usage

```python
from src.optimizer import ParameterOptimizer

optimizer = ParameterOptimizer()
best_params = optimizer.optimize_strategy("ml_basic", symbol="BTCUSDT")
```