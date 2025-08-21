# Performance Utilities

Utilities for computing common performance metrics used across backtesting, live trading, and monitoring.

## Contents
- `metrics.py`: Sharpe ratio, max drawdown, and other portfolio metrics

## Usage
```python
from performance.metrics import perf_sharpe, perf_max_drawdown

ratio = perf_sharpe(daily_balance_series)
dd = perf_max_drawdown(daily_balance_series)
```
