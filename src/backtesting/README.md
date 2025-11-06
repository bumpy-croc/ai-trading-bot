# Backtesting Engine

> **Last Updated**: 2025-11-06  
> **Related Documentation**: See [docs/backtesting.md](../../docs/backtesting.md) for comprehensive guide

Vectorized historical simulation engine for evaluating strategies.

## Highlights
- Plugs into data providers and strategies
- Optional on-disk data caching via `CachedDataProvider`
- Can log results to PostgreSQL

## CLI
```bash
atb backtest ml_basic --symbol BTCUSDT --timeframe 1h --days 90
```

## Programmatic
```python
from src.backtesting.engine import Backtester

results = Backtester(...).run(symbol="BTCUSDT", timeframe="1h", start=..., end=...)
print(results["session_id"])  # if logging enabled
```
