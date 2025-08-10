# Backtesting Engine

Vectorized historical simulation engine for evaluating strategies.

## Highlights
- Plugs into data providers and strategies
- Optional on-disk data caching via `CachedDataProvider`
- Can log results to PostgreSQL

## CLI
```bash
python scripts/run_backtest.py ml_basic --symbol BTCUSDT --days 90
```

## Programmatic
```python
from backtesting.engine import Backtester

results = Backtester(...).run(symbol="BTCUSDT", timeframe="1h", start=..., end=...)
print(results["session_id"])  # if logging enabled
```