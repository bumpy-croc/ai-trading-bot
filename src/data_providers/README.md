# Data Providers

> **Last Updated**: 2025-11-26  
> **Related Documentation**: See [docs/data_pipeline.md](../../docs/data_pipeline.md) for detailed usage

Abstractions and implementations for market and sentiment data.

## Modules
- `binance_provider.py`, `coinbase_provider.py`: Exchange price/candle data
- `feargreed_provider.py`: Sentiment data
- `cached_data_provider.py`: File-based caching wrapper
- `data_provider.py`, `exchange_interface.py`: Base interfaces

## Usage

```python
from src.data_providers.binance_provider import BinanceProvider
from src.data_providers.cached_data_provider import CachedDataProvider

provider = CachedDataProvider(BinanceProvider(), cache_ttl_hours=24)
df = provider.get_historical_data(symbol="BTCUSDT", timeframe="1h", start=..., end=...)
```

`CachedDataProvider` stores yearly partitions as Parquet files with deterministic hashes, keeps prior full years permanently valid,
and enforces the configured TTL (24 hours by default) for the current year. Pass `cache_dir` to align with the CLI cache manager
(`atb data cache-manager ...`)—`cache-manager info` reads the Parquet stats today, while the `list`/`clear`/`clear-old` subcommands still
target the legacy `.pkl` cache files. Use `CachedDataProvider.clear_cache()` (or delete the `.parquet` files) when you need to purge the
newer cache layout.
