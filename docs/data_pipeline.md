# Data pipeline

> **Last Updated**: 2025-12-21  
> **Related Documentation**: [Backtesting](backtesting.md), [Configuration](configuration.md)

Market, sentiment, and cached data access lives under `src/data_providers`. The system exposes a consistent `DataProvider`
interface so engines and CLI commands can swap sources without changing call sites.

## Market data providers

- `BinanceProvider` (`src/data_providers/binance_provider.py`) fetches OHLCV candles via the official REST API. It supports live
  sampling (`get_live_data`) and historical range queries (`get_historical_data`).
- `CoinbaseProvider` offers the same contract for Coinbase spot markets.
- `MockDataProvider` supplies deterministic candles for integration tests and reproducible optimisation runs.

The providers normalise symbols via `SymbolFactory` so CLI commands accept tickers like `BTC-USD` or `BTCUSDT` and map them to the
underlying exchange format.

## Sentiment data

`FearGreedProvider` (`src/data_providers/feargreed_provider.py`) downloads the Alternative.me Fear & Greed index and exposes
`get_historical_sentiment()` plus aggregation helpers that align the series with OHLCV data. Both the backtesting CLI (`--use-sentiment`)
and the live trading engine accept an optional `SentimentDataProvider` to enrich decisions.

## Cached access

`CachedDataProvider` wraps any market provider and persists yearly partitions as zipped Parquet files (see
`src/data_providers/cached_data_provider.py`). Each partition uses a deterministic hash-based filename so the CLI cache tools can
identify duplicates quickly even when multiple processes are warming the cache. Cached entries remain valid forever for completed
calendar years and respect a configurable TTL (24 hours by default) for the current year. If the default cache directory cannot be
created, the helper falls back to a project-local temporary directory instead of silently disabling caching.

```python
from datetime import datetime, timedelta

from src.data_providers.binance_provider import BinanceProvider
from src.data_providers.cached_data_provider import CachedDataProvider

provider = CachedDataProvider(BinanceProvider(), cache_ttl_hours=24)
start = datetime.utcnow() - timedelta(days=90)
end = datetime.utcnow()
df = provider.get_historical_data("BTCUSDT", "1h", start, end)
```

Cache metadata (file count, disk usage, entry age) is exposed through `get_cache_info()` and surfaced by the CLI cache manager.

## CLI utilities

The `atb data` command family in `cli/commands/data.py` covers the most common workflows:

- `atb data download BTCUSDT --timeframe 1h --start_date 2024-01-01 --end_date 2024-03-01` – export a CSV/Feather dataset via CCXT without
  touching the cache (both `--start_date` and `--end_date` are optional; omit `--end_date` to pull through “now”).
- `atb data prefill-cache --symbols BTCUSDT ETHUSDT --timeframes 1h 4h --years 3` – eagerly fetches year chunks so backtests can
  run offline.
- `atb data preload-offline --symbols BTCUSDT --timeframes 1h --years-back 10 --test-offline` – ensures the cache contains enough
  history for air-gapped environments, optionally forcing refreshes with `--force-refresh`, and verifies offline reads when
  `--test-offline` is set.
- `atb data cache-manager info|list|clear|clear-old` – inspect, reset, or prune cached files. The commands reuse
  `CachedDataProvider` instrumentation and normalise output sizes/timestamps for easier monitoring.
- `atb data populate-dummy --trades 100 --confirm` – write deterministic mock trades/positions into PostgreSQL so dashboards have
  data even before the first real session runs.

All subcommands honour the `--cache-dir` flag so CI and containerised deployments can isolate cache storage.
