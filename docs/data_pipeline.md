# Data pipeline

> **Last Updated**: 2026-01-12  
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

`CachedDataProvider` wraps any market provider and persists yearly partitions as `.parquet` files (see
`src/data_providers/cached_data_provider.py`). Each partition uses a deterministic hash-based filename so the CLI cache tools can
identify duplicates quickly even when multiple processes are warming the cache. Cached entries remain valid forever for completed
calendar years and respect a configurable TTL (24 hours by default) for the current year. If the default cache directory cannot be
created, the helper falls back to a project-local temporary directory instead of silently disabling caching.

```python
from datetime import UTC, datetime, timedelta

from src.data_providers.binance_provider import BinanceProvider
from src.data_providers.cached_data_provider import CachedDataProvider

provider = CachedDataProvider(BinanceProvider(), cache_ttl_hours=24)
start = datetime.now(UTC) - timedelta(days=90)
end = datetime.now(UTC)
df = provider.get_historical_data("BTCUSDT", "1h", start, end)
```

Cache metadata (file count, disk usage, entry age) is exposed through `get_cache_info()` and surfaced by the CLI cache manager.

## CLI utilities

The `atb data` command family in `cli/commands/data.py` covers the most common workflows:

- `atb data download BTCUSDT --timeframe 1h --start_date 2024-01-01 --end_date 2024-03-01` – export a CSV/Feather dataset via CCXT without
  touching the cache (both `--start_date` and `--end_date` are optional; omit `--end_date` to pull through “now”). Note that Binance
  may block the CCXT endpoint from restricted locations; if this command fails, use `prefill-cache`/`preload-offline` instead because
  those flows go through `BinanceProvider` and select a geo-appropriate endpoint automatically.
- `atb data prefill-cache --symbols BTCUSDT ETHUSDT --timeframes 1h 4h --years 3` – eagerly fetches year chunks so backtests can
  run offline.
- `atb data preload-offline --symbols BTCUSDT --timeframes 1h --years-back 10 --test-offline` – ensures the cache contains enough
  history for air-gapped environments, optionally forcing refreshes with `--force-refresh`, and verifies offline reads when
  `--test-offline` is set.
- `atb data cache-manager info|list|clear|clear-old` – inspect, reset, or prune cached files. The commands reuse
  `CachedDataProvider` instrumentation and normalise output sizes/timestamps for easier monitoring. Note that `info` reflects the
  current `.parquet` cache format; `list`/`clear`/`clear-old` may not show `.parquet` files in all environments, so deleting
  `cache/market_data/*.parquet` (or passing `--cache-dir` and deleting that directory’s `.parquet` files) is the most reliable
  cleanup option today.
- `atb data populate-dummy --trades 100 --confirm` – write deterministic mock trades/positions into PostgreSQL so dashboards have
  data even before the first real session runs.

All subcommands honour the `--cache-dir` flag so CI and containerised deployments can isolate cache storage.

## ETF net-flow signal (#803)

US spot BTC/ETH ETF net flows have been the marginal price-setter this cycle — multi-day
outflow streaks have led/tracked price legs. `ETFFlowProvider`
(`src/data_providers/etf_flow_provider.py`) makes flows available to the bot:

- **Ingest & cache**: daily net flows are fetched via a pluggable `fetch_fn` (default targets
  Farside Investors) and cached to `cache/etf_flows/etf_flows.parquet` with an atomic
  temp+`os.replace` write. Resolution order is **fresh cache → upstream fetch → stale cache →
  bundled seed** (`src/data/etf_flows_seed.csv`), so a trading loop never hard-fails on a
  source outage — it degrades to last-known/neutral flows and logs.
- **Features**: `compute_flow_features(...)` returns the 5d/20d net-flow **z-scores** (the
  z-score of the W-day rolling-mean flow standardized against its recent history, so a
  sustained outflow regime prints strongly negative) and the **consecutive-outflow-day** count.
- **Gate** (rule-based, active today): `FlowGatedSignalGenerator`
  (`src/strategies/components/flow_gate.py`) wraps a strategy's signal generator and turns a
  BUY into HOLD while the 5-day z-score is below the block threshold (default -1.0). It is a
  signal-generator decorator, so it applies in **both** the backtest and live engines through
  the strategy with no per-engine wiring. SELL/HOLD pass through; unknown flow does **not**
  block. Enabled via `FEATURE_ENABLE_ETF_FLOW_GATE` (default OFF).
- **Model feature** (inert until retrain): `ETFFlowFeatureExtractor`
  (`src/prediction/features/etf_flow.py`) exposes the same features as optional model inputs.
  Because it changes the model's feature schema, existing ONNX models cannot consume it until
  retrained (#801, human sign-off) — registered only behind `etf_flows_features.enabled`,
  default off.

> The bundled seed and the default Farside `fetch_fn` are best-effort; operators with API
> access should override `fetch_fn` with an authoritative client. The seed is illustrative
> data for offline/CI runs, not an authoritative flow record.
