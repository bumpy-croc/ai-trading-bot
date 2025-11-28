# Live Trading Engine

> **Last Updated**: 2025-11-10  
> **Related Documentation**: See [docs/live_trading.md](../../docs/live_trading.md) for comprehensive guide and safety controls

Executes strategies in real time with risk controls, data providers, and database logging.

## CLI
```bash
# Paper trading (safe)
atb live ml_basic --symbol BTCUSDT --paper-trading

# Live trading (explicit confirmation required)
atb live ml_basic --symbol BTCUSDT --live-trading --i-understand-the-risks

# Live trading with health endpoint (set PORT/HEALTH_CHECK_PORT to control HTTP port)
PORT=8000 atb live-health -- ml_basic --symbol BTCUSDT --paper-trading
```

`atb live-health` reads the `PORT` (or `HEALTH_CHECK_PORT`) environment variable, defaulting to `8000`, before launching the
embedded HTTP server.

Sentiment enrichment is currently wired manually: the CLI keeps the `--use-sentiment` flag for backwards compatibility but it
emits a warning and does not attach a provider. Pass a `SentimentDataProvider` directly when constructing the engine if you need
live sentiment features.

## Programmatic
```python
from src.data_providers.binance_provider import BinanceProvider
from src.data_providers.cached_data_provider import CachedDataProvider
from src.data_providers.feargreed_provider import FearGreedProvider
from src.live.trading_engine import LiveTradingEngine
from src.strategies.ml_basic import create_ml_basic_strategy

engine = LiveTradingEngine(
    strategy=create_ml_basic_strategy(),
    data_provider=CachedDataProvider(BinanceProvider(), cache_ttl_hours=1),
    sentiment_provider=FearGreedProvider(),
    initial_balance=10000,
)
engine.start("BTCUSDT", "1h")
```
