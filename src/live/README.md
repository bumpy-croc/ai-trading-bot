# Live Trading Engine

> **Last Updated**: 2025-10-24  
> **Related Documentation**: See [docs/live_trading.md](../../docs/live_trading.md) for comprehensive guide and safety controls

Executes strategies in real time with risk controls, data providers, and database logging.

## CLI
```bash
# Paper trading (safe)
atb live ml_basic --symbol BTCUSDT --paper-trading

# Live trading (explicit confirmation required)
atb live ml_basic --symbol BTCUSDT --live-trading --i-understand-the-risks

# Live trading with health endpoint
atb live-health --port 8000 -- ml_basic --symbol BTCUSDT --paper-trading
```

## Programmatic
```python
from src.live.trading_engine import LiveTradingEngine
from src.strategies.ml_basic import create_ml_basic_strategy
from src.data_providers.binance_provider import BinanceProvider

engine = LiveTradingEngine(
    strategy=create_ml_basic_strategy(),
    data_provider=BinanceProvider(),
    initial_balance=10000,
)
engine.start("BTCUSDT", "1h")
```
