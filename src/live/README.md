# Live Trading Engine

Executes strategies in real time with risk controls, data providers, and database logging.

## CLI
```bash
# Paper trading (safe)
python scripts/run_live_trading.py ml_basic --symbol BTCUSDT --paper-trading

# Live trading (explicit confirmation required)
python scripts/run_live_trading.py ml_basic --symbol BTCUSDT --live-trading --i-understand-the-risks
```

## Programmatic
```python
from live.trading_engine import LiveTradingEngine
from strategies.ml_basic import MlBasic
from data_providers.binance_provider import BinanceProvider

engine = LiveTradingEngine(strategy=MlBasic(), data_provider=BinanceProvider(), initial_balance=10000)
engine.start("BTCUSDT", "1h")
```
