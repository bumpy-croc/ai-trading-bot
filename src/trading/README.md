# Trading Core

Helper utilities that sit next to the strategy framework. The package currently exposes the symbol-normalisation helpers used by
CLI commands, providers, and strategy factories.

## Structure

- `symbols/factory.py` – `SymbolFactory` converts between human-friendly tickers (`BTC-USD`, `ETH/USDT`) and the formats required by each exchange.
- `symbols/README.md` – usage notes for the helper.

Strategy composition now lives under `src/strategies/components/strategy.py`, so this package intentionally stays slim and focused on
shared trading utilities.

## Usage

```python
from src.trading.symbols.factory import SymbolFactory

binance_symbol = SymbolFactory.to_exchange_symbol("BTC-USD", "binance")  # BTCUSDT
coinbase_symbol = SymbolFactory.to_exchange_symbol("ETHUSDT", "coinbase")  # ETH-USD

print(binance_symbol, coinbase_symbol)
```

Pass the `SymbolFactory` output directly to data providers, strategies, or CLI commands to guarantee consistent ticker formatting across
backtesting and live trading.

## Related docs

- `src/strategies/README.md` – component-based strategy architecture
- `docs/backtesting.md` – how the engines use the trading helpers
- `docs/live_trading.md` – live runner flags and symbol requirements
