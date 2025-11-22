# Trading Core

Minimal trading-domain helpers that sit outside the strategy/component system.

## Overview

The `src/trading` package currently focuses on symbol normalization so every subsystem (data providers, backtester, live engine, CLI)
can speak a consistent ticker format. Strategy composition, risk, and component interfaces now live under `src/strategies/components`,
so this directory intentionally stays lightweight.

## Modules

- `symbols/factory.py` – `SymbolFactory` converts between generic tickers (e.g., `BTC-USD`) and exchange-specific formats such as
  `BTCUSDT` (Binance) or `BTC-USD` (Coinbase). The CLI, providers, and engines all reuse this helper to avoid bespoke conversions.

## Usage

```python
from src.trading.symbols.factory import SymbolFactory

# Normalize user input for Binance backtests
symbol = SymbolFactory.to_exchange_symbol("btc-usd", "binance")
# -> 'BTCUSDT'

# Convert cached Binance symbols back to Coinbase style
coinbase_symbol = SymbolFactory.to_exchange_symbol(symbol, "coinbase")
# -> 'BTC-USD'
```

## See Also

- `src/strategies/README.md` – Component-based strategy architecture
- `docs/backtesting.md` – How symbols flow through the simulation engine
- `docs/live_trading.md` – Live engine configuration, including provider selection
