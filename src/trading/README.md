# Trading Core

Shared utilities and helpers for trading strategies and components.

## Overview

This module provides shared functionality used across the trading system, including symbol normalization utilities under `symbols/`. Trading strategies are built using the component-based architecture defined in `src/strategies/components/`.

## Modules

- `symbols/`: Symbol normalization helpers for exchange-specific formats (e.g., converting between `BTCUSDT` and `BTC-USD`).

## Key Components

### Symbol Utilities
Symbol conversion and validation helpers that handle exchange-specific formats:
- Convert between different exchange symbol formats
- Validate symbol strings
- Normalize symbol representations

### Shared Helpers
Common utilities used by strategies and engines including:
- Symbol normalization and validation
- Exchange-specific formatting helpers
- Data processing utilities

## Usage

```python
# Import symbol utilities
from src.trading.symbols.factory import SymbolFactory

# Convert symbol formats between exchanges
binance_symbol = SymbolFactory.to_exchange_symbol("BTC-USD", "binance")  # Returns "BTCUSDT"
coinbase_symbol = SymbolFactory.to_exchange_symbol("BTCUSDT", "coinbase")  # Returns "BTC-USD"
```

## See Also

- [strategies/README.md](../strategies/README.md) - Component-based strategy architecture
- [strategies/components/README.md](../strategies/components/README.md) - Strategy component interfaces
- [docs/backtesting.md](../../docs/backtesting.md) - Backtesting strategies
- [docs/live_trading.md](../../docs/live_trading.md) - Live trading usage
