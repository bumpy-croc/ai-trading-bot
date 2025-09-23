# Utilities

Shared helpers and cross-cutting utilities.

## Modules
- `symbol_factory.py`: symbol normalization (e.g., BTCUSDT)
- `logging_config.py`: centralized logging configuration

Note: Path helpers are in `config/paths.py`.

## Usage
```python
from src.utils.symbol_factory import SymbolFactory
print(SymbolFactory.to_binance("BTC-USD"))  # BTCUSDT

from src.utils.logging_config import configure_logging
configure_logging()  # Sets up structured logging
```
