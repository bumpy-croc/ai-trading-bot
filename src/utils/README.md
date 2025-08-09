# Utilities

Shared helpers and cross-cutting utilities.

## Modules
- `paths.py` (in `config/paths.py`): path helpers (cache, data)
- `symbol_factory.py`: symbol normalization (e.g., BTCUSDT)

## Usage
```python
from src.utils.symbol_factory import SymbolFactory
print(SymbolFactory.to_binance("BTC-USD"))  # BTCUSDT
```