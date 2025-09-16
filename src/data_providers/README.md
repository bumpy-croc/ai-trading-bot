# Data Providers

Abstractions and implementations for market and sentiment data.

## Modules
- `binance_provider.py`, `coinbase_provider.py`: Exchange price/candle data
- `feargreed_provider.py`: Sentiment data
- `cached_data_provider.py`: File-based caching wrapper
- `data_provider.py`, `exchange_interface.py`: Base interfaces

## Usage
```python
from data_providers.binance_provider import BinanceProvider
from data_providers.cached_data_provider import CachedDataProvider

provider = CachedDataProvider(BinanceProvider(), cache_ttl_hours=24)
df = provider.get_historical_data(symbol="BTCUSDT", timeframe="1h", start=..., end=...)
```
