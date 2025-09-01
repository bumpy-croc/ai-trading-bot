# Configuration

Typed configuration loader with provider priority and helpful accessors.

## Providers (priority)
1. Railway environment variables
2. Environment variables
3. .env file (local development)

## Modules
- `config_manager.py`: `get_config()` and typed getters
- `constants.py`: Common defaults and constants
- `feature_flags.py`: Feature flag access
- `paths.py`: Project paths (cache, data, etc.)

## Usage
```python
from config import get_config
config = get_config()
api_key = config.get_required("BINANCE_API_KEY")
balance = config.get_int("INITIAL_BALANCE", 1000)
```
