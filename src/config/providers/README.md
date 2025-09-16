# Configuration Providers

Environment-specific configuration providers for the configuration system.

## Providers

- **Railway Provider**: Railway environment variables (production)
- **Environment Provider**: System environment variables
- **DotEnv Provider**: `.env` file support (development)

## Priority Order

1. Railway environment variables (highest priority)
2. System environment variables
3. .env file (lowest priority)

## Usage

Configuration providers are automatically initialized by the `ConfigManager` and should not be used directly. Use the config manager instead:

```python
from config import get_config

config = get_config()
value = config.get_required("API_KEY")
```