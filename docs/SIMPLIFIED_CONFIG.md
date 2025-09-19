# Simplified Configuration System

## Overview

The configuration system uses a multi-provider approach with priority-based fallback:
- **Railway environment variables** (production deployment)
- **Environment variables** (Docker/CI/local)
- **`.env` file** (local development)
- **Strategy parameters** defined directly in each strategy class
- **Strategy-specific trading pairs** set in each strategy constructor

## Configuration Structure

### 1. Configuration Providers (Priority Order)
The system uses multiple configuration providers handled by `ConfigManager`:

```python
from src.config import get_config

config = get_config()
api_key = config.get_required('BINANCE_API_KEY')
balance = config.get_int('INITIAL_BALANCE', 1000)
```

Available providers:
1. **Railway Provider**: For Railway deployments 
2. **Environment Variables**: For Docker/CI environments
3. **DotEnv Provider**: For local development (`.env` file)

### 2. Environment Variables
Configuration can be set via environment variables or `.env` file:
```bash
BINANCE_API_KEY=your_binance_api_key
BINANCE_API_SECRET=your_binance_api_secret
DATABASE_URL=postgresql://trading_bot:dev_password_123@localhost:5432/ai_trading_bot
TRADING_MODE=paper
INITIAL_BALANCE=1000
```

### 3. Strategy Parameters (In Strategy Classes)
Each strategy defines its own parameters and trading pair:

```python
class MyStrategy(BaseStrategy):
    def __init__(self, name="MyStrategy"):
        super().__init__(name)

        # Set strategy-specific trading pair
        self.trading_pair = 'ETH-USD'  # Use SymbolFactory for conversion if needed

        # Strategy parameters
        self.risk_per_trade = 0.02
        self.stop_loss_pct = 0.015
        # ... other parameters
```

## Current Strategy Trading Pairs

| Strategy | Default Trading Pair | Description |
|----------|---------------------|-------------|
| `ml_basic` | BTCUSDT | Price-only ML model |
| `ml_sentiment` | BTCUSDT | Price + sentiment ML model |
| `ml_adaptive` | BTCUSDT | Adaptive ML with regime detection |
| `bull` | BTCUSDT | Bull market optimized strategy |
| `bear` | BTCUSDT | Bear market optimized strategy |

## Usage Examples

```bash
# Uses strategy's default trading pair
atb backtest ml_basic --symbol BTCUSDT --timeframe 1h --days 100

# Override with specific symbol
atb backtest ml_sentiment --symbol ETHUSDT --timeframe 1h --days 100

# Try different strategies
atb backtest ml_adaptive --symbol BTCUSDT --timeframe 4h --days 365
```

## Benefits

1. **Provider Flexibility**: Multiple configuration sources with automatic fallback
2. **Security**: Environment variable priority for production secrets
3. **Local Development**: Easy `.env` file support for development
4. **Type Safety**: Built-in type conversion and validation in ConfigManager
5. **Railway Integration**: Seamless deployment with Railway environment variables
6. **Strategy-Specific Trading Pairs**: Different strategies can focus on different assets

## Files Modified

- **ADDED**: `src/config/config_manager.py` - Centralized configuration management
- **ADDED**: `src/config/providers/` - Modular configuration providers
- **UPDATED**: All strategy files - Added trading pair definitions
- **UPDATED**: CLI tool `atb` - Uses strategy-specific trading pairs
- **UPDATED**: Data providers - Use ConfigManager for API keys

## Adding New Strategies

When creating new strategies:

1. **Inherit from BaseStrategy**
2. **Set trading pair** in constructor: `self.trading_pair = 'SYMBOL'`
3. **Define parameters** directly in the class
4. **Add to strategy registry** in `src/strategies/__init__.py`

Example:
```python
class NewStrategy(BaseStrategy):
    def __init__(self, name="NewStrategy"):
        super().__init__(name)
        self.trading_pair = 'ADAUSDT'  # Strategy-specific pair
        self.risk_per_trade = 0.015
        # ... other parameters
```
