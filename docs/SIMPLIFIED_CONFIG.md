# Simplified Configuration System

## Overview

The configuration system has been streamlined to follow a simple approach:
- **API keys only** in `.env` file
- **Strategy parameters** defined directly in each strategy class
- **Strategy-specific trading pairs** set in each strategy constructor

## Configuration Structure

### 1. API Keys (`.env` file)
Only API credentials are stored in environment variables:
```bash
BINANCE_API_KEY=your_binance_api_key
BINANCE_API_SECRET=your_binance_api_secret
CRYPTO_COMPARE_API_KEY=your_crypto_compare_api_key
```

Components that need API keys load them directly:
```python
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('BINANCE_API_KEY')
```

### 2. Strategy Parameters (In Strategy Classes)
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

| Strategy | Default Trading Pair |
|----------|---------------------|
| `ml_basic` | ETH-USD |

## Usage Examples

```bash
# Uses strategy's default trading pair
python scripts/run_backtest.py ml_basic --days 100          # Uses ETH-USD

# Override with specific symbol
python scripts/run_backtest.py ml_basic --symbol SOL-USD --days 100  # Use SymbolFactory for conversion if needed
```

## Benefits

1. **Simplicity**: No complex configuration management
2. **Clarity**: Parameters are clearly defined in each strategy
3. **Flexibility**: Easy to modify strategy parameters during development
4. **Security**: Only sensitive API keys are in environment variables
5. **Strategy-Specific Trading Pairs**: Different strategies can focus on different assets

## Files Modified

- **UPDATED**: All strategy files - Added trading pair definitions
- **UPDATED**: `scripts/run_backtest.py` - Uses strategy-specific trading pairs
- **UPDATED**: Data providers - Load API keys directly from environment
- **REMOVED**: Old `config.py` and `config/settings.py` files - No longer needed

## Adding New Strategies

When creating new strategies:

1. **Inherit from BaseStrategy**
2. **Set trading pair** in constructor: `self.trading_pair = 'SYMBOL'`
3. **Define parameters** directly in the class
4. **Add to strategy map** in `scripts/run_backtest.py`

Example:
```python
class NewStrategy(BaseStrategy):
    def __init__(self, name="NewStrategy"):
        super().__init__(name)
        self.trading_pair = 'ADAUSDT'  # Strategy-specific pair
        self.risk_per_trade = 0.015
        # ... other parameters
``` 