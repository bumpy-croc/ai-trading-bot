# Trading Core

Base classes and shared utilities for trading strategies and components.

## Overview

This module provides the foundational interfaces and shared functionality used across the trading system. It defines base strategy classes, component interfaces, and common helpers for backtesting and live trading, including symbol utilities under `symbols/`.

## Modules

- `symbols/`: Symbol normalization helpers for exchange-specific formats.

## Key Components

### Base Strategy Classes
Abstract base classes for implementing trading strategies. All custom strategies should inherit from `BaseStrategy`.

### Component Interfaces
Contracts defining the interface for:
- Signal generators
- Risk managers
- Position sizers
- Entry/exit logic

### Shared Utilities
Common helpers used by strategies and engines including validation, error handling, and data processing utilities.

## Usage

```python
# Import base strategy class
from src.strategies.base import BaseStrategy

class MyStrategy(BaseStrategy):
    def calculate_indicators(self, df):
        # Add indicators to dataframe
        return df
    
    def check_entry_conditions(self, df):
        # Return entry signal
        return signal
    
    def check_exit_conditions(self, df, position):
        # Return exit signal
        return signal
```

## See Also

- [strategies/README.md](../strategies/README.md) - Strategy implementation examples
- [docs/backtesting.md](../../docs/backtesting.md) - Backtesting strategies
- [docs/live_trading.md](../../docs/live_trading.md) - Live trading usage
