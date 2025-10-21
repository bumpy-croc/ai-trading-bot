# Trading Core

Base classes and shared utilities for trading strategies and components.

## Overview

This module provides the foundational interfaces and shared functionality used across the trading system. It defines base strategy classes, component interfaces, and common utilities for backtesting and live trading.

## Modules

- `shared/`: Common trading interfaces and utilities used across the system

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
# Import shared trading interfaces
from src.trading.shared import TradingInterface

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