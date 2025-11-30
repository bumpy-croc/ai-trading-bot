# Trading Core

Base classes and shared utilities for trading strategies and components.

## Overview

This module provides the foundational interfaces and shared functionality used across the trading system. It defines the component-based strategy wiring, shared helpers, and symbol utilities under `symbols/` that both the backtester and live engine reuse.

## Modules

- `symbols/`: Symbol normalization helpers for exchange-specific formats.

## Key Components

### Strategy Composition
Strategies are composed from `SignalGenerator`, `RiskManager`, and `PositionSizer` components via the `Strategy` class in
`src/strategies/components`. Each component has a clear interface so you can swap implementations without rewriting the rest of the
stack.

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
import pandas as pd

from src.strategies.components import (
    Strategy,
    Signal,
    SignalDirection,
    SignalGenerator,
    FixedRiskManager,
    FixedFractionSizer,
)


class SimpleMASignalGenerator(SignalGenerator):
    def __init__(self, fast_period: int = 10, slow_period: int = 30):
        super().__init__("simple_ma")
        self.fast_period = fast_period
        self.slow_period = slow_period

    def generate_signal(self, df: pd.DataFrame, index: int, regime=None) -> Signal:
        if index < self.slow_period:
            return Signal(SignalDirection.HOLD, strength=0.0, confidence=0.0, metadata={})
        fast = df["close"].rolling(self.fast_period).mean().iloc[index]
        slow = df["close"].rolling(self.slow_period).mean().iloc[index]
        if fast > slow:
            direction = SignalDirection.BUY
        elif fast < slow:
            direction = SignalDirection.SELL
        else:
            direction = SignalDirection.HOLD
        return Signal(direction, strength=1.0, confidence=self.get_confidence(df, index), metadata={"fast": fast, "slow": slow})

    def get_confidence(self, df: pd.DataFrame, index: int) -> float:
        return 0.6  # plug in whatever heuristic you need


strategy = Strategy(
    name="simple_ma",
    signal_generator=SimpleMASignalGenerator(),
    risk_manager=FixedRiskManager(risk_per_trade=0.02),
    position_sizer=FixedFractionSizer(fraction=0.1),
)
```

## See Also

- [strategies/README.md](../strategies/README.md) - Strategy implementation examples
- [docs/backtesting.md](../../docs/backtesting.md) - Backtesting strategies
- [docs/live_trading.md](../../docs/live_trading.md) - Live trading usage
