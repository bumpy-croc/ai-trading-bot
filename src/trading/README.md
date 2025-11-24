# Trading Core

Foundational interfaces and utilities that every strategy, engine, and CLI command
shares. The package glues together component-based strategies, registry helpers,
and exchange-agnostic symbol handling.

## Modules

- `components/` – composable strategy system (signal generators, risk managers,
  position sizers, runtime context, strategy registries).
- `symbols/` – normalizes tickers across venues (`BTCUSDT` ↔ `BTC-USD`) so data
  providers, engines, and dashboards speak the same symbol format.

## Component-first architecture

Modern strategies are composed from small building blocks:

- `Strategy` orchestrates components and exposes `process_candle`.
- `SignalGenerator`, `RiskManager`, and `PositionSizer` live under
  `src.strategies.components` and can be mixed/matched per strategy.
- `ComponentStrategyManager` wires registry metadata, hot-swapping, and lineage
  tracking for both backtests and live trading.
- `RegimeContext` + `EnhancedRegimeDetector` provide adaptive behavior without
  embedding market-state logic inside each component.

Keeping logic in components lets engines swap implementations without touching
business logic while tests focus on isolated units.

## Example: compose a simple breakout strategy

```python
from __future__ import annotations

import pandas as pd

from src.strategies.components import (
    FixedFractionSizer,
    FixedRiskManager,
    RegimeContext,
    Signal,
    SignalDirection,
    SignalGenerator,
    Strategy,
)


class CloseBreakoutSignal(SignalGenerator):
    """Emit BUY/SELL signals when the close breaks the recent window."""

    def __init__(self, window: int = 20):
        super().__init__(name="close_breakout")
        self.window = window

    def generate_signal(
        self, df: pd.DataFrame, index: int, regime: RegimeContext | None = None
    ) -> Signal:
        if index < self.window:
            return Signal(SignalDirection.HOLD, strength=0.0, confidence=0.0, metadata={})

        window_slice = df["close"].iloc[index - self.window : index]
        close = float(df["close"].iat[index - 1])

        if close >= float(window_slice.max()):
            return Signal(
                SignalDirection.BUY,
                strength=1.0,
                confidence=0.7,
                metadata={"window_high": float(window_slice.max())},
            )
        if close <= float(window_slice.min()):
            return Signal(
                SignalDirection.SELL,
                strength=1.0,
                confidence=0.7,
                metadata={"window_low": float(window_slice.min())},
            )
        return Signal(SignalDirection.HOLD, strength=0.0, confidence=0.0, metadata={})

    def get_confidence(self, df: pd.DataFrame, index: int) -> float:
        return self.generate_signal(df, index).confidence


signal = CloseBreakoutSignal(window=15)
risk = FixedRiskManager(risk_per_trade=0.02)
sizer = FixedFractionSizer(fraction=0.1)

strategy = Strategy(
    name="example_breakout",
    signal_generator=signal,
    risk_manager=risk,
    position_sizer=sizer,
)

prices = [100 + i for i in range(40)]
frame = pd.DataFrame(
    {
        "open": prices,
        "high": [p * 1.01 for p in prices],
        "low": [p * 0.99 for p in prices],
        "close": prices,
        "volume": [1_000] * len(prices),
    }
)
decision = strategy.process_candle(frame, index=len(frame) - 1, balance=10_000.0)
print(decision.signal.direction, decision.position_size)
```

The example stays entirely inside the component system, so the same strategy
definition works in backtests, live trading, and unit tests.

## See also

- [strategies/README.md](../strategies/README.md) – deeper dive into component
  strategies and available factories.
- [docs/backtesting.md](../../docs/backtesting.md) – how strategies plug into the
  vectorized engine.
- [docs/live_trading.md](../../docs/live_trading.md) – runtime wiring, safety
  controls, and live strategy management.
