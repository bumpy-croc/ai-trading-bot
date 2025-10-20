---
description: Strategy development (concise)
alwaysApply: false
---

### Implemented
- `ml_basic`, `ml_adaptive`, `ml_sentiment`, `ensemble_weighted`, `momentum_leverage` (see `src/strategies/`).
- Factory functions in respective files (e.g., `create_ml_basic_strategy()`).
- Exported in `src/strategies/__init__.py`.

### Component-Based Interface
All strategies use `Strategy` class with composed components:
- `SignalGenerator`: Generates trading signals with confidence
- `RiskManager`: Calculates risk-based position sizes
- `PositionSizer`: Determines final position size
- `RegimeDetector` (optional): Detects market regimes

Main method: `strategy.process_candle(df, index, balance, positions) -> TradingDecision`

### Add a strategy
- Create factory function in `src/strategies/my_strategy.py` that returns configured `Strategy` instance.
- Compose appropriate `SignalGenerator`, `RiskManager`, and `PositionSizer` components.
- Register exports in `src/strategies/__init__.py`.
- Backtest: `atb backtest my_strategy --symbol BTCUSDT --timeframe 1h --days 30`.
