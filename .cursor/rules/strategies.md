---
description: Strategy development (concise)
alwaysApply: false
---

### Implemented
- `MlBasic`, `MlAdaptive`, `MlWithSentiment` (see `src/strategies/`).
- Exported in `src/strategies/__init__.py`.

### Base interface
`BaseStrategy` requires:
- `calculate_indicators(df) -> df`
- `check_entry_conditions(df, i) -> bool`
- `check_exit_conditions(df, i, entry_price) -> bool`
- `calculate_position_size(df, i, balance) -> float`
- `calculate_stop_loss(df, i, price, side='long') -> float`
- `get_parameters() -> dict`

### Add a strategy
- Create `src/strategies/my_strategy.py` extending `BaseStrategy`.
- Register exports in `src/strategies/__init__.py`.
- Backtest: `python scripts/run_backtest.py my_strategy --days 30 --no-db`.