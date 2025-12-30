# Performance Utilities

Lightweight helpers for computing the performance numbers surfaced by the
backtester, live engine, dashboards, and tests.

## Module overview

- `metrics.py` – Pure, side-effect free helpers used throughout the stack.

### Included helpers

| Helper | Purpose |
| ------ | ------- |
| `pnl_percent` | Sized percentage return for a trade based on entry/exit and side |
| `cash_pnl` | Converts sized percentage returns into account currency |
| `total_return` | Cumulative percentage return between two balances |
| `cagr` | Compound annual growth rate for a period (percentage) |
| `sharpe` | Annualised Sharpe ratio for a daily equity curve |
| `max_drawdown` | Maximum drawdown (percentage) for an equity curve |
| `directional_accuracy` | Percentage of correct up/down predictions |
| `mean_absolute_error` | Absolute error between prediction and actual series |
| `mean_absolute_percentage_error` | Same as above expressed as a percentage |
| `brier_score_direction` | Brier score for binary directional probabilities |

All helpers accept pandas Series/DataFrames (or scalars) and never mutate their
inputs, which keeps them easy to test and reason about.

## Usage

```python
from datetime import datetime

import pandas as pd

from src.performance.metrics import (
    cash_pnl,
    cagr,
    max_drawdown,
    pnl_percent,
    sharpe,
    total_return,
)

# Example trade
trade_return = pnl_percent(entry_price=100, exit_price=105, side="long", fraction=0.5)
assert round(trade_return, 4) == 0.025  # +2.5 % on total balance
print(f"Cash PnL on $20k account: ${cash_pnl(trade_return, balance_before=20_000):.2f}")

# Equity curve metrics
index = pd.date_range(end=datetime.now(UTC), periods=120, freq="D")
equity_curve = pd.Series(10_000).reindex(index)
equity_curve += (pd.Series(range(len(index)), index=index) * 25)  # toy growth

print(f"Total return: {total_return(10_000, equity_curve.iloc[-1]):.2f}%")
print(f"CAGR: {cagr(10_000, equity_curve.iloc[-1], days=len(equity_curve)):.2f}%")
print(f"Sharpe: {sharpe(equity_curve):.2f}")
print(f"Max drawdown: {max_drawdown(equity_curve):.2f}%")
```

## Integration points

- **Backtesting** – `src/backtesting/utils.compute_performance_metrics` consumes
  these helpers before emitting CLI summaries or persisting results.
- **Live trading** – Live PnL, drawdown, and Sharpe readouts in
  `src/dashboards/monitoring` rely on the same functions so offline and live
  views stay consistent.
- **Prediction diagnostics** – The ML evaluation pipeline uses
  `directional_accuracy`, `mean_absolute_error`, `mean_absolute_percentage_error`,
  and `brier_score_direction` when validating model outputs.

Because each helper is a pure function, you can safely call them from unit
tests, CLI tooling, or ad-hoc notebooks without pulling in any runtime
dependencies beyond pandas/numpy.
