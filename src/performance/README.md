# Performance Utilities

Shared metrics used by the backtester, live engine, and dashboards.

## Modules

- `metrics.py` – Pure helper functions with no side effects; every function accepts pandas/NumPy inputs and returns floats.

## Available helpers

- `pnl_percent(entry_price, exit_price, side, fraction)` – Sized percentage return given trade direction.
- `cash_pnl(pnl_pct, balance_before)` – Convert percentage PnL into currency units.
- `total_return(initial_balance, final_balance)` – Cumulative return (%) for the entire run.
- `cagr(initial_balance, final_balance, days)` – Annualised growth rate (%) over a period.
- `sharpe(daily_balance_series)` – Risk-adjusted return using daily equity data.
- `max_drawdown(balance_series)` – Worst peak-to-trough decline (%).
- `directional_accuracy(pred_prices, actual_next_close)` – Percentage of correct up/down predictions.
- `mean_absolute_error(pred, actual)` / `mean_absolute_percentage_error(pred, actual)` – Regression accuracy helpers.
- `brier_score_direction(prob_up, actual_up)` – Calibration score for directional probabilities (lower is better).

## Usage

```python
import pandas as pd
from src.performance.metrics import (
    Side,
    pnl_percent,
    cash_pnl,
    total_return,
    cagr,
    sharpe,
    max_drawdown,
    directional_accuracy,
)

# Trade-level sizing
pct = pnl_percent(100, 104, Side.LONG, fraction=0.3)  # -> 0.012 (1.2 % of equity)
cash = cash_pnl(pct, balance_before=25_000)           # -> $300

# Equity-curve stats
equity = pd.Series([25_000, 25_300, 24_900, 26_200], index=pd.date_range("2025-01-01", periods=4))
ret = total_return(equity.iloc[0], equity.iloc[-1])
growth = cagr(equity.iloc[0], equity.iloc[-1], days=len(equity) - 1)
sr = sharpe(equity)
dd = max_drawdown(equity)

print(f"Total return: {ret:.2f}%  CAGR: {growth:.2f}%  Sharpe: {sr:.2f}  Max DD: {dd:.2f}%")

# Prediction diagnostics
pred = equity.pct_change().shift(-1).dropna()
truth = pred + 0.001
print(f"Directional accuracy: {directional_accuracy(pred, truth):.2f}%")
```

All helpers are deterministic and safe to call from multiprocessing contexts or async loops. Whenever possible, pass in pandas Series so
index metadata (timestamps) propagates into dashboards and post-processing steps that consume these metrics.
