# Performance Utilities

Lightweight, vectorized metrics shared by the backtester, live engine, and dashboards. Implementations live in `metrics.py`
and deliberately stick to the handful of helpers the runtime actually imports today.

## Contents
- `metrics.py` – Trade-level math (`pnl_percent`, `cash_pnl`), equity-curve helpers, and forecast validation metrics

## Available helpers

| Category | Helper | Description |
| --- | --- | --- |
| Trade sizing | `pnl_percent(entry, exit, side, fraction)` | Percentage PnL (decimal) for a sized position. |
| Trade sizing | `cash_pnl(pnl_pct, balance_before)` | Convert sized percentage PnL into cash units. |
| Equity curve | `total_return(initial, final)` | Total return (%) between two balances. |
| Equity curve | `cagr(initial, final, days)` | Compound annual growth rate (%). |
| Risk | `sharpe(daily_balance)` | Annualised Sharpe ratio (risk-free 0). |
| Risk | `max_drawdown(balance_series)` | Maximum drawdown (%) of an equity curve. |
| Forecast quality | `directional_accuracy(pred_prices, actual)` | Percent of correct up/down calls. |
| Forecast quality | `mean_absolute_error(pred, actual)` / `mean_absolute_percentage_error(pred, actual)` | Magnitude-focused error metrics. |
| Forecast quality | `brier_score_direction(prob_up, actual_up)` | Brier score for binary direction probabilities. |

All helpers expect pandas Series/DataFrames (where applicable) and return floats for easy logging.

## Usage

```python
import pandas as pd

from src.performance.metrics import Side, cash_pnl, max_drawdown, pnl_percent, sharpe

# Trade sizing helpers
pnl_pct = pnl_percent(100.0, 105.0, Side.LONG, fraction=0.25)
cash = cash_pnl(pnl_pct, balance_before=10_000.0)
print(f"PnL: {pnl_pct:.2%} -> ${cash:.2f}")

# Equity curve metrics
daily_balance = pd.Series([10_000, 10_200, 9_800, 10_400], dtype="float64")
print(f"Sharpe: {sharpe(daily_balance):.2f}")
print(f"Max drawdown: {max_drawdown(daily_balance):.2f}%")
```

```python
from src.performance.metrics import directional_accuracy, mean_absolute_percentage_error

pred = pd.Series([100, 101, 102, 101.5])
actual = pd.Series([100, 100.5, 101.4, 101.0])

print(f"Directional accuracy: {directional_accuracy(pred, actual):.2f}%")
print(f"MAPE: {mean_absolute_percentage_error(pred, actual):.2f}%")
```

## Integration notes

- Backtests call these helpers via `src/backtesting/utils.compute_performance_metrics`, so any additions should remain pure and deterministic.
- Live trading and dashboards reuse the same helpers to keep historical metrics aligned.
- Keep new helpers side-effect free and pandas-friendly so they can run inside vectorised pipelines as well as per-trade loops.
