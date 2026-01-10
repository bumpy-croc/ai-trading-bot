from __future__ import annotations

import pandas as pd

from src.performance.metrics import (
    cagr as perf_cagr,
)
from src.performance.metrics import (
    max_drawdown as perf_max_drawdown,
)
from src.performance.metrics import (
    sharpe as perf_sharpe,
)
from src.performance.metrics import (
    total_return as perf_total_return,
)
from src.tech.adapters import row_extractors

# NOTE: extraction helpers are implemented in src.tech.adapters.row_extractors so
# they can be shared by backtesting, live trading, and dashboards.

extract_indicators = row_extractors.extract_indicators
extract_sentiment_data = row_extractors.extract_sentiment_data
extract_ml_predictions = row_extractors.extract_ml_predictions


def compute_performance_metrics(
    initial_balance: float,
    final_balance: float,
    start: pd.Timestamp,
    end: pd.Timestamp | None,
    balance_history: pd.DataFrame,
) -> tuple[float, float, float, float]:
    """Compute total return (%), max drawdown (%), Sharpe, and annualized return (%).

    Parameters
    ----------
    balance_history : DataFrame with index timestamp and column 'balance'.
    """
    total_ret = perf_total_return(initial_balance, final_balance)

    if balance_history is not None and not balance_history.empty:
        daily_balance = balance_history["balance"].resample("1D").last().ffill()
        if (
            not daily_balance.empty
            and daily_balance.shape[0] >= 2
            and daily_balance.pct_change().dropna().std() != 0
        ):
            sharpe_ratio = perf_sharpe(daily_balance)
        else:
            sharpe_ratio = 0.0
        max_dd_pct = perf_max_drawdown(daily_balance)
    else:
        sharpe_ratio = 0.0
        max_dd_pct = 0.0

    days = (end - start).days if end is not None else (pd.Timestamp.now() - start).days
    annualized_ret = perf_cagr(initial_balance, final_balance, int(days))

    return total_ret, max_dd_pct, float(sharpe_ratio), float(annualized_ret)
