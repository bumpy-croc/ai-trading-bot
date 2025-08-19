from __future__ import annotations

import pandas as pd
from performance.metrics import (
    cagr as perf_cagr,
)
from performance.metrics import (
    max_drawdown as perf_max_drawdown,
)
from performance.metrics import (
    sharpe as perf_sharpe,
)
from performance.metrics import (
    total_return as perf_total_return,
)

# NOTE: extraction helpers moved to trading.shared.indicators; this file retains only performance math


def extract_indicators(df: pd.DataFrame, index: int) -> dict:
    """Extract a stable subset of indicator values from `df.iloc[index]`.

    Returns numeric values where possible and includes basic OHLCV.
    """
    if index >= len(df) or len(df) == 0:
        return {}

    indicators: dict = {}
    row = df.iloc[index]

    indicator_columns = [
        "rsi",
        "macd",
        "macd_signal",
        "macd_hist",
        "atr",
        "volatility",
        "trend_ma",
        "short_ma",
        "long_ma",
        "volume_ma",
        "trend_strength",
        "regime",
        "body_size",
        "upper_wick",
        "lower_wick",
        "onnx_pred",
        "ml_prediction",
        "prediction_confidence",
    ]

    for col in indicator_columns:
        if col in df.columns and pd.notna(row[col]):
            if col == "regime":
                indicators[col] = row[col]
            else:
                try:
                    indicators[col] = float(row[col])
                except (ValueError, TypeError):
                    continue

    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns and pd.notna(row[col]):
            indicators[col] = float(row[col])

    return indicators


def extract_sentiment_data(df: pd.DataFrame, index: int) -> dict:
    """Extract sentiment-related columns if present for logging."""
    if index >= len(df) or len(df) == 0:
        return {}

    sentiment: dict = {}
    row = df.iloc[index]

    sentiment_columns = [
        "sentiment_score",
        "sentiment_primary",
        "sentiment_momentum",
        "sentiment_volatility",
        "sentiment_confidence",
        "sentiment_freshness",
    ]

    for col in sentiment_columns:
        if col in df.columns and pd.notna(row[col]):
            try:
                sentiment[col] = float(row[col])
            except (ValueError, TypeError):
                continue

    return sentiment


def extract_ml_predictions(df: pd.DataFrame, index: int) -> dict:
    """Extract ML prediction-related columns if present for logging."""
    if index >= len(df) or len(df) == 0:
        return {}

    ml: dict = {}
    row = df.iloc[index]

    ml_columns = ["onnx_pred", "ml_prediction", "prediction_confidence"]

    for col in ml_columns:
        if col in df.columns and pd.notna(row[col]):
            try:
                ml[col] = float(row[col])
            except (ValueError, TypeError):
                continue

    return ml


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
