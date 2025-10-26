"""Shared helpers for extracting indicator-related rows from pandas DataFrames."""

from __future__ import annotations

from typing import Any

import pandas as pd

_INDICATOR_COLUMNS = [
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

_SENTIMENT_COLUMNS = [
    "sentiment_score",
    "sentiment_primary",
    "sentiment_momentum",
    "sentiment_volatility",
    "sentiment_confidence",
    "sentiment_freshness",
]

_ML_COLUMNS = ["onnx_pred", "ml_prediction", "prediction_confidence"]
_OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


def _safe_row(df: pd.DataFrame, index: int) -> pd.Series | None:
    if index >= len(df) or len(df) == 0:
        return None
    return df.iloc[index]


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def extract_indicators(df: pd.DataFrame, index: int) -> dict[str, float | str]:
    """Extract a stable subset of indicator values from ``df.iloc[index]``."""

    row = _safe_row(df, index)
    if row is None:
        return {}

    indicators: dict[str, float | str] = {}

    for col in _INDICATOR_COLUMNS:
        if col in df.columns and pd.notna(row[col]):
            if col == "regime":
                indicators[col] = row[col]
            else:
                value = _safe_float(row[col])
                if value is not None:
                    indicators[col] = value

    for col in _OHLCV_COLUMNS:
        if col in df.columns and pd.notna(row[col]):
            value = _safe_float(row[col])
            if value is not None:
                indicators[col] = value

    return indicators


def extract_sentiment_data(df: pd.DataFrame, index: int) -> dict[str, float]:
    """Extract sentiment-related columns if present for logging."""

    row = _safe_row(df, index)
    if row is None:
        return {}

    sentiment: dict[str, float] = {}

    for col in _SENTIMENT_COLUMNS:
        if col in df.columns and pd.notna(row[col]):
            value = _safe_float(row[col])
            if value is not None:
                sentiment[col] = value

    return sentiment


def extract_ml_predictions(df: pd.DataFrame, index: int) -> dict[str, float]:
    """Extract ML prediction-related columns if present for logging."""

    row = _safe_row(df, index)
    if row is None:
        return {}

    ml: dict[str, float] = {}

    for col in _ML_COLUMNS:
        if col in df.columns and pd.notna(row[col]):
            value = _safe_float(row[col])
            if value is not None:
                ml[col] = value

    return ml


__all__ = [
    "extract_indicators",
    "extract_sentiment_data",
    "extract_ml_predictions",
]
