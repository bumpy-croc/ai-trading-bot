"""Shared helpers for extracting indicator-related rows from pandas DataFrames."""

from __future__ import annotations

from typing import Any

import pandas as pd

_ML_COLUMNS = ["onnx_pred", "ml_prediction", "prediction_confidence"]

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
] + _ML_COLUMNS

_SENTIMENT_COLUMNS = [
    "sentiment_score",
    "sentiment_primary",
    "sentiment_momentum",
    "sentiment_volatility",
    "sentiment_confidence",
    "sentiment_freshness",
]
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


# Metadata keys the ML signal generators attach to their Signals
# (src/strategies/components/ml_signal_generator.py). "error"/"error_type" are
# included so richer failure detail flows through if generators start emitting it.
_ML_SIGNAL_METADATA_KEYS = (
    "prediction",
    "predicted_return",
    "current_price",
    "engine_model_name",
    "model_type",
    "model_timeframe",
    "model_symbol",
    "trading_symbol",
    "generator",
    "error",
    "error_type",
    # Live inference-timeout substitutions (see PredictionEngine._run_inference)
    "timed_out",
)

# Failure reasons only the ML generators emit. Generic reasons such as
# "insufficient_history" are shared with technical generators and would
# mislabel non-ML decisions as prediction failures.
_ML_FAILURE_REASONS = frozenset({"prediction_failed", "invalid_prediction_or_price"})


def extract_ml_predictions_from_signal(signal: Any) -> dict[str, Any]:
    """Extract ML prediction context from a Signal's metadata for DB logging.

    Model outputs (predicted price/return, model identity) live on Signal
    metadata rather than dataframe columns, so the column-based extractor
    above never sees them (#914). Returns ``{}`` for signals that no ML
    prediction informed; failed predictions return their failure reason with
    ``prediction_failed=True`` so they stay visible in ``strategy_executions``
    instead of persisting as null.
    """

    metadata = getattr(signal, "metadata", None)
    if not isinstance(metadata, dict):
        return {}

    reason = metadata.get("reason")
    is_failure = reason in _ML_FAILURE_REASONS
    has_prediction = "prediction" in metadata or "predicted_return" in metadata
    if not has_prediction and not is_failure:
        return {}

    extracted: dict[str, Any] = {
        key: metadata[key] for key in _ML_SIGNAL_METADATA_KEYS if metadata.get(key) is not None
    }
    if is_failure:
        extracted["prediction_failed"] = True
        extracted["reason"] = reason
    return extracted


__all__ = [
    "extract_indicators",
    "extract_sentiment_data",
    "extract_ml_predictions",
    "extract_ml_predictions_from_signal",
]
