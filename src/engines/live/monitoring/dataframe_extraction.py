"""Pure helpers that pull loggable values out of a strategy dataframe.

Used by the live engine when persisting trading decisions; extraction is
best-effort — missing or NaN columns are simply omitted.
"""

from __future__ import annotations

import pandas as pd

# Indicator columns commonly produced by strategy/indicator pipelines.
INDICATOR_COLUMNS = [
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
]

SENTIMENT_COLUMNS = [
    "sentiment_primary",
    "sentiment_momentum",
    "sentiment_volatility",
    "sentiment_extreme_positive",
    "sentiment_extreme_negative",
    "sentiment_ma_3",
    "sentiment_ma_7",
    "sentiment_ma_14",
    "sentiment_confidence",
    "sentiment_freshness",
]

ML_PREDICTION_COLUMNS = ["ml_prediction", "prediction_confidence", "onnx_pred"]


def extract_indicators(df: pd.DataFrame, index: int) -> dict:
    """Extract indicator values from dataframe for logging"""
    if index >= len(df):
        return {}

    indicators = {}
    current_row = df.iloc[index]

    for col in INDICATOR_COLUMNS:
        if col in df.columns and not pd.isna(current_row[col]):
            indicators[col] = float(current_row[col])

    # Add basic OHLCV data
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            indicators[col] = float(current_row[col])

    return indicators


def extract_sentiment_data(df: pd.DataFrame, index: int) -> dict:
    """Extract sentiment data from dataframe for logging"""
    if index >= len(df):
        return {}

    sentiment_data = {}
    current_row = df.iloc[index]

    for col in SENTIMENT_COLUMNS:
        if col in df.columns and not pd.isna(current_row[col]):
            sentiment_data[col] = float(current_row[col])

    return sentiment_data


def extract_ml_predictions(df: pd.DataFrame, index: int) -> dict:
    """Extract ML prediction data from dataframe for logging"""
    if index >= len(df):
        return {}

    ml_data = {}
    current_row = df.iloc[index]

    for col in ML_PREDICTION_COLUMNS:
        if col in df.columns and not pd.isna(current_row[col]):
            ml_data[col] = float(current_row[col])

    return ml_data
