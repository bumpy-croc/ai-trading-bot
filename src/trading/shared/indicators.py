from __future__ import annotations

import pandas as pd


def extract_indicators(df: pd.DataFrame, index: int) -> dict:
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
