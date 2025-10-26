"""Feature engineering utilities for model training."""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)


def assess_sentiment_data_quality(sentiment_df: pd.DataFrame, price_df: pd.DataFrame) -> dict:
    """Assess coverage and freshness of sentiment data relative to price data."""

    assessment = {
        "total_sentiment_points": len(sentiment_df),
        "total_price_points": len(price_df),
        "coverage_ratio": 0.0,
        "data_freshness_days": 999,
        "missing_periods": [],
        "quality_score": 0.0,
        "recommendation": "price_only" if sentiment_df.empty else "unknown",
    }

    if sentiment_df.empty:
        assessment["reason"] = "No sentiment data available"
        return assessment

    price_start, price_end = price_df.index.min(), price_df.index.max()
    sentiment_start, sentiment_end = sentiment_df.index.min(), sentiment_df.index.max()

    if price_start.tzinfo is not None and sentiment_start.tzinfo is None:
        price_start = price_start.tz_localize(None)
        price_end = price_end.tz_localize(None)
    elif price_start.tzinfo is None and sentiment_start.tzinfo is not None:
        sentiment_start = sentiment_start.tz_localize(None)
        sentiment_end = sentiment_end.tz_localize(None)

    overlap_start = max(price_start, sentiment_start)
    overlap_end = min(price_end, sentiment_end)

    if overlap_start >= overlap_end:
        assessment["reason"] = "No temporal overlap between price and sentiment data"
        return assessment

    total_period = (price_end - price_start).total_seconds()
    overlap_period = (overlap_end - overlap_start).total_seconds()
    assessment["coverage_ratio"] = overlap_period / total_period if total_period > 0 else 0

    current_time = pd.Timestamp.now()
    if sentiment_end.tzinfo is not None and current_time.tzinfo is None:
        current_time = current_time.tz_localize("UTC")
    elif sentiment_end.tzinfo is None and current_time.tzinfo is not None:
        current_time = current_time.tz_localize(None)
    assessment["data_freshness_days"] = (current_time - sentiment_end).days

    sentiment_dates = pd.date_range(sentiment_start, sentiment_end, freq="D")
    available_dates = set(sentiment_df.index.date)
    missing_dates = [d for d in sentiment_dates if d.date() not in available_dates]

    if missing_dates:
        gap_starts = []
        current_gap_start = missing_dates[0]
        for i in range(1, len(missing_dates)):
            if (missing_dates[i] - missing_dates[i - 1]).days > 1:
                gap_starts.append((current_gap_start, missing_dates[i - 1]))
                current_gap_start = missing_dates[i]
        gap_starts.append((current_gap_start, missing_dates[-1]))
        assessment["missing_periods"] = gap_starts

    coverage_weight = 0.6
    freshness_weight = 0.4
    coverage_score = min(assessment["coverage_ratio"] * 2, 1.0)
    freshness_score = max(0, 1 - (assessment["data_freshness_days"] / 365))
    assessment["quality_score"] = coverage_score * coverage_weight + freshness_score * freshness_weight

    if assessment["quality_score"] >= 0.8:
        assessment["recommendation"] = "full_sentiment"
    elif assessment["quality_score"] >= 0.4:
        assessment["recommendation"] = "hybrid_with_fallback"
    else:
        assessment["recommendation"] = "price_only"

    return assessment


def merge_price_sentiment_data(price_df: pd.DataFrame, sentiment_df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    if sentiment_df.empty:
        return price_df
    if timeframe != "1d":
        sentiment_resampled = sentiment_df.resample(timeframe).ffill()
    else:
        sentiment_resampled = sentiment_df
    return price_df.join(sentiment_resampled, how="left")


def create_robust_features(
    data: pd.DataFrame,
    sentiment_assessment: dict,
    time_steps: int,
) -> Tuple[pd.DataFrame, Dict[str, MinMaxScaler], List[str]]:
    feature_names: List[str] = []
    scalers: Dict[str, MinMaxScaler] = {}

    price_features = ["open", "high", "low", "close", "volume"]
    for feature in price_features:
        if feature in data.columns:
            scaler = MinMaxScaler()
            data[f"{feature}_scaled"] = scaler.fit_transform(data[[feature]])
            feature_names.append(f"{feature}_scaled")
            scalers[feature] = scaler

    if "close" in data.columns:
        for window in [7, 14, 30]:
            col = f"sma_{window}"
            data[col] = data["close"].rolling(window=window).mean()
            scaled = f"{col}_scaled"
            data[scaled] = MinMaxScaler().fit_transform(data[[col]])
            feature_names.append(scaled)

        delta = data["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data["rsi"] = 100 - (100 / (1 + rs))
        data["rsi_scaled"] = MinMaxScaler().fit_transform(data[["rsi"]])
        feature_names.append("rsi_scaled")

    if sentiment_assessment["recommendation"] in ["full_sentiment", "hybrid_with_fallback"]:
        sentiment_features = ["sentiment_score", "sentiment_volume", "sentiment_momentum"]
        for feature in sentiment_features:
            if feature in data.columns:
                data[f"{feature}_filled"] = data[feature].fillna(0)
                scaler = MinMaxScaler()
                scaled = f"{feature}_scaled"
                data[scaled] = scaler.fit_transform(data[[f"{feature}_filled"]])
                feature_names.append(scaled)
                scalers[feature] = scaler

    data = data.dropna()
    return data, scalers, feature_names
