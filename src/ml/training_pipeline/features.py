"""Feature engineering utilities for model training."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)

# Sentiment quality assessment constants
MAX_DATA_FRESHNESS_DAYS = 999  # Indicates stale or missing sentiment data
COVERAGE_WEIGHT = 0.6  # Weight for temporal coverage in quality score
FRESHNESS_WEIGHT = 0.4  # Weight for data freshness in quality score
DAYS_PER_YEAR = 365  # Days in a year for freshness calculation
QUALITY_THRESHOLD_HIGH = 0.8  # Threshold for full sentiment recommendation
QUALITY_THRESHOLD_MEDIUM = 0.4  # Threshold for hybrid sentiment recommendation

# Technical indicator window sizes
SMA_WINDOWS = [7, 14, 30]  # Simple moving average window sizes (days)
RSI_WINDOW = 14  # Relative Strength Index window size (days)
RSI_MAX = 100  # Maximum RSI value for normalization


def normalize_timezone(ts1: pd.Timestamp, ts2: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    """
    Normalize two timestamps to have compatible timezone information.

    If one timestamp is tz-aware and the other is tz-naive, removes timezone
    info from the tz-aware timestamp to enable comparison operations.

    Args:
        ts1: First timestamp
        ts2: Second timestamp

    Returns:
        Tuple of normalized timestamps that can be compared
    """
    if ts1.tzinfo is not None and ts2.tzinfo is None:
        return ts1.tz_localize(None), ts2
    if ts1.tzinfo is None and ts2.tzinfo is not None:
        return ts1, ts2.tz_localize(None)
    return ts1, ts2


def assess_sentiment_data_quality(sentiment_df: pd.DataFrame, price_df: pd.DataFrame) -> dict:
    """Assess coverage and freshness of sentiment data relative to price data."""

    assessment = {
        "total_sentiment_points": len(sentiment_df),
        "total_price_points": len(price_df),
        "coverage_ratio": 0.0,
        "data_freshness_days": MAX_DATA_FRESHNESS_DAYS,
        "missing_periods": [],
        "quality_score": 0.0,
        "recommendation": "price_only" if sentiment_df.empty else "unknown",
    }

    if sentiment_df.empty:
        assessment["reason"] = "No sentiment data available"
        return assessment

    price_start, price_end = price_df.index.min(), price_df.index.max()
    sentiment_start, sentiment_end = sentiment_df.index.min(), sentiment_df.index.max()

    # Normalize timezone info to enable timestamp comparisons
    price_start, sentiment_start = normalize_timezone(price_start, sentiment_start)
    price_end, sentiment_end = normalize_timezone(price_end, sentiment_end)

    overlap_start = max(price_start, sentiment_start)
    overlap_end = min(price_end, sentiment_end)

    if overlap_start >= overlap_end:
        assessment["reason"] = "No temporal overlap between price and sentiment data"
        return assessment

    total_period = (price_end - price_start).total_seconds()
    overlap_period = (overlap_end - overlap_start).total_seconds()
    assessment["coverage_ratio"] = overlap_period / total_period if total_period > 0 else 0

    current_time = pd.Timestamp.now()
    current_time, sentiment_end_normalized = normalize_timezone(current_time, sentiment_end)
    assessment["data_freshness_days"] = (current_time - sentiment_end_normalized).days

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

    coverage_score = min(assessment["coverage_ratio"] * 2, 1.0)
    freshness_score = max(0, 1 - (assessment["data_freshness_days"] / DAYS_PER_YEAR))
    assessment["quality_score"] = (
        coverage_score * COVERAGE_WEIGHT + freshness_score * FRESHNESS_WEIGHT
    )

    if assessment["quality_score"] >= QUALITY_THRESHOLD_HIGH:
        assessment["recommendation"] = "full_sentiment"
    elif assessment["quality_score"] >= QUALITY_THRESHOLD_MEDIUM:
        assessment["recommendation"] = "hybrid_with_fallback"
    else:
        assessment["recommendation"] = "price_only"

    return assessment


def merge_price_sentiment_data(
    price_df: pd.DataFrame, sentiment_df: pd.DataFrame, timeframe: str
) -> pd.DataFrame:
    if sentiment_df.empty:
        return price_df
    if timeframe != "1d":
        sentiment_resampled = sentiment_df.resample(timeframe).ffill()
    else:
        sentiment_resampled = sentiment_df
    return price_df.join(sentiment_resampled, how="left")


def _calculate_rsi_fast(close_prices: np.ndarray, window: int = 14) -> np.ndarray:
    """Calculate RSI using optimized numpy operations.

    Args:
        close_prices: 1D array of close prices
        window: RSI window size

    Returns:
        1D array of RSI values (with NaNs for initial window)
    """
    # Calculate price changes
    deltas = np.diff(close_prices, prepend=close_prices[0])

    # Separate gains and losses
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    # Check if we have enough data for RSI calculation
    if len(close_prices) <= window:
        # Return all NaNs if insufficient data
        return np.full(len(close_prices), np.nan, dtype=np.float32)

    # Calculate rolling averages using numpy (faster than pandas for this)
    avg_gains = np.full(len(gains), np.nan, dtype=np.float32)
    avg_losses = np.full(len(losses), np.nan, dtype=np.float32)

    # Calculate initial averages
    avg_gains[window] = np.mean(gains[1 : window + 1])
    avg_losses[window] = np.mean(losses[1 : window + 1])

    # Calculate subsequent averages using EMA-style smoothing
    for i in range(window + 1, len(gains)):
        avg_gains[i] = (avg_gains[i - 1] * (window - 1) + gains[i]) / window
        avg_losses[i] = (avg_losses[i - 1] * (window - 1) + losses[i]) / window

    # Calculate RS and RSI (only for valid indices)
    rs = avg_gains / (avg_losses + 1e-10)  # Add epsilon to avoid division by zero
    rsi = 100.0 - (100.0 / (1.0 + rs))

    return rsi.astype(np.float32)


def create_robust_features(
    data: pd.DataFrame,
    sentiment_assessment: dict,
    time_steps: int,
) -> tuple[pd.DataFrame, dict[str, MinMaxScaler], list[str]]:
    feature_names: list[str] = []
    scalers: dict[str, MinMaxScaler] = {}

    price_features = ["open", "high", "low", "close", "volume"]
    for feature in price_features:
        if feature in data.columns:
            scaler = MinMaxScaler()
            data[f"{feature}_scaled"] = scaler.fit_transform(data[[feature]])
            feature_names.append(f"{feature}_scaled")
            scalers[feature] = scaler

    if "close" in data.columns:
        # Calculate rolling features (produces NaNs for initial window)
        for window in SMA_WINDOWS:
            col = f"sma_{window}"
            data[col] = data["close"].rolling(window=window).mean()

        # Calculate RSI using optimized numpy implementation
        close_prices = data["close"].values
        rsi_values = _calculate_rsi_fast(close_prices, window=RSI_WINDOW)
        data["rsi"] = rsi_values

        # Drop NaNs before scaling to avoid MinMaxScaler ValueError
        rows_before = len(data)
        nan_counts = data.isna().sum()
        data = data.dropna()
        rows_dropped = rows_before - len(data)

        if rows_dropped > 0:
            logger.info(
                "Dropped %d rows with NaNs from %d total rows (%.1f%%). " "Features with NaNs: %s",
                rows_dropped,
                rows_before,
                (rows_dropped / rows_before) * 100,
                {col: int(count) for col, count in nan_counts[nan_counts > 0].items()},
            )

        # Validate sufficient data remains after dropping NaNs
        min_required_rows = max(SMA_WINDOWS) * 2  # Need enough data for meaningful training
        if len(data) < min_required_rows:
            raise ValueError(
                f"Insufficient data after dropping NaNs: {len(data)} rows remaining, "
                f"need at least {min_required_rows} for training with SMA windows {SMA_WINDOWS}"
            )

        # Validate data for inf values and other corruption that would break MinMaxScaler
        inf_counts = np.isinf(data.select_dtypes(include=[np.number])).sum()
        if inf_counts.sum() > 0:
            logger.error(
                "Found infinite values in training data. Columns with inf: %s",
                {col: int(count) for col, count in inf_counts[inf_counts > 0].items()},
            )
            raise ValueError(
                f"Training data contains infinite values in {inf_counts[inf_counts > 0].to_dict()} - "
                "cannot proceed with scaling. Check data provider for corrupt price data."
            )

        # Validate data is numeric where expected
        numeric_cols = ["close", "open", "high", "low", "volume", "rsi"] + [
            f"sma_{w}" for w in SMA_WINDOWS
        ]
        for col in numeric_cols:
            if col in data.columns:
                if not np.issubdtype(data[col].dtype, np.number):
                    raise ValueError(
                        f"Column '{col}' is not numeric (dtype={data[col].dtype}) - "
                        "cannot proceed with feature engineering"
                    )

        # Now scale the technical indicators (NaN-free data)
        for window in SMA_WINDOWS:
            col = f"sma_{window}"
            scaled = f"{col}_scaled"
            data[scaled] = MinMaxScaler().fit_transform(data[[col]])
            feature_names.append(scaled)

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

    # NaNs are dropped after technical indicator calculation (line 141)
    # Sentiment features use fillna(0), so no additional NaNs are expected
    return data, scalers, feature_names
