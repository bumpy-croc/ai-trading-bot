from __future__ import annotations

import hashlib
from functools import wraps
from typing import Any, Callable

import numpy as np
import pandas as pd

from src.config.constants import DEFAULT_ATR_PERIOD

# Global cache for indicator calculations
_INDICATOR_CACHE: dict[str, Any] = {}
_CACHE_MAX_SIZE = 1000  # Limit cache size to prevent memory issues


def _make_cache_key(df: pd.DataFrame, func_name: str, *args, **kwargs) -> str | None:
    """
    Create a cache key from DataFrame properties and function parameters.

    Uses last timestamp, row count, function name, positional args, and sorted parameters.
    Returns None for empty DataFrames (skip caching).
    """
    if df.empty:
        return None

    # Get last timestamp (assuming index is datetime)
    last_ts = str(df.index[-1]) if hasattr(df.index, "__getitem__") else str(len(df))
    num_rows = len(df)

    # Include positional arguments in cache key
    args_str = "_".join(str(a) for a in args)
    # Sort parameters for consistent hashing
    param_str = "_".join(f"{k}={v}" for k, v in sorted(kwargs.items()))

    # Create composite key and hash for shorter keys
    key_str = "|".join([func_name, last_ts, str(num_rows), args_str, param_str])
    return hashlib.md5(key_str.encode()).hexdigest()


def cached_indicator(func: Callable) -> Callable:
    """
    Decorator to cache indicator calculations based on DataFrame properties.

    Caches results using a composite key of function name, last timestamp,
    row count, and all function parameters. Cache is cleared when it exceeds
    _CACHE_MAX_SIZE entries.
    """

    @wraps(func)
    def wrapper(df: pd.DataFrame, *args, **kwargs):
        # Build cache key
        cache_key = _make_cache_key(df, func.__name__, *args, **kwargs)

        if cache_key is None:
            # Empty DataFrame, skip caching
            return func(df, *args, **kwargs)

        # Check cache
        if cache_key in _INDICATOR_CACHE:
            cached = _INDICATOR_CACHE[cache_key]
            # Return copy to prevent mutation
            return cached.copy() if isinstance(cached, pd.DataFrame) else cached

        # Calculate indicator
        result = func(df, *args, **kwargs)

        # Store in cache
        _INDICATOR_CACHE[cache_key] = (
            result.copy() if isinstance(result, pd.DataFrame) else result
        )

        # Limit cache size (remove oldest 20% when exceeded)
        if len(_INDICATOR_CACHE) > _CACHE_MAX_SIZE:
            keys_to_remove = list(_INDICATOR_CACHE.keys())[: int(_CACHE_MAX_SIZE * 0.2)]
            for key in keys_to_remove:
                del _INDICATOR_CACHE[key]

        return result

    return wrapper


def clear_indicator_cache() -> None:
    """Clear the global indicator cache. Useful for testing or memory management."""
    global _INDICATOR_CACHE
    _INDICATOR_CACHE = {}

# Constants for division protection and regime detection
EPSILON = 1e-10  # Prevents division by zero in all calculations

# Market regime detection parameters
VOLATILITY_BASELINE_PERIOD = 50  # 50-period mean provides stable volatility baseline
VOLATILITY_RATIO_THRESHOLD = 1.3  # Threshold for detecting volatile market conditions
TREND_STRENGTH_MULTIPLIER = 0.5  # Multiplier for determining trending vs ranging markets


@cached_indicator
def calculate_moving_averages(df: pd.DataFrame, periods: list[int]) -> pd.DataFrame:
    """
    Calculate simple moving averages for multiple periods.

    Args:
        df: DataFrame with OHLCV data containing 'close' column
        periods: List of periods for moving average calculations (must be positive integers)

    Returns:
        DataFrame with original data plus ma_{period} columns

    Raises:
        ValueError: If required columns are missing or periods are invalid
    """
    if "close" not in df.columns:
        raise ValueError("DataFrame must contain 'close' column")

    if not periods:
        raise ValueError("periods list cannot be empty")

    for period in periods:
        if period <= 0:
            raise ValueError(f"Period must be positive, got {period}")

    df = df.copy()
    for period in periods:
        df[f"ma_{period}"] = df["close"].rolling(window=period).mean()
    return df


def calculate_rsi(data: pd.DataFrame | pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index using simple rolling mean.

    TODO: Unify RSI implementations - this uses simple rolling mean while
    src/ml/training_pipeline/features.py uses Wilder's smoothing (EMA-style).
    Different algorithms produce different RSI values, risking train/inference mismatch.
    Consolidate to single implementation with smoothing_method parameter.

    Args:
        data: DataFrame with 'close' column or Series of closing prices
        period: RSI period (must be positive, default: 14)

    Returns:
        Series of RSI values ranging from 0 to 100

    Raises:
        ValueError: If period is not positive or required columns are missing
    """
    if period <= 0:
        raise ValueError(f"RSI period must be positive, got {period}")

    if isinstance(data, pd.DataFrame):
        if "close" not in data.columns:
            raise ValueError("DataFrame must contain 'close' column")
        close = data["close"]
    else:
        close = pd.Series(data)

    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    # Prevent division by zero by adding epsilon to loss
    # When loss is 0, RS approaches infinity and RSI approaches 100
    rs = gain / (loss + EPSILON)
    rsi = 100 - (100 / (1 + rs))
    return rsi


@cached_indicator
def calculate_atr(df: pd.DataFrame, period: int = DEFAULT_ATR_PERIOD) -> pd.DataFrame:
    """
    Calculate Average True Range (ATR) volatility indicator.

    Args:
        df: DataFrame with OHLCV data containing 'high', 'low', 'close' columns
        period: ATR period (must be positive, default from constants)

    Returns:
        DataFrame with original data plus 'atr' column

    Raises:
        ValueError: If required columns are missing or period is invalid
    """
    required_cols = {"high", "low", "close"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")

    if period <= 0:
        raise ValueError(f"ATR period must be positive, got {period}")

    df = df.copy()
    high_low = df["high"] - df["low"]
    high_close = np.abs(df["high"] - df["close"].shift())
    low_close = np.abs(df["low"] - df["close"].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)  # Use pandas .max() to return Series, not np.max()
    df["atr"] = true_range.rolling(window=period).mean()
    return df


@cached_indicator
def calculate_bollinger_bands(
    df: pd.DataFrame, period: int = 20, std_dev: float = 2.0
) -> pd.DataFrame:
    """
    Calculate Bollinger Bands volatility indicator.

    Args:
        df: DataFrame with OHLCV data containing 'close' column
        period: Period for moving average and standard deviation (must be positive, default: 20)
        std_dev: Standard deviation multiplier for bands (default: 2.0)

    Returns:
        DataFrame with original data plus bb_middle, bb_std, bb_upper, bb_lower columns
        Note: bb_std is included for potential use in further calculations

    Raises:
        ValueError: If required columns are missing or period is invalid
    """
    if "close" not in df.columns:
        raise ValueError("DataFrame must contain 'close' column")

    if period <= 0:
        raise ValueError(f"Bollinger Bands period must be positive, got {period}")

    if std_dev <= 0:
        raise ValueError(f"Standard deviation multiplier must be positive, got {std_dev}")

    df = df.copy()
    df["bb_middle"] = df["close"].rolling(window=period).mean()
    df["bb_std"] = df["close"].rolling(window=period).std()
    df["bb_upper"] = df["bb_middle"] + (df["bb_std"] * std_dev)
    df["bb_lower"] = df["bb_middle"] - (df["bb_std"] * std_dev)
    return df


@cached_indicator
def calculate_macd(
    df: pd.DataFrame, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9
) -> pd.DataFrame:
    """
    Calculate MACD (Moving Average Convergence Divergence) momentum indicator.

    Args:
        df: DataFrame with OHLCV data containing 'close' column
        fast_period: Fast EMA period (must be positive, default: 12)
        slow_period: Slow EMA period (must be positive, default: 26)
        signal_period: Signal line EMA period (must be positive, default: 9)

    Returns:
        DataFrame with original data plus macd_fast, macd_slow, macd, macd_signal, macd_hist columns

    Raises:
        ValueError: If required columns are missing or periods are invalid
    """
    if "close" not in df.columns:
        raise ValueError("DataFrame must contain 'close' column")

    if fast_period <= 0:
        raise ValueError(f"Fast period must be positive, got {fast_period}")
    if slow_period <= 0:
        raise ValueError(f"Slow period must be positive, got {slow_period}")
    if signal_period <= 0:
        raise ValueError(f"Signal period must be positive, got {signal_period}")
    if fast_period >= slow_period:
        raise ValueError(
            f"Fast period ({fast_period}) must be less than slow period ({slow_period})"
        )

    df = df.copy()
    df["macd_fast"] = df["close"].ewm(span=fast_period, adjust=False).mean()
    df["macd_slow"] = df["close"].ewm(span=slow_period, adjust=False).mean()
    df["macd"] = df["macd_fast"] - df["macd_slow"]
    df["macd_signal"] = df["macd"].ewm(span=signal_period, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    return df


def detect_market_regime(
    df: pd.DataFrame,
    volatility_lookback: int = 20,
    trend_lookback: int = 50,
    regime_threshold: float = 0.01,
) -> pd.DataFrame:
    """
    Detect market regime (trending, ranging, volatile).

    Args:
        df: DataFrame with OHLCV data containing 'close' column
        volatility_lookback: Period for calculating current volatility (must be positive, default: 20)
        trend_lookback: Period for trend moving average (must be positive, default: 50)
        regime_threshold: Threshold for trend strength classification (default: 0.01 = 1%)

    Returns:
        DataFrame with original data plus 'volatility', 'trend', and 'regime' columns
        Regime values: 'volatile', 'trending', or 'ranging'

    Raises:
        ValueError: If required columns are missing or parameters are invalid
    """
    if "close" not in df.columns:
        raise ValueError("DataFrame must contain 'close' column")

    if volatility_lookback <= 0:
        raise ValueError(f"Volatility lookback must be positive, got {volatility_lookback}")
    if trend_lookback <= 0:
        raise ValueError(f"Trend lookback must be positive, got {trend_lookback}")
    if regime_threshold < 0:
        raise ValueError(f"Regime threshold must be non-negative, got {regime_threshold}")

    df = df.copy()

    # Calculate volatility
    df["volatility"] = df["close"].rolling(volatility_lookback).std()
    # Prevent division by zero with epsilon protection
    vol_mean = df["volatility"].rolling(VOLATILITY_BASELINE_PERIOD).mean()
    vol_ratio = df["volatility"] / (vol_mean + EPSILON)

    # Calculate trend strength
    df["trend"] = df["close"].rolling(trend_lookback).mean()
    # Prevent division by zero with epsilon protection
    trend_strength = (df["close"] - df["trend"]) / (df["trend"] + EPSILON)

    # Determine regime
    # Note: NaN values in early rows (from rolling windows) cause comparisons to return False,
    # defaulting to "ranging". This is intentional - we classify insufficient data as neutral/ranging.
    #
    # Precedence: First matching condition wins (np.select behavior)
    # 1. Volatile: High volatility AND low trend strength
    # 2. Trending: High trend strength (regardless of volatility)
    # 3. Ranging: Default for everything else (low volatility, low trend, or NaN)
    conditions = [
        (vol_ratio > VOLATILITY_RATIO_THRESHOLD)
        & (abs(trend_strength) < regime_threshold),  # Volatile
        (abs(trend_strength) > regime_threshold * TREND_STRENGTH_MULTIPLIER),  # Trending
    ]
    choices = ["volatile", "trending"]
    df["regime"] = np.select(conditions, choices, default="ranging")

    return df


def calculate_support_resistance(
    df: pd.DataFrame, period: int = 20, num_points: int = 5
) -> tuple[pd.Series, pd.Series]:
    """
    Calculate dynamic support and resistance levels.

    WARNING: This function uses center=True in rolling windows, which looks at both past
    AND future data. DO NOT use this for generating trading signals in backtests or live
    trading as it introduces look-ahead bias. This function is intended for historical
    analysis and visualization only.

    Args:
        df: DataFrame with OHLCV data containing 'high' and 'low' columns
        period: Period for rolling window to find local extrema (must be positive, default: 20)
        num_points: Number of support/resistance points to return (must be positive, default: 5)

    Returns:
        Tuple of (support_levels, resistance_levels) as pandas Series
        May return empty Series if no local extrema are found

    Raises:
        ValueError: If required columns are missing or parameters are invalid
    """
    required_cols = {"high", "low"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")

    if period <= 0:
        raise ValueError(f"Period must be positive, got {period}")
    if num_points <= 0:
        raise ValueError(f"num_points must be positive, got {num_points}")

    df = df.copy()

    # Find local maxima and minima
    df["high_roll"] = df["high"].rolling(window=period, center=True).max()
    df["low_roll"] = df["low"].rolling(window=period, center=True).min()

    resistance_levels = df[df["high"] == df["high_roll"]]["high"].tail(num_points)
    support_levels = df[df["low"] == df["low_roll"]]["low"].tail(num_points)

    return support_levels, resistance_levels


def calculate_ema(series: pd.Series, period: int = 9) -> pd.Series:
    """
    Calculate Exponential Moving Average of a series.

    Args:
        series: Pandas Series of values
        period: EMA period (must be positive, default: 9)

    Returns:
        Series of EMA values

    Raises:
        ValueError: If period is not positive
    """
    if period <= 0:
        raise ValueError(f"EMA period must be positive, got {period}")

    return series.ewm(span=period, adjust=False).mean()
