import numpy as np
import pandas as pd

from src.config.constants import DEFAULT_ATR_PERIOD


def calculate_moving_averages(df: pd.DataFrame, periods: list) -> pd.DataFrame:
    """Calculate simple moving averages for multiple periods"""
    df = df.copy()
    for period in periods:
        df[f"ma_{period}"] = df["close"].rolling(window=period).mean()
    return df


def calculate_rsi(data, period: int = 14):
    """Calculate Relative Strength Index.

    Accepts either a DataFrame with a 'close' column or a Series of closing prices.
    Returns a pandas Series of RSI values.
    """
    if isinstance(data, pd.DataFrame):
        close = data["close"]
    else:
        close = pd.Series(data)

    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_atr(df: pd.DataFrame, period: int = DEFAULT_ATR_PERIOD) -> pd.DataFrame:
    """Calculate Average True Range"""
    df = df.copy()
    high_low = df["high"] - df["low"]
    high_close = np.abs(df["high"] - df["close"].shift())
    low_close = np.abs(df["low"] - df["close"].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df["atr"] = true_range.rolling(window=period).mean()
    return df


def calculate_bollinger_bands(
    df: pd.DataFrame, period: int = 20, std_dev: float = 2.0
) -> pd.DataFrame:
    """Calculate Bollinger Bands"""
    df = df.copy()
    df["bb_middle"] = df["close"].rolling(window=period).mean()
    df["bb_std"] = df["close"].rolling(window=period).std()
    df["bb_upper"] = df["bb_middle"] + (df["bb_std"] * std_dev)
    df["bb_lower"] = df["bb_middle"] - (df["bb_std"] * std_dev)
    return df


def calculate_macd(
    df: pd.DataFrame, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9
) -> pd.DataFrame:
    """Calculate MACD (Moving Average Convergence Divergence)"""
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
    """Detect market regime (trending, ranging, volatile)"""
    df = df.copy()

    # Calculate volatility
    df["volatility"] = df["close"].rolling(volatility_lookback).std()
    vol_ratio = df["volatility"] / df["volatility"].rolling(50).mean()

    # Calculate trend strength
    df["trend"] = df["close"].rolling(trend_lookback).mean()
    trend_strength = (df["close"] - df["trend"]) / df["trend"]

    # Determine regime
    conditions = [
        (vol_ratio > 1.3) & (abs(trend_strength) < regime_threshold),  # Volatile
        (abs(trend_strength) > regime_threshold * 0.5),  # Trending
    ]
    choices = ["volatile", "trending"]
    df["regime"] = np.select(conditions, choices, default="ranging")

    return df


def calculate_support_resistance(
    df: pd.DataFrame, period: int = 20, num_points: int = 5
) -> tuple[pd.Series, pd.Series]:
    """Calculate dynamic support and resistance levels"""
    df = df.copy()

    # Find local maxima and minima
    df["high_roll"] = df["high"].rolling(window=period, center=True).max()
    df["low_roll"] = df["low"].rolling(window=period, center=True).min()

    resistance_levels = df[df["high"] == df["high_roll"]]["high"].tail(num_points)
    support_levels = df[df["low"] == df["low_roll"]]["low"].tail(num_points)

    return support_levels, resistance_levels


def calculate_ema(series: pd.Series, period: int = 9) -> pd.Series:
    """Calculate Exponential Moving Average of a series"""
    return series.ewm(span=period, adjust=False).mean()
