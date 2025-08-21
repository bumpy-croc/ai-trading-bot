"""
Technical indicator feature extractor.

This module provides technical analysis features for prediction.
"""

import numpy as np
import pandas as pd
from typing import Optional
from indicators.technical import (
    calculate_atr,
    calculate_bollinger_bands,
    calculate_macd,
    calculate_moving_averages,
    calculate_rsi,
)

from config.constants import (
    DEFAULT_ATR_PERIOD,
    DEFAULT_BOLLINGER_PERIOD,
    DEFAULT_BOLLINGER_STD_DEV,
    DEFAULT_MA_PERIODS,
    DEFAULT_MACD_FAST_PERIOD,
    DEFAULT_MACD_SIGNAL_PERIOD,
    DEFAULT_MACD_SLOW_PERIOD,
    DEFAULT_NORMALIZATION_WINDOW,
    DEFAULT_RSI_PERIOD,
    DEFAULT_SEQUENCE_LENGTH,
)

from .base import FeatureExtractor
from .schemas import TECHNICAL_FEATURES_SCHEMA


class TechnicalFeatureExtractor(FeatureExtractor):
    """
    Extracts technical indicators and normalized price features from OHLCV data.

    This extractor consolidates technical analysis functionality into a single, reusable component.
    """

    def __init__(
        self,
        sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
        normalization_window: int = DEFAULT_NORMALIZATION_WINDOW,
        rsi_period: int = DEFAULT_RSI_PERIOD,
        atr_period: int = DEFAULT_ATR_PERIOD,
        bollinger_period: int = DEFAULT_BOLLINGER_PERIOD,
        bollinger_std_dev: float = DEFAULT_BOLLINGER_STD_DEV,
        ma_periods: Optional[list[int]] = None,
        macd_fast: int = DEFAULT_MACD_FAST_PERIOD,
        macd_slow: int = DEFAULT_MACD_SLOW_PERIOD,
        macd_signal: int = DEFAULT_MACD_SIGNAL_PERIOD,
        nan_threshold: float = 0.5,
    ):
        """
        Initialize the technical feature extractor.

        Args:
            sequence_length: Length of sequence for LSTM models
            normalization_window: Window size for price normalization
            rsi_period: Period for RSI calculation
            atr_period: Period for ATR calculation
            bollinger_period: Period for Bollinger Bands
            bollinger_std_dev: Standard deviation multiplier for Bollinger Bands
            ma_periods: Periods for moving averages
            macd_fast: Fast period for MACD
            macd_slow: Slow period for MACD
            macd_signal: Signal period for MACD
            nan_threshold: Maximum allowed ratio of NaN values in features
        """
        super().__init__("technical")

        # Store configuration
        self.sequence_length = sequence_length
        self.normalization_window = normalization_window
        self.rsi_period = rsi_period
        self.atr_period = atr_period
        self.bollinger_period = bollinger_period
        self.bollinger_std_dev = bollinger_std_dev
        self.ma_periods = ma_periods or DEFAULT_MA_PERIODS.copy()
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.nan_threshold = nan_threshold

        # Enable flags for different feature groups
        self.enable_bollinger = True
        self.enable_macd = True
        self.enable_moving_averages = True

        # Initialize feature names from schema
        self._feature_names = TECHNICAL_FEATURES_SCHEMA.get_feature_names()

    def extract(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract all technical features from OHLCV data.

        Args:
            data: DataFrame with OHLCV columns

        Returns:
            DataFrame with original data plus technical features

        Raises:
            ValueError: If input data is invalid
            RuntimeError: If feature extraction fails
        """
        if not self.validate_input(data):
            raise ValueError("Invalid input data: missing required OHLCV columns")

        try:
            # Make a copy to avoid modifying original data
            df = data.copy()

            # Extract technical indicators
            df = self._extract_technical_indicators(df)

            # Extract normalized price features
            df = self._extract_normalized_price_features(df)

            # Extract derived features
            df = self._extract_derived_features(df)

            return df

        except Exception as e:
            raise RuntimeError(f"Technical feature extraction failed: {str(e)}") from e

    def _extract_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract core technical indicators."""
        # Calculate ATR
        df = calculate_atr(df, period=self.atr_period)

        # Calculate moving averages
        df = calculate_moving_averages(df, periods=self.ma_periods)

        # Calculate Bollinger Bands
        df = calculate_bollinger_bands(
            df, period=self.bollinger_period, std_dev=self.bollinger_std_dev
        )

        # Calculate MACD
        df = calculate_macd(
            df,
            fast_period=self.macd_fast,
            slow_period=self.macd_slow,
            signal_period=self.macd_signal,
        )

        # Calculate RSI
        df["rsi"] = calculate_rsi(df, period=self.rsi_period)

        return df

    def _extract_normalized_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract normalized price features using rolling min-max normalization."""
        price_features = ["close", "volume", "high", "low", "open"]

        for feature in price_features:
            if feature in df.columns:
                # Rolling min-max normalization using vectorized operations
                rolling_min = (
                    df[feature].rolling(window=self.normalization_window, min_periods=1).min()
                )
                rolling_max = (
                    df[feature].rolling(window=self.normalization_window, min_periods=1).max()
                )

                # Handle NaN values explicitly before division
                valid_mask = (
                    (rolling_max != rolling_min) & rolling_max.notna() & rolling_min.notna()
                )
                df[f"{feature}_normalized"] = np.where(
                    valid_mask,
                    (df[feature] - rolling_min) / (rolling_max - rolling_min),
                    0.5,  # Handle cases where min == max or NaN - use neutral value instead of minimum
                )
        return df

    def _extract_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract derived features like volatility and trend measures."""
        # Calculate returns
        df["returns"] = df["close"].pct_change()

        # Calculate volatility metrics (from MlAdaptive)
        df["volatility_20"] = df["returns"].rolling(window=20).std()
        df["volatility_50"] = df["returns"].rolling(window=50).std()

        # Calculate ATR as percentage of price
        df["atr_pct"] = df["atr"] / df["close"]

        # Calculate trend measures (from MlAdaptive)
        df["trend_strength"] = (df["close"] - df["ma_50"]) / df["ma_50"]
        df["trend_direction"] = np.where(df["ma_20"] > df["ma_50"], 1, -1)

        return df

    def get_feature_names(self) -> list[str]:
        """Return list of feature names this extractor produces."""
        return self._feature_names.copy()

    def get_normalized_features(self) -> list[str]:
        """Get list of normalized price feature names."""
        return [f"{feature}_normalized" for feature in ["close", "volume", "high", "low", "open"]]

    def get_technical_indicators(self) -> list[str]:
        """Get list of technical indicator names."""
        indicators = ["rsi", "atr", "atr_pct"]
        if self.enable_bollinger:
            indicators.extend(["bb_upper", "bb_lower", "bb_width", "bb_position"])
        if self.enable_macd:
            indicators.extend(["macd", "macd_signal", "macd_histogram"])
        if self.enable_moving_averages:
            for period in self.ma_periods:
                indicators.extend([f"ma_{period}", f"ma_{period}_pct"])
        return indicators

    def get_derived_features(self) -> list[str]:
        """Get list of derived feature names."""
        return ["returns", "volatility_20", "volatility_50", "trend_strength", "trend_direction"]

    def validate_features(self, data: pd.DataFrame) -> dict[str, bool]:
        """
        Validate that all expected features are present and valid.

        Args:
            data: DataFrame with extracted features

        Returns:
            Dictionary mapping feature names to validation status
        """
        validation_results = {}
        expected_features = self.get_feature_names()

        for feature in expected_features:
            if feature not in data.columns:
                validation_results[feature] = False
                continue

            # Check for infinite values
            if np.isinf(data[feature]).any():
                validation_results[feature] = False
                continue

            # Check for excessive NaN values
            nan_ratio = data[feature].isna().sum() / len(data[feature])
            if nan_ratio > self.nan_threshold:
                validation_results[feature] = False
                continue

            validation_results[feature] = True

        return validation_results

    def get_config(self) -> dict:
        """Get configuration parameters for this extractor."""
        config = super().get_config()
        config.update(
            {
                "sequence_length": self.sequence_length,
                "normalization_window": self.normalization_window,
                "rsi_period": self.rsi_period,
                "atr_period": self.atr_period,
                "bollinger_period": self.bollinger_period,
                "bollinger_std_dev": self.bollinger_std_dev,
                "ma_periods": self.ma_periods,
                "macd_fast": self.macd_fast,
                "macd_slow": self.macd_slow,
                "macd_signal": self.macd_signal,
            }
        )
        return config

    def get_feature_importance_weights(self) -> dict[str, float]:
        """
        Get feature importance weights based on common usage in strategies.

        Returns:
            Dictionary mapping feature names to importance weights
        """
        weights = {}
        for feature in self.get_feature_names():
            # Normalized features get highest weights (most important for ML models)
            if feature.endswith("_normalized"):
                weights[feature] = 1.0
            # Technical indicators get high weights
            elif feature in ["rsi", "atr", "macd", "bb_position"]:
                weights[feature] = 0.8
            # Derived features get medium weights
            elif feature in ["returns", "volatility_20", "trend_strength"]:
                weights[feature] = 0.8
            # Moving averages get lower weights
            elif feature.startswith("ma_"):
                weights[feature] = 0.5
            else:
                weights[feature] = 0.6
        return weights
