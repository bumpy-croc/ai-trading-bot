"""MarketDataHandler manages data fetching and preparation for live trading.

Handles data retrieval from providers, sentiment integration, indicator extraction,
and data freshness validation.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import pandas as pd

from src.config.constants import (
    DEFAULT_MARKET_DATA_LIMIT,
    DEFAULT_MARKET_DATA_STALENESS_THRESHOLD,
    DEFAULT_SENTIMENT_RECENT_WINDOW_HOURS,
)

if TYPE_CHECKING:
    from src.data_providers.base import DataProvider
    from src.sentiment.base import SentimentDataProvider

logger = logging.getLogger(__name__)


class MarketDataHandler:
    """Manages data fetching and preparation for live trading.

    This class encapsulates data-related operations including:
    - Fetching latest market data from providers
    - Adding sentiment data to price series
    - Checking data freshness
    - Extracting indicators for logging
    - Checking context readiness for strategy processing
    """

    def __init__(
        self,
        data_provider: DataProvider,
        sentiment_provider: SentimentDataProvider | None = None,
        data_limit: int = DEFAULT_MARKET_DATA_LIMIT,
        data_freshness_threshold: int = DEFAULT_MARKET_DATA_STALENESS_THRESHOLD,
        sentiment_lookback_hours: int = DEFAULT_SENTIMENT_RECENT_WINDOW_HOURS,
    ) -> None:
        """Initialize market data handler.

        Args:
            data_provider: Provider for market data.
            sentiment_provider: Provider for sentiment data.
            data_limit: Number of candles to fetch.
            data_freshness_threshold: Max age in seconds for data to be considered fresh.
            sentiment_lookback_hours: Hours to apply live sentiment.
        """
        self.data_provider = data_provider
        self.sentiment_provider = sentiment_provider
        self.data_limit = data_limit
        self.data_freshness_threshold = data_freshness_threshold
        self.sentiment_lookback_hours = sentiment_lookback_hours
        self.last_data_update: datetime | None = None

    def get_latest_data(
        self,
        symbol: str,
        timeframe: str,
    ) -> pd.DataFrame | None:
        """Fetch latest market data with error handling.

        Args:
            symbol: Trading symbol.
            timeframe: Candle timeframe.

        Returns:
            DataFrame with market data, or None if fetch failed.
        """
        try:
            df = self.data_provider.get_live_data(
                symbol, timeframe, limit=self.data_limit
            )
            self.last_data_update = datetime.now(UTC)
            return df
        except (ConnectionError, TimeoutError) as e:
            logger.error("Network error fetching market data: %s", e)
            return None
        except (ValueError, KeyError) as e:
            logger.error("Data parsing error in market data: %s", e)
            return None
        except (AttributeError, TypeError) as e:
            logger.error("Invalid data provider interface: %s", e, exc_info=True)
            return None

    def add_sentiment_data(
        self,
        df: pd.DataFrame,
        symbol: str,
    ) -> pd.DataFrame:
        """Add sentiment data to price dataframe.

        Args:
            df: Price data dataframe.
            symbol: Trading symbol.

        Returns:
            DataFrame with sentiment data added.
        """
        if self.sentiment_provider is None:
            return df

        try:
            if hasattr(self.sentiment_provider, "get_live_sentiment"):
                live_sentiment = self.sentiment_provider.get_live_sentiment()

                # Limit sentiment application to recent candles because live sentiment
                # reflects current market mood while historical candles need their
                # original context preserved for accurate strategy training and validation
                recent_mask = df.index >= (
                    df.index.max()
                    - pd.Timedelta(hours=self.sentiment_lookback_hours)
                )
                for feature, value in live_sentiment.items():
                    if feature not in df.columns:
                        df[feature] = 0.0
                    df.loc[recent_mask, feature] = value

                # Strategies use freshness flag to weight live sentiment vs historical context,
                # ensuring decisions prioritize current market mood when available
                df["sentiment_freshness"] = 0
                df.loc[recent_mask, "sentiment_freshness"] = 1

                logger.debug(
                    "Applied live sentiment to %d recent candles",
                    recent_mask.sum(),
                )
            else:
                logger.debug("Using historical sentiment data")

        except (AttributeError, TypeError) as e:
            logger.error("Sentiment provider interface error: %s", e)
        except (KeyError, ValueError) as e:
            logger.error("Invalid sentiment data format: %s", e)
        except (IndexError, pd.errors.InvalidIndexError) as e:
            logger.error("Failed to apply sentiment to dataframe indices: %s", e, exc_info=True)

        return df

    def is_data_fresh(self, df: pd.DataFrame) -> bool:
        """Check if the data is fresh enough to warrant processing.

        Args:
            df: Price data dataframe.

        Returns:
            True if data is fresh, False otherwise.
        """
        if df is None or df.empty:
            return False

        latest_timestamp = (
            df.index[-1]
            if hasattr(df.index[-1], "timestamp")
            else datetime.now(UTC)
        )
        if isinstance(latest_timestamp, str):
            try:
                latest_timestamp = pd.to_datetime(latest_timestamp)
            except (ValueError, TypeError):
                return True  # Assume fresh if we can't parse timestamp

        age_seconds = (datetime.now(UTC) - latest_timestamp).total_seconds()
        return age_seconds <= self.data_freshness_threshold

    def check_context_ready(
        self,
        df: pd.DataFrame,
        strategy: Any,
    ) -> tuple[bool, str]:
        """Check if the dataframe has enough context for strategy decisions.

        Args:
            df: Price data dataframe.
            strategy: Strategy instance for context requirements.

        Returns:
            Tuple of (ready, reason_if_not_ready).
        """
        try:
            rows = len(df)

            # Required rows from ML sequence length
            try:
                seq_len = int(getattr(strategy, "sequence_length", 0) or 0)
            except (TypeError, ValueError, AttributeError):
                seq_len = 0

            # Indicator window requirement
            try:
                max_window_attr = getattr(strategy, "max_indicator_window", 0)
                max_window = int(max_window_attr or 0)
            except (TypeError, ValueError, AttributeError):
                max_window = 0

            min_needed_base = max(seq_len, max_window)
            min_needed = (min_needed_base + 1) if min_needed_base > 0 else 2

            if rows < min_needed:
                return False, f"insufficient_rows:{rows}<min_needed:{min_needed}"

            # Strategies rely on essential OHLCV data being present to avoid
            # NaN-induced errors during indicator calculations and signal generation
            idx = rows - 1
            essentials = ["open", "high", "low", "close", "volume"]
            for col in essentials:
                try:
                    if pd.isna(df.iloc[idx][col]):
                        return False, f"nan_in_essentials:{col}"
                except (KeyError, IndexError):
                    return False, f"missing_essential:{col}"

            # ML strategies require predictions at current index to generate signals,
            # ensuring we don't attempt to trade without model inference results
            if seq_len > 0:
                if "onnx_pred" in df.columns:
                    try:
                        if pd.isna(df["onnx_pred"].iloc[idx]):
                            return False, "prediction_unavailable_at_current_index"
                    except (KeyError, IndexError):
                        return False, "prediction_column_access_error"

            # Data freshness check
            if not self.is_data_fresh(df):
                return False, "stale_data"

            return True, ""

        except (AttributeError, TypeError, ValueError) as e:
            logger.debug("Context readiness check failed: %s", e)
            return False, "readiness_check_error"

    def extract_indicators(
        self,
        df: pd.DataFrame,
        index: int,
    ) -> dict:
        """Extract indicator values from dataframe for logging.

        Args:
            df: Price data dataframe.
            index: Row index to extract from.

        Returns:
            Dictionary of indicator values.
        """
        if index >= len(df):
            return {}

        indicators: dict[str, float] = {}
        current_row = df.iloc[index]

        # Common indicators to extract
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
        ]

        for col in indicator_columns:
            if col in df.columns and not pd.isna(current_row[col]):
                indicators[col] = float(current_row[col])

        # Add basic OHLCV data
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                indicators[col] = float(current_row[col])

        return indicators

    def extract_sentiment_data(
        self,
        df: pd.DataFrame,
        index: int,
    ) -> dict:
        """Extract sentiment data from dataframe for logging.

        Args:
            df: Price data dataframe.
            index: Row index to extract from.

        Returns:
            Dictionary of sentiment values.
        """
        if index >= len(df):
            return {}

        sentiment_data: dict[str, float] = {}
        current_row = df.iloc[index]

        sentiment_columns = [
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

        for col in sentiment_columns:
            if col in df.columns and not pd.isna(current_row[col]):
                sentiment_data[col] = float(current_row[col])

        return sentiment_data

    def extract_ml_predictions(
        self,
        df: pd.DataFrame,
        index: int,
    ) -> dict:
        """Extract ML prediction data from dataframe for logging.

        Args:
            df: Price data dataframe.
            index: Row index to extract from.

        Returns:
            Dictionary of ML prediction values.
        """
        if index >= len(df):
            return {}

        ml_data: dict[str, float] = {}
        current_row = df.iloc[index]

        ml_columns = ["ml_prediction", "prediction_confidence", "onnx_pred"]

        for col in ml_columns:
            if col in df.columns and not pd.isna(current_row[col]):
                ml_data[col] = float(current_row[col])

        return ml_data
