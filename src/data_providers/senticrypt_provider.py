import logging
import os
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import requests

from .sentiment_provider import SentimentDataProvider

logger = logging.getLogger(__name__)


class SentiCryptProvider(SentimentDataProvider):
    """
    SentiCrypt sentiment data provider using their API or local CSV data.
    Provides normalized sentiment scores correlated with Bitcoin price data.
    """

    def __init__(
        self,
        csv_path: Optional[str] = None,
        api_url: str = "https://api.senticrypt.com/v2/all.json",
        live_mode: bool = False,
        cache_duration_minutes: int = 30,
    ):
        super().__init__()
        self.csv_path = csv_path
        self.api_url = api_url
        self.live_mode = live_mode
        self.cache_duration_minutes = cache_duration_minutes
        self.data = None
        self._live_cache = {}
        self._last_api_call = None
        self._load_data()

    def _load_data(self):
        """Load sentiment data from CSV file or API"""
        try:
            if self.csv_path and os.path.exists(self.csv_path) and not self.live_mode:
                logger.info(f"Loading sentiment data from CSV: {self.csv_path}")
                self.data = pd.read_csv(self.csv_path)
            else:
                logger.info(f"Fetching sentiment data from API: {self.api_url}")
                response = requests.get(self.api_url, timeout=30)
                response.raise_for_status()
                self.data = pd.DataFrame(response.json())

            # Process the data
            self._process_data()

        except Exception as e:
            logger.error(f"Failed to load sentiment data: {e}")
            self.data = pd.DataFrame()

    def _process_data(self):
        """Process and normalize the sentiment data"""
        if self.data.empty:
            return

        # Convert date to datetime
        self.data["date"] = pd.to_datetime(self.data["date"])
        self.data.set_index("date", inplace=True)
        self.data.sort_index(inplace=True)

        # Create normalized sentiment features
        self._create_sentiment_features()

        logger.info(
            f"Processed {len(self.data)} sentiment records from {self.data.index.min()} to {self.data.index.max()}"
        )

    def _create_sentiment_features(self):
        """Create normalized sentiment features from raw scores"""
        # Normalize individual scores to [-1, 1] range
        for col in ["score1", "score2", "score3"]:
            if col in self.data.columns:
                # Use tanh to normalize to [-1, 1] range, preserving extreme values
                self.data[f"{col}_norm"] = np.tanh(self.data[col])

        # Create composite sentiment score
        if all(col in self.data.columns for col in ["score1", "score2", "score3"]):
            # Weighted average of the three scores (you can adjust weights based on importance)
            weights = [0.4, 0.3, 0.3]  # score1 gets higher weight
            self.data["sentiment_composite"] = (
                weights[0] * self.data["score1_norm"]
                + weights[1] * self.data["score2_norm"]
                + weights[2] * self.data["score3_norm"]
            )

        # Use the mean score as primary sentiment if available
        if "mean" in self.data.columns:
            self.data["sentiment_primary"] = np.tanh(self.data["mean"])
        else:
            self.data["sentiment_primary"] = self.data.get("sentiment_composite", 0)

        # Create sentiment momentum (rate of change)
        self.data["sentiment_momentum"] = self.data["sentiment_primary"].pct_change().fillna(0)

        # Create sentiment volatility (rolling standard deviation)
        self.data["sentiment_volatility"] = (
            self.data["sentiment_primary"].rolling(window=7).std().fillna(0)
        )

        # Create sentiment extremes (binary flags for very positive/negative sentiment)
        sentiment_std = self.data["sentiment_primary"].std()
        sentiment_mean = self.data["sentiment_primary"].mean()

        self.data["sentiment_extreme_positive"] = (
            self.data["sentiment_primary"] > sentiment_mean + 2 * sentiment_std
        ).astype(int)

        self.data["sentiment_extreme_negative"] = (
            self.data["sentiment_primary"] < sentiment_mean - 2 * sentiment_std
        ).astype(int)

        # Create sentiment moving averages
        for window in [3, 7, 14]:
            self.data[f"sentiment_ma_{window}"] = (
                self.data["sentiment_primary"].rolling(window=window).mean()
            )

    def get_historical_sentiment(
        self, symbol: str, start: datetime, end: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Get historical sentiment data for the specified period.

        Args:
            symbol: Trading pair symbol (currently only supports BTC-related pairs)
            start: Start datetime
            end: End datetime (optional, defaults to current time)

        Returns:
            DataFrame with sentiment features aligned to the date range
        """
        if self.data.empty:
            logger.warning("No sentiment data available")
            return pd.DataFrame()

        # SentiCrypt data is BTC-focused, so we'll use it for all BTC pairs
        if "BTC" not in symbol.upper():
            logger.warning(f"SentiCrypt data is BTC-focused, using for {symbol} with caution")

        # Filter data by date range
        if end is None:
            end = datetime.now()

        # Ensure timezone consistency for comparison
        if self.data.index.tz is not None:
            # Data index is timezone-aware, make start/end timezone-aware too
            if start.tzinfo is None:
                start = start.replace(tzinfo=self.data.index.tz)
            if end.tzinfo is None:
                end = end.replace(tzinfo=self.data.index.tz)
        else:
            # Data index is timezone-naive, make start/end timezone-naive too
            if start.tzinfo is not None:
                start = start.replace(tzinfo=None)
            if end.tzinfo is not None:
                end = end.replace(tzinfo=None)

        mask = (self.data.index >= start) & (self.data.index <= end)
        filtered_data = self.data.loc[mask].copy()

        if filtered_data.empty:
            logger.warning(f"No sentiment data found for period {start} to {end}")
            return pd.DataFrame()

        # Select relevant columns for ML model
        sentiment_columns = [
            "sentiment_primary",
            "sentiment_momentum",
            "sentiment_volatility",
            "sentiment_extreme_positive",
            "sentiment_extreme_negative",
            "sentiment_ma_3",
            "sentiment_ma_7",
            "sentiment_ma_14",
        ]

        # Add raw scores if needed
        for col in ["score1_norm", "score2_norm", "score3_norm"]:
            if col in filtered_data.columns:
                sentiment_columns.append(col)

        result = filtered_data[sentiment_columns].copy()

        # Forward fill missing values
        result = result.ffill().fillna(0)

        logger.info(f"Retrieved {len(result)} sentiment records for {symbol} from {start} to {end}")
        return result

    def calculate_sentiment_score(self, sentiment_data: list[dict]) -> float:
        """
        Calculate a single normalized sentiment score from raw sentiment data.

        Args:
            sentiment_data: List of sentiment data points

        Returns:
            Normalized sentiment score between -1 and 1
        """
        if not sentiment_data:
            return 0.0

        # Extract mean scores and calculate average
        scores = []
        for item in sentiment_data:
            if "mean" in item:
                scores.append(item["mean"])
            elif all(key in item for key in ["score1", "score2", "score3"]):
                # Calculate weighted average if individual scores are available
                composite = 0.4 * item["score1"] + 0.3 * item["score2"] + 0.3 * item["score3"]
                scores.append(composite)

        if not scores:
            return 0.0

        # Return normalized average
        avg_score = np.mean(scores)
        return np.tanh(avg_score)  # Normalize to [-1, 1]

    def get_live_sentiment(self, force_refresh: bool = False) -> dict[str, float]:
        """
        Get the most recent sentiment data for live trading.

        Args:
            force_refresh: Force API call even if cache is valid

        Returns:
            Dictionary of current sentiment features
        """
        current_time = datetime.now()

        # Check if we need to refresh the cache
        should_refresh = (
            force_refresh
            or self._last_api_call is None
            or (current_time - self._last_api_call).total_seconds()
            > (self.cache_duration_minutes * 60)
        )

        if should_refresh:
            try:
                logger.info("Fetching fresh sentiment data for live trading")
                response = requests.get(self.api_url, timeout=10)
                response.raise_for_status()

                fresh_data = response.json()
                if fresh_data:
                    # Get the most recent entry
                    latest_entry = max(fresh_data, key=lambda x: x.get("date", ""))

                    # Calculate live sentiment features
                    live_sentiment = self._calculate_live_sentiment_features(
                        latest_entry, fresh_data[-10:]
                    )

                    # Update cache
                    self._live_cache = live_sentiment
                    self._last_api_call = current_time

                    logger.info(
                        f"Updated live sentiment cache: {live_sentiment.get('sentiment_primary', 0):.3f}"
                    )
                    return live_sentiment

            except Exception as e:
                logger.error(f"Failed to fetch live sentiment: {e}")

        # Return cached data or fallback
        if self._live_cache:
            logger.debug("Using cached sentiment data")
            return self._live_cache
        else:
            logger.warning("No live sentiment data available, using neutral values")
            return self._get_neutral_sentiment()

    def _calculate_live_sentiment_features(
        self, latest_entry: dict, recent_entries: list[dict]
    ) -> dict[str, float]:
        """
        Calculate sentiment features from fresh API data.

        Args:
            latest_entry: Most recent sentiment data point
            recent_entries: List of recent entries for momentum/volatility calculation

        Returns:
            Dictionary of calculated sentiment features
        """
        features = {}

        # Primary sentiment score
        if "mean" in latest_entry:
            features["sentiment_primary"] = np.tanh(latest_entry["mean"])
        elif all(key in latest_entry for key in ["score1", "score2", "score3"]):
            composite = (
                0.4 * latest_entry["score1"]
                + 0.3 * latest_entry["score2"]
                + 0.3 * latest_entry["score3"]
            )
            features["sentiment_primary"] = np.tanh(composite)
        else:
            features["sentiment_primary"] = 0.0

        # Calculate momentum from recent entries
        if len(recent_entries) >= 2:
            recent_scores = []
            for entry in recent_entries:
                if "mean" in entry:
                    recent_scores.append(np.tanh(entry["mean"]))
                elif all(key in entry for key in ["score1", "score2", "score3"]):
                    composite = (
                        0.4 * entry["score1"] + 0.3 * entry["score2"] + 0.3 * entry["score3"]
                    )
                    recent_scores.append(np.tanh(composite))

            if len(recent_scores) >= 2:
                # Calculate momentum as rate of change
                features["sentiment_momentum"] = (recent_scores[-1] - recent_scores[-2]) / abs(
                    recent_scores[-2] + 1e-8
                )

                # Calculate volatility
                features["sentiment_volatility"] = (
                    np.std(recent_scores) if len(recent_scores) > 1 else 0.0
                )

                # Moving averages
                features["sentiment_ma_3"] = (
                    np.mean(recent_scores[-3:]) if len(recent_scores) >= 3 else recent_scores[-1]
                )
                features["sentiment_ma_7"] = (
                    np.mean(recent_scores[-7:])
                    if len(recent_scores) >= 7
                    else np.mean(recent_scores)
                )
                features["sentiment_ma_14"] = (
                    np.mean(recent_scores) if len(recent_scores) >= 10 else np.mean(recent_scores)
                )
            else:
                features["sentiment_momentum"] = 0.0
                features["sentiment_volatility"] = 0.0
                features["sentiment_ma_3"] = features["sentiment_primary"]
                features["sentiment_ma_7"] = features["sentiment_primary"]
                features["sentiment_ma_14"] = features["sentiment_primary"]
        else:
            features["sentiment_momentum"] = 0.0
            features["sentiment_volatility"] = 0.0
            features["sentiment_ma_3"] = features["sentiment_primary"]
            features["sentiment_ma_7"] = features["sentiment_primary"]
            features["sentiment_ma_14"] = features["sentiment_primary"]

        # Extreme sentiment flags
        sentiment_threshold = 0.5  # Adjust based on historical data
        features["sentiment_extreme_positive"] = (
            1.0 if features["sentiment_primary"] > sentiment_threshold else 0.0
        )
        features["sentiment_extreme_negative"] = (
            1.0 if features["sentiment_primary"] < -sentiment_threshold else 0.0
        )

        return features

    def _get_neutral_sentiment(self) -> dict[str, float]:
        """Return neutral sentiment values as fallback"""
        return {
            "sentiment_primary": 0.0,
            "sentiment_momentum": 0.0,
            "sentiment_volatility": 0.0,
            "sentiment_extreme_positive": 0.0,
            "sentiment_extreme_negative": 0.0,
            "sentiment_ma_3": 0.0,
            "sentiment_ma_7": 0.0,
            "sentiment_ma_14": 0.0,
        }

    def get_sentiment_for_date(self, date: datetime, use_live: bool = False) -> dict[str, float]:
        """
        Get sentiment features for a specific date.

        Args:
            date: The date to get sentiment for
            use_live: If True and date is recent, use live API data

        Returns:
            Dictionary of sentiment features
        """
        # If requesting very recent data and live mode is enabled, use live API
        if use_live and self.live_mode:
            time_diff = datetime.now() - date
            if time_diff.total_seconds() < 3600:  # Within last hour
                return self.get_live_sentiment()

        if self.data.empty:
            return self._get_neutral_sentiment()

        # Find the closest date
        closest_date = self.data.index[self.data.index.get_indexer([date], method="nearest")[0]]

        if abs((closest_date - date).days) > 7:  # If more than 7 days away, return zeros
            return {col: 0.0 for col in self.data.columns if col.startswith("sentiment_")}

        row = self.data.loc[closest_date]
        return {col: row[col] for col in self.data.columns if col.startswith("sentiment_")}

    def resample_to_timeframe(self, timeframe: str = "1h") -> pd.DataFrame:
        """
        Resample sentiment data to match price data timeframe.

        Args:
            timeframe: Target timeframe (e.g., '1h', '4h', '1d')

        Returns:
            Resampled DataFrame
        """
        if self.data.empty:
            return pd.DataFrame()

        # Define aggregation methods for different columns
        agg_methods = {}

        # Sentiment scores: use mean
        for col in self.data.columns:
            if col.startswith("sentiment_"):
                if "extreme" in col:
                    agg_methods[col] = "max"  # Use max for extreme flags
                else:
                    agg_methods[col] = "mean"
            elif col in ["score1_norm", "score2_norm", "score3_norm"]:
                agg_methods[col] = "mean"
            elif col in ["count", "volume"]:
                agg_methods[col] = "sum"
            elif col in ["price"]:
                agg_methods[col] = "last"  # Use last price in period
            else:
                agg_methods[col] = "mean"

        # Resample the data
        resampled = self.data.resample(timeframe).agg(agg_methods)

        # Forward fill missing values
        resampled = resampled.ffill()

        return resampled
