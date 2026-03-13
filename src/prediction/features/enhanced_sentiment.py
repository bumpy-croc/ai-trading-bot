"""
Enhanced sentiment feature extractor.

Combines fear/greed index, social volume, and news sentiment
into a composite sentiment signal.
"""

import logging

import numpy as np
import pandas as pd

from src.config.constants import DEFAULT_ENABLE_ENHANCED_SENTIMENT
from src.data_providers.feargreed_provider import FearGreedProvider
from src.tech.features.base import FeatureExtractor

logger = logging.getLogger(__name__)


class EnhancedSentimentExtractor(FeatureExtractor):
    """
    Extracts enhanced sentiment features combining multiple sentiment sources.

    Produces normalized [-1, 1] features from fear/greed index, simulated social
    volume, and news sentiment. Integrates with the existing FearGreedProvider
    and provides clear interfaces for adding real social/news data sources.
    """

    def __init__(self, enabled: bool = DEFAULT_ENABLE_ENHANCED_SENTIMENT):
        """
        Initialize the enhanced sentiment extractor.

        Args:
            enabled: Whether enhanced sentiment extraction is enabled
        """
        super().__init__("enhanced_sentiment")
        self.enabled = enabled
        self._provider: FearGreedProvider | None = None
        self._feature_names = [
            "fear_greed_normalized",
            "social_volume_zscore",
            "news_sentiment_score",
            "composite_sentiment",
        ]

    def extract(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract enhanced sentiment features.

        When disabled, returns neutral values (0.0) for all features.

        Args:
            data: DataFrame with OHLCV market data

        Returns:
            DataFrame with original data plus enhanced sentiment features
        """
        if not self.validate_input(data):
            raise ValueError("Invalid input data: missing required OHLCV columns")

        df = data.copy()

        if not self.enabled:
            for feature in self._feature_names:
                df[feature] = 0.0
            return df

        df = self._compute_fear_greed(df)
        df = self._compute_social_volume(df)
        df = self._compute_news_sentiment(df)
        df = self._compute_composite(df)

        return df

    def _compute_fear_greed(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute normalized fear/greed signal from FearGreedProvider.

        Maps the [0, 1] fear/greed index to [-1, 1] where -1 is extreme fear
        and +1 is extreme greed. Falls back to price-derived proxy on failure.
        """
        fear_greed_values = self._get_fear_greed_values(df)

        if fear_greed_values is not None:
            # Map [0, 1] to [-1, 1]
            df["fear_greed_normalized"] = np.clip(fear_greed_values * 2.0 - 1.0, -1.0, 1.0)
        else:
            # Fallback: derive from RSI-like oscillator on price
            returns = df["close"].pct_change().fillna(0.0)
            cumulative = returns.rolling(window=14, min_periods=1).sum()
            df["fear_greed_normalized"] = np.clip(cumulative * 10.0, -1.0, 1.0)

        return df

    def _get_fear_greed_values(self, df: pd.DataFrame) -> pd.Series | None:
        """
        Attempt to get fear/greed values from the provider.

        Returns:
            Series of sentiment_primary values aligned to df index, or None on failure
        """
        if self._provider is None:
            try:
                self._provider = FearGreedProvider()
            except Exception as e:
                logger.warning("FearGreedProvider init failed: %s", e)
                return None

        if self._provider.data.empty:
            return None

        try:
            # Get the sentiment_primary column from provider data
            sentiment_data = self._provider.data[["sentiment_primary"]].copy()

            # Normalize timezone for safe join
            if isinstance(sentiment_data.index, pd.DatetimeIndex):
                sentiment_data.index = (
                    pd.to_datetime(sentiment_data.index, utc=True)
                    .tz_convert("UTC")
                    .tz_localize(None)
                )

            work_index = df.index
            if isinstance(work_index, pd.DatetimeIndex):
                work_index = (
                    pd.to_datetime(work_index, utc=True)
                    .tz_convert("UTC")
                    .tz_localize(None)
                )

            # Reindex to data's index with forward fill
            aligned = sentiment_data.reindex(work_index, method="ffill")
            if aligned["sentiment_primary"].isna().all():
                return None
            return aligned["sentiment_primary"].fillna(0.5)
        except Exception as e:
            logger.warning("Failed to align fear/greed data: %s", e)
            return None

    def _compute_social_volume(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute social volume z-score signal.

        Simulates social media volume using trading volume as a proxy.
        High social volume relative to recent history indicates heightened interest.

        Replace with real social volume data from LunarCrush or Santiment.
        """
        log_volume = np.log1p(df["volume"])
        rolling_mean = log_volume.rolling(window=20, min_periods=1).mean()
        rolling_std = log_volume.rolling(window=20, min_periods=1).std().fillna(1e-9).replace(0, 1e-9)
        zscore = ((log_volume - rolling_mean) / rolling_std).fillna(0.0)
        df["social_volume_zscore"] = np.clip(zscore / 3.0, -1.0, 1.0)
        return df

    def _compute_news_sentiment(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute news sentiment score.

        Simulates NLP-based news sentiment using momentum of price returns.
        Sustained directional moves proxy for positive/negative news flow.

        Replace with real news NLP data from CryptoPanic or custom NLP pipeline.
        """
        returns = df["close"].pct_change().fillna(0.0)
        # Exponential moving average of returns as sentiment proxy
        ema_returns = returns.ewm(span=10, min_periods=1).mean()
        df["news_sentiment_score"] = np.clip(ema_returns * 20.0, -1.0, 1.0)
        return df

    def _compute_composite(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute composite sentiment signal from all sentiment sources.

        Weighted average: fear/greed (40%), social volume (30%), news (30%).
        """
        df["composite_sentiment"] = np.clip(
            0.4 * df["fear_greed_normalized"]
            + 0.3 * df["social_volume_zscore"]
            + 0.3 * df["news_sentiment_score"],
            -1.0,
            1.0,
        )
        return df

    def get_feature_names(self) -> list[str]:
        """Return list of feature names this extractor produces."""
        return self._feature_names.copy()

    def get_config(self) -> dict:
        """Get configuration parameters for this extractor."""
        config = super().get_config()
        config.update({
            "enabled": self.enabled,
            "has_provider": self._provider is not None,
        })
        return config
