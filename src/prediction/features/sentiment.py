"""
Sentiment Feature Extractor

This module extracts sentiment-based features from market sentiment data.
For MVP, this extractor is disabled and returns neutral sentiment values.
"""

from typing import List

import pandas as pd

from src.config.constants import DEFAULT_ENABLE_SENTIMENT

from .base import FeatureExtractor
from .schemas import SENTIMENT_FEATURES_SCHEMA
from src.data_providers.feargreed_provider import FearGreedProvider
from datetime import datetime, timezone


class SentimentFeatureExtractor(FeatureExtractor):
    """
    Extracts sentiment features from market sentiment data.

    For MVP, this extractor is disabled and provides neutral sentiment values
    to maintain model compatibility.
    """

    def __init__(self, enabled: bool = DEFAULT_ENABLE_SENTIMENT):
        """
        Initialize the sentiment feature extractor.

        Args:
            enabled: Whether sentiment extraction is enabled (False for MVP)
        """
        super().__init__("sentiment")
        self.enabled = enabled
        self._feature_names = SENTIMENT_FEATURES_SCHEMA.get_feature_names()
        # Lazily instantiate the provider only if/when needed
        self._provider = None
    
    def extract(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract sentiment features from data.

        For MVP, this returns neutral sentiment values.

        Args:
            data: DataFrame with market data

        Returns:
            DataFrame with original data plus sentiment features
        """
        if not self.validate_input(data):
            raise ValueError("Invalid input data: missing required OHLCV columns")

        df = data.copy()

        if not self.enabled:
            # MVP: Return neutral sentiment values
            return self._add_neutral_sentiment_features(df)
        else:
            # Implement actual sentiment extraction (Fear & Greed)
            return self._extract_sentiment_features(df)

    def _add_neutral_sentiment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add neutral sentiment feature values for MVP compatibility."""
        # Add neutral sentiment values as defined in schema
        for feature_def in SENTIMENT_FEATURES_SCHEMA.features:
            if feature_def.default_value is not None:
                df[feature_def.name] = feature_def.default_value
            else:
                # Default neutral values
                if "primary" in feature_def.name:
                    df[feature_def.name] = 0.5  # Neutral sentiment
                elif "momentum" in feature_def.name:
                    df[feature_def.name] = 0.0  # No momentum
                elif "volatility" in feature_def.name:
                    df[feature_def.name] = 0.3  # Low-moderate volatility
                elif "confidence" in feature_def.name:
                    df[feature_def.name] = 0.7  # Moderate confidence
                else:
                    df[feature_def.name] = 0.0  # Default neutral

        return df

    def _extract_sentiment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract actual sentiment features using FearGreedProvider.
        Joins on timestamp index (expects df indexed by datetime or with a 'timestamp' column).
        """
        # Ensure provider is created only when sentiment is enabled and needed
        if self._provider is None:
            try:
                self._provider = FearGreedProvider()
            except Exception:
                # If provider init fails (e.g., no network), fallback to neutral
                return self._add_neutral_sentiment_features(df)
        # Ensure datetime index
        work = df.copy()
        if 'timestamp' in work.columns and not isinstance(work.index, pd.DatetimeIndex):
            work = work.set_index('timestamp')
        # Normalize price data index to timezone-naive (UTC without tzinfo)
        if isinstance(work.index, pd.DatetimeIndex):
            try:
                # Convert everything to UTC then drop tz to avoid tz-aware/naive join errors
                work.index = pd.to_datetime(work.index, utc=True).tz_convert('UTC').tz_localize(None)
            except Exception:
                # Best effort standardization; if it fails, fallback to neutral to be safe
                return self._add_neutral_sentiment_features(df)
        if not isinstance(work.index, pd.DatetimeIndex):
            # Cannot merge robustly; fall back to neutral
            return self._add_neutral_sentiment_features(df)

        start = work.index.min()
        end = work.index.max()
        try:
            # Freshness gate: if provider data is stale, fallback to neutral
            now_ts = end if isinstance(end, pd.Timestamp) else pd.Timestamp(end)
            if not self._provider._is_fresh(now_ts.to_pydatetime()):
                return self._add_neutral_sentiment_features(df)
            sents = self._provider.get_historical_sentiment(symbol="BTCUSDT", start=start, end=end)
        except Exception:
            sents = pd.DataFrame()
        
        if sents.empty:
            return self._add_neutral_sentiment_features(df)
        
        # Resample sentiment to the data's frequency (infer)
        try:
            inferred = pd.infer_freq(work.index)
        except Exception:
            inferred = None
        if inferred is None:
            # Default to daily if cannot infer; forward fill will align
            inferred = '1D'
        sents_resampled = self._provider.aggregate_sentiment(sents, window=inferred)
        # Normalize sentiment index to timezone-naive (UTC without tzinfo)
        if isinstance(sents_resampled.index, pd.DatetimeIndex):
            try:
                sents_resampled.index = (
                    pd.to_datetime(sents_resampled.index, utc=True)
                    .tz_convert('UTC')
                    .tz_localize(None)
                )
            except Exception:
                return self._add_neutral_sentiment_features(df)
        
        # Join and forward-fill sentiment features; do not backfill into the future
        merged = work.join(sents_resampled, how='left')
        merged = merged.ffill()
        
        # If still missing (all NaN), fill with neutral defaults from schema
        for feature_def in SENTIMENT_FEATURES_SCHEMA.features:
            if feature_def.name not in merged.columns or merged[feature_def.name].isna().all():
                default_val = feature_def.default_value
                if default_val is None:
                    if 'primary' in feature_def.name:
                        default_val = 0.5
                    elif 'momentum' in feature_def.name:
                        default_val = 0.0
                    elif 'volatility' in feature_def.name:
                        default_val = 0.3
                    elif 'confidence' in feature_def.name:
                        default_val = 0.7
                    else:
                        default_val = 0.0
                merged[feature_def.name] = merged.get(feature_def.name, pd.Series(index=merged.index, dtype=float)).fillna(default_val)
        
        # Return on original index/columns order plus new features
        result = merged
        if 'timestamp' in df.columns and not isinstance(df.index, pd.DatetimeIndex):
            # If original had timestamp column as data, keep original shape
            result = merged.reset_index().rename(columns={'index': 'timestamp'})
        return result

    def get_feature_names(self) -> List[str]:
        """Return list of feature names this extractor produces."""
        return self._feature_names.copy()

    def get_config(self) -> dict:
        """Get configuration parameters for this extractor."""
        config = super().get_config()
        config.update({"enabled": self.enabled, "mvp_mode": not self.enabled})
        return config
