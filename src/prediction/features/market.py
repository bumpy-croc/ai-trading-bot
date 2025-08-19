"""
Market microstructure feature extractor.

This module provides features related to market microstructure analysis.
"""

import pandas as pd

from config.constants import DEFAULT_ENABLE_MARKET_MICROSTRUCTURE

from .base import FeatureExtractor


class MarketFeatureExtractor(FeatureExtractor):
    """
    Extracts market microstructure features from order book and trade data.

    For MVP, this extractor is disabled and not included in the feature pipeline.
    """

    def __init__(self, enabled: bool = DEFAULT_ENABLE_MARKET_MICROSTRUCTURE):
        """
        Initialize the market microstructure feature extractor.

        Args:
            enabled: Whether market microstructure extraction is enabled (False for MVP)
        """
        super().__init__("market_microstructure")
        self.enabled = enabled
        self._feature_names = [
            "bid_ask_spread",
            "order_book_imbalance",
            "volume_profile_delta",
            "trade_size_distribution",
            "liquidity_score",
        ]

    def extract(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract market microstructure features from data.

        For MVP, this is not implemented and should not be called.

        Args:
            data: DataFrame with market data

        Returns:
            DataFrame with original data plus market microstructure features
        """
        if not self.enabled:
            raise RuntimeError("Market microstructure extraction is disabled for MVP")

        # Post-MVP implementation will go here
        # Will extract features like:
        # - Bid-ask spread analysis
        # - Order book depth and imbalance
        # - Volume profile and delta
        # - Trade size distribution
        # - Liquidity metrics

        return data.copy()

    def get_feature_names(self) -> list[str]:
        """Return list of feature names this extractor produces."""
        return self._feature_names.copy() if self.enabled else []

    def get_config(self) -> dict:
        """Get configuration parameters for this extractor."""
        config = super().get_config()
        config.update({"enabled": self.enabled, "mvp_mode": not self.enabled})
        return config
