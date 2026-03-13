"""
On-chain feature extractor.

Extracts blockchain-derived features for crypto market analysis.
Uses simulated data with clear interfaces for real API integration later.
"""

import hashlib
import logging

import numpy as np
import pandas as pd

from src.config.constants import DEFAULT_ONCHAIN_CACHE_TTL
from src.tech.features.base import FeatureExtractor

logger = logging.getLogger(__name__)


class OnChainFeatureExtractor(FeatureExtractor):
    """
    Extracts on-chain features from blockchain data.

    Features are normalized to [-1, 1] range. Currently uses deterministic
    simulated data derived from price action; designed for drop-in replacement
    with real on-chain data providers (e.g., Glassnode, CryptoQuant).
    """

    def __init__(
        self,
        enabled: bool = False,
        cache_ttl: int = DEFAULT_ONCHAIN_CACHE_TTL,
    ):
        """
        Initialize the on-chain feature extractor.

        Args:
            enabled: Whether on-chain feature extraction is enabled
            cache_ttl: Cache TTL in seconds for on-chain data
        """
        super().__init__("onchain")
        self.enabled = enabled
        self.cache_ttl = cache_ttl
        self._feature_names = [
            "exchange_netflow",
            "whale_ratio",
            "supply_in_profit_pct",
            "hodl_wave_signal",
            "active_addresses_zscore",
        ]

    def extract(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract on-chain features from market data.

        When disabled, returns neutral values (0.0) for all features.

        Args:
            data: DataFrame with OHLCV market data

        Returns:
            DataFrame with original data plus on-chain features
        """
        if not self.validate_input(data):
            raise ValueError("Invalid input data: missing required OHLCV columns")

        df = data.copy()

        if not self.enabled:
            for feature in self._feature_names:
                df[feature] = 0.0
            return df

        df = self._compute_exchange_netflow(df)
        df = self._compute_whale_ratio(df)
        df = self._compute_supply_in_profit(df)
        df = self._compute_hodl_wave_signal(df)
        df = self._compute_active_addresses_zscore(df)

        return df

    def _compute_exchange_netflow(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute exchange netflow signal.

        Simulates net flow of assets into/out of exchanges using volume and
        price momentum. Positive values indicate net inflows (bearish),
        negative values indicate net outflows (bullish).

        Replace with real exchange flow data from Glassnode/CryptoQuant.
        """
        returns = df["close"].pct_change().fillna(0.0)
        volume_change = df["volume"].pct_change().fillna(0.0)
        # Netflow proxy: high volume on down moves = exchange inflows
        raw = volume_change * (-returns)
        df["exchange_netflow"] = np.clip(raw * 5.0, -1.0, 1.0)
        return df

    def _compute_whale_ratio(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute whale transaction ratio signal.

        Simulates the ratio of large transactions to total volume.
        High whale ratio during down moves is bearish; during up moves, bullish.

        Replace with real whale alert / large transaction data.
        """
        returns = df["close"].pct_change().fillna(0.0)
        # Use high-low range as proxy for large-player activity
        price_range = (df["high"] - df["low"]) / (df["close"] + 1e-9)
        raw = price_range * np.sign(returns)
        # Normalize using rolling statistics
        rolling_mean = raw.rolling(window=20, min_periods=1).mean()
        rolling_std = raw.rolling(window=20, min_periods=1).std().fillna(1e-9).replace(0, 1e-9)
        zscore = ((raw - rolling_mean) / rolling_std).fillna(0.0)
        df["whale_ratio"] = np.clip(zscore / 3.0, -1.0, 1.0)
        return df

    def _compute_supply_in_profit(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute percentage of supply in profit signal.

        Simulates using price relative to rolling percentiles.
        High values indicate most holders are in profit (potential selling pressure).

        Replace with real UTXO-based supply profit data.
        """
        rolling_max = df["close"].rolling(window=90, min_periods=1).max()
        rolling_min = df["close"].rolling(window=90, min_periods=1).min()
        price_range = rolling_max - rolling_min
        # Position within range as proxy for supply in profit
        position = (df["close"] - rolling_min) / (price_range + 1e-9)
        # Map [0, 1] to [-1, 1]
        df["supply_in_profit_pct"] = np.clip(position * 2.0 - 1.0, -1.0, 1.0)
        return df

    def _compute_hodl_wave_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute HODL wave signal.

        Simulates long-term holder behavior using slow moving average crossovers.
        Positive values indicate accumulation phase, negative indicate distribution.

        Replace with real HODL wave / coin age data.
        """
        ma_short = df["close"].rolling(window=30, min_periods=1).mean()
        ma_long = df["close"].rolling(window=90, min_periods=1).mean()
        # Crossover signal as proxy for holder behavior
        raw = (ma_short - ma_long) / (ma_long + 1e-9)
        df["hodl_wave_signal"] = np.clip(raw * 10.0, -1.0, 1.0)
        return df

    def _compute_active_addresses_zscore(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute active addresses z-score signal.

        Simulates network activity using volume as a proxy for on-chain activity.
        High activity relative to recent history suggests growing network usage.

        Replace with real active address count data.
        """
        log_volume = np.log1p(df["volume"])
        rolling_mean = log_volume.rolling(window=30, min_periods=1).mean()
        rolling_std = log_volume.rolling(window=30, min_periods=1).std().fillna(1e-9).replace(0, 1e-9)
        zscore = ((log_volume - rolling_mean) / rolling_std).fillna(0.0)
        df["active_addresses_zscore"] = np.clip(zscore / 3.0, -1.0, 1.0)
        return df

    def get_feature_names(self) -> list[str]:
        """Return list of feature names this extractor produces."""
        return self._feature_names.copy()

    def get_config(self) -> dict:
        """Get configuration parameters for this extractor."""
        config = super().get_config()
        config.update({
            "enabled": self.enabled,
            "cache_ttl": self.cache_ttl,
        })
        return config
