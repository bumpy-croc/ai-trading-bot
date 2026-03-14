"""
Macro economic feature extractor.

Extracts macroeconomic trend features for cross-asset analysis.
Uses simulated data with clear interfaces for real API integration later.
"""

import logging

import numpy as np
import pandas as pd

from src.config.constants import DEFAULT_MACRO_CACHE_TTL
from src.tech.features.base import FeatureExtractor

logger = logging.getLogger(__name__)


class MacroFeatureExtractor(FeatureExtractor):
    """
    Extracts macroeconomic trend features from external market data.

    Features are normalized to [-1, 1] range. Currently uses deterministic
    simulated data derived from crypto price action; designed for drop-in
    replacement with real macro data providers (e.g., FRED, Alpha Vantage).
    """

    def __init__(
        self,
        enabled: bool = False,
        cache_ttl: int = DEFAULT_MACRO_CACHE_TTL,
    ):
        """
        Initialize the macro feature extractor.

        Args:
            enabled: Whether macro feature extraction is enabled
            cache_ttl: Cache TTL in seconds for macro data
        """
        super().__init__("macro")
        self.enabled = enabled
        self.cache_ttl = cache_ttl
        self._feature_names = [
            "spx_trend",
            "dxy_trend",
            "treasury_10y_change",
            "gold_trend",
            "oil_trend",
        ]

    def extract(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract macro economic features from market data.

        When disabled, returns neutral values (0.0) for all features.

        Args:
            data: DataFrame with OHLCV market data

        Returns:
            DataFrame with original data plus macro features
        """
        if not self.validate_input(data):
            raise ValueError("Invalid input data: missing required OHLCV columns")

        # Strict validation rejects non-positive prices, negative volume,
        # and non-finite values that would produce misleading [-1, 1] signals.
        if not self.validate_input(data, strict=True):
            raise ValueError(
                "Invalid input data: prices must be positive and finite, "
                "volume must be non-negative"
            )

        df = data.copy()

        if not self.enabled:
            for feature in self._feature_names:
                df[feature] = 0.0
            return df

        df = self._compute_spx_trend(df)
        df = self._compute_dxy_trend(df)
        df = self._compute_treasury_change(df)
        df = self._compute_gold_trend(df)
        df = self._compute_oil_trend(df)

        return df

    def _momentum_signal(
        self, series: pd.Series, fast: int = 10, slow: int = 30
    ) -> pd.Series:
        """
        Compute a normalized momentum signal from a price series.

        Uses fast/slow moving average crossover, normalized to [-1, 1].

        Args:
            series: Price series to compute momentum for
            fast: Fast moving average period
            slow: Slow moving average period

        Returns:
            Momentum signal clipped to [-1, 1]
        """
        ma_fast = series.rolling(window=fast, min_periods=1).mean()
        ma_slow = series.rolling(window=slow, min_periods=1).mean()
        raw = (ma_fast - ma_slow) / (ma_slow + 1e-9)
        return np.clip(raw * 10.0, -1.0, 1.0)

    def _compute_spx_trend(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute S&P 500 trend signal.

        Simulates equity market momentum using crypto close as a correlated proxy.
        Crypto and equities share risk-on/risk-off cycles.

        Replace with real SPX data from Alpha Vantage or Yahoo Finance.
        """
        df["spx_trend"] = self._momentum_signal(df["close"], fast=10, slow=30)
        return df

    def _compute_dxy_trend(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute US Dollar Index trend signal.

        Simulates dollar strength using inverse of crypto momentum, reflecting
        the typical negative correlation between USD and crypto.

        Replace with real DXY data.
        """
        df["dxy_trend"] = -self._momentum_signal(df["close"], fast=14, slow=40)
        return df

    def _compute_treasury_change(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute 10-year Treasury yield change signal.

        Simulates yield changes using rate of change of volatility.
        Rising yields correlate with risk-off moves in crypto.

        Replace with real Treasury yield data from FRED.
        """
        returns = df["close"].pct_change().fillna(0.0)
        vol = returns.rolling(window=20, min_periods=1).std()
        vol_change = vol.pct_change().fillna(0.0)
        df["treasury_10y_change"] = np.clip(vol_change * 5.0, -1.0, 1.0)
        return df

    def _compute_gold_trend(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute gold trend signal.

        Simulates gold momentum using a slow momentum of close prices,
        reflecting crypto's partial correlation with gold as an alternative asset.

        Replace with real gold (XAU) data.
        """
        df["gold_trend"] = self._momentum_signal(df["close"], fast=20, slow=50)
        return df

    def _compute_oil_trend(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute oil trend signal.

        Simulates oil price momentum using volume-weighted price momentum.
        Energy prices affect mining costs and macro inflation expectations.

        Replace with real crude oil (WTI/Brent) data.
        """
        vwap = (df["close"] * df["volume"]).rolling(window=20, min_periods=1).sum() / (
            df["volume"].rolling(window=20, min_periods=1).sum() + 1e-9
        )
        df["oil_trend"] = self._momentum_signal(vwap, fast=10, slow=30)
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
