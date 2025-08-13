"""
Price-Only Feature Extractor

Produces only normalized OHLCV features using rolling min-max normalization
with the same semantics as used by MlBasic (sequence_length window).
"""

from typing import List

import numpy as np
import pandas as pd

from config.constants import DEFAULT_NORMALIZATION_WINDOW

from .base import FeatureExtractor


class PriceOnlyFeatureExtractor(FeatureExtractor):
    """Extract only normalized OHLCV price features."""

    def __init__(self, normalization_window: int = DEFAULT_NORMALIZATION_WINDOW):
        super().__init__("price_only")
        self.normalization_window = normalization_window
        # Enforce feature order to match model expectations
        self._feature_names = [
            "close_normalized",
            "volume_normalized",
            "high_normalized",
            "low_normalized",
            "open_normalized",
        ]

    def extract(self, data: pd.DataFrame) -> pd.DataFrame:
        if not self.validate_input(data):
            raise ValueError("Invalid input data: missing required OHLCV columns")

        df = data.copy()

        # Rolling min-max normalization; vectorized and consistent with TechnicalFeatureExtractor
        def _norm(col: str, out_col: str):
            rolling_min = df[col].rolling(window=self.normalization_window, min_periods=1).min()
            rolling_max = df[col].rolling(window=self.normalization_window, min_periods=1).max()
            valid = (rolling_max != rolling_min) & rolling_max.notna() & rolling_min.notna()
            df[out_col] = np.where(
                valid, (df[col] - rolling_min) / (rolling_max - rolling_min), 0.5
            )

        _norm("close", "close_normalized")
        _norm("volume", "volume_normalized")
        _norm("high", "high_normalized")
        _norm("low", "low_normalized")
        _norm("open", "open_normalized")
        return df

    def get_feature_names(self) -> List[str]:
        return self._feature_names.copy()
