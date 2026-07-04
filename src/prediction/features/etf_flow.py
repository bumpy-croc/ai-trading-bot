"""ETF net-flow feature extractor (#803).

Exposes US spot ETF net-flow features (5d/20d net-flow z-scores, consecutive
outflow days) as optional model inputs, mirroring ``OnChainFeatureExtractor``.

**Inert until retrain.** Adding these to a live model's feature vector changes
its schema, so existing ONNX models cannot consume them until retrained (#801,
human sign-off). The extractor therefore ships **disabled by default** and is
registered only behind ``config["etf_flows_features"]["enabled"]``. The
rule-based flow *gate* (`src/strategies/components/flow_gate.py`) is what acts on
flows today; this extractor is the model-input path for a future retrain.
"""

from __future__ import annotations

import logging
from datetime import timedelta

import pandas as pd

from src.config.constants import (
    DEFAULT_ETF_FLOW_ZSCORE_LONG_WINDOW,
    DEFAULT_ETF_FLOW_ZSCORE_SHORT_WINDOW,
)
from src.data_providers.etf_flow_provider import (
    BTC_FLOW_COL,
    ETFFlowProvider,
    compute_flow_features,
)
from src.tech.features.base import FeatureExtractor

logger = logging.getLogger(__name__)


class ETFFlowFeatureExtractor(FeatureExtractor):
    """Extracts ETF net-flow features, joined onto the price frame by date."""

    def __init__(
        self,
        enabled: bool = False,
        *,
        provider: ETFFlowProvider | None = None,
        short_window: int = DEFAULT_ETF_FLOW_ZSCORE_SHORT_WINDOW,
        long_window: int = DEFAULT_ETF_FLOW_ZSCORE_LONG_WINDOW,
    ) -> None:
        super().__init__("etf_flow")
        self.enabled = enabled
        self._provider = provider
        self._short_window = short_window
        self._long_window = long_window
        self._feature_names = [
            "btc_etf_netflow_zscore_5d",
            "btc_etf_netflow_zscore_20d",
            "etf_consecutive_outflow_days",
        ]

    def _get_provider(self) -> ETFFlowProvider:
        if self._provider is None:
            self._provider = ETFFlowProvider()
        return self._provider

    def extract(self, data: pd.DataFrame) -> pd.DataFrame:
        if not self.validate_input(data):
            raise ValueError("Invalid input data: missing required OHLCV columns")
        df = data.copy()

        # Neutral (0.0) when disabled or when the frame lacks a usable date index.
        if not self.enabled or not isinstance(df.index, pd.DatetimeIndex):
            for name in self._feature_names:
                df[name] = 0.0
            return df

        idx = pd.to_datetime(df.index, utc=True)
        start = idx.min().to_pydatetime() - timedelta(days=self._long_window * 3 + 10)
        end = idx.max().to_pydatetime()
        try:
            flows = self._get_provider().get_flows(start, end)
        except Exception as exc:  # noqa: BLE001 - never break feature extraction
            logger.warning("ETF flow features unavailable (%s); using neutral values", exc)
            for name in self._feature_names:
                df[name] = 0.0
            return df

        # Flow features change daily; compute once per unique date, then map.
        unique_dates = pd.DatetimeIndex(idx.normalize().unique())
        feats_by_date = {
            d: compute_flow_features(
                flows,
                d.to_pydatetime(),
                asset_col=BTC_FLOW_COL,
                short_window=self._short_window,
                long_window=self._long_window,
            )
            for d in unique_dates
        }
        norm = idx.normalize()
        df["btc_etf_netflow_zscore_5d"] = [
            _num(feats_by_date[d]["netflow_zscore_5d"]) for d in norm
        ]
        df["btc_etf_netflow_zscore_20d"] = [
            _num(feats_by_date[d]["netflow_zscore_20d"]) for d in norm
        ]
        df["etf_consecutive_outflow_days"] = [
            _num(feats_by_date[d]["consecutive_outflow_days"]) for d in norm
        ]
        return df

    def get_feature_names(self) -> list[str]:
        return self._feature_names.copy()

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({"enabled": self.enabled})
        return config


def _num(value: float | None) -> float:
    return float(value) if value is not None else 0.0
