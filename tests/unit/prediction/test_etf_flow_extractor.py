"""Unit tests for the ETF net-flow feature extractor (#803)."""

from __future__ import annotations

import pandas as pd
import pytest

from src.prediction.features.etf_flow import ETFFlowFeatureExtractor

pytestmark = pytest.mark.unit

FEATURES = [
    "btc_etf_netflow_zscore_5d",
    "btc_etf_netflow_zscore_20d",
    "etf_consecutive_outflow_days",
]


def _ohlcv(n=45, start="2026-05-01"):
    idx = pd.date_range(start, periods=n, freq="D", tz="UTC")
    return pd.DataFrame(
        {"open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 100.0}, index=idx
    )


def test_disabled_returns_neutral_zero():
    out = ETFFlowFeatureExtractor(enabled=False).extract(_ohlcv())
    for f in FEATURES:
        assert f in out.columns
        assert (out[f] == 0.0).all()


def test_enabled_populates_from_seed():
    out = ETFFlowFeatureExtractor(enabled=True).extract(_ohlcv())
    for f in FEATURES:
        assert f in out.columns
    # The seed encodes an outflow regime in early June -> negative 5d z-score there.
    z = float(out.loc[pd.Timestamp("2026-06-03", tz="UTC"), "btc_etf_netflow_zscore_5d"])
    assert z < 0


def test_non_datetime_index_is_neutral():
    df = _ohlcv().reset_index(drop=True)
    out = ETFFlowFeatureExtractor(enabled=True).extract(df)
    for f in FEATURES:
        assert (out[f] == 0.0).all()


def test_missing_ohlcv_raises():
    with pytest.raises(ValueError, match="OHLCV"):
        ETFFlowFeatureExtractor(enabled=True).extract(pd.DataFrame({"close": [1, 2, 3]}))


def test_feature_names():
    assert ETFFlowFeatureExtractor().get_feature_names() == FEATURES
