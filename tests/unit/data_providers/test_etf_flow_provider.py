"""Unit tests for the ETF net-flow provider + feature math (#803)."""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import pytest

from src.data_providers.etf_flow_provider import (
    BTC_FLOW_COL,
    ETH_FLOW_COL,
    ETFFlowProvider,
    compute_flow_features,
)

pytestmark = pytest.mark.unit


def _frame(dates, btc):
    idx = pd.DatetimeIndex([pd.Timestamp(d, tz="UTC") for d in dates], name="date")
    return pd.DataFrame({BTC_FLOW_COL: btc, ETH_FLOW_COL: [b * 0.3 for b in btc]}, index=idx)


# --- compute_flow_features -------------------------------------------------


def test_zscore_negative_during_outflow_streak():
    # Steady inflows, then a sharp multi-day outflow -> negative 5d z-score.
    dates = pd.date_range("2026-01-01", periods=30, freq="D")
    btc = [200] * 22 + [-800, -900, -1000, -1100, -1200, -1300, -1400, -1500]
    feats = compute_flow_features(_frame(dates, btc), datetime(2026, 1, 30, tzinfo=UTC))
    assert feats["netflow_zscore_5d"] is not None
    assert feats["netflow_zscore_5d"] < -1.0
    assert feats["consecutive_outflow_days"] == 8.0


def test_zscore_positive_during_recovery():
    dates = pd.date_range("2026-01-01", periods=30, freq="D")
    btc = [-500] * 22 + [100, 300, 500, 700, 900, 1100, 1300, 1500]
    feats = compute_flow_features(_frame(dates, btc), datetime(2026, 1, 30, tzinfo=UTC))
    assert feats["netflow_zscore_5d"] > 0
    assert feats["consecutive_outflow_days"] == 0.0


def test_features_none_on_insufficient_history():
    dates = pd.date_range("2026-01-01", periods=3, freq="D")
    feats = compute_flow_features(
        _frame(dates, [100, -100, -200]), datetime(2026, 1, 3, tzinfo=UTC)
    )
    assert feats["netflow_zscore_20d"] is None
    # 5d z-score needs 5 rolling obs (window 5 + baseline 20) -> None here.
    assert feats["netflow_zscore_5d"] is None


def test_features_respect_as_of_cutoff():
    dates = pd.date_range("2026-01-01", periods=30, freq="D")
    btc = [200] * 22 + [-800, -900, -1000, -1100, -1200, -1300, -1400, -1500]
    # As of mid-inflow period, the later outflows must not leak in.
    feats = compute_flow_features(_frame(dates, btc), datetime(2026, 1, 15, tzinfo=UTC))
    assert feats["consecutive_outflow_days"] == 0.0


def test_features_empty_frame():
    feats = compute_flow_features(pd.DataFrame(), datetime(2026, 1, 1, tzinfo=UTC))
    assert feats == {
        "netflow_zscore_5d": None,
        "netflow_zscore_20d": None,
        "consecutive_outflow_days": 0.0,
    }


# --- provider: sources + cache ---------------------------------------------


def test_fetch_used_and_cached(tmp_path):
    dates = pd.date_range("2026-02-01", periods=25, freq="D")
    fetched = _frame(dates, [100] * 25)
    calls = {"n": 0}

    def fetch(start, end):
        calls["n"] += 1
        return fetched

    p = ETFFlowProvider(cache_dir=tmp_path, fetch_fn=fetch)
    out = p.get_flows(datetime(2026, 2, 1, tzinfo=UTC), datetime(2026, 2, 20, tzinfo=UTC))
    assert not out.empty
    assert (tmp_path / "etf_flows.parquet").exists()
    assert calls["n"] == 1

    # Second call within TTL and covered by cache -> no new fetch.
    p2 = ETFFlowProvider(cache_dir=tmp_path, fetch_fn=fetch)
    p2.get_flows(datetime(2026, 2, 1, tzinfo=UTC), datetime(2026, 2, 20, tzinfo=UTC))
    assert calls["n"] == 1


def test_falls_back_to_seed_when_fetch_none(tmp_path):
    # No cache, fetch returns None -> bundled seed dataset is served.
    p = ETFFlowProvider(cache_dir=tmp_path, fetch_fn=lambda s, e: None)
    out = p.get_flows(datetime(2026, 5, 1, tzinfo=UTC), datetime(2026, 6, 10, tzinfo=UTC))
    assert not out.empty
    assert BTC_FLOW_COL in out.columns


def test_fetch_exception_degrades_gracefully(tmp_path):
    def boom(start, end):
        raise ConnectionError("network down")

    p = ETFFlowProvider(cache_dir=tmp_path, fetch_fn=boom)
    # Must not raise; degrades to seed.
    out = p.get_flows(datetime(2026, 5, 1, tzinfo=UTC), datetime(2026, 6, 1, tzinfo=UTC))
    assert isinstance(out, pd.DataFrame)


def test_flow_features_from_seed(tmp_path):
    p = ETFFlowProvider(cache_dir=tmp_path, fetch_fn=lambda s, e: None)
    feats = p.flow_features(datetime(2026, 6, 3, tzinfo=UTC))
    assert feats["netflow_zscore_5d"] is not None
    assert feats["netflow_zscore_5d"] < 0  # outflow regime
