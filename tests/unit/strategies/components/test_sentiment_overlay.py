"""Unit tests for the sentiment-extreme mean-reversion overlay (#804)."""

from __future__ import annotations

import pandas as pd
import pytest

from src.strategies.components.sentiment_overlay import (
    SentimentExtremeOverlay,
    maybe_wrap_with_sentiment_overlay,
)
from src.strategies.components.signal_generator import Signal, SignalDirection, SignalGenerator

pytestmark = pytest.mark.unit


class _Const(SignalGenerator):
    def __init__(self, direction):
        super().__init__("const")
        self._d = direction

    def generate_signal(self, df, index, regime=None):
        return Signal(self._d, 0.8, 0.8, {})

    def get_confidence(self, df, index):
        return 0.8


class _FakeFG:
    def __init__(self, value):
        self._v = value

    def get_sentiment_for_date(self, date):
        return {"sentiment_primary": self._v}


class _Regime:
    def __init__(self, trend):
        self.trend = type("T", (), {"value": trend})()


def _df(close=54000.0, n=5):
    idx = pd.date_range("2026-06-01", periods=n, freq="D", tz="UTC")
    return pd.DataFrame(
        {"open": 1.0, "high": 2.0, "low": 0.5, "close": close, "volume": 1.0}, index=idx
    )


def test_extreme_fear_blocks_shorts():
    ov = SentimentExtremeOverlay(_Const(SignalDirection.SELL), _FakeFG(0.08))
    sig = ov.generate_signal(_df(), 2)
    assert sig.direction == SignalDirection.HOLD
    assert "short_block" in sig.metadata["sentiment_overlay_blocked"]


def test_extreme_fear_allows_longs_without_support_level():
    ov = SentimentExtremeOverlay(_Const(SignalDirection.BUY), _FakeFG(0.08))
    assert ov.generate_signal(_df(), 2).direction == SignalDirection.BUY


def test_extreme_fear_allows_long_near_support():
    ov = SentimentExtremeOverlay(
        _Const(SignalDirection.BUY), _FakeFG(0.08), support_level=54000.0, support_band_pct=0.05
    )
    assert ov.generate_signal(_df(close=54000.0), 2).direction == SignalDirection.BUY


def test_extreme_fear_blocks_long_far_from_support():
    ov = SentimentExtremeOverlay(
        _Const(SignalDirection.BUY), _FakeFG(0.08), support_level=40000.0, support_band_pct=0.05
    )
    sig = ov.generate_signal(_df(close=54000.0), 2)
    assert sig.direction == SignalDirection.HOLD
    assert "outside_support" in sig.metadata["sentiment_overlay_blocked"]


def test_neutral_sentiment_no_action():
    ov = SentimentExtremeOverlay(_Const(SignalDirection.SELL), _FakeFG(0.5))
    assert ov.generate_signal(_df(), 2).direction == SignalDirection.SELL


def test_extreme_greed_downtrend_permits_shorts():
    ov = SentimentExtremeOverlay(_Const(SignalDirection.SELL), _FakeFG(0.85))
    sig = ov.generate_signal(_df(), 2, regime=_Regime("trend_down"))
    assert sig.direction == SignalDirection.SELL


def test_hold_passthrough():
    ov = SentimentExtremeOverlay(_Const(SignalDirection.HOLD), _FakeFG(0.08))
    assert ov.generate_signal(_df(), 2).direction == SignalDirection.HOLD


def test_non_datetime_index_passthrough():
    ov = SentimentExtremeOverlay(_Const(SignalDirection.SELL), _FakeFG(0.08))
    df = _df().reset_index(drop=True)
    assert ov.generate_signal(df, 2).direction == SignalDirection.SELL


def test_composes_with_flow_gate_most_restrictive_wins():
    """Overlay + flow gate chained: a short blocked by the overlay stays blocked
    regardless of the (long-only) flow gate; a long blocked by the flow gate
    stays blocked regardless of the overlay."""
    from src.strategies.components.flow_gate import FlowGatedSignalGenerator

    class _BlockLongGate:
        def should_block_long(self, as_of):
            return True, "etf_flow_outflow_z5_-1.50"

    # SELL through both: overlay (extreme fear) blocks the short -> HOLD.
    short_chain = SentimentExtremeOverlay(
        FlowGatedSignalGenerator(_Const(SignalDirection.SELL), _BlockLongGate()),
        _FakeFG(0.08),
    )
    assert short_chain.generate_signal(_df(), 2).direction == SignalDirection.HOLD

    # BUY through both: flow gate blocks the long -> HOLD (overlay would allow).
    long_chain = SentimentExtremeOverlay(
        FlowGatedSignalGenerator(_Const(SignalDirection.BUY), _BlockLongGate()),
        _FakeFG(0.08),
    )
    assert long_chain.generate_signal(_df(), 2).direction == SignalDirection.HOLD


def test_maybe_wrap_respects_flag(monkeypatch):
    base = _Const(SignalDirection.BUY)
    monkeypatch.delenv("FEATURE_ENABLE_SENTIMENT_EXTREME_OVERLAY", raising=False)
    assert maybe_wrap_with_sentiment_overlay(base) is base
    monkeypatch.setenv("FEATURE_ENABLE_SENTIMENT_EXTREME_OVERLAY", "true")
    assert isinstance(
        maybe_wrap_with_sentiment_overlay(base, provider=_FakeFG(0.5)), SentimentExtremeOverlay
    )
