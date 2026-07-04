"""Unit tests for the ETF net-flow entry gate + signal wrapper (#803)."""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import pytest

from src.strategies.components.flow_gate import (
    ETFFlowGate,
    FlowGatedSignalGenerator,
    maybe_wrap_with_flow_gate,
)
from src.strategies.components.signal_generator import Signal, SignalDirection, SignalGenerator

pytestmark = pytest.mark.unit


class _FakeGate:
    """Deterministic gate: block longs on the flagged dates."""

    def __init__(self, block_dates):
        self._block = {pd.Timestamp(d).date() for d in block_dates}

    def should_block_long(self, as_of: datetime):
        if as_of.date() in self._block:
            return True, "etf_flow_outflow_z5_-1.50"
        return False, None


class _ConstSignal(SignalGenerator):
    def __init__(self, direction):
        super().__init__("const")
        self._d = direction

    def generate_signal(self, df, index, regime=None):
        return Signal(self._d, 0.8, 0.8, {})

    def get_confidence(self, df, index):
        return 0.8


def _df(n=10):
    idx = pd.date_range("2026-06-01", periods=n, freq="D", tz="UTC")
    return pd.DataFrame({"open": 1, "high": 2, "low": 0.5, "close": 1.5, "volume": 1}, index=idx)


def test_buy_blocked_becomes_hold_on_outflow_day():
    df = _df()
    gate = _FakeGate(block_dates=["2026-06-03"])
    w = FlowGatedSignalGenerator(_ConstSignal(SignalDirection.BUY), gate)
    idx = list(df.index).index(pd.Timestamp("2026-06-03", tz="UTC"))
    signal = w.generate_signal(df, idx)
    assert signal.direction == SignalDirection.HOLD
    assert signal.metadata["flow_gate_blocked"].startswith("etf_flow_outflow")


def test_buy_passes_on_non_outflow_day():
    df = _df()
    gate = _FakeGate(block_dates=["2026-06-03"])
    w = FlowGatedSignalGenerator(_ConstSignal(SignalDirection.BUY), gate)
    idx = list(df.index).index(pd.Timestamp("2026-06-05", tz="UTC"))
    assert w.generate_signal(df, idx).direction == SignalDirection.BUY


def test_sell_passes_through_even_on_outflow_day():
    df = _df()
    gate = _FakeGate(block_dates=["2026-06-03"])
    w = FlowGatedSignalGenerator(_ConstSignal(SignalDirection.SELL), gate)
    idx = list(df.index).index(pd.Timestamp("2026-06-03", tz="UTC"))
    assert w.generate_signal(df, idx).direction == SignalDirection.SELL


def test_non_datetime_index_passes_through():
    df = _df().reset_index(drop=True)  # RangeIndex -> no timestamp
    gate = _FakeGate(block_dates=["2026-06-03"])
    w = FlowGatedSignalGenerator(_ConstSignal(SignalDirection.BUY), gate)
    assert w.generate_signal(df, 2).direction == SignalDirection.BUY


def test_gate_unknown_flow_allows(tmp_path):
    # A real gate with no data (empty provider fetch) must NOT block.
    from src.data_providers.etf_flow_provider import ETFFlowProvider

    provider = ETFFlowProvider(
        cache_dir=tmp_path, fetch_fn=lambda s, e: None, seed_path=tmp_path / "missing.csv"
    )
    gate = ETFFlowGate(provider=provider)
    block, reason = gate.should_block_long(datetime(2026, 6, 3, tzinfo=UTC))
    assert block is False
    assert reason is None


def test_maybe_wrap_respects_flag(monkeypatch):
    base = _ConstSignal(SignalDirection.BUY)
    monkeypatch.delenv("FEATURE_ENABLE_ETF_FLOW_GATE", raising=False)
    assert maybe_wrap_with_flow_gate(base) is base
    monkeypatch.setenv("FEATURE_ENABLE_ETF_FLOW_GATE", "true")
    wrapped = maybe_wrap_with_flow_gate(base, gate=_FakeGate([]))
    assert isinstance(wrapped, FlowGatedSignalGenerator)
