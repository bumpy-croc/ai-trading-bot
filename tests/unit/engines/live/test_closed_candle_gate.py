"""Tests for ClosedCandleGate — closed-candle gating for live signal decisions.

Parity plan P1.0, decision D1 (docs/refactor/backtest_live_parity_plan.md):
backtest is inherently closed-bar; when the ``closed_candle_gating`` flag is ON,
live signal evaluation runs exactly once per newly closed bar, at that bar's
index, so live's decision input layer matches backtest by construction.
"""

from datetime import UTC, datetime

import pandas as pd
import pytest

from src.engines.live.closed_candle_gate import ClosedCandleGate, stamp_decision_signal
from src.strategies.components import Signal, SignalDirection
from src.strategies.components.strategy import TradingDecision


def _make_df(n: int, start: str = "2026-07-01 00:00:00", freq: str = "1h") -> pd.DataFrame:
    idx = pd.date_range(start=start, periods=n, freq=freq, name="timestamp")
    return pd.DataFrame(
        {
            "open": [100.0 + i for i in range(n)],
            "high": [110.0 + i for i in range(n)],
            "low": [90.0 + i for i in range(n)],
            "close": [105.0 + i for i in range(n)],
            "volume": [1000.0] * n,
        },
        index=idx,
    )


def _make_decision(metadata: dict | None = None) -> TradingDecision:
    return TradingDecision(
        timestamp=datetime.now(UTC),
        signal=Signal(
            direction=SignalDirection.HOLD,
            strength=0.5,
            confidence=0.5,
            metadata=metadata if metadata is not None else {},
        ),
        position_size=0.0,
        regime=None,
        risk_metrics={},
        execution_time_ms=0.0,
        metadata={},
    )


# --------------------------------------------------------------------------- #
# Disabled gate (flag OFF): evaluate every tick on the tail — today's behavior
# --------------------------------------------------------------------------- #


@pytest.mark.fast
class TestGateDisabled:
    def test_always_evaluates_on_tail(self):
        gate = ClosedCandleGate(enabled=False)
        df = _make_df(5)

        view = gate.resolve(df, buffer_frontier=None)

        assert view.evaluate is True
        assert view.index == 4
        assert view.bar_time == df.index[-1]

    def test_tail_reported_forming_without_close_evidence(self):
        gate = ClosedCandleGate(enabled=False)
        df = _make_df(5)

        view = gate.resolve(df, buffer_frontier=None)

        assert view.bar_closed is False

    def test_tail_reported_closed_when_buffer_confirms(self):
        gate = ClosedCandleGate(enabled=False)
        df = _make_df(5)

        view = gate.resolve(df, buffer_frontier=df.index[-1])

        assert view.bar_closed is True
        assert view.evaluate is True
        assert view.index == 4

    def test_evaluates_every_tick_even_after_resolve(self):
        """Disabled gate never rate-limits: repeated resolves all evaluate."""
        gate = ClosedCandleGate(enabled=False)
        df = _make_df(5)

        views = [gate.resolve(df, buffer_frontier=None) for _ in range(3)]

        assert all(v.evaluate for v in views)


# --------------------------------------------------------------------------- #
# Enabled gate (flag ON): once per newly closed bar, at that bar's index
# --------------------------------------------------------------------------- #


@pytest.mark.fast
class TestGateEnabled:
    def test_first_resolve_targets_second_to_last_bar(self):
        """Without buffer evidence, the newest provably closed bar is index[-2]
        (a successor row exists)."""
        gate = ClosedCandleGate(enabled=True)
        df = _make_df(5)

        view = gate.resolve(df, buffer_frontier=None)

        assert view.evaluate is True
        assert view.index == 3
        assert view.bar_time == df.index[-2]
        assert view.bar_closed is True

    def test_buffer_frontier_promotes_tail_to_decision_bar(self):
        """When the buffer saw x=true for the tail, decide on the tail itself."""
        gate = ClosedCandleGate(enabled=True)
        df = _make_df(5)

        view = gate.resolve(df, buffer_frontier=df.index[-1])

        assert view.evaluate is True
        assert view.index == 4
        assert view.bar_time == df.index[-1]

    def test_same_bar_never_evaluates_twice(self):
        """Idempotency: after mark_evaluated, ticks on the same frontier skip."""
        gate = ClosedCandleGate(enabled=True)
        df = _make_df(5)

        first = gate.resolve(df, buffer_frontier=None)
        gate.mark_evaluated(first.bar_time)
        second = gate.resolve(df, buffer_frontier=None)
        third = gate.resolve(df, buffer_frontier=None)

        assert first.evaluate is True
        assert second.evaluate is False
        assert third.evaluate is False

    def test_new_closed_bar_evaluates_exactly_once(self):
        gate = ClosedCandleGate(enabled=True)
        df5 = _make_df(5)
        first = gate.resolve(df5, buffer_frontier=None)
        gate.mark_evaluated(first.bar_time)

        df6 = _make_df(6)  # a new bar arrived; old tail is now closed
        second = gate.resolve(df6, buffer_frontier=None)
        gate.mark_evaluated(second.bar_time)
        third = gate.resolve(df6, buffer_frontier=None)

        assert second.evaluate is True
        assert second.bar_time == df6.index[-2]
        assert third.evaluate is False

    def test_backfill_to_older_data_does_not_reevaluate(self):
        """Reconnect/backfill idempotency: a resync that rewinds the frame must
        not re-trigger evaluation of already-decided bars."""
        gate = ClosedCandleGate(enabled=True)
        df6 = _make_df(6)
        first = gate.resolve(df6, buffer_frontier=None)
        gate.mark_evaluated(first.bar_time)

        df5 = _make_df(5)  # resync returned an older frame
        view = gate.resolve(df5, buffer_frontier=None)

        assert view.evaluate is False

    def test_stale_buffer_frontier_ahead_of_frame_clamps_to_tail(self):
        """A buffer frontier newer than the frame (stale REST fallback) proves the
        tail closed — decide on the tail, not beyond it."""
        gate = ClosedCandleGate(enabled=True)
        df = _make_df(5)
        ahead = df.index[-1] + pd.Timedelta(hours=1)

        view = gate.resolve(df, buffer_frontier=ahead)

        assert view.evaluate is True
        assert view.index == 4
        assert view.bar_time == df.index[-1]

    def test_single_row_frame_has_nothing_closed(self):
        gate = ClosedCandleGate(enabled=True)
        df = _make_df(1)

        view = gate.resolve(df, buffer_frontier=None)

        assert view.evaluate is False

    def test_empty_frame_never_evaluates(self):
        gate = ClosedCandleGate(enabled=True)
        df = _make_df(0)

        view = gate.resolve(df, buffer_frontier=None)

        assert view.evaluate is False


# --------------------------------------------------------------------------- #
# Signal.metadata stamping (BOTH modes) — A/B + parity-ledger attribution
# --------------------------------------------------------------------------- #


@pytest.mark.fast
class TestStampDecisionSignal:
    def test_stamps_open_close_and_closed_flag(self):
        decision = _make_decision()
        bar_time = pd.Timestamp("2026-07-01 03:00:00")

        stamp_decision_signal(decision, bar_time=bar_time, bar_closed=True, timeframe="1h")

        md = decision.signal.metadata
        assert md["decision_bar_open_time"] == "2026-07-01T03:00:00"
        assert md["decision_bar_close_time"] == "2026-07-01T04:00:00"
        assert md["decision_bar_closed"] is True

    def test_stamps_forming_bar_as_not_closed(self):
        decision = _make_decision()

        stamp_decision_signal(
            decision,
            bar_time=pd.Timestamp("2026-07-01 03:00:00"),
            bar_closed=False,
            timeframe="4h",
        )

        md = decision.signal.metadata
        assert md["decision_bar_closed"] is False
        assert md["decision_bar_close_time"] == "2026-07-01T07:00:00"

    def test_unknown_timeframe_omits_close_time(self):
        decision = _make_decision()

        stamp_decision_signal(
            decision,
            bar_time=pd.Timestamp("2026-07-01 03:00:00"),
            bar_closed=False,
            timeframe="7h",
        )

        md = decision.signal.metadata
        assert md["decision_bar_close_time"] is None
        assert md["decision_bar_open_time"] == "2026-07-01T03:00:00"

    def test_none_decision_is_a_noop(self):
        stamp_decision_signal(None, bar_time=pd.Timestamp("2026-07-01"), bar_closed=True, timeframe="1h")

    def test_preserves_existing_metadata(self):
        decision = _make_decision(metadata={"onnx_pred": 0.7})

        stamp_decision_signal(
            decision,
            bar_time=pd.Timestamp("2026-07-01 03:00:00"),
            bar_closed=True,
            timeframe="1h",
        )

        assert decision.signal.metadata["onnx_pred"] == 0.7
