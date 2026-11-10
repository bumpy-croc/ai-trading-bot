"""Hardening tests for closed-candle gating (#1106 review round).

Each class here pins one defect the review found, so a regression fails on the
specific property rather than on a distant symptom:

- the frontier must be captured atomically with the frame it describes;
- a closed bar's OHLCV must survive a late duplicate WebSocket event;
- incomparable timestamps must fail closed, never escape into the loop;
- the decision bar must resolve against the dataset the runtime actually
  indexes, even when ``dropna`` shifts frame positions;
- a bar must not be consumed until entry execution has had its attempt;
- a strategy hot-swap must not let the retired strategy's decision survive;
- a jammed gate must be *observable*, because its only symptom is silence.
"""

from __future__ import annotations

import threading
from datetime import UTC, datetime
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from src.data_providers.data_provider import DataProvider
from src.engines.live.closed_candle_gate import ClosedCandleGate
from src.engines.live.kline_buffer import KlineBuffer
from src.engines.live.trading_engine import LiveTradingEngine
from src.strategies.components import Signal, SignalDirection
from src.strategies.components.strategy import TradingDecision


def _frame(n: int, start: str = "2026-07-01 00:00:00", tz: str | None = None) -> pd.DataFrame:
    idx = pd.date_range(start=start, periods=n, freq="1h", tz=tz, name="timestamp")
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


def _kline(ts: pd.Timestamp, close: float, *, closed: bool, high: float | None = None) -> dict:
    return {
        "k": {
            "t": int(ts.timestamp() * 1000),
            "o": "100.0",
            "h": str(high if high is not None else close + 1.0),
            "l": "90.0",
            "c": str(close),
            "v": "1000.0",
            "x": closed,
        }
    }


def _buffer(rows: int = 5) -> KlineBuffer:
    provider = Mock()
    provider.get_live_data.return_value = _frame(rows)
    return KlineBuffer("BTCUSDT", "1h", provider)


def _decision(metadata: dict | None = None) -> TradingDecision:
    return TradingDecision(
        timestamp=datetime.now(UTC),
        signal=Signal(
            direction=SignalDirection.HOLD,
            strength=0.5,
            confidence=0.5,
            metadata={} if metadata is None else metadata,
        ),
        position_size=0.0,
        regime=None,
        risk_metrics={},
        execution_time_ms=0.0,
        metadata={},
    )


@pytest.mark.fast
class TestAtomicSnapshot:
    """The frontier must describe the frame it is returned with."""

    def test_snapshot_returns_frame_and_frontier_together(self):
        buf = _buffer(5)
        df, frontier = buf.snapshot()
        # REST seed: tail is the in-progress bar, so the frontier is index[-2].
        assert frontier == df.index[-2]

    def test_snapshot_frontier_matches_tail_once_closed(self):
        buf = _buffer(5)
        tail = buf.get_dataframe().index[-1]
        buf.on_kline(_kline(tail, 106.0, closed=True))

        df, frontier = buf.snapshot()
        assert frontier == df.index[-1] == tail

    def test_snapshot_is_consistent_under_concurrent_closes(self):
        """The pair must never disagree, however the WS thread interleaves.

        A separate get_dataframe()/last_closed_bar_time pair can observe the
        tail as forming and the frontier as closed — the race that let gating
        decide on a bar the frame still held mid-formation.
        """
        buf = _buffer(5)
        stop = threading.Event()

        def churn():
            while not stop.is_set():
                tail = buf.get_dataframe().index[-1]
                buf.on_kline(_kline(tail, 107.0, closed=True))
                buf.on_kline(_kline(tail + pd.Timedelta(hours=1), 108.0, closed=False))

        writer = threading.Thread(target=churn, daemon=True)
        writer.start()
        try:
            for _ in range(300):
                df, frontier = buf.snapshot()
                if frontier is not None:
                    # The frontier must exist in the frame it came with, and
                    # never point past its tail.
                    assert frontier <= df.index[-1]
                    assert frontier in df.index
        finally:
            stop.set()
            writer.join(timeout=5)


@pytest.mark.fast
class TestClosedBarIsImmutable:
    """A closed bar's values are final — the latch protects data, not just a flag."""

    def test_late_forming_duplicate_cannot_rewrite_a_closed_bar(self):
        buf = _buffer(5)
        tail = buf.get_dataframe().index[-1]
        buf.on_kline(_kline(tail, 150.0, closed=True, high=155.0))

        settled = buf.get_dataframe().iloc[-1]
        # A stale mid-bar event for the same bar arrives after the close.
        buf.on_kline(_kline(tail, 99.0, closed=False, high=99.5))

        after = buf.get_dataframe().iloc[-1]
        assert after["close"] == settled["close"] == 150.0
        assert after["high"] == settled["high"] == 155.0
        assert buf.last_closed_bar_time == tail

    def test_repeated_close_event_is_still_applied(self):
        """Re-delivery of x=true carries identical final values — idempotent."""
        buf = _buffer(5)
        tail = buf.get_dataframe().index[-1]
        buf.on_kline(_kline(tail, 150.0, closed=True))
        buf.on_kline(_kline(tail, 150.0, closed=True))

        assert buf.get_dataframe().iloc[-1]["close"] == 150.0
        assert buf.last_closed_bar_time == tail


@pytest.mark.fast
class TestIncomparableTimestampsFailClosed:
    """Mixed tz-awareness must never raise into the loop's error counter."""

    def test_frontier_mismatch_does_not_raise_and_does_not_evaluate(self):
        gate = ClosedCandleGate(enabled=True, timeframe="1h")
        df = _frame(5, tz="UTC")
        naive_frontier = pd.Timestamp("2026-07-01 03:00:00")

        view = gate.resolve(df, naive_frontier)

        # Frame-shape evidence is tz-aware and usable, so the gate still works;
        # what must not happen is an exception escaping.
        assert view.bar_time == df.index[-2]

    def test_monotonic_guard_mismatch_does_not_raise(self):
        """The already-evaluated comparison is guarded like the frontier one.

        Previously ``bar_time <= self._last_evaluated_bar`` repeated the very
        comparison ``_closed_frontier`` catches, unguarded.
        """
        gate = ClosedCandleGate(enabled=True, timeframe="1h")
        gate.mark_evaluated(pd.Timestamp("2026-07-01 03:00:00"))  # tz-naive

        view = gate.resolve(_frame(5, tz="UTC"))  # tz-aware frame

        assert isinstance(view.evaluate, bool)

    def test_warns_once_while_the_condition_holds(self, caplog):
        gate = ClosedCandleGate(enabled=True, timeframe="1h")
        df = _frame(1, tz="UTC")  # single row: no frame-shape evidence
        naive = pd.Timestamp("2026-07-01 03:00:00")

        with caplog.at_level("WARNING"):
            for _ in range(5):
                gate.resolve(df, naive)

        warnings = [r for r in caplog.records if "not comparable" in r.message]
        assert len(warnings) == 1
        assert warnings[0].levelname == "WARNING"


@pytest.mark.fast
class TestDegradedFrontierIsLoud:
    """Running without WebSocket closure evidence must not be silent."""

    def test_missing_frontier_warns_once_then_recovers(self, caplog):
        gate = ClosedCandleGate(enabled=True, timeframe="1h")
        df = _frame(5)

        with caplog.at_level("INFO"):
            gate.resolve(df, None)
            gate.resolve(df, None)
            gate.resolve(df, df.index[-1])

        degraded = [r for r in caplog.records if "without a WebSocket closed-bar" in r.message]
        restored = [r for r in caplog.records if "frontier restored" in r.message]
        assert len(degraded) == 1
        assert len(restored) == 1

    def test_flag_off_never_warns(self, caplog):
        """Inertness: the degraded path must not log when gating is disabled."""
        gate = ClosedCandleGate(enabled=False, timeframe="1h")
        with caplog.at_level("WARNING"):
            for _ in range(3):
                gate.resolve(_frame(5), None)
        assert not [r for r in caplog.records if "WebSocket closed-bar" in r.message]


@pytest.mark.fast
class TestStallObservation:
    """A jammed gate must be observable — its only symptom is silence."""

    def test_healthy_while_evaluating(self):
        now = [1000.0]
        gate = ClosedCandleGate(enabled=True, timeframe="1h", clock=lambda: now[0])

        now[0] += 60.0
        assert gate.stall_observation().stalled is False

    def test_stalls_after_the_threshold(self):
        now = [1000.0]
        gate = ClosedCandleGate(enabled=True, timeframe="1h", clock=lambda: now[0])

        now[0] += 3 * 3600 + 1
        observation = gate.stall_observation()

        assert observation.stalled is True
        assert "has not evaluated a bar" in (observation.reason or "")

    def test_evaluating_clears_the_stall(self):
        now = [1000.0]
        gate = ClosedCandleGate(enabled=True, timeframe="1h", clock=lambda: now[0])
        now[0] += 3 * 3600 + 1
        assert gate.stall_observation().stalled is True

        gate.mark_evaluated(pd.Timestamp("2026-07-01 03:00:00"))
        assert gate.stall_observation().stalled is False

    def test_never_stalls_with_the_flag_off(self):
        now = [1000.0]
        gate = ClosedCandleGate(enabled=False, timeframe="1h", clock=lambda: now[0])
        now[0] += 10 * 86400
        assert gate.stall_observation().stalled is False

    def test_monitor_reports_the_stall_as_a_latched_condition(self):
        from src.engines.live.monitoring.latched_condition_monitor import LatchedConditionMonitor

        now = [1000.0]
        gate = ClosedCandleGate(enabled=True, timeframe="1h", clock=lambda: now[0])
        state = Mock()
        state._close_only_mode = False
        state._closed_candle_gate = gate
        monitor = LatchedConditionMonitor(engine_state=state, dispatch_async=False)

        assert monitor.observe_closed_candle_gate().active is False
        now[0] += 3 * 3600 + 1
        observation = monitor.observe_closed_candle_gate()

        assert observation.active is True
        assert "closed-candle gating" in (observation.reason or "")

    def test_monitor_tolerates_an_absent_gate(self):
        from src.engines.live.monitoring.latched_condition_monitor import LatchedConditionMonitor

        state = Mock(spec=[])
        monitor = LatchedConditionMonitor(engine_state=state, dispatch_async=False)
        assert monitor.observe_closed_candle_gate().active is False


@pytest.mark.fast
class TestResetOnHotSwap:
    def test_reset_clears_the_high_water_mark(self):
        gate = ClosedCandleGate(enabled=True, timeframe="1h")
        df = _frame(5)
        first = gate.resolve(df)
        gate.mark_evaluated(first.bar_time)
        assert gate.resolve(df).evaluate is False

        gate.reset()

        assert gate.resolve(df).evaluate is True


def _engine(monkeypatch, *, enabled: bool) -> LiveTradingEngine:
    monkeypatch.setenv("FEATURE_CLOSED_CANDLE_GATING", "true" if enabled else "false")
    from src.strategies.ml_basic import create_ml_basic_strategy

    with patch("src.engines.live.trading_engine.DatabaseManager"):
        return LiveTradingEngine(
            strategy=create_ml_basic_strategy(),
            data_provider=Mock(spec=DataProvider),
            initial_balance=10_000,
            enable_live_trading=False,
        )


def _stub_periphery(engine: LiveTradingEngine, frames: list[pd.DataFrame]) -> None:
    engine.is_running = True
    engine._sleep_with_interrupt = Mock()
    engine._ensure_ws_health_monitor_alive = Mock()
    engine._get_latest_data = Mock(side_effect=list(frames))
    engine._is_data_fresh = Mock(return_value=True)
    engine.strategy_manager = None
    engine._prepare_strategy_dataframe = Mock(side_effect=lambda df: df)
    engine._is_context_ready = Mock(return_value=(True, ""))
    engine._runtime_process_decision = Mock(side_effect=lambda *a, **k: _decision())
    engine._check_exit_conditions = Mock()
    engine.entry_coordinator.process_legacy_short_entry = Mock()
    engine.live_exit_handler.update_trailing_stops = Mock()
    engine.live_exit_handler.check_partial_operations = Mock()
    engine.live_position_tracker.update_pnl = Mock()
    engine.live_position_tracker.update_mfe_mae = Mock()
    engine._update_performance_metrics = Mock()
    engine._check_max_drawdown = Mock()
    engine._log_periodic_account_state = Mock()
    engine._log_status = Mock()


@pytest.mark.fast
class TestBarConsumedOnlyAfterExecution:
    def test_entry_failure_leaves_the_bar_retryable(self, monkeypatch):
        """A raising entry check must not cost a whole timeframe of entries."""
        engine = _engine(monkeypatch, enabled=True)
        frames = [_frame(5), _frame(5), _frame(5)]
        _stub_periphery(engine, frames)
        engine._check_entry_conditions = Mock(side_effect=RuntimeError("exchange down"))

        engine._trading_loop("BTCUSDT", "1h", max_steps=len(frames))

        # Every tick retried the same bar rather than consuming it once.
        assert engine._check_entry_conditions.call_count == len(frames)
        assert engine._closed_candle_gate._last_evaluated_bar is None

    def test_successful_entry_consumes_the_bar_once(self, monkeypatch):
        engine = _engine(monkeypatch, enabled=True)
        frames = [_frame(5), _frame(5), _frame(5)]
        _stub_periphery(engine, frames)
        engine._check_entry_conditions = Mock()

        engine._trading_loop("BTCUSDT", "1h", max_steps=len(frames))

        assert engine._check_entry_conditions.call_count == 1
        assert engine._closed_candle_gate._last_evaluated_bar == frames[0].index[-2]


@pytest.mark.fast
class TestHotSwapInvalidatesCachedDecision:
    def test_swap_clears_gate_state_and_cached_decision(self, monkeypatch):
        engine = _engine(monkeypatch, enabled=True)
        engine._last_closed_bar_decision = _decision()
        engine._closed_candle_gate.mark_evaluated(pd.Timestamp("2026-07-01 03:00:00"))

        engine._reset_closed_candle_gate()

        assert engine._last_closed_bar_decision is None
        assert engine._closed_candle_gate._last_evaluated_bar is None


@pytest.mark.fast
class TestCachedDecisionIsNotMutated:
    """The cached decision is handed to every tick until the next bar close.

    ``TradingDecision`` and its nested ``Signal`` are plain (non-frozen)
    dataclasses, so the exit path receives a shared mutable object rather than
    a copy. Nothing in production mutates it today, and this test is what keeps
    that true; immutability itself is tracked separately.
    """

    def test_exit_path_does_not_mutate_the_cached_decision(self, monkeypatch):
        engine = _engine(monkeypatch, enabled=True)
        frames = [_frame(5), _frame(5), _frame(5)]
        _stub_periphery(engine, frames)
        engine._check_entry_conditions = Mock()

        seen: list[tuple[int, int, str, float]] = []

        def _record_exit(df, index, price, runtime_decision=None, candle=None, safety_mode=False):
            if runtime_decision is not None:
                seen.append(
                    (
                        id(runtime_decision),
                        len(runtime_decision.signal.metadata),
                        runtime_decision.signal.direction.value,
                        runtime_decision.position_size,
                    )
                )

        engine._check_exit_conditions = Mock(side_effect=_record_exit)
        engine._trading_loop("BTCUSDT", "1h", max_steps=len(frames))

        assert len(seen) >= 2, "cached decision was never replayed to the exit path"
        # Same object every tick, and every observable field unchanged.
        assert len({entry for entry in seen}) == 1


@pytest.mark.fast
class TestRuntimeIndexResolvedByTimestamp:
    """The runtime indexes its own dataset, not the loop's post-dropna frame.

    ``StrategyRuntime.process`` ignores the frame it is handed and indexes
    ``dataset.data`` positionally. When the loop's ``dropna`` removes a row,
    frame positions and dataset positions diverge — and the frontier bar is
    precisely the row the gate's own comment says dropna may drop.
    """

    def test_resolves_the_dataset_position_not_the_frame_position(self, monkeypatch):
        engine = _engine(monkeypatch, enabled=True)
        dataset_frame = _frame(6)
        engine.strategy_coordinator._state._runtime_dataset = Mock(data=dataset_frame)

        # dropna removed the first two rows: frame position 2 is dataset 4.
        assert engine.strategy_coordinator.runtime_index_for(dataset_frame.index[4]) == 4

    def test_returns_none_for_an_absent_bar(self, monkeypatch):
        engine = _engine(monkeypatch, enabled=True)
        engine.strategy_coordinator._state._runtime_dataset = Mock(data=_frame(6))

        assert engine.strategy_coordinator.runtime_index_for(pd.Timestamp("2020-01-01")) is None

    def test_returns_none_without_a_dataset(self, monkeypatch):
        engine = _engine(monkeypatch, enabled=True)
        engine.strategy_coordinator._state._runtime_dataset = None

        assert engine.strategy_coordinator.runtime_index_for(pd.Timestamp("2026-07-01")) is None

    def test_decision_uses_the_dataset_position_when_dropna_shifted_the_frame(self, monkeypatch):
        """End to end: the index handed to the runtime is the dataset's."""
        engine = _engine(monkeypatch, enabled=True)
        dataset_frame = _frame(6)
        # The loop's frame is the dataset with its first two rows dropped, so
        # every frame position is 2 lower than the dataset position.
        loop_frame = dataset_frame.iloc[2:]
        engine.strategy_coordinator._state._runtime_dataset = Mock(data=dataset_frame)

        captured: list[int] = []
        engine._runtime_process_decision = Mock(
            side_effect=lambda df, index, *a, **k: (captured.append(index), _decision())[1]
        )

        engine._evaluate_signal_decision(
            loop_frame,
            len(loop_frame) - 1,
            105.0,
            datetime.now(UTC),
            "1h",
            safety_mode=False,
        )

        # Frame position of the decision bar is 2; dataset position is 4.
        assert captured == [4]


@pytest.mark.fast
class TestEndToEndWithARealKlineBuffer:
    """Drive a real KlineBuffer through the real coordinator into the gate.

    Every other gating test injects the frontier directly, so the WS path —
    the ``x: true`` latch, the snapshot clamp, the coordinator's hand-off —
    was never exercised end to end. Both P1 defects lived in exactly that gap.
    """

    def _engine_with_buffer(self, monkeypatch, rows: int = 5):
        engine = _engine(monkeypatch, enabled=True)
        buf = _buffer(rows)
        engine._kline_buffer = buf
        engine._ws_kline_provider = Mock(ws_healthy=True, _kline_ws_state=None)
        return engine, buf

    def test_frontier_flows_from_buffer_through_coordinator_to_gate(self, monkeypatch):
        engine, buf = self._engine_with_buffer(monkeypatch)

        df = engine._get_latest_data("BTCUSDT", "1h")

        assert df is not None
        # REST seed: tail still forming, so the frontier is the previous bar.
        assert engine._buffer_frontier == df.index[-2]
        view = engine._closed_candle_gate.resolve(df, engine._buffer_frontier)
        assert view.evaluate is True
        assert view.bar_time == df.index[-2]

    def test_close_event_promotes_the_tail_end_to_end(self, monkeypatch):
        engine, buf = self._engine_with_buffer(monkeypatch)
        tail = buf.get_dataframe().index[-1]
        buf.on_kline(_kline(tail, 123.0, closed=True))

        df = engine._get_latest_data("BTCUSDT", "1h")

        assert engine._buffer_frontier == tail
        view = engine._closed_candle_gate.resolve(df, engine._buffer_frontier)
        assert view.bar_time == tail
        assert view.bar_closed is True
        # The row the strategy will read holds the bar's settled close.
        assert float(df.loc[tail, "close"]) == 123.0

    def test_rest_fallback_clears_a_stale_frontier(self, monkeypatch):
        """An unhealthy WS must not leave the previous snapshot's frontier behind."""
        engine, buf = self._engine_with_buffer(monkeypatch)
        tail = buf.get_dataframe().index[-1]
        buf.on_kline(_kline(tail, 123.0, closed=True))
        engine._get_latest_data("BTCUSDT", "1h")
        assert engine._buffer_frontier is not None

        engine._ws_kline_provider = Mock(ws_healthy=False, _kline_ws_state=None)
        engine.data_provider.get_live_data = Mock(return_value=_frame(5))
        engine._get_latest_data("BTCUSDT", "1h")

        assert engine._buffer_frontier is None
