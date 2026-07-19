"""Shadow observability for config-suppressed short entries (GH #1020, C6).

When ``allow_shorts=False`` suppresses a short ENTRY at signal generation,
the LIVE engine must write a DB-durable "would-have-entered-short"
``system_events`` row so the counterfactual stays measurable — bounded by the
#1016 episode-dedup pattern (first suppression + every Nth + episode-end
summary). The event path is entirely fault-isolated: bookkeeping or DB
failures never affect the trading decision. Backtests never construct the
monitor, so they write nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.database.models import EventType
from src.engines.live.execution.entry_coordinator import LiveEntryCoordinator
from src.engines.live.execution.entry_handler import LiveEntrySignal
from src.engines.live.execution.short_suppression_monitor import (
    SHORT_SUPPRESSION_EMIT_EVERY_N,
    SHORT_SUPPRESSION_EPISODE_GAP_SECONDS,
    ShortSuppressionMonitor,
)
from src.strategies.components import SignalDirection
from src.strategies.components.ml_signal_generator import SHORT_ENTRY_SUPPRESSED_KEY

pytestmark = [pytest.mark.unit, pytest.mark.fast, pytest.mark.mock_only]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _logged_events(db_manager: MagicMock) -> list[dict[str, Any]]:
    return [call.kwargs for call in db_manager.log_event.call_args_list]


def _record(monitor: ShortSuppressionMonitor, db_manager: MagicMock, **overrides: Any) -> None:
    kwargs: dict[str, Any] = {
        "db_manager": db_manager,
        "session_id": 7,
        "price": 2000.0,
        "position_size_notional": 250.0,
        "signal_strength": 0.8,
        "signal_confidence": 0.24,
        "signal_metadata": {
            "prediction": 1960.0,
            "predicted_return": -0.02,
            "model_type": "basic",
            "trading_symbol": "ETHUSDT",
        },
    }
    kwargs.update(overrides)
    monitor.record_suppression("ETHUSDT", **kwargs)


# ---------------------------------------------------------------------------
# Monitor: payload
# ---------------------------------------------------------------------------


class TestMonitorPayload:
    def test_first_suppression_writes_event_with_payload(self):
        monitor = ShortSuppressionMonitor()
        db = MagicMock()

        _record(monitor, db)

        events = _logged_events(db)
        assert len(events) == 1
        event = events[0]
        assert event["event_type"] == EventType.SHORT_ENTRY_SUPPRESSED
        assert event["severity"] == "info"
        assert event["error_code"] == "WOULD_ENTER_SHORT"
        assert event["session_id"] == 7

        details = event["details"]
        assert details["symbol"] == "ETHUSDT"
        assert details["side"] == "short"
        assert details["reason"] == "allow_shorts_false"
        assert details["price"] == pytest.approx(2000.0)
        assert details["position_size_notional"] == pytest.approx(250.0)
        assert details["signal"]["strength"] == pytest.approx(0.8)
        assert details["signal"]["confidence"] == pytest.approx(0.24)
        assert details["signal"]["predicted_return"] == pytest.approx(-0.02)
        assert details["signal"]["model_type"] == "basic"
        assert details["episode"]["suppression_count"] == 1

    def test_unserializable_metadata_values_are_dropped(self):
        """Only JSON-safe whitelisted values reach the row; junk is dropped."""
        monitor = ShortSuppressionMonitor()
        db = MagicMock()

        _record(monitor, db, signal_metadata={"prediction": object(), "model_type": "basic"})

        details = _logged_events(db)[0]["details"]
        assert "prediction" not in details["signal"]
        assert details["signal"]["model_type"] == "basic"


# ---------------------------------------------------------------------------
# Monitor: episode dedup
# ---------------------------------------------------------------------------


class TestMonitorEpisodeDedup:
    def test_episode_dedup_bounds_rows(self):
        """30 consecutive suppressions write 4 rows (1st + every Nth), not 30."""
        monitor = ShortSuppressionMonitor()
        db = MagicMock()

        for _ in range(30):
            _record(monitor, db)

        events = _logged_events(db)
        assert len(events) == 4
        counts = [e["details"]["episode"]["suppression_count"] for e in events]
        assert counts == [1, SHORT_SUPPRESSION_EMIT_EVERY_N, 2 * SHORT_SUPPRESSION_EMIT_EVERY_N, 30]
        assert all(e["error_code"] == "WOULD_ENTER_SHORT" for e in events)

    def test_inactivity_gap_closes_episode_and_starts_fresh(self):
        """A suppression after a quiet gap writes the old episode's summary
        (with the TRUE total) and starts a fresh first-of-episode row."""
        monitor = ShortSuppressionMonitor()
        db = MagicMock()
        now = 1000.0
        monitor._monotonic = lambda: now

        for _ in range(3):
            _record(monitor, db)
        assert len(_logged_events(db)) == 1  # first suppression only

        now += SHORT_SUPPRESSION_EPISODE_GAP_SECONDS + 1.0
        _record(monitor, db)

        events = _logged_events(db)
        assert len(events) == 3
        summary = events[1]
        assert summary["error_code"] == "SHORT_SUPPRESSION_EPISODE_END"
        assert summary["event_type"] == EventType.SHORT_ENTRY_SUPPRESSED
        assert summary["details"]["end_reason"] == "inactivity_gap"
        assert summary["details"]["suppressions_total"] == 3
        fresh = events[2]
        assert fresh["error_code"] == "WOULD_ENTER_SHORT"
        assert fresh["details"]["episode"]["suppression_count"] == 1

    def test_episodes_are_tracked_per_symbol(self):
        """A second symbol starts its own episode (first row emits again)."""
        monitor = ShortSuppressionMonitor()
        db = MagicMock()

        _record(monitor, db)
        monitor.record_suppression(
            "SOLUSDT",
            db_manager=db,
            session_id=7,
            price=150.0,
            position_size_notional=100.0,
            signal_strength=0.5,
            signal_confidence=0.2,
            signal_metadata=None,
        )

        events = _logged_events(db)
        assert len(events) == 2
        assert events[0]["details"]["symbol"] == "ETHUSDT"
        assert events[1]["details"]["symbol"] == "SOLUSDT"
        assert events[1]["details"]["episode"]["suppression_count"] == 1


# ---------------------------------------------------------------------------
# Monitor: fault isolation
# ---------------------------------------------------------------------------


class TestMonitorFaultIsolation:
    def test_event_write_failure_does_not_propagate(self):
        monitor = ShortSuppressionMonitor()
        db = MagicMock()
        db.log_event.side_effect = RuntimeError("db down")

        _record(monitor, db)  # must not raise

    def test_without_db_or_session_no_write_no_crash(self):
        monitor = ShortSuppressionMonitor()

        _record(monitor, MagicMock(), session_id=None)
        _record(monitor, None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Coordinator wiring: check_entry_conditions
# ---------------------------------------------------------------------------


@dataclass
class _FakeSignal:
    direction: SignalDirection
    strength: float = 0.8
    confidence: float = 0.24
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class _FakeDecision:
    signal: _FakeSignal
    position_size: float
    metadata: dict[str, Any] | None
    regime: Any = None
    risk_metrics: Any = None


def _suppressed_decision(position_size: float = 250.0) -> _FakeDecision:
    return _FakeDecision(
        signal=_FakeSignal(
            direction=SignalDirection.SELL,
            metadata={
                SHORT_ENTRY_SUPPRESSED_KEY: True,
                "prediction": 1960.0,
                "predicted_return": -0.02,
            },
        ),
        position_size=position_size,
        metadata={"enter_short": False},
    )


def _make_state() -> MagicMock:
    """Engine-state mock for the runtime decision path of check_entry_conditions."""
    state = MagicMock()
    state._is_runtime_strategy.return_value = True
    state._close_only_mode = False
    state.current_balance = 1000.0
    state.max_position_size = 0.5
    state.timeframe = "1h"
    state.trading_session_id = 7
    state.db_manager = MagicMock()
    state._extract_indicators.return_value = {}
    state._extract_sentiment_data.return_value = {}
    state._extract_ml_predictions.return_value = {}
    state.live_entry_handler.process_runtime_decision.return_value = LiveEntrySignal(
        should_enter=False, reasons=["runtime_hold"]
    )
    state.live_position_tracker.has_position_for_symbol.return_value = False
    state.live_position_tracker.position_count = 0
    state.risk_manager.get_max_concurrent_positions.return_value = 1
    return state


def _df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": [2000.0],
            "high": [2010.0],
            "low": [1990.0],
            "close": [2000.0],
            "volume": [1000.0],
        },
        index=pd.date_range("2026-07-14", periods=1, freq="1h"),
    )


def _check_entry(coordinator: LiveEntryCoordinator, decision: Any) -> None:
    coordinator.check_entry_conditions(
        df=_df(),
        current_index=0,
        symbol="ETHUSDT",
        current_price=2000.0,
        current_time=pd.Timestamp("2026-07-14T12:00:00Z").to_pydatetime(),
        runtime_decision=decision,
    )


def _suppression_events(state: MagicMock) -> list[dict[str, Any]]:
    return [
        call.kwargs
        for call in state.db_manager.log_event.call_args_list
        if call.kwargs.get("event_type") == EventType.SHORT_ENTRY_SUPPRESSED
    ]


class TestCoordinatorWiring:
    def test_suppressed_decision_emits_shadow_event_and_no_entry(self):
        state = _make_state()
        coordinator = LiveEntryCoordinator(state)

        _check_entry(coordinator, _suppressed_decision())

        events = _suppression_events(state)
        assert len(events) == 1
        assert events[0]["details"]["symbol"] == "ETHUSDT"
        assert events[0]["details"]["position_size_notional"] == pytest.approx(250.0)
        state.live_entry_handler.execute_entry.assert_not_called()

    def test_plain_sell_without_marker_emits_nothing(self):
        state = _make_state()
        coordinator = LiveEntryCoordinator(state)
        decision = _FakeDecision(
            signal=_FakeSignal(direction=SignalDirection.SELL, metadata={"enter_short": True}),
            position_size=250.0,
            metadata={"enter_short": True},
        )
        # Keep should_enter False so no execution machinery is needed here.
        _check_entry(coordinator, decision)

        assert _suppression_events(state) == []

    def test_zero_size_suppression_emits_nothing(self):
        """No event when the strategy would not have sized a position anyway."""
        state = _make_state()
        coordinator = LiveEntryCoordinator(state)

        _check_entry(coordinator, _suppressed_decision(position_size=0.0))

        assert _suppression_events(state) == []

    def test_existing_position_on_symbol_emits_nothing(self):
        """A held position would have blocked the short too — not a counterfactual."""
        state = _make_state()
        state.live_position_tracker.has_position_for_symbol.return_value = True
        coordinator = LiveEntryCoordinator(state)

        _check_entry(coordinator, _suppressed_decision())

        assert _suppression_events(state) == []

    def test_shadow_event_failure_never_breaks_entry_check(self):
        """Fault isolation at the coordinator: a raising tracker is swallowed."""
        state = _make_state()
        state.live_position_tracker.has_position_for_symbol.side_effect = RuntimeError("boom")
        coordinator = LiveEntryCoordinator(state)

        _check_entry(coordinator, _suppressed_decision())  # must not raise

        assert _suppression_events(state) == []
