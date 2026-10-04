"""REST-fallback freshness is candle-aware and a stale streak is announced (#983).

With the kline WebSocket down the loop reads REST frames whose tail bar is the
*forming* candle. Its open time is up to one interval old by construction, so a
fixed 120s age limit froze a 1h strategy (entries AND in-engine exits) for ~58
minutes of every hour, and the only trace was a debug line.
"""

from __future__ import annotations

import logging
import threading
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.database.models import EventType
from src.engines.live.loop_timing import (
    STALE_DATA_LATCH_SECONDS,
    LiveLoopTimingCoordinator,
)
from src.engines.live.monitoring.latched_condition_monitor import LatchedConditionMonitor
from src.engines.live.system_halt import SystemHaltState

pytestmark = pytest.mark.fast


class _State:
    """Engine surface the coordinator reads, with the kline WebSocket DOWN."""

    def __init__(self, timeframe: str | None = "1h") -> None:
        self.stop_event = threading.Event()
        self.base_check_interval = 60
        self.min_check_interval = 10
        self.max_check_interval = 300
        self.data_freshness_threshold = 120
        self.timeframe = timeframe
        self.live_position_tracker = MagicMock()
        self._ws_kline_active = False
        self._kline_buffer = None
        self._ws_kline_provider = None


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def _frame(tail_open_age: timedelta) -> pd.DataFrame:
    opened = datetime.now(UTC) - tail_open_age
    return pd.DataFrame({"close": [1.0]}, index=pd.DatetimeIndex([opened]))


def _coordinator(timeframe: str | None = "1h", clock: _Clock | None = None):
    return LiveLoopTimingCoordinator(_State(timeframe), clock=clock or _Clock())


class TestRestFallbackFreshness:
    @pytest.mark.parametrize("minutes", [0, 5, 30, 58])
    def test_forming_candle_is_fresh_for_the_whole_interval(self, minutes):
        """The 1h bug: only the first ~2 minutes of each candle used to pass."""
        assert _coordinator("1h").is_data_fresh(_frame(timedelta(minutes=minutes))) is True

    def test_just_closed_tail_bar_is_fresh_until_the_threshold(self):
        """New bar not yet in the REST response right after the close."""
        assert _coordinator("1h").is_data_fresh(_frame(timedelta(minutes=61))) is True

    def test_missing_current_candle_is_stale(self):
        """Tail bar older than interval + threshold: the exchange data is lagging."""
        assert _coordinator("1h").is_data_fresh(_frame(timedelta(minutes=63))) is False
        assert _coordinator("1h").is_data_fresh(_frame(timedelta(hours=3))) is False

    def test_four_hour_timeframe_scales_the_limit(self):
        assert _coordinator("4h").is_data_fresh(_frame(timedelta(hours=3, minutes=50))) is True
        assert _coordinator("4h").is_data_fresh(_frame(timedelta(hours=4, minutes=5))) is False

    def test_short_timeframe_stays_strict(self):
        assert _coordinator("1m").is_data_fresh(_frame(timedelta(seconds=150))) is True
        assert _coordinator("1m").is_data_fresh(_frame(timedelta(minutes=5))) is False

    @pytest.mark.parametrize("timeframe", [None, "7q", ""])
    def test_unknown_timeframe_keeps_the_strict_threshold(self, timeframe):
        """Never relax the gate on a timeframe that cannot be sized."""
        coordinator = _coordinator(timeframe)
        assert coordinator.is_data_fresh(_frame(timedelta(minutes=30))) is False
        assert coordinator.is_data_fresh(_frame(timedelta(seconds=30))) is True

    def test_empty_frame_is_stale(self):
        assert _coordinator("1h").is_data_fresh(pd.DataFrame()) is False

    def test_healthy_websocket_path_is_unchanged(self):
        state = _State("1h")
        state._ws_kline_active = True
        state._kline_buffer = MagicMock(is_fresh=False)
        state._ws_kline_provider = MagicMock(ws_healthy=True)
        coordinator = LiveLoopTimingCoordinator(state, clock=_Clock())
        assert coordinator.is_data_fresh(_frame(timedelta(minutes=1))) is False
        state._kline_buffer.is_fresh = True
        assert coordinator.is_data_fresh(_frame(timedelta(hours=5))) is True


class TestStaleStreak:
    def test_short_blip_is_not_announced(self):
        clock = _Clock()
        coordinator = _coordinator("1h", clock)
        assert coordinator.is_data_fresh(_frame(timedelta(hours=3))) is False
        clock.now += STALE_DATA_LATCH_SECONDS - 1
        assert coordinator.stale_data_observation().stale is False

    def test_sustained_staleness_is_announced_with_reason(self):
        clock = _Clock()
        coordinator = _coordinator("1h", clock)
        coordinator.is_data_fresh(_frame(timedelta(hours=3)))
        clock.now += STALE_DATA_LATCH_SECONDS + 1
        coordinator.is_data_fresh(_frame(timedelta(hours=3)))
        observation = coordinator.stale_data_observation()
        assert observation.stale is True
        assert "timeframe 1h" in (observation.reason or "")
        assert "skipping entries and in-engine exits" in (observation.reason or "")

    def test_fresh_frame_ends_the_streak(self):
        clock = _Clock()
        coordinator = _coordinator("1h", clock)
        coordinator.is_data_fresh(_frame(timedelta(hours=3)))
        clock.now += STALE_DATA_LATCH_SECONDS + 1
        assert coordinator.is_data_fresh(_frame(timedelta(minutes=10))) is True
        assert coordinator.stale_data_observation().stale is False
        # A later failure starts a new streak from zero.
        coordinator.is_data_fresh(_frame(timedelta(hours=3)))
        assert coordinator.stale_data_observation().stale is False

    def test_edges_are_logged_once_not_every_tick(self, caplog):
        coordinator = _coordinator("1h")
        with caplog.at_level(logging.INFO, logger="src.engines.live.loop_timing"):
            for _ in range(5):
                coordinator.is_data_fresh(_frame(timedelta(hours=3)))
            coordinator.is_data_fresh(_frame(timedelta(minutes=5)))
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        infos = [r for r in caplog.records if "fresh again" in r.getMessage()]
        assert len(warnings) == 1
        assert len(infos) == 1


class _EventState:
    def __init__(self, coordinator) -> None:
        self._close_only_mode = False
        self._close_only_reason = None
        self.loop_timing_coordinator = coordinator
        self.events: list[dict] = []

    def _record_event(self, event_type: EventType, message: str, **kwargs) -> None:
        self.events.append({"event_type": event_type, "message": message, **kwargs})


class TestLatchedCondition:
    @pytest.fixture
    def setup(self, monkeypatch):
        monkeypatch.setattr(
            "src.engines.live.execution.entry_pause.is_enabled",
            lambda name, default=False: False,
        )
        clock = _Clock()
        coordinator = _coordinator("1h", clock)
        state = _EventState(coordinator)
        monitor = LatchedConditionMonitor(
            state,
            SystemHaltState(active=False, reason=None, established=True),
            clock=clock,
            dispatch_async=False,
        )
        return state, clock, coordinator, monitor

    def test_stale_streak_latches_a_durable_event_and_blocks_the_enabled_row(self, setup):
        state, clock, coordinator, monitor = setup
        coordinator.is_data_fresh(_frame(timedelta(hours=3)))
        clock.now += STALE_DATA_LATCH_SECONDS + 1
        coordinator.is_data_fresh(_frame(timedelta(hours=3)))
        monitor.check()

        latched = [e for e in state.events if e["error_code"] == "MARKET_DATA_STALE"]
        assert len(latched) == 1
        assert latched[0]["component"] == "data"
        assert "in-engine exits" in latched[0]["message"]
        assert "until a human clears it" not in latched[0]["message"]
        assert [e for e in state.events if e["error_code"] == "ENTRIES_ENABLED"] == []

    def test_clearing_is_recorded(self, setup):
        state, clock, coordinator, monitor = setup
        coordinator.is_data_fresh(_frame(timedelta(hours=3)))
        clock.now += STALE_DATA_LATCH_SECONDS + 1
        coordinator.is_data_fresh(_frame(timedelta(hours=3)))
        monitor.check()
        coordinator.is_data_fresh(_frame(timedelta(minutes=2)))
        clock.now += 1
        monitor.check()

        cleared = [e for e in state.events if e["error_code"] == "MARKET_DATA_STALE_CLEARED"]
        assert len(cleared) == 1
        assert "processing market data again" in cleared[0]["message"]

    def test_fresh_data_never_latches(self, setup):
        state, clock, coordinator, monitor = setup
        coordinator.is_data_fresh(_frame(timedelta(minutes=30)))
        clock.now += 10 * STALE_DATA_LATCH_SECONDS
        coordinator.is_data_fresh(_frame(timedelta(minutes=30)))
        monitor.check()
        assert [e for e in state.events if e["error_code"] == "MARKET_DATA_STALE"] == []
