"""Tests for the latched-condition re-announcer (#1095/#1096)."""

from __future__ import annotations

import pytest

from src.database.models import EventType
from src.engines.live.monitoring.latched_condition_monitor import (
    HEARTBEAT_INTERVAL_SECONDS,
    PAGE_SCHEDULE_SECONDS,
    PAGE_STEADY_INTERVAL_SECONDS,
    LatchedConditionMonitor,
    format_elapsed,
)
from src.engines.live.system_halt import SystemHaltState

pytestmark = pytest.mark.fast

HOUR = 3600.0
DAY = 86400.0


class FakeEngineState:
    """Minimal engine surface: the close-only latch plus an event recorder."""

    def __init__(self) -> None:
        self._close_only_mode = False
        self._close_only_reason: str | None = None
        self.events: list[dict] = []

    def _record_event(
        self,
        event_type: EventType,
        message: str,
        *,
        severity: str = "error",
        component: str | None = None,
        error_code: str | None = None,
        exc: BaseException | None = None,
        alert: bool = False,
    ) -> None:
        self.events.append(
            {
                "event_type": event_type,
                "message": message,
                "severity": severity,
                "component": component,
                "error_code": error_code,
                "alert": alert,
            }
        )


class FakeClock:
    """Monotonic-shaped clock the test advances explicitly."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def setup(monkeypatch):
    """Monitor + state with the entry-pause flag forced off."""
    monkeypatch.setattr(
        "src.engines.live.monitoring.latched_condition_monitor.is_enabled",
        lambda name, default=False: False,
    )
    state = FakeEngineState()
    clock = FakeClock()
    monitor = LatchedConditionMonitor(state, SystemHaltState(), clock=clock, dispatch_async=False)
    return state, clock, monitor


def codes(state: FakeEngineState, code: str) -> list[dict]:
    return [e for e in state.events if e["error_code"] == code]


class TestCloseOnlyReAnnouncement:
    def test_no_events_while_nothing_is_latched(self, setup):
        state, clock, monitor = setup
        for _ in range(5):
            monitor.check()
            clock.advance(DAY)
        assert state.events == []

    def test_first_observation_does_not_double_page(self, setup):
        """The latch's own CRITICAL has just fired; don't page again at t=0."""
        state, clock, monitor = setup
        state._close_only_mode = True
        monitor.check()
        assert state.events == []

    def test_heartbeat_row_is_written_hourly_without_paging(self, setup):
        state, clock, monitor = setup
        state._close_only_mode = True
        state._close_only_reason = "stop-loss re-placement failed"
        monitor.check()

        # First due point is both a heartbeat and the first re-page (1h).
        clock.advance(PAGE_SCHEDULE_SECONDS[0])
        monitor.check()
        rows = codes(state, "CLOSE_ONLY_LATCHED")
        assert len(rows) == 1
        assert rows[0]["alert"] is True

        # The next hours (before the 4h re-page) produce durable rows only.
        for _ in range(2):
            clock.advance(HEARTBEAT_INTERVAL_SECONDS)
            monitor.check()
        rows = codes(state, "CLOSE_ONLY_LATCHED")
        assert len(rows) == 3
        assert [r["alert"] for r in rows] == [True, False, False]
        assert all(r["severity"] == "warning" for r in rows[1:])

    def test_message_carries_elapsed_time_and_reason(self, setup):
        state, clock, monitor = setup
        state._close_only_mode = True
        state._close_only_reason = "stop-loss re-placement failed"
        monitor.check()
        clock.advance(3 * DAY + 4 * HOUR)
        monitor.check()
        message = codes(state, "CLOSE_ONLY_LATCHED")[0]["message"]
        assert "3d 4h" in message
        assert "stop-loss re-placement failed" in message

    def test_paging_is_bounded_over_a_month(self, setup):
        """<=3 re-pages in the first 24h, then at most one per day."""
        state, clock, monitor = setup
        state._close_only_mode = True
        monitor.check()

        # Step in 10-minute ticks for 30 days (loop-cadence-like).
        pages_by_day: list[int] = []
        for _day in range(30):
            before = len([e for e in codes(state, "CLOSE_ONLY_LATCHED") if e["alert"]])
            for _ in range(int(DAY // 600)):
                clock.advance(600)
                monitor.check()
            after = len([e for e in codes(state, "CLOSE_ONLY_LATCHED") if e["alert"]])
            pages_by_day.append(after - before)

        assert pages_by_day[0] == len(PAGE_SCHEDULE_SECONDS)
        assert all(count <= 1 for count in pages_by_day[1:])
        assert sum(pages_by_day) <= len(PAGE_SCHEDULE_SECONDS) + 30

    def test_steady_interval_is_daily(self, setup):
        state, clock, monitor = setup
        state._close_only_mode = True
        monitor.check()
        # Consume the escalating schedule.
        for due in PAGE_SCHEDULE_SECONDS:
            clock.now = 1000.0 + due
            monitor.check()
        pages = len([e for e in codes(state, "CLOSE_ONLY_LATCHED") if e["alert"]])
        clock.advance(PAGE_STEADY_INTERVAL_SECONDS - 60)
        monitor.check()
        assert len([e for e in codes(state, "CLOSE_ONLY_LATCHED") if e["alert"]]) == pages
        clock.advance(120)
        monitor.check()
        assert len([e for e in codes(state, "CLOSE_ONLY_LATCHED") if e["alert"]]) == pages + 1

    def test_clearing_writes_a_durable_row_but_does_not_page(self, setup):
        state, clock, monitor = setup
        state._close_only_mode = True
        monitor.check()
        clock.advance(5 * HOUR)
        monitor.check()

        state._close_only_mode = False
        monitor.check()
        cleared = codes(state, "CLOSE_ONLY_LATCH_CLEARED")
        assert len(cleared) == 1
        assert cleared[0]["alert"] is False
        assert cleared[0]["event_type"] is EventType.WARNING
        assert "5h" in cleared[0]["message"]

        # A cleared condition stops producing rows entirely.
        state.events.clear()
        clock.advance(3 * DAY)
        monitor.check()
        assert state.events == []

    def test_relatch_restarts_the_schedule(self, setup):
        state, clock, monitor = setup
        state._close_only_mode = True
        monitor.check()
        clock.advance(2 * DAY)
        monitor.check()
        state._close_only_mode = False
        monitor.check()
        state.events.clear()

        state._close_only_mode = True
        monitor.check()
        assert state.events == []  # fresh latch: no immediate page
        clock.advance(30 * 60)
        monitor.check()
        assert state.events == []  # first re-page is an hour out again


class TestOtherConditions:
    def test_system_halt_is_re_announced(self, setup):
        state, clock, monitor = setup
        halt = monitor._halt
        assert halt is not None
        halt.active = True
        halt.reason = "board decision"
        monitor.check()
        clock.advance(PAGE_SCHEDULE_SECONDS[0])
        monitor.check()
        rows = codes(state, "SYSTEM_HALT_LATCHED")
        assert len(rows) == 1
        assert rows[0]["alert"] is True
        assert "board decision" in rows[0]["message"]
        assert rows[0]["component"] == "ops"

    def test_entry_pause_inside_a_macro_window_is_not_a_finding(self, monkeypatch):
        monkeypatch.setattr(
            "src.engines.live.monitoring.latched_condition_monitor.is_enabled",
            lambda name, default=False: name == "entry_pause",
        )
        state = FakeEngineState()
        clock = FakeClock()
        monitor = LatchedConditionMonitor(state, None, clock=clock, dispatch_async=False)
        monkeypatch.setattr(monitor, "_active_macro_window", lambda: "FOMC")
        monitor.check()
        clock.advance(2 * DAY)
        monitor.check()
        assert codes(state, "ENTRY_PAUSE_LATCHED") == []

    def test_entry_pause_outside_a_macro_window_is_re_announced(self, monkeypatch):
        monkeypatch.setattr(
            "src.engines.live.monitoring.latched_condition_monitor.is_enabled",
            lambda name, default=False: name == "entry_pause",
        )
        state = FakeEngineState()
        clock = FakeClock()
        monitor = LatchedConditionMonitor(state, None, clock=clock, dispatch_async=False)
        monkeypatch.setattr(monitor, "_active_macro_window", lambda: None)
        monitor.check()
        clock.advance(PAGE_SCHEDULE_SECONDS[0])
        monitor.check()
        rows = codes(state, "ENTRY_PAUSE_LATCHED")
        assert len(rows) == 1 and rows[0]["alert"] is True

    def test_conditions_are_tracked_independently(self, setup):
        state, clock, monitor = setup
        state._close_only_mode = True
        monitor.check()
        clock.advance(PAGE_SCHEDULE_SECONDS[0])
        halt = monitor._halt
        assert halt is not None
        halt.active = True
        monitor.check()
        assert len(codes(state, "CLOSE_ONLY_LATCHED")) == 1
        assert codes(state, "SYSTEM_HALT_LATCHED") == []
        assert set(monitor.latched_summary()) == {"close_only", "system_halt"}


class TestFaultIsolation:
    def test_a_failing_event_write_never_propagates(self, setup):
        state, clock, monitor = setup

        def boom(*args, **kwargs):
            raise RuntimeError("db down")

        state._record_event = boom  # type: ignore[method-assign]
        state._close_only_mode = True
        monitor.check()
        clock.advance(2 * DAY)
        monitor.check()  # must not raise

    def test_a_failing_condition_read_never_propagates(self, setup):
        state, clock, monitor = setup

        class Boom:
            @property
            def active(self):
                raise RuntimeError("halt state exploded")

        monitor._halt = Boom()  # type: ignore[assignment]
        monitor.check()  # must not raise

    def test_alerts_are_delivered_off_the_calling_thread(self, monkeypatch):
        """The webhook (up to 10s) must never run on the trading-loop thread."""
        monkeypatch.setattr(
            "src.engines.live.monitoring.latched_condition_monitor.is_enabled",
            lambda name, default=False: False,
        )
        state = FakeEngineState()
        clock = FakeClock()
        monitor = LatchedConditionMonitor(state, None, clock=clock)  # async dispatch
        started: list[str] = []

        class FakeThread:
            def __init__(self, target=None, name=None, daemon=None):
                self._target = target
                started.append(name or "")

            def start(self):
                self._target()

        monkeypatch.setattr(
            "src.engines.live.monitoring.latched_condition_monitor.threading.Thread", FakeThread
        )
        state._close_only_mode = True
        monitor.check()
        clock.advance(PAGE_SCHEDULE_SECONDS[0])
        monitor.check()
        assert started == ["latched-condition-alert"]
        # Heartbeat rows stay inline — only pages get a thread.
        clock.advance(HEARTBEAT_INTERVAL_SECONDS)
        monitor.check()
        assert started == ["latched-condition-alert"]


class TestFormatElapsed:
    @pytest.mark.parametrize(
        ("seconds", "expected"),
        [
            (0, "0m"),
            (-5, "0m"),
            (90, "1m"),
            (3600, "1h 0m"),
            (3600 * 5 + 720, "5h 12m"),
            (DAY * 3 + HOUR * 4, "3d 4h"),
        ],
    )
    def test_formats(self, seconds, expected):
        assert format_elapsed(seconds) == expected
