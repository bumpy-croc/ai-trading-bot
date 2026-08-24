"""Tests for the latched-condition re-announcer (#1095/#1096)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from src.database.models import EventType
from src.engines.live.monitoring.latched_condition_monitor import (
    ENTRIES_ENABLED_CODE,
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


def _established() -> SystemHaltState:
    """A halt mirror that has been read successfully and is not halted."""
    return SystemHaltState(active=False, reason=None, established=True)


@pytest.fixture
def setup(monkeypatch):
    """Monitor + state with the entry-pause flag forced off."""
    monkeypatch.setattr(
        "src.engines.live.execution.entry_pause.is_enabled",
        lambda name, default=False: False,
    )
    state = FakeEngineState()
    clock = FakeClock()
    monitor = LatchedConditionMonitor(state, _established(), clock=clock, dispatch_async=False)
    return state, clock, monitor


def codes(state: FakeEngineState, code: str) -> list[dict]:
    return [e for e in state.events if e["error_code"] == code]


def pages(state: FakeEngineState, code: str) -> list[dict]:
    return [e for e in codes(state, code) if e["alert"]]


class TestPositiveAssertion:
    def test_entries_enabled_is_asserted_hourly(self, setup):
        """Absence of rows must mean the loop is dead, not 'probably fine'."""
        state, clock, monitor = setup
        monitor.check()
        assert len(codes(state, ENTRIES_ENABLED_CODE)) == 1
        assert codes(state, ENTRIES_ENABLED_CODE)[0]["alert"] is False

        # No spam between heartbeats.
        for _ in range(6):
            clock.advance(HEARTBEAT_INTERVAL_SECONDS / 10)
            monitor.check()
        assert len(codes(state, ENTRIES_ENABLED_CODE)) == 1

        clock.advance(HEARTBEAT_INTERVAL_SECONDS)
        monitor.check()
        assert len(codes(state, ENTRIES_ENABLED_CODE)) == 2

    def test_no_enabled_row_while_anything_is_latched(self, setup):
        state, clock, monitor = setup
        state._close_only_mode = True
        monitor.check()
        clock.advance(2 * DAY)
        monitor.check()
        assert codes(state, ENTRIES_ENABLED_CODE) == []


class TestCloseOnlyReAnnouncement:
    def test_first_observation_writes_a_row_immediately_without_paging(self, setup):
        """A restart loop must not blank the record for a whole hour."""
        state, clock, monitor = setup
        state._close_only_mode = True
        state._close_only_reason = "stop-loss re-placement failed"
        monitor.check()
        rows = codes(state, "CLOSE_ONLY_LATCHED")
        assert len(rows) == 1
        assert rows[0]["alert"] is False  # the latch's own CRITICAL just fired
        assert "stop-loss re-placement failed" in rows[0]["message"]

    def test_state_row_hourly_and_page_on_the_ladder(self, setup):
        state, clock, monitor = setup
        state._close_only_mode = True
        monitor.check()

        clock.advance(PAGE_SCHEDULE_SECONDS[0])
        monitor.check()
        rows = codes(state, "CLOSE_ONLY_LATCHED")
        assert [r["alert"] for r in rows] == [False, True]

        # Next hours (before the 4h re-page) produce durable rows only.
        for _ in range(2):
            clock.advance(HEARTBEAT_INTERVAL_SECONDS)
            monitor.check()
        rows = codes(state, "CLOSE_ONLY_LATCHED")
        assert [r["alert"] for r in rows] == [False, True, False, False]
        assert all(r["severity"] == "warning" for r in rows if not r["alert"])

    def test_message_carries_elapsed_time_reason_and_day_prefix(self, setup):
        state, clock, monitor = setup
        state._close_only_mode = True
        state._close_only_reason = "stop-loss re-placement failed"
        monitor.check()
        clock.advance(3 * DAY + 4 * HOUR)
        monitor.check()
        message = codes(state, "CLOSE_ONLY_LATCHED")[-1]["message"]
        assert "3d 4h" in message
        assert "stop-loss re-placement failed" in message
        assert message.startswith("STILL BLOCKED — DAY 3: ")

    def test_paging_is_bounded_over_a_month(self, setup):
        """<=3 re-pages in the first 24h, then at most one per day."""
        state, clock, monitor = setup
        state._close_only_mode = True
        monitor.check()

        pages_by_day: list[int] = []
        for _day in range(30):
            before = len(pages(state, "CLOSE_ONLY_LATCHED"))
            for _ in range(int(DAY // 600)):  # 10-minute loop-like ticks
                clock.advance(600)
                monitor.check()
            pages_by_day.append(len(pages(state, "CLOSE_ONLY_LATCHED")) - before)

        assert pages_by_day[0] == len(PAGE_SCHEDULE_SECONDS)
        assert all(count <= 1 for count in pages_by_day[1:])
        assert sum(pages_by_day) <= len(PAGE_SCHEDULE_SECONDS) + 30

    def test_steady_interval_is_daily(self, setup):
        state, clock, monitor = setup
        state._close_only_mode = True
        monitor.check()
        for due in PAGE_SCHEDULE_SECONDS:
            clock.now = 1000.0 + due
            monitor.check()
        sent = len(pages(state, "CLOSE_ONLY_LATCHED"))
        clock.advance(PAGE_STEADY_INTERVAL_SECONDS - 60)
        monitor.check()
        assert len(pages(state, "CLOSE_ONLY_LATCHED")) == sent
        clock.advance(120)
        monitor.check()
        assert len(pages(state, "CLOSE_ONLY_LATCHED")) == sent + 1

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
        assert "5h" in cleared[0]["message"]

        # And the positive assertion resumes.
        state.events.clear()
        clock.advance(HEARTBEAT_INTERVAL_SECONDS)
        monitor.check()
        assert len(codes(state, ENTRIES_ENABLED_CODE)) == 1

    def test_relatch_restarts_the_page_ladder(self, setup):
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
        assert pages(state, "CLOSE_ONLY_LATCHED") == []  # fresh latch: row, no page
        clock.advance(30 * 60)
        monitor.check()
        assert pages(state, "CLOSE_ONLY_LATCHED") == []  # first re-page is an hour out


class TestEntryGateConditions:
    def test_unverified_halt_state_is_reported(self, monkeypatch):
        """Fail-closed 'never successfully read' blocks entries — #1094's shape."""
        monkeypatch.setattr(
            "src.engines.live.execution.entry_pause.is_enabled",
            lambda name, default=False: False,
        )
        state = FakeEngineState()
        clock = FakeClock()
        # established=False: entries are refused while `active` is still False.
        monitor = LatchedConditionMonitor(
            state, SystemHaltState(), clock=clock, dispatch_async=False
        )
        monitor.check()
        rows = codes(state, "SYSTEM_HALT_LATCHED")
        assert len(rows) == 1
        assert "UNVERIFIED" in rows[0]["message"]
        assert codes(state, ENTRIES_ENABLED_CODE) == []

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
        assert [r["alert"] for r in rows] == [False, True]
        assert "board decision" in rows[0]["message"]
        assert rows[0]["component"] == "ops"

    def test_elapsed_uses_the_durable_halt_timestamp(self, setup):
        """A restart must not re-announce a four-day halt as '0m'."""
        state, clock, monitor = setup
        halt = monitor._halt
        assert halt is not None
        halt.active = True
        halt.since = datetime.now(UTC) - timedelta(days=4, hours=2)
        monitor.check()
        message = codes(state, "SYSTEM_HALT_LATCHED")[0]["message"]
        assert "4d 2h" in message
        assert message.startswith("STILL BLOCKED — DAY 4: ")

    def test_entry_pause_inside_a_macro_window_is_not_a_finding(self, monkeypatch):
        monkeypatch.setattr(
            "src.engines.live.execution.entry_pause.is_enabled",
            lambda name, default=False: name == "entry_pause",
        )
        state = FakeEngineState()
        clock = FakeClock()
        monitor = LatchedConditionMonitor(state, _established(), clock=clock, dispatch_async=False)
        monkeypatch.setattr(monitor, "_macro_window_excuses_the_pause", lambda: True)
        monitor.check()
        clock.advance(2 * DAY)
        monitor.check()
        assert codes(state, "ENTRY_PAUSE_LATCHED") == []
        assert codes(state, ENTRIES_ENABLED_CODE) != []

    def test_entry_pause_outside_a_macro_window_is_re_announced(self, monkeypatch):
        monkeypatch.setattr(
            "src.engines.live.execution.entry_pause.is_enabled",
            lambda name, default=False: name == "entry_pause",
        )
        state = FakeEngineState()
        clock = FakeClock()
        monitor = LatchedConditionMonitor(state, _established(), clock=clock, dispatch_async=False)
        monkeypatch.setattr(monitor, "_macro_window_excuses_the_pause", lambda: False)
        monitor.check()
        clock.advance(PAGE_SCHEDULE_SECONDS[0])
        monitor.check()
        rows = codes(state, "ENTRY_PAUSE_LATCHED")
        assert [r["alert"] for r in rows] == [False, True]

    def test_macro_window_only_excuses_a_pause_when_the_guard_is_enabled(self, monkeypatch):
        """With the guard off, no window suppresses entries — nor the finding."""
        monkeypatch.setattr(
            "src.engines.live.execution.entry_pause.is_enabled",
            lambda name, default=False: name == "entry_pause",
        )
        monkeypatch.setattr(
            "src.engines.live.monitoring.latched_condition_monitor.is_enabled",
            lambda name, default=False: False,
        )
        state = FakeEngineState()
        monitor = LatchedConditionMonitor(
            state, _established(), clock=FakeClock(), dispatch_async=False
        )
        assert monitor._macro_window_excuses_the_pause() is False
        monitor.check()
        assert len(codes(state, "ENTRY_PAUSE_LATCHED")) == 1

    def test_conditions_are_tracked_independently(self, setup):
        state, clock, monitor = setup
        state._close_only_mode = True
        monitor.check()
        clock.advance(PAGE_SCHEDULE_SECONDS[0])
        halt = monitor._halt
        assert halt is not None
        halt.active = True
        monitor.check()
        assert len(pages(state, "CLOSE_ONLY_LATCHED")) == 1
        assert pages(state, "SYSTEM_HALT_LATCHED") == []
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

    def test_one_failing_condition_does_not_blind_the_others(self, setup):
        state, clock, monitor = setup

        def boom(self):
            raise RuntimeError("close-only read exploded")

        monitor.observe_close_only = lambda: boom(monitor)  # type: ignore[assignment]
        halt = monitor._halt
        assert halt is not None
        halt.active = True
        monitor.check()
        # The halt was still seen despite close-only raising first.
        assert len(codes(state, "SYSTEM_HALT_LATCHED")) == 1

    def test_alerts_are_delivered_off_the_calling_thread(self, monkeypatch):
        """The webhook (up to 10s) must never run on the trading-loop thread."""
        monkeypatch.setattr(
            "src.engines.live.execution.entry_pause.is_enabled",
            lambda name, default=False: False,
        )
        state = FakeEngineState()
        clock = FakeClock()
        monitor = LatchedConditionMonitor(state, _established(), clock=clock)  # async dispatch
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
        assert started == []  # the first row is not a page
        clock.advance(PAGE_SCHEDULE_SECONDS[0])
        monitor.check()
        assert started == ["latched-condition-alert"]
        clock.advance(HEARTBEAT_INTERVAL_SECONDS)
        monitor.check()
        assert started == ["latched-condition-alert"]  # state rows stay inline


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
