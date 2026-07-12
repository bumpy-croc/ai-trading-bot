"""Unit tests for the account circuit-breaker loop enforcer (#807 follow-up)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.database.models import EventType
from src.engines.live.monitoring.circuit_breaker_enforcer import (
    MAX_SEED_ATTEMPTS,
    CircuitBreakerEnforcer,
)
from src.risk.circuit_breaker import AccountCircuitBreaker

pytestmark = pytest.mark.unit


class _State:
    def __init__(self, balance=1000.0, day_snapshot=None, session_id=1):
        self.current_balance = balance
        self.trading_session_id = session_id
        self._recovered_inactive_session_id = None
        self._close_only_mode = False
        self.events: list[str | None] = []
        self.event_calls: list[tuple[tuple, dict]] = []
        self._day_snapshot = day_snapshot
        self.db_manager = SimpleNamespace(get_first_snapshot_of_day=lambda **kw: self._day_snapshot)

    def _enter_close_only_mode(self):
        self._close_only_mode = True

    def _record_event(self, *args, **kwargs):
        self.events.append(kwargs.get("error_code"))
        self.event_calls.append((args, kwargs))


def _breaker(mode, **kw):
    kw.setdefault("daily_loss_limit", 0.025)
    return AccountCircuitBreaker(mode=mode, **kw)


def test_active_trip_enters_close_only_and_records_event():
    s = _State()
    enf = CircuitBreakerEnforcer(s, _breaker("active"))
    enf.check()  # anchor baseline 1000
    s.current_balance = 970.0  # -3%
    enf.check()
    assert s._close_only_mode is True
    assert "ACCOUNT_CIRCUIT_BREAKER_TRIP" in s.events


def test_dry_run_trips_but_takes_no_protective_action():
    s = _State()
    enf = CircuitBreakerEnforcer(s, _breaker("dry_run"))
    enf.check()
    s.current_balance = 950.0
    enf.check()
    assert s._close_only_mode is False


def test_dry_run_trip_records_durable_system_event():
    # #964: dry_run exists to accumulate would-have-tripped evidence; stdout
    # logs are ephemeral on Railway, so the trip must land in system_events.
    s = _State()
    enf = CircuitBreakerEnforcer(s, _breaker("dry_run"))
    enf.check()
    s.current_balance = 950.0
    enf.check()

    assert s._close_only_mode is False
    assert s.events == ["CIRCUIT_BREAKER_DRY_RUN"]
    (args, kwargs) = s.event_calls[0]
    assert args[0] == EventType.CIRCUIT_BREAKER_DRY_RUN
    assert "daily_loss_halt" in args[1]
    assert "950.00" in args[1]
    assert kwargs.get("severity") == "warning"
    assert kwargs.get("component") == "risk"
    # No alert requested: alert_sent must stay honestly False downstream.
    assert kwargs.get("alert", False) is False


def test_dry_run_event_recorded_once_per_trip_episode():
    s = _State()
    enf = CircuitBreakerEnforcer(s, _breaker("dry_run"))
    enf.check()
    s.current_balance = 950.0
    enf.check()
    enf.check()  # still tripped: must not write a second row
    assert s.events == ["CIRCUIT_BREAKER_DRY_RUN"]


def test_dry_run_event_failure_does_not_crash_loop():
    s = _State()

    def _boom(*args, **kwargs):
        raise RuntimeError("db down")

    s._record_event = _boom
    enf = CircuitBreakerEnforcer(s, _breaker("dry_run"))
    enf.check()
    s.current_balance = 950.0
    enf.check()  # must not raise
    assert s._close_only_mode is False


def test_off_mode_is_inert():
    s = _State()
    enf = CircuitBreakerEnforcer(s, _breaker("off"))
    enf.check()
    s.current_balance = 100.0  # catastrophic but off
    enf.check()
    assert s._close_only_mode is False


def test_seeds_daily_baseline_from_day_snapshot_restart_safety():
    # Restart mid-day at 985 while the day opened at 1000: a -3% intraday move
    # (to 970) must still trip against the persisted 1000, not the 985 restart.
    s = _State(balance=985.0, day_snapshot=SimpleNamespace(balance=1000.0))
    enf = CircuitBreakerEnforcer(s, _breaker("active"))
    enf.check()  # seeds baseline 1000 (not 985)
    assert s._close_only_mode is False  # -1.5% not yet a trip
    s.current_balance = 970.0
    enf.check()  # -3% vs seeded 1000
    assert s._close_only_mode is True


def test_no_day_snapshot_self_anchors():
    s = _State(balance=1000.0, day_snapshot=None)
    enf = CircuitBreakerEnforcer(s, _breaker("active"))
    enf.check()  # no snapshot -> baseline = current 1000
    s.current_balance = 980.0  # -2% < 2.5%
    enf.check()
    assert s._close_only_mode is False


def test_seed_deferred_until_session_ready_then_gives_up():
    s = _State(session_id=None)  # session not resolved yet
    enf = CircuitBreakerEnforcer(s, _breaker("active"))
    for _ in range(MAX_SEED_ATTEMPTS):
        enf.check()
    assert enf._seeded is True  # armed from current balance after bounded deferrals


def test_check_never_raises_on_bad_state():
    s = _State()
    s.db_manager = SimpleNamespace(
        get_first_snapshot_of_day=lambda **kw: (_ for _ in ()).throw(RuntimeError("db down"))
    )
    enf = CircuitBreakerEnforcer(s, _breaker("active"))
    enf.check()  # must not raise despite DB failure
    s.current_balance = 970.0
    enf.check()
    assert s._close_only_mode is True  # still enforces against self-anchored baseline
