"""Unit tests for the account circuit-breaker loop enforcer (#807 follow-up)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.database.models import EventType
from src.engines.live.monitoring.circuit_breaker_enforcer import (
    MAX_SEED_ATTEMPTS,
    CircuitBreakerEnforcer,
)
from src.engines.live.monitoring.seed_lineage import PEAK_SEED_UNAVAILABLE
from src.risk.circuit_breaker import AccountCircuitBreaker

pytestmark = pytest.mark.unit


class _State:
    def __init__(self, balance=1000.0, day_snapshot=None, session_id=1, session_peak_equity=None):
        self.current_balance = balance
        self.trading_session_id = session_id
        self._recovered_inactive_session_id = None
        self._close_only_mode = False
        self.events: list[str | None] = []
        self.event_calls: list[tuple[tuple, dict]] = []
        self._day_snapshot = day_snapshot
        self._session_peak_equity = session_peak_equity
        self.db_manager = SimpleNamespace(
            get_first_snapshot_of_day=lambda **kw: self._day_snapshot,
            get_session_peak_equity=lambda **kw: self._session_peak_equity,
        )

    def _enter_close_only_mode(self):
        self._close_only_mode = True

    def _record_event(self, *args, **kwargs):
        self.events.append(kwargs.get("error_code"))
        self.event_calls.append((args, kwargs))


class _Unrealized:
    """Mutable unrealized-P&L stub standing in for the position tracker sum."""

    def __init__(self, value=0.0):
        self.value = value

    def __call__(self):
        if isinstance(self.value, BaseException):
            raise self.value
        return self.value


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
        get_first_snapshot_of_day=lambda **kw: (_ for _ in ()).throw(RuntimeError("db down")),
        get_session_peak_equity=lambda **kw: None,
    )
    enf = CircuitBreakerEnforcer(s, _breaker("active"))
    enf.check()  # must not raise despite DB failure
    s.current_balance = 970.0
    enf.check()
    assert s._close_only_mode is True  # still enforces against self-anchored baseline


# --- equity-based evaluation (#986 gap A) ------------------------------------


def test_open_position_loss_trips_on_equity_where_cash_would_not():
    # Cash stays flat at 1000 (unrealized P&L never touches current_balance)
    # while an open position bleeds to -3% of equity: the breaker must trip on
    # TRUE equity. Pre-fix, this exact scenario was structurally invisible.
    s = _State()
    unrealized = _Unrealized(0.0)
    enf = CircuitBreakerEnforcer(s, _breaker("active"), unrealized_pnl_provider=unrealized)
    enf.check()  # anchor baseline at equity 1000 (flat)
    unrealized.value = -30.0  # open position marks -3%; cash unchanged
    enf.check()
    assert s._close_only_mode is True
    assert "ACCOUNT_CIRCUIT_BREAKER_TRIP" in s.events


def test_cash_only_control_does_not_trip_without_provider():
    # Control for the test above: identical cash series without an equity feed
    # never trips — proving the trip comes from the unrealized P&L.
    s = _State()
    enf = CircuitBreakerEnforcer(s, _breaker("active"))
    enf.check()
    enf.check()
    assert s._close_only_mode is False


def test_daily_baseline_seeds_from_snapshot_equity_not_balance():
    # Day opened with equity 1000 (balance 950 + 50 unrealized). The daily-loss
    # baseline must anchor to the snapshot's EQUITY: equity 970 is a -3% move
    # against 1000 (trip) but would read as a +2.1% gain against balance 950.
    s = _State(balance=970.0, day_snapshot=SimpleNamespace(balance=950.0, equity=1000.0))
    enf = CircuitBreakerEnforcer(s, _breaker("active"))
    enf.check()
    assert s._close_only_mode is True


# --- restart-safe peak seeding (#986 gap B) -----------------------------------


def test_restart_reseeds_drawdown_peak_from_durable_session_max():
    # Restart at 1000 while the durable session equity peak is 1100: a slide to
    # 930 is a 15.5% drawdown from the true peak (trip) but only 7% from the
    # restart-time self-anchor (silently blind, the #845/#847 class).
    s = _State(balance=1000.0, session_peak_equity=1100.0)
    enf = CircuitBreakerEnforcer(
        s, _breaker("active", daily_loss_limit=0.99, drawdown_halt=0.15, drawdown_recovery=0.05)
    )
    enf.check()  # seeds peak 1100 from account_history
    assert s._close_only_mode is False
    s.current_balance = 930.0
    enf.check()
    assert s._close_only_mode is True


def test_no_durable_peak_self_anchors():
    s = _State(balance=1000.0, session_peak_equity=None)
    enf = CircuitBreakerEnforcer(
        s, _breaker("active", daily_loss_limit=0.99, drawdown_halt=0.15, drawdown_recovery=0.05)
    )
    enf.check()
    s.current_balance = 930.0  # -7% from self-anchored 1000: below the cap
    enf.check()
    assert s._close_only_mode is False


def test_peak_seed_failure_defers_then_reports_unavailable():
    """A never-resolved session is a seeding MISS, not a fresh account (#1036).

    ``self_anchored`` means "there was genuinely nothing to seed from"; a
    lookup that never happened must not borrow that reassurance.
    """
    s = _State(session_id=None)  # session not resolved: seeding must defer
    enf = CircuitBreakerEnforcer(s, _breaker("active"))
    for _ in range(MAX_SEED_ATTEMPTS):
        enf.check()
    assert enf._seeded is True
    assert enf.peak_seed_provenance == PEAK_SEED_UNAVAILABLE


# --- equity-read fault isolation ----------------------------------------------


def test_unrealized_failure_degrades_to_balance_with_warning(caplog):
    s = _State()
    unrealized = _Unrealized(RuntimeError("mark price unavailable"))
    enf = CircuitBreakerEnforcer(s, _breaker("active"), unrealized_pnl_provider=unrealized)
    with caplog.at_level("WARNING"):
        enf.check()  # must not raise
        enf.check()
    assert s._close_only_mode is False  # balance alone is healthy: no halt
    degraded_warnings = [r for r in caplog.records if "balance-only" in r.message]
    assert len(degraded_warnings) == 1  # warn on transition, not every iteration


def test_unrealized_recovery_resumes_equity_evaluation(caplog):
    s = _State()
    unrealized = _Unrealized(RuntimeError("mark price unavailable"))
    enf = CircuitBreakerEnforcer(s, _breaker("active"), unrealized_pnl_provider=unrealized)
    enf.check()  # degraded (balance-only): frozen, nothing anchors
    unrealized.value = -30.0  # feed recovers: baseline anchors at equity 970
    with caplog.at_level("INFO"):
        enf.check()
    assert s._close_only_mode is False  # anchored at recovery, no loss yet
    assert any("recovered" in r.message for r in caplog.records)
    unrealized.value = -60.0  # equity 940: a real -3.1% move vs the 970 anchor
    enf.check()
    assert s._close_only_mode is True  # equity evaluation resumed and tripped


def test_non_finite_unrealized_degrades_to_balance():
    s = _State()
    unrealized = _Unrealized(float("nan"))
    enf = CircuitBreakerEnforcer(s, _breaker("active"), unrealized_pnl_provider=unrealized)
    enf.check()
    enf.check()
    assert s._close_only_mode is False


def test_non_positive_equity_degrades_to_balance(caplog):
    # A garbage unrealized read that drives computed equity <= 0 must not feed
    # the breaker (evaluate() would silently no-op on it) — degrade explicitly.
    s = _State()
    unrealized = _Unrealized(-2000.0)
    enf = CircuitBreakerEnforcer(s, _breaker("active"), unrealized_pnl_provider=unrealized)
    with caplog.at_level("WARNING"):
        enf.check()
        enf.check()
    assert s._close_only_mode is False
    assert any("balance-only" in r.message for r in caplog.records)


# --- dry_run observability payload (#968) -------------------------------------


def test_dry_run_event_carries_equity_numbers_and_peak_provenance():
    s = _State(session_peak_equity=1100.0)
    unrealized = _Unrealized(0.0)
    enf = CircuitBreakerEnforcer(s, _breaker("dry_run"), unrealized_pnl_provider=unrealized)
    enf.check()  # seed: peak 1100 (db), baseline 1000 (self-anchor)
    unrealized.value = -30.0  # equity 970: -3% daily loss
    enf.check()

    assert s.events == ["CIRCUIT_BREAKER_DRY_RUN"]
    (args, _kwargs) = s.event_calls[0]
    message = args[1]
    assert "equity $970.00" in message
    assert "balance $1,000.00" in message
    assert "unrealized $-30.00" in message
    assert "peak $1,100.00" in message
    assert "db_session_max" in message


# --- degraded-basis latch freeze (#1032 fix round) -----------------------------
# A degraded reading is CASH measured against EQUITY-basis anchors — a mixed
# basis must never move latch state: no new latches, no clears, frozen until
# the basis recovers.


def test_degraded_winning_position_does_not_latch_daily_halt():
    # Winning open position (+50) anchors the day at equity 1050; the provider
    # then faults, collapsing the reading to cash 1000 — an apparent -4.8%
    # "loss" that pre-fix would spuriously LATCH the daily halt for the rest
    # of the UTC day.
    s = _State()
    unrealized = _Unrealized(50.0)
    enf = CircuitBreakerEnforcer(s, _breaker("active"), unrealized_pnl_provider=unrealized)
    enf.check()  # healthy anchor: baseline 1050
    unrealized.value = RuntimeError("mark price unavailable")
    enf.check()  # degraded: frozen, no latch
    assert s._close_only_mode is False
    assert s.events == []
    unrealized.value = 40.0  # recovered: equity 1040, a real -0.95% (no trip)
    enf.check()
    assert s._close_only_mode is False
    assert s.events == []


def test_degraded_reading_does_not_clear_drawdown_latch_or_duplicate_rows():
    # In-drawdown latch (equity 840 vs peak 1000 = -16%), then the provider
    # faults and the reading jumps to cash par with the peak — pre-fix that
    # spuriously CLEARED the latch, and the healthy re-trip wrote a duplicate
    # dry_run row. The latch must stay frozen and the episode must stay one row.
    s = _State()
    unrealized = _Unrealized(0.0)
    enf = CircuitBreakerEnforcer(
        s,
        _breaker("dry_run", daily_loss_limit=0.99, drawdown_halt=0.15, drawdown_recovery=0.05),
        unrealized_pnl_provider=unrealized,
    )
    enf.check()  # anchor peak/baseline at 1000
    unrealized.value = -160.0  # equity 840: -16% drawdown latches (row 1)
    enf.check()
    assert s.events == ["CIRCUIT_BREAKER_DRY_RUN"]
    unrealized.value = RuntimeError("mark price unavailable")
    enf.check()  # degraded at cash par: latch must NOT clear
    unrealized.value = -160.0  # recovered, still in real drawdown
    enf.check()
    assert s.events == ["CIRCUIT_BREAKER_DRY_RUN"]  # same episode: exactly one row


def test_degraded_realized_loss_does_not_trip_while_frozen():
    # Mandated freeze semantics: while the basis is degraded even a realized
    # cash move makes no latch transitions — the WARNING plus the basis field
    # in observability surface the outage; the halt re-arms on recovery.
    s = _State()
    unrealized = _Unrealized(RuntimeError("mark price unavailable"))
    enf = CircuitBreakerEnforcer(
        s, _breaker("dry_run", daily_loss_limit=0.025), unrealized_pnl_provider=unrealized
    )
    enf.check()
    s.current_balance = 950.0  # -5% realized while degraded: frozen, no row
    enf.check()
    assert s.events == []
    assert s._close_only_mode is False
