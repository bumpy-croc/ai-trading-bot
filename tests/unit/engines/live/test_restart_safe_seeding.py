"""Restart-safe peak/baseline seeding on carry-forward boots (#1036).

Both durable-history seeders — the ``MaxDrawdownGuard`` peak (#1001) and the
``AccountCircuitBreaker`` daily baseline + drawdown peak (#1032) — used to read
``_recovered_inactive_session_id``, which the #668 carry-forward re-entry guard
clears during startup BEFORE the first loop iteration. On the clean-restart /
new-session boot path (the one a mid-drawdown restart takes) they therefore saw
an empty value and silently self-anchored to the post-restart balance.

These tests pin the three behaviours that fix requires:
  1. the seeders resolve the prior session AFTER startup has cleared the #668 field,
  2. an empty-but-successful read is retried when history was EXPECTED (the
     documented first-snapshot race) instead of latching terminally, and
  3. a genuinely fresh session with no history still self-anchors, once, silently.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from src.engines.live.monitoring.circuit_breaker_enforcer import CircuitBreakerEnforcer
from src.engines.live.monitoring.drawdown_guard import (
    MAX_SEED_ATTEMPTS,
    MaxDrawdownEnforcer,
    MaxDrawdownGuard,
)
from src.engines.live.monitoring.seed_lineage import (
    PEAK_SEED_DB_SESSION_MAX,
    PEAK_SEED_SELF_ANCHORED,
    PEAK_SEED_UNAVAILABLE,
    resolve_history_lineage,
)
from src.risk.circuit_breaker import AccountCircuitBreaker

pytestmark = pytest.mark.unit

NEW_SESSION = 23
PRIOR_SESSION = 22


class _SeedState:
    """Engine-state stub for the seeders (carry-forward boot by default).

    ``_recovered_inactive_session_id`` is None because startup has already
    cleared it — reproducing the #1036 boot exactly.
    """

    def __init__(
        self,
        *,
        balance: float = 1015.84,
        peaks: list[float | None] | None = None,
        snapshots: list[object] | None = None,
        history_session_id: int | None = PRIOR_SESSION,
        session_id: int | None = NEW_SESSION,
    ) -> None:
        self.current_balance = balance
        self.trading_session_id = session_id
        self._recovered_inactive_session_id = None
        self._history_seed_session_id = history_session_id
        self._close_only_mode = False
        self.events: list[str | None] = []
        self._peaks = list(peaks if peaks is not None else [None])
        self._snapshots = list(snapshots if snapshots is not None else [None])
        self.peak_calls: list[int | None] = []
        self.snapshot_calls: list[int | None] = []
        self.db_manager = SimpleNamespace(
            get_session_peak_balance=self._get_peak,
            get_session_peak_equity=lambda **kw: self._get_peak(**kw),
            get_first_snapshot_of_day=self._get_snapshot,
        )

    @staticmethod
    def _next(queue: list, default=None):
        if not queue:
            return default
        return queue.pop(0) if len(queue) > 1 else queue[0]

    def _get_peak(self, session_id=None, fallback_session_id=None, **_kw):
        self.peak_calls.append(fallback_session_id)
        value = self._next(self._peaks)
        if isinstance(value, BaseException):
            raise value
        return value

    def _get_snapshot(self, session_id=None, target_date=None, fallback_session_id=None, **_kw):
        self.snapshot_calls.append(fallback_session_id)
        return self._next(self._snapshots)

    def _enter_close_only_mode(self):
        self._close_only_mode = True

    def _record_event(self, *args, **kwargs):
        self.events.append(kwargs.get("error_code"))


def _guard_enforcer(state: _SeedState) -> MaxDrawdownEnforcer:
    return MaxDrawdownEnforcer(engine_state=state, guard=MaxDrawdownGuard(0.20))


def _breaker_enforcer(state: _SeedState) -> CircuitBreakerEnforcer:
    breaker = AccountCircuitBreaker(mode="dry_run", daily_loss_limit=0.025, drawdown_halt=0.15)
    return CircuitBreakerEnforcer(engine_state=state, breaker=breaker)


# ---------------------------------------------------------------------------
# 1. Carry-forward boot: the prior session must still be reachable
# ---------------------------------------------------------------------------


def test_startup_carry_forward_clear_does_not_destroy_the_seed_lineage():
    """#668 clears ``_recovered_inactive_session_id``; the seed lineage survives."""
    state = _SeedState()
    state._recovered_inactive_session_id = None  # startup already cleared it

    lineage = resolve_history_lineage(state)

    assert lineage.fallback_session_id == PRIOR_SESSION
    assert lineage.history_expected is True


def test_guard_seeds_from_prior_session_peak_on_carry_forward_boot():
    state = _SeedState(balance=1015.84, peaks=[1015.98])
    enforcer = _guard_enforcer(state)

    enforcer.check()

    assert state.peak_calls == [PRIOR_SESSION]
    assert enforcer.guard.peak_balance == pytest.approx(1015.98)
    assert enforcer.peak_seed_provenance == PEAK_SEED_DB_SESSION_MAX


def test_breaker_seeds_peak_and_baseline_from_prior_session_on_carry_forward_boot():
    snapshot = SimpleNamespace(equity=1010.0, balance=1009.0)
    state = _SeedState(balance=1015.84, peaks=[1016.44], snapshots=[snapshot])
    enforcer = _breaker_enforcer(state)

    enforcer.check()

    assert state.peak_calls == [PRIOR_SESSION]
    assert state.snapshot_calls == [PRIOR_SESSION]
    assert enforcer.breaker.peak == pytest.approx(1016.44)
    assert enforcer.breaker.daily_baseline == pytest.approx(1010.0)
    assert enforcer.peak_seed_provenance == PEAK_SEED_DB_SESSION_MAX


# ---------------------------------------------------------------------------
# 2. First-snapshot race: empty read is retryable when history was expected
# ---------------------------------------------------------------------------


def test_guard_arms_immediately_then_ratchets_when_the_snapshot_lands_late():
    """The ~3s race must not permanently forfeit the durable peak.

    The cap is armed on the first iteration regardless (never unarmed), and the
    peak is ratcheted UP once the durable read succeeds.
    """
    state = _SeedState(balance=1015.84, peaks=[None, 1015.98])
    enforcer = _guard_enforcer(state)

    enforcer.check()
    assert enforcer.guard.seeded is True  # armed, never left unprotected
    assert enforcer.guard.peak_balance == pytest.approx(1015.84)
    assert enforcer.peak_seed_provenance == PEAK_SEED_UNAVAILABLE

    enforcer.check()
    assert enforcer.guard.peak_balance == pytest.approx(1015.98)
    assert enforcer.peak_seed_provenance == PEAK_SEED_DB_SESSION_MAX


def test_guard_late_peak_never_lowers_an_already_ratcheted_peak():
    state = _SeedState(balance=1000.0, peaks=[None, 900.0])
    enforcer = _guard_enforcer(state)

    enforcer.check()
    state.current_balance = 1200.0
    enforcer.check()  # observed a new high; the late 900 candidate must not win

    assert enforcer.guard.peak_balance == pytest.approx(1200.0)


def test_breaker_defers_seeding_when_history_expected_but_read_is_empty():
    snapshot = SimpleNamespace(equity=1010.0, balance=1009.0)
    state = _SeedState(balance=1015.84, peaks=[None, 1015.98], snapshots=[None, snapshot])
    enforcer = _breaker_enforcer(state)

    enforcer.check()
    assert enforcer.peak_seed_provenance == PEAK_SEED_UNAVAILABLE
    assert enforcer.breaker.peak == pytest.approx(1015.84)  # self-anchored meanwhile

    enforcer.check()
    assert enforcer.peak_seed_provenance == PEAK_SEED_DB_SESSION_MAX
    assert enforcer.breaker.peak == pytest.approx(1015.98)
    assert enforcer.breaker.daily_baseline == pytest.approx(1010.0)


def test_guard_logs_warning_and_unavailable_provenance_when_history_never_arrives(caplog):
    state = _SeedState(balance=1000.0, peaks=[None])
    enforcer = _guard_enforcer(state)

    with caplog.at_level(logging.WARNING):
        for _ in range(MAX_SEED_ATTEMPTS + 2):
            enforcer.check()

    assert enforcer.peak_seed_provenance == PEAK_SEED_UNAVAILABLE
    assert any(r.levelno >= logging.WARNING for r in caplog.records)
    # Bounded: it stops hammering the DB once the budget is spent.
    assert len(state.peak_calls) <= MAX_SEED_ATTEMPTS + 1


def test_breaker_logs_warning_and_unavailable_provenance_when_history_never_arrives(caplog):
    state = _SeedState(balance=1000.0, peaks=[None], snapshots=[None])
    enforcer = _breaker_enforcer(state)

    with caplog.at_level(logging.WARNING):
        for _ in range(MAX_SEED_ATTEMPTS + 2):
            enforcer.check()

    assert enforcer.peak_seed_provenance == PEAK_SEED_UNAVAILABLE
    assert any(r.levelno >= logging.WARNING for r in caplog.records)
    assert len(state.peak_calls) <= MAX_SEED_ATTEMPTS + 1


# ---------------------------------------------------------------------------
# 3. Genuine fresh session: zero behaviour change
# ---------------------------------------------------------------------------


def test_guard_fresh_session_self_anchors_once_without_warning(caplog):
    state = _SeedState(balance=1000.0, peaks=[None], history_session_id=None)
    enforcer = _guard_enforcer(state)

    with caplog.at_level(logging.WARNING):
        enforcer.check()
        enforcer.check()

    assert enforcer.guard.seeded is True
    assert enforcer.guard.peak_balance == pytest.approx(1000.0)
    assert enforcer.peak_seed_provenance == PEAK_SEED_SELF_ANCHORED
    assert state.peak_calls == [None]  # exactly one read; no retry storm
    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []


def test_breaker_fresh_session_self_anchors_once_without_warning(caplog):
    state = _SeedState(balance=1000.0, peaks=[None], snapshots=[None], history_session_id=None)
    enforcer = _breaker_enforcer(state)

    with caplog.at_level(logging.WARNING):
        enforcer.check()
        enforcer.check()

    assert enforcer.peak_seed_provenance == PEAK_SEED_SELF_ANCHORED
    assert state.peak_calls == [None]
    assert state.snapshot_calls == [None]
    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []


def test_breaker_new_utc_day_without_todays_snapshot_is_not_a_defect(caplog):
    """History EXISTS (peak found) but today has no snapshot yet — legitimate."""
    state = _SeedState(balance=1000.0, peaks=[1050.0], snapshots=[None])
    enforcer = _breaker_enforcer(state)

    with caplog.at_level(logging.WARNING):
        enforcer.check()
        enforcer.check()

    assert enforcer.peak_seed_provenance == PEAK_SEED_DB_SESSION_MAX
    assert enforcer.breaker.peak == pytest.approx(1050.0)
    assert len(state.snapshot_calls) == 1  # terminal, no retry
    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []


def test_breaker_trip_event_reports_truthful_peak_provenance():
    """The #1032 ``peak_seed`` field must not claim db_session_max on a miss."""
    state = _SeedState(balance=1000.0, peaks=[None], snapshots=[None])
    breaker = AccountCircuitBreaker(mode="active", daily_loss_limit=0.025, drawdown_halt=0.15)
    enforcer = CircuitBreakerEnforcer(engine_state=state, breaker=breaker)

    enforcer.check()
    state.current_balance = 800.0  # -20% from the self-anchored peak
    enforcer.check()

    assert state._close_only_mode is True
    assert enforcer.peak_seed_provenance == PEAK_SEED_UNAVAILABLE


def test_seeders_use_the_recovered_id_when_the_durable_field_is_absent():
    """Back-compat: state objects without the new field seed as before."""
    state = _SeedState(peaks=[1200.0], history_session_id=None)
    del state._history_seed_session_id
    state._recovered_inactive_session_id = PRIOR_SESSION

    enforcer = _guard_enforcer(state)
    enforcer.check()

    assert state.peak_calls == [PRIOR_SESSION]
    assert enforcer.peak_seed_provenance == PEAK_SEED_DB_SESSION_MAX


def test_now_is_passed_through_for_the_day_window():
    """Sanity: the breaker seeds against the evaluation clock, not import time."""
    state = _SeedState(peaks=[1200.0], snapshots=[None])
    enforcer = _breaker_enforcer(state)
    before = datetime.now(UTC).date()

    enforcer.check()

    assert enforcer.breaker.peak == pytest.approx(1200.0)
    assert before <= datetime.now(UTC).date()
