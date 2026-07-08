"""Unit tests for the manual system-halt loop enforcer (#922).

The enforcer polls the DB ``system_halt`` control flag once per trading-loop
iteration and mirrors it into the loop-owned ``SystemHaltState`` that the
entry/scale-in gates consult. Transitions emit operator-visible events;
poll failures keep the last-known state (a DB outage can never silently
clear an active halt).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.database.manager import SystemHaltStatus
from src.engines.live.monitoring.system_halt_enforcer import SystemHaltEnforcer
from src.engines.live.system_halt import SystemHaltState

pytestmark = [pytest.mark.unit, pytest.mark.fast]


class _State:
    """Minimal engine-state stub: a swappable halt status + event recorder."""

    def __init__(self, status: SystemHaltStatus | Exception):
        self.status = status
        self.events: list[dict] = []
        self.db_manager = SimpleNamespace(get_system_halt=self._get_system_halt)

    def _get_system_halt(self) -> SystemHaltStatus:
        if isinstance(self.status, Exception):
            raise self.status
        return self.status

    def _record_event(self, event_type, message, **kwargs):
        self.events.append({"message": message, **kwargs})


def _status(active: bool, reason: str | None = None) -> SystemHaltStatus:
    return SystemHaltStatus(active=active, reason=reason, source="cli:test", updated_at=None)


def test_halt_flag_activates_state_and_records_critical_event():
    state = _State(_status(True, "kill-switch drill"))
    halt = SystemHaltState()

    SystemHaltEnforcer(state, halt).check()

    assert halt.active is True
    assert halt.reason == "kill-switch drill"
    assert len(state.events) == 1
    event = state.events[0]
    assert event["error_code"] == "SYSTEM_HALT"
    assert event["severity"] == "critical"
    assert event["alert"] is True
    assert "kill-switch drill" in event["message"]


def test_repeated_active_checks_do_not_duplicate_events():
    state = _State(_status(True, "drill"))
    halt = SystemHaltState()
    enforcer = SystemHaltEnforcer(state, halt)

    for _ in range(3):
        enforcer.check()

    assert halt.active is True
    assert len(state.events) == 1


def test_clear_flag_deactivates_and_records_resume_event():
    state = _State(_status(True, "drill"))
    halt = SystemHaltState()
    enforcer = SystemHaltEnforcer(state, halt)
    enforcer.check()

    state.status = _status(False, "drill complete")
    enforcer.check()

    assert halt.active is False
    assert halt.reason is None
    assert len(state.events) == 2
    resume = state.events[1]
    assert resume["error_code"] == "SYSTEM_HALT_CLEARED"
    assert resume["severity"] == "warning"
    assert resume["alert"] is True


def test_inactive_flag_records_no_events():
    state = _State(_status(False))
    halt = SystemHaltState()

    SystemHaltEnforcer(state, halt).check()

    assert halt.active is False
    assert state.events == []


def test_poll_failure_keeps_active_halt_latched():
    """A DB outage must never silently release an active halt."""
    state = _State(_status(True, "drill"))
    halt = SystemHaltState()
    enforcer = SystemHaltEnforcer(state, halt)
    enforcer.check()

    state.status = RuntimeError("db unreachable")
    enforcer.check()

    assert halt.active is True
    assert halt.reason == "drill"
    assert len(state.events) == 1  # no spurious clear/halt events


def test_poll_failure_when_not_halted_stays_not_halted():
    state = _State(RuntimeError("db unreachable"))
    halt = SystemHaltState()

    SystemHaltEnforcer(state, halt).check()  # must not raise

    assert halt.active is False
    assert state.events == []


def test_reason_updates_while_halted_without_new_event():
    state = _State(_status(True, "first reason"))
    halt = SystemHaltState()
    enforcer = SystemHaltEnforcer(state, halt)
    enforcer.check()

    state.status = _status(True, "amended reason")
    enforcer.check()

    assert halt.reason == "amended reason"
    assert len(state.events) == 1


def test_event_failure_does_not_undo_protective_action():
    """Protective action first: a failing event sink cannot unset the halt."""

    class _ExplodingState(_State):
        def _record_event(self, *args, **kwargs):
            raise RuntimeError("event sink down")

    state = _ExplodingState(_status(True, "drill"))
    halt = SystemHaltState()

    SystemHaltEnforcer(state, halt).check()  # must not raise

    assert halt.active is True


# ---------------------------------------------------------------------------
# Fail-closed startup: never-successfully-polled state is not trusted
# ---------------------------------------------------------------------------


def test_state_unestablished_until_first_successful_poll():
    """`established` flips only on a successful poll — never on failure."""
    state = _State(RuntimeError("db unreachable"))
    halt = SystemHaltState()
    enforcer = SystemHaltEnforcer(state, halt)

    assert halt.established is False
    enforcer.check()
    assert halt.established is False  # failed poll cannot establish truth

    state.status = _status(False)
    enforcer.check()
    assert halt.established is True


def test_prime_failure_fails_closed_and_records_unverified_event():
    """Boot with a dead DB: state stays unestablished and the operator is paged."""
    state = _State(RuntimeError("db unreachable at boot"))
    halt = SystemHaltState()

    SystemHaltEnforcer(state, halt).prime()

    assert halt.established is False
    assert len(state.events) == 1
    event = state.events[0]
    assert event["error_code"] == "SYSTEM_HALT_UNVERIFIED"
    assert event["severity"] == "critical"
    assert event["alert"] is True


def test_prime_healthy_boot_establishes_without_events():
    """Boot with DB up and no halt row: trading starts immediately, quietly."""
    state = _State(_status(False))
    halt = SystemHaltState()

    SystemHaltEnforcer(state, halt).prime()

    assert halt.established is True
    assert halt.active is False
    assert state.events == []


def test_prime_with_active_row_activates_and_announces():
    """Boot with an active halt row: the halt is enforced from construction."""
    state = _State(_status(True, "pre-restart drill"))
    halt = SystemHaltState()

    SystemHaltEnforcer(state, halt).prime()

    assert halt.established is True
    assert halt.active is True
    assert len(state.events) == 1
    assert state.events[0]["error_code"] == "SYSTEM_HALT"


def test_prime_event_failure_still_fails_closed():
    """The unverified-boot page failing cannot flip the state to trading."""

    class _ExplodingState(_State):
        def _record_event(self, *args, **kwargs):
            raise RuntimeError("event sink down")

    state = _ExplodingState(RuntimeError("db unreachable"))
    halt = SystemHaltState()

    SystemHaltEnforcer(state, halt).prime()  # must not raise

    assert halt.established is False


def test_non_boolean_active_value_does_not_halt():
    """Only an explicit boolean True halts — a mocked/garbage status must not
    phantom-halt an engine (real reads coerce the column to bool)."""
    state = _State(
        SystemHaltStatus(active=object(), reason=None, source=None, updated_at=None)  # type: ignore[arg-type]
    )
    halt = SystemHaltState()

    SystemHaltEnforcer(state, halt).check()

    assert halt.active is False
    assert halt.established is True
    assert state.events == []
