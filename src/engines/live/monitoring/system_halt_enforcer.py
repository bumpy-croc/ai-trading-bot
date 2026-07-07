"""Manual kill-switch enforcement on the live trading loop (#922).

``atb live-control halt`` durably sets the ``system_halt`` control flag in the
target environment's database. This enforcer polls that flag once per
trading-loop iteration (before entry evaluation, so a halt takes effect within
the SAME iteration) and mirrors it into the loop-owned ``SystemHaltState``
consumed by the entry/scale-in gates. Semantics match FEATURE_ENTRY_PAUSE: no
exposure increases — new entries and scale-ins stop; exits, partial exits,
stop-loss management and reconciliation continue. Nothing is liquidated.

Transitions are announced once (CRITICAL system_event + alert on halt, WARNING
on clear). Fail-safe: a failed poll keeps the last-known state, so a database
outage can never silently release an active halt — and never trips one either
(the existing DB-outage close-only guard covers prolonged outages). Startup is
fail-CLOSED: until the FIRST successful poll (a priming read at engine
construction, retried every loop iteration) the shared state is unestablished
and the entry gates refuse new risk — a reboot behind a dead database cannot
trade past an operator halt it never managed to read.

Fault-isolated: a failing check never crashes the trading loop, and the
protective action (mirroring the halt) is taken before observability so an
event failure cannot leave entries running.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Protocol

from src.database.models import EventType
from src.engines.live.system_halt import SystemHaltState

if TYPE_CHECKING:
    from src.database.manager import DatabaseManager

logger = logging.getLogger(__name__)


class SystemHaltEngineState(Protocol):
    """Live engine state the system-halt enforcer reads and acts through."""

    db_manager: DatabaseManager

    def _record_event(
        self,
        event_type: EventType,
        message: str,
        *,
        severity: str = ...,
        component: str | None = ...,
        error_code: str | None = ...,
        exc: BaseException | None = ...,
        alert: bool = ...,
    ) -> None: ...


class SystemHaltEnforcer:
    """Polls the DB ``system_halt`` flag and drives the shared halt state."""

    def __init__(self, engine_state: SystemHaltEngineState, halt_state: SystemHaltState) -> None:
        """Bind to the engine's live state and the shared halt-state holder."""
        self._state = engine_state
        self._halt = halt_state

    def prime(self) -> None:
        """One authoritative startup read; loudly fail-closed if unverifiable.

        Called at engine construction so a healthy boot establishes the halt
        state BEFORE the trading loop starts. If the read fails, the state
        stays unestablished — the entry gates refuse new risk until the first
        successful in-loop poll — and the operator is paged.
        """
        self.check()
        if self._halt.established:
            return
        message = (
            "System-halt state UNVERIFIED at startup (the system_halt flag could not "
            "be read) — failing CLOSED: new entries and scale-ins stay blocked until "
            "the flag is successfully polled. Exits, stop-losses and reconciliation "
            "run normally."
        )
        logger.critical("🛑 %s", message)
        try:
            self._state._record_event(
                EventType.ALERT,
                message,
                severity="critical",
                component="ops",
                error_code="SYSTEM_HALT_UNVERIFIED",
                alert=True,
            )
        except Exception as e:
            logger.warning("Unverified-halt boot announcement failed: %s", e)

    def check(self) -> None:
        """Mirror the DB flag into the loop-owned halt state; announce transitions."""
        try:
            status = self._state.db_manager.get_system_halt()
        except Exception as e:
            # Fail-safe: keep the last-known state. An active halt stays
            # latched through a DB outage; an inactive one is not tripped —
            # and an UNESTABLISHED state stays fail-closed at the gates.
            logger.warning(
                "system_halt flag poll failed: %s — keeping last state "
                "(halted=%s, established=%s)",
                e,
                self._halt.active,
                self._halt.established,
            )
            return

        # Only an explicit boolean True halts. The real read coerces the DB
        # column to bool; this guards test doubles/garbage from phantom-halting
        # a live engine.
        active = status.active is True
        self._halt.established = True

        if active == self._halt.active:
            if active:
                # Reason may be amended while halted; keep the mirror fresh
                # without re-announcing.
                self._halt.reason = status.reason
            return

        if active:
            self._activate(status.reason, status.source)
        else:
            self._deactivate()

    def _activate(self, reason: str | None, source: str | None) -> None:
        """Enforce the halt (protective action first), then page the operator."""
        self._halt.active = True
        self._halt.reason = reason
        try:
            message = (
                f"MANUAL SYSTEM HALT ENFORCED (set by {source or 'unknown'}, "
                f"reason: {reason or 'no reason recorded'}). New entries and scale-ins "
                "are blocked; exits, stop-losses and reconciliation continue. "
                "Clear with 'atb live-control resume'."
            )
            logger.critical("🛑 %s", message)
            self._state._record_event(
                EventType.ALERT,
                message,
                severity="critical",
                component="ops",
                error_code="SYSTEM_HALT",
                alert=True,
            )
        except Exception as e:
            logger.critical("System-halt announcement failed after enforcement: %s", e)

    def _deactivate(self) -> None:
        """Release the halt and announce that entries are live again."""
        self._halt.active = False
        self._halt.reason = None
        try:
            message = "Manual system halt cleared — new entries and scale-ins are enabled again."
            logger.warning("✅ %s", message)
            self._state._record_event(
                EventType.ALERT,
                message,
                severity="warning",
                component="ops",
                error_code="SYSTEM_HALT_CLEARED",
                alert=True,
            )
        except Exception as e:
            logger.warning("System-halt clear announcement failed: %s", e)
