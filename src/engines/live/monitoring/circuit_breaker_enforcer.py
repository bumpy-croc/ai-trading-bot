"""Account circuit-breaker enforcement on the live trading loop (#807 follow-up).

The #807 ``AccountCircuitBreaker`` is evaluated in the pre-order gate (it blocks
new entries when tripped). This enforcer adds the loop-driven half:

- **Restart-safe daily baseline**: on boot it seeds the breaker's daily-loss
  baseline from the day's first ``account_history`` snapshot
  (``get_first_snapshot_of_day``), so an intraday restart does not re-anchor the
  baseline to current equity and silently disarm the daily-loss halt.
- **Close-only on trip**: in ``active`` mode a trip flips the engine's existing
  **close-only mode** (new entries AND scale-ins stop; exits and stop-losses keep
  running — nothing is liquidated), matching the ``MaxDrawdownGuard`` precedent
  (the codebase deliberately does not force-liquidate into a dip). In ``dry_run``
  it logs "would halt" and takes no action. ``off`` is fully inert.
- **Surfacing**: a trip emits a ``risk_event`` + a CRITICAL ``system_events`` row
  so the monitoring dashboard and alerting pick it up.

Fault-isolated: a failing check never crashes the trading loop, and the
protective action is taken before observability so an event failure cannot leave
the account unprotected.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Protocol

from src.database.models import EventType
from src.infrastructure.logging.events import log_risk_event
from src.risk.circuit_breaker import MODE_ACTIVE, MODE_DRY_RUN, AccountCircuitBreaker

if TYPE_CHECKING:
    from src.database.manager import DatabaseManager

logger = logging.getLogger(__name__)

# Bounded seeding deferrals (mirrors MaxDrawdownEnforcer): a failed/unavailable
# day-snapshot read defers seeding to the next cycle; after this many attempts we
# arm from the current balance so the breaker is never left unseeded indefinitely.
MAX_SEED_ATTEMPTS = 10


class CircuitBreakerEngineState(Protocol):
    """Live engine state the circuit-breaker enforcer reads and acts through."""

    current_balance: float
    trading_session_id: int | None
    _recovered_inactive_session_id: int | None
    db_manager: DatabaseManager
    _close_only_mode: bool

    def _enter_close_only_mode(self) -> None: ...

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


class CircuitBreakerEnforcer:
    """Runs the account circuit breaker on the trading loop."""

    def __init__(
        self, engine_state: CircuitBreakerEngineState, breaker: AccountCircuitBreaker
    ) -> None:
        self._state = engine_state
        self._breaker = breaker
        self._seeded = False
        self._seed_attempts = 0
        self._halt_notified = False

    @property
    def breaker(self) -> AccountCircuitBreaker:
        return self._breaker

    def check(self) -> None:
        """Evaluate the breaker against current equity; enforce halts."""
        state = self._state
        mode = self._breaker.mode
        if mode not in (MODE_DRY_RUN, MODE_ACTIVE):
            return  # off -> fully inert

        try:
            balance = float(state.current_balance)
            now = datetime.now(UTC)
            if not self._seeded:
                self._try_seed(balance, now)  # best-effort; evaluate proceeds regardless
            decision = self._breaker.evaluate(balance, now)
        except Exception as e:  # noqa: BLE001 - monitoring must never crash the loop
            logger.error("Circuit-breaker check failed: %s", e, exc_info=True)
            return

        if not decision.tripped:
            self._halt_notified = False  # cleared (e.g. next UTC day) -> allow re-notify
            return

        if mode == MODE_DRY_RUN:
            if not self._halt_notified:
                logger.warning(
                    "🟡 Account circuit breaker WOULD HALT (dry_run): %s (balance $%.2f). "
                    "Set account_circuit_breakers=active to enforce.",
                    decision.reason,
                    balance,
                )
                self._halt_notified = True
            return

        # active mode: trip close-only (protective action first, then observe).
        if self._halt_notified and state._close_only_mode:
            return
        try:
            state._enter_close_only_mode()
            self._halt_notified = True
            message = (
                f"ACCOUNT CIRCUIT BREAKER TRIPPED: {decision.reason} (balance ${balance:,.2f}). "
                "Close-only mode in force — no new entries or scale-ins; exits and stop-losses "
                "remain active. Operator action required to review and clear."
            )
            logger.critical("🛑 %s", message)
            log_risk_event(
                "account_circuit_breaker_trip",
                reason=decision.reason,
                balance=balance,
                mode=mode,
            )
            state._record_event(
                EventType.ALERT,
                message,
                severity="critical",
                component="risk",
                error_code="ACCOUNT_CIRCUIT_BREAKER_TRIP",
                alert=True,
            )
        except Exception as e:  # noqa: BLE001
            logger.critical(
                "Circuit-breaker trip handling failed after close-only: %s", e, exc_info=True
            )

    def _try_seed(self, balance: float, now: datetime) -> None:
        """Seed the daily baseline from the day's first snapshot (restart-safety).

        Best-effort: on any failure or missing snapshot, leaves the breaker to
        self-anchor from current equity. After ``MAX_SEED_ATTEMPTS`` deferrals we
        stop trying (the breaker is already self-anchored to a live baseline)."""
        state = self._state
        self._seed_attempts += 1
        if state.db_manager is None or state.trading_session_id is None:
            if self._seed_attempts >= MAX_SEED_ATTEMPTS:
                self._seeded = True  # give up; breaker self-anchors from current equity
            return
        try:
            snapshot = state.db_manager.get_first_snapshot_of_day(
                session_id=state.trading_session_id,
                target_date=now.date(),
                fallback_session_id=getattr(state, "_recovered_inactive_session_id", None),
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("Circuit-breaker daily-baseline seed deferred: %s", e)
            if self._seed_attempts >= MAX_SEED_ATTEMPTS:
                self._seeded = True
            return

        if snapshot is not None and getattr(snapshot, "balance", None):
            baseline = float(snapshot.balance)
            self._breaker.seed_daily_baseline(baseline, now.date())
            logger.info(
                "Circuit-breaker daily baseline seeded from day-start snapshot: $%.2f",
                baseline,
            )
        # No snapshot for today yet (fresh day / first run) -> self-anchor is fine.
        self._seeded = True
