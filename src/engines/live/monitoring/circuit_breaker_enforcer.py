"""Account circuit-breaker enforcement on the live trading loop (#807 follow-up).

The #807 ``AccountCircuitBreaker`` is evaluated in the pre-order gate (it blocks
new entries when tripped). This enforcer adds the loop-driven half:

- **True-equity evaluation (#986 gap A)**: the breaker is fed
  ``current_balance + unrealized P&L of open positions``, not raw cash.
  ``current_balance`` only moves on realized events, so a cash-fed breaker is
  structurally blind to an open position's adverse move — the exact loss it
  exists to halt — for the entire holding period. The unrealized read is
  fault-isolated: if it is unavailable on an iteration the check degrades
  explicitly to balance-only with a WARNING (never crashes the loop, never
  silently halts).
- **Restart-safe daily baseline**: on boot it seeds the breaker's daily-loss
  baseline from the day's first ``account_history`` snapshot
  (``get_first_snapshot_of_day``, equity basis), so an intraday restart does not
  re-anchor the baseline to current equity and silently disarm the daily-loss
  halt.
- **Restart-safe drawdown peak (#986 gap B)**: on boot it seeds the breaker's
  drawdown peak from the durable ``account_history`` session equity max
  (``get_session_peak_equity``), the same session-scoped pattern the
  ``MaxDrawdownGuard`` uses since #1001 — a restart cannot silently zero the
  15% halt's memory (the #845/#847 peak-reset class).
- **Close-only on trip**: in ``active`` mode a trip flips the engine's existing
  **close-only mode** (new entries AND scale-ins stop; exits and stop-losses keep
  running — nothing is liquidated), matching the ``MaxDrawdownGuard`` precedent
  (the codebase deliberately does not force-liquidate into a dip). In ``dry_run``
  it logs "would halt", writes a ``CIRCUIT_BREAKER_DRY_RUN`` ``system_events``
  row (durable would-have-tripped evidence carrying the equity breakdown and
  peak provenance, #968), and takes no protective action. ``off`` is fully inert.
- **Surfacing**: a trip emits a ``risk_event`` + a CRITICAL ``system_events`` row
  so the monitoring dashboard and alerting pick it up.

Fault-isolated: a failing check never crashes the trading loop, and the
protective action is taken before observability so an event failure cannot leave
the account unprotected.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
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
        self,
        engine_state: CircuitBreakerEngineState,
        breaker: AccountCircuitBreaker,
        *,
        unrealized_pnl_provider: Callable[[], float] | None = None,
    ) -> None:
        """Bind to the engine's live state.

        Args:
            engine_state: Live engine state (see protocol for the surface).
            breaker: The shared breaker instance (also wired to the entry gate).
            unrealized_pnl_provider: Mark-to-market unrealized P&L of open
                positions (the live position tracker's sum), so the breaker
                evaluates TRUE equity. ``None`` evaluates cash balance only.
        """
        self._state = engine_state
        self._breaker = breaker
        self._unrealized_provider = unrealized_pnl_provider
        self._seeded = False
        self._seed_attempts = 0
        self._halt_notified = False
        self._equity_degraded = False
        self._peak_seed_provenance = "self_anchored"

    @property
    def breaker(self) -> AccountCircuitBreaker:
        return self._breaker

    def check(self) -> None:
        """Evaluate the breaker against current TRUE equity; enforce halts."""
        state = self._state
        mode = self._breaker.mode
        if mode not in (MODE_DRY_RUN, MODE_ACTIVE):
            return  # off -> fully inert

        try:
            balance = float(state.current_balance)
            equity, unrealized, basis = self._equity_reading(balance)
            now = datetime.now(UTC)
            if not self._seeded:
                self._try_seed(now)  # best-effort; evaluate proceeds regardless
            decision = self._breaker.evaluate(equity, now)
        except Exception as e:  # noqa: BLE001 - monitoring must never crash the loop
            logger.error("Circuit-breaker check failed: %s", e, exc_info=True)
            return

        if not decision.tripped:
            self._halt_notified = False  # cleared (e.g. next UTC day) -> allow re-notify
            return

        # Auditable measurement snapshot (#968): what was measured, from what
        # basis, against which peak, and where that peak came from.
        measurement = (
            f"equity ${equity:,.2f} = balance ${balance:,.2f} "
            f"+ unrealized ${unrealized:,.2f}, basis={basis}, "
            f"peak ${self._breaker.peak:,.2f} [{self._peak_seed_provenance}]"
        )

        if mode == MODE_DRY_RUN:
            if not self._halt_notified:
                self._halt_notified = True
                message = (
                    f"Account circuit breaker WOULD HALT (dry_run): {decision.reason} "
                    f"({measurement}). "
                    "Set account_circuit_breakers=active to enforce."
                )
                logger.warning("🟡 %s", message)
                # dry_run exists to accumulate would-have-tripped evidence for the
                # promote/kill verdict (#964); container stdout is ephemeral, so
                # the trip must also land in system_events. Best-effort: an
                # observability failure never crashes the trading loop.
                try:
                    log_risk_event(
                        "account_circuit_breaker_dry_run",
                        reason=decision.reason,
                        equity=equity,
                        balance=balance,
                        unrealized_pnl=unrealized,
                        equity_basis=basis,
                        peak=self._breaker.peak,
                        peak_seed=self._peak_seed_provenance,
                        mode=mode,
                    )
                    state._record_event(
                        EventType.CIRCUIT_BREAKER_DRY_RUN,
                        message,
                        severity="warning",
                        component="risk",
                        error_code="CIRCUIT_BREAKER_DRY_RUN",
                    )
                except Exception as e:  # noqa: BLE001
                    logger.error(
                        "Failed to record circuit-breaker dry-run event: %s", e, exc_info=True
                    )
            return

        # active mode: trip close-only (protective action first, then observe).
        if self._halt_notified and state._close_only_mode:
            return
        try:
            state._enter_close_only_mode()
            self._halt_notified = True
            message = (
                f"ACCOUNT CIRCUIT BREAKER TRIPPED: {decision.reason} ({measurement}). "
                "Close-only mode in force — no new entries or scale-ins; exits and stop-losses "
                "remain active. Operator action required to review and clear."
            )
            logger.critical("🛑 %s", message)
            log_risk_event(
                "account_circuit_breaker_trip",
                reason=decision.reason,
                equity=equity,
                balance=balance,
                unrealized_pnl=unrealized,
                equity_basis=basis,
                peak=self._breaker.peak,
                peak_seed=self._peak_seed_provenance,
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

    def _equity_reading(self, balance: float) -> tuple[float, float, str]:
        """Read TRUE equity for the breaker: ``(equity, unrealized, basis)``.

        ``basis`` is ``"equity"`` (cash + unrealized), ``"balance"`` (no
        provider wired), or ``"balance_degraded"`` (provider unavailable this
        iteration — degraded explicitly with a WARNING on the transition, not
        on every loop iteration). Never raises.
        """
        provider = self._unrealized_provider
        if provider is None:
            return balance, 0.0, "balance"
        try:
            unrealized = float(provider())
            if not math.isfinite(unrealized):
                raise ValueError(f"non-finite unrealized P&L: {unrealized}")
            equity = balance + unrealized
            if equity <= 0 < balance:
                # evaluate() ignores non-positive equity entirely — surface the
                # degenerate read instead of letting the breaker silently no-op.
                raise ValueError(f"non-positive equity {equity} from unrealized {unrealized}")
        except Exception as e:  # noqa: BLE001 - equity read must never crash the loop
            if not self._equity_degraded:
                self._equity_degraded = True
                logger.warning(
                    "Circuit-breaker unrealized P&L unavailable — degrading to "
                    "balance-only evaluation (balance $%.2f): %s",
                    balance,
                    e,
                )
            return balance, 0.0, "balance_degraded"
        if self._equity_degraded:
            self._equity_degraded = False
            logger.info("Circuit-breaker equity read recovered — resuming equity evaluation")
        return equity, unrealized, "equity"

    def _try_seed(self, now: datetime) -> None:
        """Seed the daily baseline and drawdown peak from durable history.

        Restart-safety (both halts): the daily baseline comes from the day's
        first ``account_history`` snapshot (equity basis) and the drawdown peak
        from the session's ``account_history`` equity max — the same
        session-scoped seeding the ``MaxDrawdownGuard`` uses (#1001), including
        the recovered-inactive-session fallback on clean restarts.

        Best-effort: on any failure or missing rows, leaves the breaker to
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
            session_peak = state.db_manager.get_session_peak_equity(
                session_id=state.trading_session_id,
                fallback_session_id=getattr(state, "_recovered_inactive_session_id", None),
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("Circuit-breaker baseline/peak seed deferred: %s", e)
            if self._seed_attempts >= MAX_SEED_ATTEMPTS:
                self._seeded = True
            return

        if snapshot is not None:
            # Equity basis to match evaluate(); legacy/stub rows without a
            # usable equity fall back to balance. Numeric columns load as
            # Decimal — coerce before mixing with floats.
            baseline = self._as_positive_float(
                getattr(snapshot, "equity", None)
            ) or self._as_positive_float(getattr(snapshot, "balance", None))
            if baseline is not None:
                self._breaker.seed_daily_baseline(baseline, now.date())
                logger.info(
                    "Circuit-breaker daily baseline seeded from day-start snapshot: $%.2f",
                    baseline,
                )
        # No snapshot for today yet (fresh day / first run) -> self-anchor is fine.

        peak = self._as_positive_float(session_peak)
        if peak is not None:
            self._breaker.seed_peak(peak)
            self._peak_seed_provenance = "db_session_max"
            logger.info(
                "Circuit-breaker drawdown peak seeded from account_history session max: $%.2f",
                peak,
            )
        # No session history yet -> peak self-anchors from current equity.
        self._seeded = True

    @staticmethod
    def _as_positive_float(value: object) -> float | None:
        """Coerce a DB-loaded numeric (possibly Decimal/None) to a finite positive float."""
        if value is None:
            return None
        try:
            result = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None
        if not math.isfinite(result) or result <= 0:
            return None
        return result
