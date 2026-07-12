"""Portfolio max-drawdown hard-cap enforcement for the live engine.

``MaxDrawdownGuard`` tracks the session peak balance and classifies the
current drawdown into escalation tiers (risk-limits.json ``escalation``):
WARNING at 50% of the cap, CRITICAL at 80%, and a latched BREACH at the cap
itself (``portfolio.max_drawdown_pct``).

``MaxDrawdownEnforcer`` runs the guard on every trading-loop iteration and, on
breach, trips the EXISTING close-only mode, which gates every exposure
increase: entry evaluation, the ``execute_entry_locked`` chokepoint (covers
the legacy duck-typed short path and any direct caller), and scale-ins. Exits,
partial exits, stop-loss management, and reconciliation keep running. It never
liquidates anything — it only stops new risk.

Because the loop-level check runs AFTER entry evaluation, each of those
chokepoints ALSO re-assesses the guard in-line via ``check_before_new_risk``
(through the engine's ``_refresh_drawdown_gate``) so a breach realized earlier
in the same iteration — e.g. a stop-loss fill — blocks that iteration's entry
instead of leaking one bar of fresh exposure (2026-07-12 risk audit P1).

Peak-equity source: the AUTHORITATIVE baseline is the ``account_history``
session max (active session plus the recovered inactive session on clean
restarts), recomputed on boot so a restart cannot reset the drawdown baseline
— a bot restarted mid-breach re-trips naturally on its first loop iteration.
Fallback order is DB session max → current recovered balance; the in-memory
PerformanceTracker peak is NEVER a seed candidate (it can initialize from the
CONFIGURED balance — the optimistic book value that mis-seeded prod at $100
vs true equity ~$84 on 2026-07-04). A failed DB read defers seeding to the
next cycle (bounded by ``MAX_SEED_ATTEMPTS``) instead of latching a
half-seeded baseline.

Baseline policy (PM decision, 2026-07-04): the peak baseline is the peak TRUE
equity since the last reconciled reset — i.e. session-scoped history, NOT
all-time book value. Pre-reset ledger history is deliberately excluded: the
Mar–Jun 2026 rows carry a phantom-era book peak (a software-held $100.00
while true equity was ~$84; ``margin_equity_sync`` wrote the books down to
true equity on 2026-06-03/05 — see the capital-erosion postmortem). Measuring
drawdown from that phantom peak would falsely report an immediate >20% breach
on deploy.

Known limitation (accepted, follow-up #847): because the peak is
session-scoped, a future CLEAN restart that creates a NEW session
re-baselines the peak — the cap degrades to "20% per session" rather than a
rolling 20%. Dormant today: prod reuses the active session across restarts.
The follow-up proposal is a durable cross-session peak anchored to the last
human-verified reconciliation marker.

Clearing a trip is an operator decision: restart with
``FEATURE_MAX_DRAWDOWN_RESET_PEAK=true`` to re-baseline the peak to the
current balance (the guard stays armed from the new baseline; remove the flag
afterwards so later restarts don't silently re-baseline). ``resume_trading()``
alone does not clear it — the guard re-trips on the next iteration while the
drawdown persists.

Thread-safety: all state is read and mutated on the trading-loop thread only,
matching the loop-owned close-only flag it drives.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING, Protocol

from src.config.constants import (
    DRAWDOWN_CRITICAL_AT_PCT_OF_LIMIT,
    DRAWDOWN_GUARD_LOG_INTERVAL_SECONDS,
    DRAWDOWN_WARNING_AT_PCT_OF_LIMIT,
)
from src.config.feature_flags import is_enabled
from src.database.models import EventType
from src.infrastructure.logging.events import log_risk_event

if TYPE_CHECKING:
    from src.database.manager import DatabaseManager

logger = logging.getLogger(__name__)

# Tier thresholds are products of two config floats (e.g. 0.20 * 0.80); binary
# float rounding can land the product 1 ulp above the intended boundary, so an
# exact drawdown reading at the boundary would spuriously miss the tier.
_TIER_EPSILON = 1e-12

RESET_PEAK_FLAG = "max_drawdown_reset_peak"

# Bounded seeding deferrals: a failed/unavailable account_history read defers
# seeding to the next loop cycle rather than latching a half-seeded baseline;
# after this many attempts the guard arms from the current balance so the cap
# is never left unarmed indefinitely.
MAX_SEED_ATTEMPTS = 10


class DrawdownTier(IntEnum):
    """Escalation tier for the current drawdown, ordered by severity."""

    NONE = 0
    WARNING = 1
    CRITICAL = 2
    BREACH = 3


@dataclass(frozen=True)
class DrawdownAssessment:
    """One drawdown reading: level, peak it was measured from, and tier."""

    drawdown: float  # fraction of peak (0..1)
    peak_balance: float
    tier: DrawdownTier
    tripped: bool  # latched hard-cap breach


class MaxDrawdownGuard:
    """Rolling drawdown-from-peak tracker with latched hard-cap breach.

    Pure state machine plus rate-limited tier logging; taking action on a
    breach is the enforcer's job. The trip latches for the process lifetime:
    a partial balance recovery below the cap does not silently re-enable
    entries — that is an operator decision.
    """

    def __init__(
        self,
        max_drawdown_pct: float,
        *,
        warning_at_pct_of_limit: float = DRAWDOWN_WARNING_AT_PCT_OF_LIMIT,
        critical_at_pct_of_limit: float = DRAWDOWN_CRITICAL_AT_PCT_OF_LIMIT,
        log_interval_seconds: float = DRAWDOWN_GUARD_LOG_INTERVAL_SECONDS,
    ) -> None:
        """Configure thresholds; the guard is inert until ``seed_peak``."""
        if not (0 < max_drawdown_pct <= 1):
            raise ValueError("max_drawdown_pct must be in (0, 1]")
        self.max_drawdown_pct = float(max_drawdown_pct)
        self._warning_threshold = self.max_drawdown_pct * warning_at_pct_of_limit
        self._critical_threshold = self.max_drawdown_pct * critical_at_pct_of_limit
        self._log_interval = log_interval_seconds
        self._peak = 0.0
        self._seeded = False
        self._tripped = False
        self._last_tier_log: float | None = None
        self._last_logged_tier = DrawdownTier.NONE

    @property
    def seeded(self) -> bool:
        """Whether the peak baseline has been established."""
        return self._seeded

    @property
    def tripped(self) -> bool:
        """Whether the hard cap has been breached (latched)."""
        return self._tripped

    @property
    def peak_balance(self) -> float:
        """Current peak balance the drawdown is measured from."""
        return self._peak

    def seed_peak(self, current_balance: float, *history_candidates: float | None) -> None:
        """Establish the peak baseline from boot-time candidates.

        Takes the max of the current balance and any finite positive
        candidates (persisted account_history peak, performance-tracker peak).
        Under ``FEATURE_MAX_DRAWDOWN_RESET_PEAK`` the history is ignored and
        the peak re-baselines to the current balance — the documented operator
        override for clearing a drawdown halt.
        """
        if is_enabled(RESET_PEAK_FLAG, default=False):
            self._peak = max(self._as_valid(current_balance) or 0.0, 0.0)
            logger.critical(
                "FEATURE_MAX_DRAWDOWN_RESET_PEAK active — drawdown peak re-baselined "
                "to current balance $%.2f (historical peak ignored). Remove the flag "
                "after recovery or every restart will silently re-baseline.",
                self._peak,
            )
        else:
            valid = [
                v
                for v in (self._as_valid(c) for c in (current_balance, *history_candidates))
                if v is not None
            ]
            self._peak = max(valid, default=0.0)
        self._seeded = True

    @staticmethod
    def _as_valid(candidate: float | None) -> float | None:
        """Coerce a peak candidate to a finite positive float, else None."""
        if candidate is None:
            return None
        try:
            value = float(candidate)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(value) or value <= 0:
            return None
        return value

    def observe(self, balance: float) -> DrawdownAssessment:
        """Record a balance sample; ratchet the peak and classify the tier."""
        if not self._seeded:
            self.seed_peak(balance)
        if not math.isfinite(balance):
            # A garbage sample must neither move the peak nor trip the cap.
            logger.warning("Non-finite balance in drawdown check: %s — sample ignored", balance)
            tier = DrawdownTier.BREACH if self._tripped else DrawdownTier.NONE
            return DrawdownAssessment(0.0, self._peak, tier, self._tripped)

        if balance > self._peak:
            self._peak = balance
        drawdown = (self._peak - balance) / self._peak if self._peak > 0 else 0.0

        tier = DrawdownTier.NONE
        if drawdown + _TIER_EPSILON >= self.max_drawdown_pct:
            tier = DrawdownTier.BREACH
            self._tripped = True
        elif drawdown + _TIER_EPSILON >= self._critical_threshold:
            tier = DrawdownTier.CRITICAL
        elif drawdown + _TIER_EPSILON >= self._warning_threshold:
            tier = DrawdownTier.WARNING
        if self._tripped:
            tier = DrawdownTier.BREACH  # latched: recovery below the cap does not un-trip

        self._log_tier(tier, drawdown, balance)
        return DrawdownAssessment(drawdown, self._peak, tier, self._tripped)

    def _log_tier(self, tier: DrawdownTier, drawdown: float, balance: float) -> None:
        """Log the active tier: immediately on escalation, else rate-limited."""
        if tier is DrawdownTier.NONE:
            if self._last_logged_tier is not DrawdownTier.NONE:
                logger.info(
                    "Drawdown receded below warning tier (%.2f%% of peak $%.2f)",
                    drawdown * 100,
                    self._peak,
                )
                self._last_logged_tier = DrawdownTier.NONE
                self._last_tier_log = None
            return

        now = time.monotonic()
        escalated = tier > self._last_logged_tier
        rate_limit_elapsed = (
            self._last_tier_log is None or now - self._last_tier_log >= self._log_interval
        )
        if not escalated and not rate_limit_elapsed:
            return
        self._last_logged_tier = tier
        self._last_tier_log = now

        detail = (
            f"drawdown {drawdown * 100:.2f}% of peak ${self._peak:,.2f} "
            f"(balance ${balance:,.2f}, hard cap {self.max_drawdown_pct * 100:.1f}%)"
        )
        if tier is DrawdownTier.BREACH:
            logger.critical(
                "🛑 Max-drawdown hard cap breached — close-only mode in force: %s", detail
            )
        elif tier is DrawdownTier.CRITICAL:
            logger.critical(
                "🚨 Drawdown CRITICAL (>= %.1f%% of the cap): %s",
                self._critical_threshold / self.max_drawdown_pct * 100,
                detail,
            )
        else:
            logger.warning(
                "⚠️ Drawdown warning (>= %.1f%% of the cap): %s",
                self._warning_threshold / self.max_drawdown_pct * 100,
                detail,
            )


class DrawdownEngineState(Protocol):
    """Live engine state the drawdown enforcer reads and acts through.

    Accessed dynamically (not captured at construction) because balance and
    session ids mutate throughout the engine lifecycle.
    """

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


class MaxDrawdownEnforcer:
    """Runs the guard on the trading loop and trips close-only mode on breach.

    Fault-isolated: a failing check can never crash the trading loop. The
    protective action (close-only) is taken before observability so an event
    failure cannot leave the account unprotected.
    """

    def __init__(self, engine_state: DrawdownEngineState, guard: MaxDrawdownGuard) -> None:
        """Bind to the engine's live state (see protocol for the surface)."""
        self._state = engine_state
        self._guard = guard
        self._breach_notified = False
        self._seed_attempts = 0

    @property
    def guard(self) -> MaxDrawdownGuard:
        """The underlying drawdown guard (exposed for status/diagnostics)."""
        return self._guard

    def check(self) -> None:
        """Assess current drawdown; enforce the hard cap on breach."""
        self._run_check(count_seed_deferral=True)

    def check_before_new_risk(self) -> None:
        """In-line pre-order variant of :meth:`check` for exposure chokepoints.

        The loop-level check runs AFTER entry evaluation, so on the bar that
        crosses the cap a fresh entry would execute before close-only latches
        (one-iteration leak). Entry evaluation, the ``execute_entry_locked``
        chokepoint, and the scale-in gate call this first — mirroring the
        #807 in-line circuit-breaker gate — so the cap binds on the SAME
        iteration, including the first iteration after a mid-breach restart.

        Identical enforcement, but a deferred seeding attempt is NOT counted
        against ``MAX_SEED_ATTEMPTS``: the bounded current-balance fallback
        stays owned by the once-per-iteration loop check, so several
        chokepoint calls per iteration cannot burn through the deferral
        budget prematurely.
        """
        self._run_check(count_seed_deferral=False)

    def _run_check(self, *, count_seed_deferral: bool) -> None:
        state = self._state
        try:
            balance = float(state.current_balance)
            if not self._guard.seeded and not self._try_seed(
                balance, count_deferral=count_seed_deferral
            ):
                return  # seeding deferred (DB/session not ready); retry next cycle
            assessment = self._guard.observe(balance)
        except Exception as e:
            # Monitoring must never take down the trading loop.
            logger.error("Max-drawdown check failed: %s", e, exc_info=True)
            return

        if not assessment.tripped:
            return
        if self._breach_notified and state._close_only_mode:
            return  # already tripped and halted; guard logs rate-limited reminders

        # Protective action first (idempotent), then page the operator. This
        # also re-fires if close-only is cleared while the breach persists, so
        # resume_trading() alone cannot silently restart entries mid-breach.
        try:
            state._enter_close_only_mode()
            self._breach_notified = True
            message = (
                f"MAX DRAWDOWN HARD CAP BREACHED: drawdown {assessment.drawdown * 100:.2f}% "
                f">= limit {self._guard.max_drawdown_pct * 100:.1f}% "
                f"(peak ${assessment.peak_balance:,.2f} → balance ${balance:,.2f}). "
                "Close-only mode in force — no new entries; exits and stop-losses remain "
                "active. Operator action required: see docs/live_trading.md "
                "(max-drawdown hard cap) to review and clear."
            )
            logger.critical("🛑 %s", message)
            log_risk_event(
                "max_drawdown_breach",
                drawdown=assessment.drawdown,
                peak_balance=assessment.peak_balance,
                balance=balance,
                max_drawdown_pct=self._guard.max_drawdown_pct,
            )
            state._record_event(
                EventType.ALERT,
                message,
                severity="critical",
                component="risk",
                error_code="MAX_DRAWDOWN_BREACH",
                alert=True,
            )
        except Exception as e:
            logger.critical(
                "Max-drawdown breach handling failed after close-only trip: %s",
                e,
                exc_info=True,
            )

    def _try_seed(self, balance: float, *, count_deferral: bool = True) -> bool:
        """Seed the peak baseline; the DB session max is AUTHORITATIVE.

        Returns False when seeding must be deferred to the next loop cycle
        (session id not resolved yet, or the account_history read failed).
        After ``MAX_SEED_ATTEMPTS`` deferrals the guard falls back to the
        current balance with a WARNING so the cap is never left unarmed
        indefinitely. With ``count_deferral=False`` (the in-line pre-order
        gate) a failed attempt defers without consuming that budget and can
        never trigger the fallback — only the loop check arms it.

        The in-memory PerformanceTracker peak is deliberately NOT a seed
        candidate: it can initialize from the CONFIGURED initial balance,
        which mis-seeded the prod guard at the optimistic $100 book value vs
        true session equity ~$84 (2026-07-04) and produced a phantom 15.6%
        drawdown warning on boot.
        """
        state = self._state
        if is_enabled(RESET_PEAK_FLAG, default=False):
            # Operator override: re-baseline to current balance; history
            # (including the DB peak) is intentionally ignored.
            self._guard.seed_peak(balance)
            return True

        if count_deferral:
            self._seed_attempts += 1
        session_id = state.trading_session_id
        db_read_ok = False
        db_peak: float | None = None
        if state.db_manager is None or session_id is None:
            logger.info(
                "Max-drawdown guard seeding deferred: trading session not resolved yet "
                "(attempt %d/%d)",
                self._seed_attempts,
                MAX_SEED_ATTEMPTS,
            )
        else:
            try:
                db_peak = state.db_manager.get_session_peak_balance(
                    session_id,
                    fallback_session_id=state._recovered_inactive_session_id,
                )
                # None here means the read SUCCEEDED but the session has no
                # snapshots yet — a fresh session legitimately baselines at
                # the current balance.
                db_read_ok = True
            except Exception as e:
                logger.warning(
                    "Could not recompute session peak from account_history "
                    "(attempt %d/%d): %s — will retry next cycle",
                    self._seed_attempts,
                    MAX_SEED_ATTEMPTS,
                    e,
                )

        if not db_read_ok:
            if not count_deferral or self._seed_attempts < MAX_SEED_ATTEMPTS:
                return False
            logger.warning(
                "Max-drawdown guard: no successful account_history read after %d "
                "attempts — falling back to current balance $%.2f as the peak "
                "baseline (drawdown measured from here on)",
                self._seed_attempts,
                balance,
            )

        self._guard.seed_peak(balance, db_peak)
        logger.info(
            "Max-drawdown guard armed: peak=$%.2f, hard cap=%.1f%% (session %s, "
            "account_history peak %s)",
            self._guard.peak_balance,
            self._guard.max_drawdown_pct * 100,
            session_id,
            f"${db_peak:,.2f}" if db_peak is not None else "unavailable",
        )
        return True
