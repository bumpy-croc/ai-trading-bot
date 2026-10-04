"""Portfolio max-drawdown hard-cap enforcement for the live engine.

``MaxDrawdownGuard`` tracks two drawdown legs and classifies the WORSE of them
into escalation tiers (risk-limits.json ``escalation``): WARNING at 50% of the
cap, CRITICAL at 80%, and a latched BREACH at the cap itself
(``portfolio.max_drawdown_pct``).

- Balance leg: realized balance vs the session peak balance. This is the
  original guard, unchanged — it can only move on realized fills.
- Equity leg: mark-to-market equity (balance + unrealized P&L of open
  positions) vs its own session peak equity, so a position running deep
  underwater trips the cap before it is stopped out. The equity peak is floored
  at the balance peak, so with no open position the two legs are identical.

Breach is ``max(balance leg, equity leg) >= cap``: the old balance condition is
a literal disjunct, so the guard can never be looser than the balance-only
version. The equity leg is skipped (never guessed) when no trustworthy mark is
available — invalid price, position recovered but not yet re-marked, or a stale
price feed — and escalates to a CRITICAL alert if that persists. The equity
peak only ratchets up once two consecutive distinct marks agree, so a single
bad tick cannot inflate it into a phantom latch.

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
session max (active session plus the prior session on clean restarts, resolved
via ``seed_lineage``), recomputed on boot so a restart cannot reset the
drawdown baseline — a bot restarted mid-breach re-trips naturally on its first
loop iteration. Fallback order is DB session max → current recovered balance;
the in-memory PerformanceTracker peak is NEVER a seed candidate (it can
initialize from the CONFIGURED balance — the optimistic book value that
mis-seeded prod at $100 vs true equity ~$84 on 2026-07-04). A failed DB read
defers seeding to the next cycle (bounded by ``MAX_SEED_ATTEMPTS``) instead of
latching a half-seeded baseline.

When the durable peak is EXPECTED (a prior session exists) but the read comes
back empty — the first-snapshot race of #1036, where the new session's opening
``account_history`` row lands seconds after the first loop iteration — the
guard arms provisionally from the current balance so the cap is never unarmed,
records ``peak_seed=seed_unavailable``, and keeps retrying to ratchet the peak
UP (``raise_peak``) until the durable value is readable or the attempt budget
is spent. An empty read with NO prior session is a genuinely fresh account and
self-anchors terminally, as before.

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
from collections.abc import Callable
from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING, Protocol

from src.config.constants import (
    DRAWDOWN_CRITICAL_AT_PCT_OF_LIMIT,
    DRAWDOWN_GUARD_LOG_INTERVAL_SECONDS,
    DRAWDOWN_MARK_MAX_AGE_SECONDS,
    DRAWDOWN_WARNING_AT_PCT_OF_LIMIT,
)
from src.config.feature_flags import is_enabled
from src.database.models import EventType
from src.engines.live.monitoring.seed_lineage import (
    PEAK_SEED_DB_SESSION_MAX,
    PEAK_SEED_SELF_ANCHORED,
    PEAK_SEED_UNAVAILABLE,
    HistoryLineage,
    resolve_history_lineage,
)
from src.infrastructure.logging.events import log_risk_event

if TYPE_CHECKING:
    from src.database.manager import DatabaseManager
    from src.engines.live.execution.position_tracker import MarkReading

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

# Seconds the equity leg may stay blind (no fresh mark) with a position open before the
# operator is paged: the balance leg alone cannot see the unrealized loss it exists to catch.
MARK_BLIND_ESCALATE_SECONDS = DRAWDOWN_MARK_MAX_AGE_SECONDS


class DrawdownTier(IntEnum):
    """Escalation tier for the current drawdown, ordered by severity."""

    NONE = 0
    WARNING = 1
    CRITICAL = 2
    BREACH = 3


@dataclass(frozen=True)
class DrawdownAssessment:
    """One drawdown reading: level, peak it was measured from, and tier."""

    drawdown: float  # worst leg, fraction of that leg's peak (0..1)
    peak_balance: float
    tier: DrawdownTier
    tripped: bool  # latched hard-cap breach
    balance_drawdown: float = 0.0
    equity_drawdown: float = 0.0
    equity_peak: float = 0.0
    equity: float | None = None  # None: equity leg not evaluated for this sample


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
        self._equity_peak = 0.0
        # Consecutive distinct marks, so a lone bad tick cannot ratchet the equity peak.
        self._prev_equity: float | None = None
        self._cur_equity: float | None = None
        self._equity_mark_sequence: int | None = None
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

    @property
    def equity_peak(self) -> float:
        """Peak mark-to-market equity the equity leg is measured from.

        Floored at the balance peak: equity equals balance with no open
        position, so the equity leg can never sit on a lower peak than the
        balance leg.
        """
        return max(self._equity_peak, self._peak)

    def seed_peak(
        self,
        current_balance: float,
        *history_candidates: float | None,
        equity_peak: float | None = None,
    ) -> None:
        """Establish the peak baselines from boot-time candidates.

        Takes the max of the current balance and any finite positive
        candidates (persisted account_history peak, performance-tracker peak).
        ``equity_peak`` is the persisted session equity max for the equity leg.
        Under ``FEATURE_MAX_DRAWDOWN_RESET_PEAK`` the history is ignored and
        both peaks re-baseline to the current balance — the documented operator
        override for clearing a drawdown halt.
        """
        reset = is_enabled(RESET_PEAK_FLAG, default=False)
        if reset:
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
        self._equity_peak = self._peak
        if not reset:
            self._equity_peak = max(self._peak, self._as_valid(equity_peak) or 0.0)
        self._seeded = True

    def raise_peak(self, candidate: float | None) -> bool:
        """Ratchet the peak UP from a late-arriving durable candidate (#1036).

        Used when the boot-time ``account_history`` read came back empty
        because the session's first snapshot had not landed yet: the guard arms
        immediately from the current balance (never unarmed) and the true peak
        is raised as soon as it becomes readable. Only ever raises — a stale or
        lower candidate can never erase drawdown already observed.
        """
        value = self._as_valid(candidate)
        if value is None or value <= self._peak:
            return False
        self._peak = value
        return True

    def raise_equity_peak(self, candidate: float | None) -> bool:
        """Ratchet the equity peak UP from a late-arriving durable candidate."""
        value = self._as_valid(candidate)
        if value is None or value <= self.equity_peak:
            return False
        self._equity_peak = value
        return True

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

    def observe(
        self,
        balance: float,
        equity: float | None = None,
        mark_sequence: int | None = None,
    ) -> DrawdownAssessment:
        """Record a sample; ratchet the peaks and classify the tier.

        Args:
            balance: Realized balance (drives the balance leg).
            equity: Mark-to-market equity, or None when no trustworthy mark is
                available — the equity leg is then skipped, never guessed.
            mark_sequence: Identifies the price mark ``equity`` was built from;
                repeated reads of one mark do not count as confirmation of the
                equity peak. None treats every call as a new mark.
        """
        if not self._seeded:
            self.seed_peak(balance)
        if not math.isfinite(balance):
            # A garbage sample must neither move the peak nor trip the cap.
            logger.warning("Non-finite balance in drawdown check: %s — sample ignored", balance)
            tier = DrawdownTier.BREACH if self._tripped else DrawdownTier.NONE
            return DrawdownAssessment(0.0, self._peak, tier, self._tripped)

        if balance > self._peak:
            self._peak = balance
        balance_drawdown = (self._peak - balance) / self._peak if self._peak > 0 else 0.0

        valid_equity = equity if equity is not None and math.isfinite(equity) else None
        equity_drawdown = 0.0
        if valid_equity is None:
            # Without a trustworthy mark nothing may build toward a peak.
            self._prev_equity = self._cur_equity = self._equity_mark_sequence = None
        else:
            self._confirm_equity_peak(valid_equity, mark_sequence)
            peak_equity = self.equity_peak
            if peak_equity > 0:
                equity_drawdown = max((peak_equity - valid_equity) / peak_equity, 0.0)
        drawdown = max(balance_drawdown, equity_drawdown)

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

        self._log_tier(tier, drawdown, balance, balance_drawdown, equity_drawdown, valid_equity)
        return DrawdownAssessment(
            drawdown,
            self._peak,
            tier,
            self._tripped,
            balance_drawdown=balance_drawdown,
            equity_drawdown=equity_drawdown,
            equity_peak=self.equity_peak,
            equity=valid_equity,
        )

    def _confirm_equity_peak(self, equity: float, mark_sequence: int | None) -> None:
        """Ratchet the equity peak to the lower of the last two distinct marks."""
        if mark_sequence is None or mark_sequence != self._equity_mark_sequence:
            self._prev_equity, self._cur_equity = self._cur_equity, equity
            self._equity_mark_sequence = mark_sequence
        else:
            self._cur_equity = equity
        if self._prev_equity is not None and self._cur_equity is not None:
            self.raise_equity_peak(min(self._prev_equity, self._cur_equity))

    def _log_tier(
        self,
        tier: DrawdownTier,
        drawdown: float,
        balance: float,
        balance_drawdown: float,
        equity_drawdown: float,
        equity: float | None,
    ) -> None:
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

        equity_part = (
            f"equity ${equity:,.2f} {equity_drawdown * 100:.2f}% of peak ${self.equity_peak:,.2f}"
            if equity is not None
            else "equity leg not evaluated (no fresh mark)"
        )
        detail = (
            f"drawdown {drawdown * 100:.2f}% (balance ${balance:,.2f} "
            f"{balance_drawdown * 100:.2f}% of peak ${self._peak:,.2f}; {equity_part}; "
            f"hard cap {self.max_drawdown_pct * 100:.1f}%)"
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
    # Durable seeding lineage (#1036): set once at recovery, never cleared.
    # ``_recovered_inactive_session_id`` is deliberately NOT part of this
    # contract — the #668 carry-forward guard clears it before the first loop
    # iteration, and depending on it is what disarmed the seeders.
    _history_seed_session_id: int | None
    _history_seed_lookup_failed: bool
    db_manager: DatabaseManager
    _close_only_mode: bool

    def _enter_close_only_mode(self, reason: str | None = None) -> None: ...

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

    def __init__(
        self,
        engine_state: DrawdownEngineState,
        guard: MaxDrawdownGuard,
        *,
        mark_provider: Callable[[], MarkReading] | None = None,
    ) -> None:
        """Bind to the engine's live state (see protocol for the surface).

        Args:
            engine_state: Live engine state.
            guard: The drawdown state machine.
            mark_provider: Returns the open positions' unrealized P&L with a
                freshness verdict. ``None`` runs the balance leg only.
        """
        self._state = engine_state
        self._guard = guard
        self._mark_provider = mark_provider
        self._mark_blind_since: float | None = None
        self._mark_blind_alerted = False
        self._breach_notified = False
        self._seed_attempts = 0
        # Separate budget from _seed_attempts: the pre-order chokepoint runs the
        # upgrade too (so it never gates on a stale provisional peak), and several
        # chokepoint calls land per iteration. Sharing one counter would let them
        # burn the budget within seconds and forfeit the durable peak.
        self._upgrade_attempts = 0
        self._peak_seed_provenance = PEAK_SEED_SELF_ANCHORED
        self._history_upgrade_pending = False

    @property
    def guard(self) -> MaxDrawdownGuard:
        """The underlying drawdown guard (exposed for status/diagnostics)."""
        return self._guard

    @property
    def peak_seed_provenance(self) -> str:
        """Where the armed peak came from (see ``seed_lineage`` constants)."""
        return self._peak_seed_provenance

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

        Identical enforcement — including ratcheting a provisionally-armed peak
        up to the durable one, so the gate never admits an entry against a
        stale peak the loop check is about to trip on. Neither the deferred
        seeding attempt nor the upgrade attempt is counted against
        ``MAX_SEED_ATTEMPTS``: both budgets stay owned by the once-per-iteration
        loop check, so several chokepoint calls per iteration cannot burn
        through them prematurely.
        """
        self._run_check(count_seed_deferral=False)

    def _run_check(self, *, count_seed_deferral: bool) -> None:
        state = self._state
        try:
            balance = float(state.current_balance)
            was_seeded = self._guard.seeded
            if not was_seeded and not self._try_seed(balance, count_deferral=count_seed_deferral):
                return  # seeding deferred (DB/session not ready); retry next cycle
            # Only once the guard is already armed: re-reading the same DB state
            # inside the call that just armed provisionally cannot tell us
            # anything new. The pre-order chokepoint upgrades too — otherwise it
            # would admit an entry against the stale provisional peak that the
            # loop check trips on moments later (the very leak it exists to
            # close) — but without consuming the attempt budget.
            if was_seeded and self._history_upgrade_pending:
                # Armed provisionally from the current balance; keep trying to
                # ratchet up to the durable peak.
                self._upgrade_peak_from_history(count_attempt=count_seed_deferral)
            equity, mark_sequence, unrealized = self._read_equity(balance)
            assessment = self._guard.observe(balance, equity, mark_sequence)
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
            state._enter_close_only_mode(
                f"max-drawdown hard cap breached: drawdown "
                f"{assessment.drawdown * 100:.2f}% >= limit "
                f"{self._guard.max_drawdown_pct * 100:.1f}%"
            )
            self._breach_notified = True
            equity_part = (
                f"; equity leg {assessment.equity_drawdown * 100:.2f}% "
                f"(peak ${assessment.equity_peak:,.2f} → equity ${assessment.equity:,.2f}, "
                f"unrealized ${unrealized:,.2f})"
                if assessment.equity is not None and unrealized is not None
                else ""
            )
            message = (
                f"MAX DRAWDOWN HARD CAP BREACHED: drawdown {assessment.drawdown * 100:.2f}% "
                f">= limit {self._guard.max_drawdown_pct * 100:.1f}% "
                f"(balance leg {assessment.balance_drawdown * 100:.2f}%: peak "
                f"${assessment.peak_balance:,.2f} → balance ${balance:,.2f}{equity_part}). "
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
                balance_drawdown=assessment.balance_drawdown,
                equity_drawdown=assessment.equity_drawdown,
                equity=assessment.equity,
                peak_equity=assessment.equity_peak,
                unrealized_pnl=unrealized,
                max_drawdown_pct=self._guard.max_drawdown_pct,
                # Container stdout is ephemeral; without this a breach measured
                # from a provisional/unseeded peak is indistinguishable in
                # system_events from one measured off durable history (mirrors
                # the breaker's `peak_seed` on trip/dry-run payloads).
                peak_seed=self._peak_seed_provenance,
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

    def _read_equity(self, balance: float) -> tuple[float | None, int | None, float | None]:
        """Mark-to-market ``(equity, mark_sequence, unrealized)``, or Nones when blind.

        Fault-isolated: an unavailable or non-fresh mark degrades the check to
        the balance leg alone (exactly the pre-equity behaviour) with a WARNING
        on the transition, and pages once if the blindness outlasts
        ``MARK_BLIND_ESCALATE_SECONDS`` while a position is open.
        """
        provider = self._mark_provider
        if provider is None:
            return None, None, None
        reason: str | None = None
        reading: MarkReading | None = None
        try:
            reading = provider()
            if not reading.fresh:
                reason = "position marks are stale or missing"
            elif not math.isfinite(reading.unrealized_pnl):
                reason = f"non-finite unrealized P&L {reading.unrealized_pnl}"
        except Exception as e:  # noqa: BLE001 - the mark read must never crash the check
            reason = f"mark read failed: {e}"

        if reason is None and reading is not None:
            if self._mark_blind_since is not None:
                logger.info("Drawdown equity leg: fresh position marks restored")
            self._mark_blind_since = None
            self._mark_blind_alerted = False
            return balance + reading.unrealized_pnl, reading.sequence, reading.unrealized_pnl

        self._note_blind(reason or "unknown")
        return None, None, None

    def _note_blind(self, reason: str) -> None:
        """Log the equity-leg blind spot; page the operator once if it persists."""
        now = time.monotonic()
        if self._mark_blind_since is None:
            self._mark_blind_since = now
            logger.warning(
                "Drawdown equity leg blind (%s) — guarding on realized balance only "
                "until fresh marks return",
                reason,
            )
            return
        blind_for = now - self._mark_blind_since
        if self._mark_blind_alerted or blind_for < MARK_BLIND_ESCALATE_SECONDS:
            return
        self._mark_blind_alerted = True
        message = (
            f"Drawdown guard equity leg blind for {blind_for:.0f}s with open positions "
            f"({reason}): unrealized losses are NOT being measured; only the realized "
            "balance leg is active."
        )
        logger.critical("🚨 %s", message)
        try:
            self._state._record_event(
                EventType.ALERT,
                message,
                severity="critical",
                component="risk",
                error_code="DRAWDOWN_EQUITY_MARK_UNAVAILABLE",
                alert=True,
            )
        except Exception as e:  # noqa: BLE001
            logger.error("Failed to record equity-leg blind alert: %s", e, exc_info=True)

    def _read_equity_peak(
        self, session_id: int | None, lineage: HistoryLineage
    ) -> tuple[bool, float | None]:
        """Durable session equity max as ``(read_ok, peak)``; never raises."""
        try:
            peak = self._state.db_manager.get_session_peak_equity(
                session_id=session_id, fallback_session_id=lineage.fallback_session_id
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("Could not read session peak equity from account_history: %s", e)
            return False, None
        return True, peak

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
        lineage = resolve_history_lineage(state)
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
                    fallback_session_id=lineage.fallback_session_id,
                )
                # None here means the read SUCCEEDED but neither session has a
                # snapshot yet — legitimate for a genuinely fresh session, a
                # DEFECT (or the first-snapshot race) when history was expected.
                db_read_ok = True
            except Exception as e:
                logger.warning(
                    "Could not recompute session peak from account_history "
                    "(attempt %d/%d): %s — will retry next cycle",
                    self._seed_attempts,
                    MAX_SEED_ATTEMPTS,
                    e,
                )

        equity_read_ok, db_equity_peak = False, None
        if db_read_ok and session_id is not None:
            equity_read_ok, db_equity_peak = self._read_equity_peak(session_id, lineage)

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
            self._arm(balance, None, session_id, lineage, PEAK_SEED_UNAVAILABLE)
            return True

        if db_peak is None and lineage.history_expected:
            # The durable peak is expected but not readable yet (the documented
            # first-snapshot race, or a lost lineage). Arm NOW from the current
            # balance so the cap is never left unarmed, flag the provenance
            # honestly, and keep ratcheting toward the durable peak.
            self._history_upgrade_pending = True
            self._arm(balance, None, session_id, lineage, PEAK_SEED_UNAVAILABLE, db_equity_peak)
            return True

        if not equity_read_ok:
            # Balance leg seeds normally; keep retrying the equity peak so a lost
            # read cannot silently leave the equity leg on a lower baseline.
            self._history_upgrade_pending = True
        self._arm(
            balance,
            db_peak,
            session_id,
            lineage,
            PEAK_SEED_DB_SESSION_MAX if db_peak is not None else PEAK_SEED_SELF_ANCHORED,
            db_equity_peak,
        )
        return True

    def _arm(
        self,
        balance: float,
        db_peak: float | None,
        session_id: int | None,
        lineage: HistoryLineage,
        provenance: str,
        db_equity_peak: float | None = None,
    ) -> None:
        """Seed the guard's peaks and log where the baseline came from."""
        self._guard.seed_peak(balance, db_peak, equity_peak=db_equity_peak)
        self._peak_seed_provenance = provenance
        log = logger.warning if provenance == PEAK_SEED_UNAVAILABLE else logger.info
        log(
            "Max-drawdown guard armed: peak=$%.2f, equity peak=$%.2f, hard cap=%.1f%% "
            "(session %s, %s, account_history peak %s, peak_seed=%s)",
            self._guard.peak_balance,
            self._guard.equity_peak,
            self._guard.max_drawdown_pct * 100,
            session_id,
            lineage.describe,
            f"${db_peak:,.2f}" if db_peak is not None else "unavailable",
            provenance,
        )

    def _upgrade_peak_from_history(self, *, count_attempt: bool = True) -> None:
        """Retry the durable peak read after a provisional self-anchored arm.

        Bounded by ``MAX_SEED_ATTEMPTS`` against its OWN counter, consumed only
        by the loop check. The pre-order chokepoint calls this with
        ``count_attempt=False``: it must see the ratcheted peak, but it can run
        several times per iteration and would otherwise exhaust the budget in
        seconds. Success ratchets the peak UP (never down) and corrects the
        provenance; exhaustion stops the retries and leaves a WARNING naming
        the sessions that were searched, so a silently self-anchored cap is
        never mistaken for a seeded one.
        """
        state = self._state
        if count_attempt:
            self._upgrade_attempts += 1
        lineage = resolve_history_lineage(state)
        session_id = state.trading_session_id
        db_peak: float | None = None
        read_ok = False
        if state.db_manager is not None and session_id is not None:
            try:
                db_peak = state.db_manager.get_session_peak_balance(
                    session_id, fallback_session_id=lineage.fallback_session_id
                )
                read_ok = True
            except Exception as e:
                logger.warning("Max-drawdown guard peak upgrade read failed: %s", e)

        if read_ok and db_peak is not None:
            equity_read_ok, db_equity_peak = self._read_equity_peak(session_id, lineage)
            if equity_read_ok:
                self._guard.raise_equity_peak(db_equity_peak)
            if equity_read_ok or (count_attempt and self._upgrade_attempts >= MAX_SEED_ATTEMPTS):
                self._history_upgrade_pending = False
            self._peak_seed_provenance = PEAK_SEED_DB_SESSION_MAX
            raised = self._guard.raise_peak(db_peak)
            logger.info(
                "Max-drawdown guard peak seeded from account_history after a deferred "
                "read: $%.2f (%s) — peak now $%.2f%s",
                db_peak,
                lineage.describe,
                self._guard.peak_balance,
                "" if raised else " (already higher; unchanged)",
            )
            return

        if count_attempt and self._upgrade_attempts >= MAX_SEED_ATTEMPTS:
            self._history_upgrade_pending = False
            logger.warning(
                "Max-drawdown guard: durable account_history peak never became available "
                "after %d attempts (session %s, %s) — the cap stays anchored to the "
                "boot balance $%.2f (peak_seed=%s)",
                self._upgrade_attempts,
                session_id,
                lineage.describe,
                self._guard.peak_balance,
                PEAK_SEED_UNAVAILABLE,
            )
