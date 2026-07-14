"""Account-level circuit breakers (#807).

Hard, account-level safety limits enforced independent of strategy logic:

- **Daily-loss halt**: if equity falls ``daily_loss_limit`` below a
  **UTC-day-anchored baseline**, halt new entries for the rest of the day.
- **Drawdown halt**: if peak-to-trough drawdown reaches ``drawdown_halt``, halt
  new entries until equity recovers to within ``drawdown_recovery`` of the peak.

These are the *hard* stops. Graduated drawdown throttling stays with dynamic
risk (this breaker does not double-count it — it only halts at the deeper
thresholds).

Mode is a **string** flag ``account_circuit_breakers`` ∈ ``off`` / ``dry_run`` /
``active`` (read via ``get_flag`` — ``is_enabled`` would collapse ``"dry_run"``
to False and silently disarm it): ``off`` = inert; ``dry_run`` = evaluate + log
"would halt/flatten" but take no action; ``active`` = block new entries (and the
engine flattens via the RLock'd exit path — a money-mover, wired separately).

Persistence note: the daily baseline is anchored to the equity at the first
observation of each UTC day. In a long-lived process this is correct; on a
mid-day restart the in-memory baseline re-anchors to current equity, which would
disarm the daily-loss halt for the rest of that day. Callers that persist a
daily baseline (DB) can seed it via ``seed_daily_baseline`` on boot to preserve
the halt across restarts (full DB persistence is a follow-up, see the PR).
The drawdown peak has the same restart fragility: it self-anchors to the first
post-restart equity sample unless seeded from durable history via ``seed_peak``
(the ``CircuitBreakerEnforcer`` seeds it from the ``account_history`` session
equity max — the #845/#847 peak-reset class otherwise silently zeroes the 15%
halt's memory on every deploy).

Measurement basis: ``evaluate`` expects TRUE EQUITY (cash balance plus
mark-to-market unrealized P&L of open positions). Cash-only input is blind to
an open position's adverse move — the exact loss these halts exist to catch —
so callers feed equity and degrade to balance-only explicitly (with a WARNING)
only when the unrealized read is unavailable.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import date, datetime

from src.config.constants import (
    DEFAULT_CIRCUIT_DRAWDOWN_HALT,
    DEFAULT_CIRCUIT_DRAWDOWN_RECOVERY,
    DEFAULT_DAILY_LOSS_LIMIT,
)
from src.config.feature_flags import get_flag

logger = logging.getLogger(__name__)

FEATURE_FLAG = "account_circuit_breakers"

MODE_OFF = "off"
MODE_DRY_RUN = "dry_run"
MODE_ACTIVE = "active"
_VALID_MODES = frozenset({MODE_OFF, MODE_DRY_RUN, MODE_ACTIVE})


def _normalize_mode(value: object) -> str:
    """Coerce any flag value to a valid mode string (unknown -> ``off``)."""
    return value if value in _VALID_MODES else MODE_OFF  # type: ignore[return-value]


@dataclass
class BreakerDecision:
    """Result of a circuit-breaker evaluation."""

    entries_blocked: bool  # True when new entries must be blocked (active mode only)
    tripped: bool  # True when a limit is breached (regardless of mode)
    reason: str | None
    mode: str


class AccountCircuitBreaker:
    """Daily-loss + drawdown hard halts with a UTC-day-anchored baseline.

    Args:
        daily_loss_limit: Fractional daily loss (vs the day's baseline) that halts.
        drawdown_halt: Peak-to-trough fraction that halts.
        drawdown_recovery: Resume once within this fraction of the peak.
        mode: Force a mode (bypass the flag) for tests/wiring; ``None`` reads the
            ``account_circuit_breakers`` string flag.
    """

    def __init__(
        self,
        *,
        daily_loss_limit: float = DEFAULT_DAILY_LOSS_LIMIT,
        drawdown_halt: float = DEFAULT_CIRCUIT_DRAWDOWN_HALT,
        drawdown_recovery: float = DEFAULT_CIRCUIT_DRAWDOWN_RECOVERY,
        mode: str | bool | None = None,
    ) -> None:
        self.daily_loss_limit = float(daily_loss_limit)
        self.drawdown_halt = float(drawdown_halt)
        self.drawdown_recovery = float(drawdown_recovery)
        self._mode_override = _normalize_mode(mode) if mode is not None else None
        # State
        self._day: date | None = None
        self._daily_baseline: float | None = None
        self._daily_halt_latched = False  # daily-loss halt latches for the rest of the day
        self._drawdown_halted = False  # drawdown halt clears on recovery
        self._peak: float = 0.0

    @property
    def mode(self) -> str:
        if self._mode_override is not None:
            return self._mode_override
        return _normalize_mode(get_flag(FEATURE_FLAG, default=MODE_OFF))

    @property
    def peak(self) -> float:
        """Current peak equity the drawdown halt measures from."""
        return self._peak

    @property
    def daily_baseline(self) -> float | None:
        """Equity baseline the daily-loss halt measures from (None until anchored)."""
        return self._daily_baseline

    def seed_daily_baseline(self, baseline: float, day: date) -> None:
        """Seed a persisted daily baseline on boot (restart-safety hook)."""
        if math.isfinite(baseline) and baseline > 0:
            self._daily_baseline = float(baseline)
            self._day = day

    def seed_peak(self, peak: float) -> None:
        """Seed the drawdown peak from durable history on boot (restart-safety hook).

        Only ever RAISES the peak — a stale or lower candidate can never erase
        drawdown the breaker has already observed in this process.
        """
        if math.isfinite(peak) and peak > self._peak:
            self._peak = float(peak)

    def _roll_day(self, equity: float, today: date) -> None:
        """Re-anchor the daily baseline and clear the daily latch on a new UTC day."""
        if self._day != today:
            self._day = today
            self._daily_baseline = equity
            self._daily_halt_latched = False

    def evaluate(self, equity: float, now: datetime) -> BreakerDecision:
        """Update state from the latest equity and return the halt decision.

        Args:
            equity: Current account equity (balance + unrealized), in quote currency.
            now: Current time (UTC-aware; naive treated as UTC).

        Returns:
            A :class:`BreakerDecision`. ``entries_blocked`` is only True in
            ``active`` mode; in ``dry_run`` the trip is logged but not enforced.
        """
        mode = self.mode
        if mode == MODE_OFF or not math.isfinite(equity) or equity <= 0:
            return BreakerDecision(False, False, None, mode)

        today = now.date()  # naive datetimes are treated as UTC by contract
        self._roll_day(equity, today)
        self._peak = max(self._peak, equity)

        reason: str | None = None

        # Daily-loss halt (latches for the day once tripped).
        baseline = self._daily_baseline or equity
        daily_loss = (baseline - equity) / baseline if baseline > 0 else 0.0
        if daily_loss >= self.daily_loss_limit:
            self._daily_halt_latched = True
        if self._daily_halt_latched:
            reason = f"daily_loss_halt_{daily_loss:.3f}"

        # Drawdown halt (clears on recovery toward the peak).
        drawdown = (self._peak - equity) / self._peak if self._peak > 0 else 0.0
        if drawdown >= self.drawdown_halt:
            self._drawdown_halted = True
        elif self._drawdown_halted and drawdown <= self.drawdown_recovery:
            self._drawdown_halted = False
        if self._drawdown_halted:
            reason = reason or f"drawdown_halt_{drawdown:.3f}"

        tripped = reason is not None
        if tripped:
            log = logger.error if mode == MODE_ACTIVE else logger.warning
            log(
                "Account circuit breaker %s (mode=%s): %s (equity=%.2f, baseline=%.2f, peak=%.2f)",
                "HALT" if mode == MODE_ACTIVE else "would halt",
                mode,
                reason,
                equity,
                baseline,
                self._peak,
            )
        return BreakerDecision(
            entries_blocked=tripped and mode == MODE_ACTIVE,
            tripped=tripped,
            reason=reason,
            mode=mode,
        )
