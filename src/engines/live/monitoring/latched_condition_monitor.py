"""Re-announce operator-latched conditions until a human clears them (#1096).

A latched condition — close-only mode, the manual system halt (including its
fail-closed "unverified" state), an entry pause outside a macro-event window —
blocks new entries until somebody acts. Before this monitor, each announced
itself exactly once, at minute zero. On 2026-08-20 production latched
close-only, paged correctly, and then sat halted for four days because nothing
re-raised the condition and nothing recorded that it was still true (#1094).

Every pass writes exactly one **state row** per active condition, or a single
``ENTRIES_ENABLED`` row when nothing is blocking:

- The rows are a **positive assertion**, not just an alarm. ``ENTRIES_ENABLED``
  means the loop ran and found the bot able to trade; ``*_LATCHED`` means it ran
  and found it blocked. The *absence* of any row therefore means the loop thread
  itself is dead — a third condition a monitor must catch — so one query answers
  all three cases without an external cross-check (#1096 review).
- The close-only latch is an in-process bool that is never persisted, so these
  rows are what makes "the bot is currently unable to trade" observable from the
  database at all (#1095).

On top of the rows, a **bounded operator page** (``alert=True``) fires on an
escalating schedule while a condition holds: ``PAGE_SCHEDULE_SECONDS`` after the
latch, then once per ``PAGE_STEADY_INTERVAL_SECONDS``. A condition latched for a
month produces ~31 pages, not ~720 — three in the first day, one a day after.
Messages carry the elapsed time and the recorded reason, and pages past the
first day are prefixed ``STILL BLOCKED — DAY N`` so day four does not read
identically to day one.

"Are entries blocked?" is answered by ``EntryPauseGate.entry_block()`` — the
same authority the entry and scale-in paths gate on — so the monitor cannot
report a state the enforcement path disagrees with.

Fault isolation: each condition is evaluated inside its own try/except (one
failure must not blind the others), and the webhook POST (up to 10s) is
dispatched on a short-lived daemon thread so re-announcement cannot stall the
trading loop or run under any lock (LESSONS: no webhook under locks).
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Protocol

from src.config.feature_flags import is_enabled
from src.database.models import EventType
from src.engines.live.execution.entry_pause import EntryPauseGate

if TYPE_CHECKING:
    from collections.abc import Callable

    from src.engines.live.system_halt import SystemHaltState

logger = logging.getLogger(__name__)

# One state row per hour: cheap, and it bounds how stale a monitor's read of
# "can the bot trade right now?" can be.
HEARTBEAT_INTERVAL_SECONDS = 3600.0
# Escalating re-pages measured from the latch: 1h, 4h, 12h...
PAGE_SCHEDULE_SECONDS = (3600.0, 14400.0, 43200.0)
# ...then once a day for as long as it holds. Bound: <=3 re-pages in the first
# 24h and <=1 per 24h afterwards.
PAGE_STEADY_INTERVAL_SECONDS = 86400.0
# Row written when nothing is blocking entries — the positive assertion.
ENTRIES_ENABLED_CODE = "ENTRIES_ENABLED"


class LatchedConditionEngineState(Protocol):
    """The slice of live-engine state this monitor reads and announces through."""

    _close_only_mode: bool
    _close_only_reason: str | None

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


@dataclass(frozen=True)
class Observation:
    """One condition's current state."""

    active: bool
    reason: str | None = None
    # Durable start time where one exists (e.g. the halt flag's updated_at), so
    # elapsed survives the restarts an in-process counter would reset.
    started_at: datetime | None = None


@dataclass
class _Condition:
    """A latchable condition, its announcement codes, and how to observe it."""

    key: str
    component: str
    latched_code: str
    cleared_code: str
    label: str
    predicate: Callable[[LatchedConditionMonitor], Observation]


@dataclass
class _LatchRecord:
    """Bookkeeping for one currently-latched condition (loop thread only)."""

    since: float
    reason: str | None
    next_page_at: float
    started_at: datetime | None = None
    pages_sent: int = 0


def format_elapsed(seconds: float) -> str:
    """Human-readable elapsed time, e.g. ``3d 4h``, ``5h 12m``, ``45m``."""
    total = max(int(seconds), 0)
    days, rem = divmod(total, 86400)
    hours, rem = divmod(rem, 3600)
    minutes = rem // 60
    if days:
        return f"{days}d {hours}h"
    if hours:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"


CONDITIONS: tuple[_Condition, ...] = (
    _Condition(
        key="close_only",
        component="risk",
        latched_code="CLOSE_ONLY_LATCHED",
        cleared_code="CLOSE_ONLY_LATCH_CLEARED",
        label="Close-only mode",
        predicate=lambda monitor: monitor.observe_close_only(),
    ),
    _Condition(
        key="system_halt",
        component="ops",
        latched_code="SYSTEM_HALT_LATCHED",
        cleared_code="SYSTEM_HALT_LATCH_CLEARED",
        label="Manual system halt",
        predicate=lambda monitor: monitor.observe_entry_gate("system_halt"),
    ),
    _Condition(
        key="entry_pause",
        component="ops",
        latched_code="ENTRY_PAUSE_LATCHED",
        cleared_code="ENTRY_PAUSE_LATCH_CLEARED",
        label="FEATURE_ENTRY_PAUSE",
        predicate=lambda monitor: monitor.observe_entry_gate("entry_pause"),
    ),
)


@dataclass
class _State:
    """Monitor bookkeeping, all owned by the trading-loop thread."""

    latched: dict[str, _LatchRecord] = field(default_factory=dict)
    next_row_at: float = 0.0


class LatchedConditionMonitor:
    """Assert the entry state every pass; re-announce blocks until cleared."""

    def __init__(
        self,
        engine_state: LatchedConditionEngineState,
        halt_state: SystemHaltState | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
        dispatch_async: bool = True,
    ) -> None:
        """Bind engine state and build a gate over the shared halt mirror.

        ``clock`` is monotonic so a system-clock step cannot trigger a page
        storm; where a condition carries a durable start time (the halt flag's
        ``updated_at``) the reported elapsed uses that instead, so a restart
        cannot re-announce a four-day halt as "45m". ``dispatch_async`` exists
        so tests can assert synchronously; production keeps it True to keep the
        webhook off the trading-loop thread.
        """
        self._state = engine_state
        self._halt = halt_state
        self._gate = EntryPauseGate(halt_state)
        self._clock = clock
        self._dispatch_async = dispatch_async
        self._bookkeeping = _State()

    def check(self) -> None:
        """Evaluate every condition once and assert the entry state. Never raises."""
        blocked = False
        now = self._clock()
        row_due = now >= self._bookkeeping.next_row_at
        for condition in CONDITIONS:
            # Per-condition isolation: a failure evaluating one must not blind
            # the monitor to the others in the same pass (#1096 review).
            try:
                observation = condition.predicate(self)
                if observation.active:
                    blocked = True
                    self._on_active(condition, observation, now, row_due)
                else:
                    self._on_inactive(condition, now)
            except Exception as e:
                logger.warning("latched-condition check failed for %s: %s", condition.key, e)
        try:
            if row_due:
                self._bookkeeping.next_row_at = now + HEARTBEAT_INTERVAL_SECONDS
                if not blocked:
                    self._assert_entries_enabled()
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("entries-enabled assertion failed: %s", e)

    def latched_summary(self) -> dict[str, str]:
        """``{condition key: elapsed}`` for everything currently latched.

        Read-only helper for status surfaces (health endpoint, status log).
        """
        now = self._clock()
        return {
            key: format_elapsed(self._elapsed(record, now))
            for key, record in sorted(self._bookkeeping.latched.items())
        }

    # ---- observations -----------------------------------------------------

    def observe_close_only(self) -> Observation:
        """The engine's in-process close-only latch."""
        if getattr(self._state, "_close_only_mode", False) is not True:
            return Observation(False)
        return Observation(True, getattr(self._state, "_close_only_reason", None))

    def observe_entry_gate(self, key: str) -> Observation:
        """Read one lever from the gate the entry path itself consults.

        Deliberately NOT a re-derivation: ``EntryPauseGate.entry_block()`` is the
        authority, so the monitor reports exactly what blocks entries, including
        the fail-closed "halt state unverified" case that a naive ``active`` read
        misses. The gate reports one cause at a time (its own precedence), which
        is the effective block.
        """
        block = self._gate.entry_block()
        if block is None or block.key != key:
            return Observation(False)
        if key == "entry_pause" and self._macro_window_excuses_the_pause():
            return Observation(False)
        started_at = self._halt.since if (key == "system_halt" and self._halt) else None
        return Observation(True, block.reason, started_at)

    def _macro_window_excuses_the_pause(self) -> bool:
        """True when a live macro-event window explains an entry pause.

        Only when the macro-event guard is ENABLED: with the flag off no window
        suppresses entries, so a calendar entry must not suppress the finding
        either — the monitor and the guard have to agree about when a window
        means anything (#1096 review). Fault-isolated: an unreadable calendar
        excuses nothing, so a broken config makes the pause visible.
        """
        try:
            if not is_enabled("enable_macro_event_guard", default=False):
                return False
            from src.position_management.macro_events import MacroEventCalendar

            return MacroEventCalendar.from_config().active_event(datetime.now(UTC)) is not None
        except Exception as e:
            logger.debug("macro-event calendar unavailable: %s", e)
            return False

    # ---- announcements ----------------------------------------------------

    def _elapsed(self, record: _LatchRecord, now: float) -> float:
        """Seconds the condition has held, preferring a durable start time."""
        if record.started_at is not None:
            try:
                started = record.started_at
                if started.tzinfo is None:
                    started = started.replace(tzinfo=UTC)
                return (datetime.now(UTC) - started).total_seconds()
            except Exception:  # pragma: no cover - defensive
                pass
        return now - record.since

    def _on_active(
        self, condition: _Condition, observation: Observation, now: float, row_due: bool
    ) -> None:
        """Track the latch and write its state row / escalating page when due."""
        record = self._bookkeeping.latched.get(condition.key)
        if record is None:
            # First observation. Write the row NOW rather than an hour out: the
            # halt flag and the entry-pause env var both survive restarts, so an
            # engine restarting more often than the heartbeat interval would
            # otherwise stay blocked while emitting nothing (#1096 review).
            record = _LatchRecord(
                since=now,
                reason=observation.reason,
                next_page_at=now + PAGE_SCHEDULE_SECONDS[0],
                started_at=observation.started_at,
            )
            self._bookkeeping.latched[condition.key] = record
            row_due = True
        else:
            if observation.reason and record.reason != observation.reason:
                record.reason = observation.reason
            if observation.started_at is not None:
                record.started_at = observation.started_at

        page_due = now >= record.next_page_at
        if not (page_due or row_due):
            return

        elapsed_seconds = self._elapsed(record, now)
        elapsed = format_elapsed(elapsed_seconds)
        days = int(elapsed_seconds // 86400)
        # A day-4 page that reads exactly like the day-0 one is easy to scroll
        # past twice — the failure #1096 is about.
        prefix = f"STILL BLOCKED — DAY {days}: " if days >= 1 else ""
        message = (
            f"{prefix}{condition.label} still active after {elapsed} "
            f"(reason: {record.reason or 'not recorded — see the latch event at minute zero'}). "
            "New entries remain blocked until a human clears it; exits, stop-losses "
            "and reconciliation continue."
        )
        if page_due:
            record.pages_sent += 1
            record.next_page_at = now + self._next_page_delay(record.pages_sent)
            logger.critical("🚨 %s", message)
        self._emit(
            EventType.ALERT if page_due else EventType.WARNING,
            message,
            severity="critical" if page_due else "warning",
            component=condition.component,
            error_code=condition.latched_code,
            alert=page_due,
        )

    def _next_page_delay(self, pages_sent: int) -> float:
        """Delay until the next page after ``pages_sent`` re-pages.

        Walks ``PAGE_SCHEDULE_SECONDS`` (measured from the latch) then settles at
        ``PAGE_STEADY_INTERVAL_SECONDS`` — the bound on re-alerting.
        """
        if pages_sent < len(PAGE_SCHEDULE_SECONDS):
            return PAGE_SCHEDULE_SECONDS[pages_sent] - PAGE_SCHEDULE_SECONDS[pages_sent - 1]
        return PAGE_STEADY_INTERVAL_SECONDS

    def _on_inactive(self, condition: _Condition, now: float) -> None:
        """Record the resolution of a previously-latched condition."""
        record = self._bookkeeping.latched.pop(condition.key, None)
        if record is None:
            return
        elapsed = format_elapsed(self._elapsed(record, now))
        message = f"{condition.label} cleared after {elapsed} — new entries are enabled again."
        logger.info("✅ %s", message)
        # Durable row only: a resolution is what monitors query for, and the
        # clearing operator does not need to be paged about their own action.
        self._emit(
            EventType.WARNING,
            message,
            severity="info",
            component=condition.component,
            error_code=condition.cleared_code,
            alert=False,
        )

    def _assert_entries_enabled(self) -> None:
        """Write the positive assertion: the loop ran and entries are allowed."""
        self._emit(
            EventType.WARNING,
            "Entries enabled — no close-only latch, manual halt or entry pause in force.",
            severity="info",
            component="engine",
            error_code=ENTRIES_ENABLED_CODE,
            alert=False,
        )

    def _emit(
        self,
        event_type: EventType,
        message: str,
        *,
        severity: str,
        component: str,
        error_code: str,
        alert: bool,
    ) -> None:
        """Write the event, keeping any webhook off the trading-loop thread."""

        def _write() -> None:
            self._state._record_event(
                event_type,
                message,
                severity=severity,
                component=component,
                error_code=error_code,
                alert=alert,
            )

        if not alert or not self._dispatch_async:
            # State rows are a single DB insert; no need for a thread.
            try:
                _write()
            except Exception as e:  # pragma: no cover - _record_event is fault-isolated
                logger.warning("latched-condition event failed: %s", e)
            return
        try:
            threading.Thread(target=_write, name="latched-condition-alert", daemon=True).start()
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("latched-condition alert dispatch failed: %s", e)
