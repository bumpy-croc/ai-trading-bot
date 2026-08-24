"""Re-announce operator-latched conditions until a human clears them (#1096).

A latched condition — close-only mode, the manual system halt, an entry pause
outside any macro-event window — blocks new entries until somebody acts. Before
this monitor, each announced itself exactly once, at minute zero. On 2026-08-20
production latched close-only, paged correctly, and then sat halted for four
days because nothing re-raised the condition and nothing recorded that it was
still true (#1094).

Two outputs, deliberately separated:

- **A durable heartbeat row** in ``system_events`` every
  ``HEARTBEAT_INTERVAL_SECONDS`` while the condition holds (``alert=False``).
  The close-only latch itself is an in-process bool that is never persisted, so
  these rows are what makes "the bot is currently unable to trade" observable
  from the database at all — the daily standup asserts on them (#1095).
- **A bounded operator page** (``alert=True``) on an escalating schedule:
  ``PAGE_SCHEDULE_SECONDS`` after the latch, then once per
  ``PAGE_STEADY_INTERVAL_SECONDS``. A condition latched for a month produces
  ~31 pages, not ~720 — three in the first day, one a day thereafter.

Every message carries the elapsed time and the recorded reason, because
"close-only for 3d 4h" reads differently from the minute-zero message.

Fault isolation: the whole check is wrapped, so a failing observability write
can never break the trading loop, and the webhook POST (up to 10s) is dispatched
on a short-lived daemon thread so re-announcement cannot stall the loop or run
under any lock (LESSONS §5.7 / no-webhook-under-lock rule).
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Protocol

from src.config.feature_flags import is_enabled
from src.database.models import EventType
from src.engines.live.system_halt import SystemHaltState

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

# One durable system_events row per hour while latched: cheap, and it gives any
# monitor a "still latched" signal with at most an hour of staleness.
HEARTBEAT_INTERVAL_SECONDS = 3600.0
# Escalating re-pages measured from the latch: 1h, 4h, 12h...
PAGE_SCHEDULE_SECONDS = (3600.0, 14400.0, 43200.0)
# ...then once a day for as long as it holds. Bound: <=3 re-pages in the first
# 24h and <=1 per 24h afterwards.
PAGE_STEADY_INTERVAL_SECONDS = 86400.0


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


@dataclass
class _Condition:
    """A latchable condition and the codes its announcements carry."""

    key: str
    component: str
    latched_code: str
    cleared_code: str
    label: str


@dataclass
class _LatchRecord:
    """Bookkeeping for one currently-latched condition (loop thread only)."""

    since: float
    reason: str | None
    next_heartbeat_at: float
    next_page_at: float
    pages_sent: int = 0


CONDITIONS: tuple[_Condition, ...] = (
    _Condition(
        key="close_only",
        component="risk",
        latched_code="CLOSE_ONLY_LATCHED",
        cleared_code="CLOSE_ONLY_LATCH_CLEARED",
        label="Close-only mode",
    ),
    _Condition(
        key="system_halt",
        component="ops",
        latched_code="SYSTEM_HALT_LATCHED",
        cleared_code="SYSTEM_HALT_LATCH_CLEARED",
        label="Manual system halt",
    ),
    _Condition(
        key="entry_pause",
        component="ops",
        latched_code="ENTRY_PAUSE_LATCHED",
        cleared_code="ENTRY_PAUSE_LATCH_CLEARED",
        label="FEATURE_ENTRY_PAUSE",
    ),
)


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


class LatchedConditionMonitor:
    """Re-announce blocking conditions on a bounded, escalating schedule."""

    def __init__(
        self,
        engine_state: LatchedConditionEngineState,
        halt_state: SystemHaltState | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
        dispatch_async: bool = True,
    ) -> None:
        """Bind engine state and the manual-halt mirror.

        ``clock`` is monotonic so a system-clock step cannot re-page a storm;
        elapsed times in messages are therefore uptime-relative, which is what
        "how long has this process been refusing entries" means. ``dispatch_async``
        exists so tests can assert synchronously; production keeps it True to
        keep the webhook off the trading-loop thread.
        """
        self._state = engine_state
        self._halt = halt_state
        self._clock = clock
        self._dispatch_async = dispatch_async
        self._latched: dict[str, _LatchRecord] = {}

    def check(self) -> None:
        """Evaluate every condition once. Never raises."""
        try:
            for condition in CONDITIONS:
                active, reason = self._evaluate(condition)
                if active:
                    self._on_active(condition, reason)
                else:
                    self._on_inactive(condition)
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("latched-condition monitor failed: %s", e)

    def latched_summary(self) -> dict[str, str]:
        """``{condition key: elapsed}`` for everything currently latched.

        Read-only helper for status surfaces (health endpoint, status log).
        """
        now = self._clock()
        return {
            key: format_elapsed(now - record.since) for key, record in sorted(self._latched.items())
        }

    def _evaluate(self, condition: _Condition) -> tuple[bool, str | None]:
        """Return ``(active, reason)`` for one condition."""
        if condition.key == "close_only":
            if getattr(self._state, "_close_only_mode", False) is not True:
                return (False, None)
            return (True, getattr(self._state, "_close_only_reason", None))
        if condition.key == "system_halt":
            if self._halt is None or self._halt.active is not True:
                return (False, None)
            return (True, self._halt.reason)
        # entry_pause: only a finding when no macro-event window explains it.
        if not is_enabled("entry_pause", default=False):
            return (False, None)
        window = self._active_macro_window()
        if window is not None:
            return (False, None)
        return (True, "env flag set with no active macro-event window")

    def _active_macro_window(self) -> str | None:
        """Name of the macro-event window covering now, if any.

        Fault-isolated: an unreadable calendar reports "no window", so a broken
        config makes the pause visible rather than silently excused.
        """
        try:
            from src.position_management.macro_events import MacroEventCalendar

            event = MacroEventCalendar.from_config().active_event(datetime.now(UTC))
            return None if event is None else getattr(event, "name", "macro event")
        except Exception as e:
            logger.debug("macro-event calendar unavailable: %s", e)
            return None

    def _on_active(self, condition: _Condition, reason: str | None) -> None:
        """Track a latch and emit its heartbeat / escalating page when due."""
        now = self._clock()
        record = self._latched.get(condition.key)
        if record is None:
            # First observation. The latch's own one-shot CRITICAL page has just
            # fired (or fired before this process saw it), so schedule the first
            # re-announcement rather than paging immediately.
            self._latched[condition.key] = _LatchRecord(
                since=now,
                reason=reason,
                next_heartbeat_at=now + HEARTBEAT_INTERVAL_SECONDS,
                next_page_at=now + PAGE_SCHEDULE_SECONDS[0],
            )
            return

        if reason and record.reason != reason:
            record.reason = reason

        page_due = now >= record.next_page_at
        if not page_due and now < record.next_heartbeat_at:
            return

        elapsed = format_elapsed(now - record.since)
        message = (
            f"{condition.label} still active after {elapsed} "
            f"(reason: {record.reason or 'not recorded — see the latch event at minute zero'}). "
            "New entries remain blocked until a human clears it; exits, stop-losses "
            "and reconciliation continue."
        )
        record.next_heartbeat_at = now + HEARTBEAT_INTERVAL_SECONDS
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

    def _on_inactive(self, condition: _Condition) -> None:
        """Record the resolution of a previously-latched condition."""
        record = self._latched.pop(condition.key, None)
        if record is None:
            return
        elapsed = format_elapsed(self._clock() - record.since)
        message = f"{condition.label} cleared after {elapsed} — new entries are enabled again."
        logger.info("✅ %s", message)
        # Durable row only: a resolution is what monitors query for, and the
        # clearing operator does not need to be paged about their own action.
        self._emit(
            EventType.WARNING,
            message,
            severity="warning",
            component=condition.component,
            error_code=condition.cleared_code,
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
            # Heartbeat rows are a single DB insert; no need for a thread.
            try:
                _write()
            except Exception as e:  # pragma: no cover - _record_event is fault-isolated
                logger.warning("latched-condition event failed: %s", e)
            return
        try:
            threading.Thread(target=_write, name="latched-condition-alert", daemon=True).start()
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("latched-condition alert dispatch failed: %s", e)
