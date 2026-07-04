"""Event-aware de-risking windows (#806).

Around high-impact macro events (FOMC, CPI) the bot de-risks: it blocks new
entries and halves regime exposure caps from ``hours_before`` before to
``hours_after`` after the event. The calendar is a maintained config file
(``config/macro_events.json``), not hardcoded — a stale/empty calendar is
fail-safe (no windows, no de-risking).

The guard plugs into the shared pre-order gate (`SharedEntryHandlerMixin.
apply_pre_order_gates`) alongside the #802 exposure governor, so it applies
identically in the backtest and live engines. Note (deviation from the plan):
the plan proposed extending `TimeExitPolicy`, but that class is exit-only;
entry-gating + an exposure factor live more cleanly in this dedicated guard.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from src.config.constants import (
    DEFAULT_MACRO_EVENT_EXPOSURE_FACTOR,
    DEFAULT_MACRO_EVENT_HOURS_AFTER,
    DEFAULT_MACRO_EVENT_HOURS_BEFORE,
)
from src.config.feature_flags import is_enabled
from src.infrastructure.runtime.paths import get_project_root

logger = logging.getLogger(__name__)

FEATURE_FLAG = "enable_macro_event_guard"
DEFAULT_CALENDAR_PATH = get_project_root() / "config" / "macro_events.json"


@dataclass(frozen=True)
class MacroEvent:
    """A single macro event with its de-risking window."""

    name: str
    event_type: str
    time: datetime
    hours_before: float
    hours_after: float

    @property
    def window_start(self) -> datetime:
        return self.time - timedelta(hours=self.hours_before)

    @property
    def window_end(self) -> datetime:
        return self.time + timedelta(hours=self.hours_after)

    def contains(self, now: datetime) -> bool:
        return self.window_start <= _as_utc(now) <= self.window_end


class MacroEventCalendar:
    """Loads macro events from config and answers window queries."""

    def __init__(self, events: list[MacroEvent]):
        # Sorted for deterministic "active event" selection.
        self._events = sorted(events, key=lambda e: e.time)

    @classmethod
    def from_config(cls, path: str | Path | None = None) -> MacroEventCalendar:
        """Load the calendar from JSON. Missing/malformed file -> empty (fail-safe)."""
        cfg_path = Path(path) if path is not None else DEFAULT_CALENDAR_PATH
        if not cfg_path.exists():
            logger.warning("Macro event calendar not found at %s; no de-risk windows", cfg_path)
            return cls([])
        try:
            data = json.loads(cfg_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.error("Macro event calendar unreadable (%s); no de-risk windows", exc)
            return cls([])

        default_before = float(data.get("default_hours_before", DEFAULT_MACRO_EVENT_HOURS_BEFORE))
        default_after = float(data.get("default_hours_after", DEFAULT_MACRO_EVENT_HOURS_AFTER))
        events: list[MacroEvent] = []
        for raw in data.get("events", []):
            try:
                events.append(
                    MacroEvent(
                        name=str(raw["name"]),
                        event_type=str(raw.get("type", "event")),
                        time=_parse_time(raw["time"]),
                        hours_before=float(raw.get("hours_before", default_before)),
                        hours_after=float(raw.get("hours_after", default_after)),
                    )
                )
            except (KeyError, ValueError, TypeError) as exc:
                logger.warning("Skipping malformed macro event %r: %s", raw, exc)
        return cls(events)

    def active_event(self, now: datetime) -> MacroEvent | None:
        """Return an event whose de-risk window contains ``now`` (or None)."""
        now = _as_utc(now)
        for event in self._events:
            if event.contains(now):
                return event
        return None

    def __len__(self) -> int:
        return len(self._events)


class MacroEventGuard:
    """Entry-block + exposure-halving around macro events for the pre-order gate.

    Args:
        calendar: The event calendar (defaults to the config file).
        exposure_factor_in_window: Multiplier applied to exposure caps inside a
            window (0.5 = halve).
        enabled: Force enabled/disabled, bypassing the feature flag (tests/wiring).
    """

    def __init__(
        self,
        calendar: MacroEventCalendar | None = None,
        *,
        exposure_factor_in_window: float = DEFAULT_MACRO_EVENT_EXPOSURE_FACTOR,
        enabled: bool | None = None,
    ) -> None:
        self._calendar = calendar if calendar is not None else MacroEventCalendar.from_config()
        self._exposure_factor = float(exposure_factor_in_window)
        self._enabled_override = enabled

    @property
    def enabled(self) -> bool:
        if self._enabled_override is not None:
            return self._enabled_override
        return is_enabled(FEATURE_FLAG, default=False)

    def entry_allowed(self, now: datetime) -> bool:
        """False while inside a macro de-risk window (blocks new entries)."""
        return self._calendar.active_event(now) is None

    def exposure_factor(self, now: datetime) -> float:
        """Cap multiplier for ``now`` (``exposure_factor_in_window`` inside a window, else 1.0)."""
        return self._exposure_factor if self._calendar.active_event(now) is not None else 1.0

    def active_event_name(self, now: datetime) -> str | None:
        event = self._calendar.active_event(now)
        return event.name if event is not None else None


def _parse_time(value: str) -> datetime:
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)


def _as_utc(dt: datetime) -> datetime:
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)


__all__ = ["MacroEvent", "MacroEventCalendar", "MacroEventGuard"]
