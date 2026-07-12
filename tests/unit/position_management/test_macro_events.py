"""Unit tests for event-aware de-risking windows (#806)."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from src.engines.shared.execution.entry_handler_mixin import SharedEntryHandlerMixin
from src.position_management.macro_events import (
    DEFAULT_CALENDAR_PATH,
    MacroEvent,
    MacroEventCalendar,
    MacroEventGuard,
)

pytestmark = pytest.mark.unit


def _event(hours_before=12, hours_after=6):
    return MacroEvent(
        name="CPI",
        event_type="CPI",
        time=datetime(2026, 7, 14, 12, 30, tzinfo=UTC),
        hours_before=hours_before,
        hours_after=hours_after,
    )


# --- calendar / window boundaries ------------------------------------------


def test_window_boundaries_inclusive():
    event = _event(hours_before=12, hours_after=6)
    cal = MacroEventCalendar([event])
    # Just inside the leading edge (12h before) and trailing edge (6h after).
    assert cal.active_event(datetime(2026, 7, 14, 0, 30, tzinfo=UTC)) is not None  # exactly -12h
    assert cal.active_event(datetime(2026, 7, 14, 18, 30, tzinfo=UTC)) is not None  # exactly +6h
    # Just outside each edge.
    assert cal.active_event(datetime(2026, 7, 14, 0, 29, tzinfo=UTC)) is None
    assert cal.active_event(datetime(2026, 7, 14, 18, 31, tzinfo=UTC)) is None


def test_naive_datetime_treated_as_utc():
    cal = MacroEventCalendar([_event()])
    assert cal.active_event(datetime(2026, 7, 14, 12, 30)) is not None  # noqa: DTZ001


def test_empty_calendar_never_active():
    cal = MacroEventCalendar([])
    assert cal.active_event(datetime(2026, 7, 14, 12, 30, tzinfo=UTC)) is None


def test_from_config_loads_default_file():
    cal = MacroEventCalendar.from_config()
    assert len(cal) >= 1


def test_from_config_missing_file_is_empty(tmp_path):
    cal = MacroEventCalendar.from_config(tmp_path / "nope.json")
    assert len(cal) == 0


def test_from_config_applies_defaults_and_overrides(tmp_path):
    cfg = tmp_path / "events.json"
    cfg.write_text(
        json.dumps(
            {
                "default_hours_before": 10,
                "default_hours_after": 4,
                "events": [
                    {"name": "A", "type": "CPI", "time": "2026-07-14T12:30:00Z"},
                    {
                        "name": "B",
                        "type": "FOMC",
                        "time": "2026-06-17T18:00:00Z",
                        "hours_before": 18,
                        "hours_after": 18,
                    },
                ],
            }
        )
    )
    cal = MacroEventCalendar.from_config(cfg)
    assert len(cal) == 2
    # Event A uses defaults: 10h before -> window opens 02:30.
    assert cal.active_event(datetime(2026, 7, 14, 2, 30, tzinfo=UTC)) is not None
    assert cal.active_event(datetime(2026, 7, 14, 2, 29, tzinfo=UTC)) is None


# --- default config integrity (regression: #962, wrong FOMC date) ----------
#
# #962: config/macro_events.json listed "FOMC July 2026" as 2026-07-29 when the
# real meeting (and the one production actually scheduled entry-pause windows
# around) was 2026-07-08. Nothing caught the typo because the file parses fine
# and MacroEventCalendar.from_config() fails open (silently drops malformed
# entries, never raises) — a config typo produces no visible symptom short of
# an empirical audit of `strategy_executions.reasons`. These tests give the
# file itself a few load-bearing invariants.


def test_default_config_events_have_valid_iso8601_utc_times():
    raw = json.loads(DEFAULT_CALENDAR_PATH.read_text(encoding="utf-8"))
    assert raw["events"], "default macro event calendar is empty"
    for event in raw["events"]:
        time_str = event["time"]
        assert time_str.endswith(
            "Z"
        ), f"{event['name']!r} time is not UTC ('Z' suffix): {time_str!r}"
        parsed = datetime.fromisoformat(time_str[:-1] + "+00:00")
        assert parsed.tzinfo is not None


def test_default_config_no_events_silently_dropped():
    """`from_config` swallows malformed entries via try/except+log instead of raising,
    so a schema regression would otherwise shrink the calendar with no test failure."""
    raw = json.loads(DEFAULT_CALENDAR_PATH.read_text(encoding="utf-8"))
    cal = MacroEventCalendar.from_config()
    assert len(cal) == len(raw["events"])


def test_default_config_has_upcoming_coverage():
    """Canary: if nobody maintains this file, the guard silently stops de-risking
    anything (fail-safe-by-design, but that's exactly how #962 went unnoticed for
    an entire staging trial). Flag it loudly instead."""
    raw = json.loads(DEFAULT_CALENDAR_PATH.read_text(encoding="utf-8"))
    times = [
        (
            datetime.fromisoformat(e["time"][:-1] + "+00:00")
            if e["time"].endswith("Z")
            else datetime.fromisoformat(e["time"])
        )
        for e in raw["events"]
    ]
    most_recent = max(times)
    now = datetime.now(UTC)
    assert most_recent >= now - timedelta(days=14), (
        "macro_events.json's most recent listed event is over 14 days in the past — "
        "the calendar needs a maintenance pass to keep upcoming de-risk coverage (#962)."
    )


# --- guard -----------------------------------------------------------------


def test_guard_blocks_and_halves_inside_window():
    guard = MacroEventGuard(MacroEventCalendar([_event()]), enabled=True)
    inside = datetime(2026, 7, 14, 12, 0, tzinfo=UTC)
    assert guard.entry_allowed(inside) is False
    assert guard.exposure_factor(inside) == 0.5
    assert guard.active_event_name(inside) == "CPI"


def test_guard_inert_outside_window():
    guard = MacroEventGuard(MacroEventCalendar([_event()]), enabled=True)
    outside = datetime(2026, 1, 1, tzinfo=UTC)
    assert guard.entry_allowed(outside) is True
    assert guard.exposure_factor(outside) == 1.0


def test_guard_respects_feature_flag(monkeypatch):
    monkeypatch.delenv("FEATURE_ENABLE_MACRO_EVENT_GUARD", raising=False)
    assert MacroEventGuard(MacroEventCalendar([_event()])).enabled is False
    monkeypatch.setenv("FEATURE_ENABLE_MACRO_EVENT_GUARD", "true")
    assert MacroEventGuard(MacroEventCalendar([_event()])).enabled is True


# --- pre-order gate integration --------------------------------------------


class _Handler(SharedEntryHandlerMixin):
    def __init__(self, guard):
        self.configure_exposure_gate(None, None)
        self.configure_macro_guard(guard)


def test_pre_order_gate_blocks_entry_in_macro_window():
    guard = MacroEventGuard(MacroEventCalendar([_event()]), enabled=True)
    h = _Handler(guard)
    inside = datetime(2026, 7, 14, 12, 0, tzinfo=UTC)
    allowed, reason = h.apply_pre_order_gates(0.3, regime=None, equity=1000.0, now=inside)
    assert allowed == 0.0
    assert reason is not None and "macro_event_block" in reason


def test_pre_order_gate_passthrough_outside_window():
    guard = MacroEventGuard(MacroEventCalendar([_event()]), enabled=True)
    h = _Handler(guard)
    outside = datetime(2026, 1, 1, tzinfo=UTC)
    allowed, reason = h.apply_pre_order_gates(0.3, regime=None, equity=1000.0, now=outside)
    assert allowed == 0.3
    assert reason is None


def test_pre_order_gate_inert_when_guard_disabled():
    guard = MacroEventGuard(MacroEventCalendar([_event()]), enabled=False)
    h = _Handler(guard)
    inside = datetime(2026, 7, 14, 12, 0, tzinfo=UTC)
    allowed, _ = h.apply_pre_order_gates(0.3, regime=None, equity=1000.0, now=inside)
    assert allowed == 0.3
