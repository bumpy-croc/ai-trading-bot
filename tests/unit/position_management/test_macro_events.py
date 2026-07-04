"""Unit tests for event-aware de-risking windows (#806)."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from src.engines.shared.execution.entry_handler_mixin import SharedEntryHandlerMixin
from src.position_management.macro_events import (
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
