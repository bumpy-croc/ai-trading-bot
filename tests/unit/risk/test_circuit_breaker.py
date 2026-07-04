"""Unit tests for account-level circuit breakers (#807)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from src.engines.shared.execution.entry_handler_mixin import SharedEntryHandlerMixin
from src.risk.circuit_breaker import AccountCircuitBreaker

pytestmark = pytest.mark.unit

D0 = datetime(2026, 7, 4, 9, 0, tzinfo=UTC)


def _breaker(mode="active", **kw):
    kw.setdefault("daily_loss_limit", 0.025)
    kw.setdefault("drawdown_halt", 0.15)
    kw.setdefault("drawdown_recovery", 0.05)
    return AccountCircuitBreaker(mode=mode, **kw)


# --- daily-loss halt -------------------------------------------------------


def test_daily_loss_halt_trips_and_blocks():
    b = _breaker()
    b.evaluate(1000.0, D0)  # anchor baseline
    d = b.evaluate(970.0, D0 + timedelta(hours=1))  # -3% > 2.5%
    assert d.tripped and d.entries_blocked
    assert "daily_loss_halt" in d.reason


def test_daily_loss_halt_latches_for_the_day():
    b = _breaker()
    b.evaluate(1000.0, D0)
    b.evaluate(970.0, D0 + timedelta(hours=1))
    # Recover within the same day -> still halted (latched).
    d = b.evaluate(1000.0, D0 + timedelta(hours=3))
    assert d.entries_blocked


def test_daily_baseline_resets_next_utc_day():
    b = _breaker()
    b.evaluate(1000.0, D0)
    b.evaluate(970.0, D0 + timedelta(hours=1))  # halted
    d = b.evaluate(1000.0, datetime(2026, 7, 5, 0, 30, tzinfo=UTC))  # new UTC day
    assert not d.entries_blocked


def test_small_daily_loss_does_not_trip():
    b = _breaker()
    b.evaluate(1000.0, D0)
    d = b.evaluate(985.0, D0 + timedelta(hours=1))  # -1.5% < 2.5%
    assert not d.tripped


# --- drawdown halt ---------------------------------------------------------


def test_drawdown_halt_trips_and_recovers():
    b = _breaker(daily_loss_limit=0.99)  # disable daily-loss so we isolate drawdown
    b.evaluate(1000.0, D0)  # peak 1000
    d = b.evaluate(840.0, D0 + timedelta(hours=1))  # -16% > 15%
    assert d.entries_blocked and "drawdown_halt" in d.reason
    # Still halted at -8% (above the 5% recovery threshold).
    assert b.evaluate(920.0, D0 + timedelta(hours=2)).entries_blocked
    # Recovered within 5% of peak -> clears.
    assert not b.evaluate(960.0, D0 + timedelta(hours=3)).entries_blocked


# --- modes -----------------------------------------------------------------


def test_dry_run_trips_but_does_not_block():
    b = _breaker(mode="dry_run")
    b.evaluate(1000.0, D0)
    d = b.evaluate(950.0, D0 + timedelta(hours=1))
    assert d.tripped
    assert not d.entries_blocked


def test_off_mode_is_inert():
    b = _breaker(mode="off")
    d = b.evaluate(500.0, D0)  # catastrophic but off
    assert not d.tripped and not d.entries_blocked


def test_mode_reads_string_flag(monkeypatch):
    monkeypatch.delenv("FEATURE_ACCOUNT_CIRCUIT_BREAKERS", raising=False)
    assert AccountCircuitBreaker().mode == "off"
    monkeypatch.setenv("FEATURE_ACCOUNT_CIRCUIT_BREAKERS", "dry_run")
    assert AccountCircuitBreaker().mode == "dry_run"
    monkeypatch.setenv("FEATURE_ACCOUNT_CIRCUIT_BREAKERS", "active")
    assert AccountCircuitBreaker().mode == "active"


def test_non_finite_equity_is_safe():
    b = _breaker()
    d = b.evaluate(float("nan"), D0)
    assert not d.tripped


def test_seed_daily_baseline_preserves_halt_across_restart():
    # Simulate a restart: seed the pre-restart baseline so an intraday loss that
    # already occurred still trips instead of re-anchoring to current equity.
    b = _breaker()
    b.seed_daily_baseline(1000.0, D0.date())
    d = b.evaluate(970.0, D0 + timedelta(hours=1))  # -3% vs seeded baseline
    assert d.entries_blocked


# --- pre-order gate integration --------------------------------------------


class _Handler(SharedEntryHandlerMixin):
    def __init__(self, breaker):
        self.configure_exposure_gate(None, None)
        self.configure_circuit_breaker(breaker)


def test_pre_order_gate_blocks_when_breaker_halts():
    b = _breaker()
    b.evaluate(1000.0, D0)
    h = _Handler(b)
    allowed, reason = h.apply_pre_order_gates(
        0.3, regime=None, equity=970.0, now=D0 + timedelta(hours=1)
    )
    assert allowed == 0.0
    assert reason is not None and "circuit_breaker" in reason


def test_pre_order_gate_passthrough_when_healthy():
    b = _breaker()
    h = _Handler(b)
    allowed, reason = h.apply_pre_order_gates(0.3, regime=None, equity=1000.0, now=D0)
    assert allowed == 0.3
    assert reason is None
