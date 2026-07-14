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


# --- restart-safe peak seeding (#986 gap B) ---------------------------------


def test_seed_peak_preserves_drawdown_halt_across_restart():
    # Simulate a restart mid-drawdown: without seeding, the peak would
    # self-anchor to the depressed post-restart equity and the 15% halt would
    # silently lose its memory (the #845/#847 peak-reset class).
    b = _breaker(daily_loss_limit=0.99)  # isolate the drawdown halt
    b.seed_peak(1000.0)
    d = b.evaluate(840.0, D0)  # -16% vs the durable peak, 0% vs self-anchor
    assert d.entries_blocked
    assert "drawdown_halt" in d.reason


def test_seed_peak_never_lowers_existing_peak():
    b = _breaker(daily_loss_limit=0.99)
    b.evaluate(1000.0, D0)
    b.seed_peak(900.0)  # stale/lower candidate must not lower the live peak
    assert b.peak == 1000.0
    d = b.evaluate(920.0, D0 + timedelta(hours=1))  # -8% < 15%
    assert not d.tripped


def test_seed_peak_rejects_garbage_candidates():
    b = _breaker(daily_loss_limit=0.99)
    for garbage in (float("nan"), float("inf"), -100.0, 0.0):
        b.seed_peak(garbage)
    assert b.peak == 0.0


# --- degraded-basis latch freeze (#1032 fix round) ---------------------------
# While the equity feed is degraded the reading is CASH measured against
# EQUITY-basis anchors (baseline/peak) — a mixed basis must never move latch
# state in either direction: no new latches, no clears, no anchor mutations.


def test_frozen_evaluation_does_not_latch_daily_loss():
    b = _breaker()
    b.evaluate(1050.0, D0)  # healthy anchor: equity 1050 (winning open position)
    # Provider fault: reading collapses to cash 1000 → apparent -4.8% "loss".
    d = b.evaluate(1000.0, D0 + timedelta(hours=1), allow_transitions=False)
    assert not d.tripped
    # Healthy again near the anchor: the spurious latch must not have stuck.
    d2 = b.evaluate(1049.0, D0 + timedelta(hours=2))
    assert not d2.tripped


def test_frozen_evaluation_does_not_clear_drawdown_latch():
    b = _breaker(daily_loss_limit=0.99)
    b.evaluate(1000.0, D0)
    assert b.evaluate(840.0, D0 + timedelta(hours=1)).tripped  # -16% latches
    # Provider fault: cash reads at par with the peak → apparent full recovery.
    d = b.evaluate(1000.0, D0 + timedelta(hours=2), allow_transitions=False)
    assert d.tripped  # latch frozen: still reported while degraded
    assert "drawdown_halt" in d.reason
    # Healthy again, still in real drawdown: latch must never have cleared.
    d2 = b.evaluate(840.0, D0 + timedelta(hours=3))
    assert d2.tripped and "drawdown_halt" in d2.reason


def test_frozen_evaluation_does_not_ratchet_peak():
    b = _breaker(daily_loss_limit=0.99)
    b.evaluate(1000.0, D0)
    b.evaluate(1200.0, D0 + timedelta(hours=1), allow_transitions=False)
    assert b.peak == 1000.0  # a degraded high reading must not inflate the peak


def test_frozen_evaluation_does_not_roll_utc_day():
    b = _breaker()
    b.evaluate(1000.0, D0)
    assert b.evaluate(970.0, D0 + timedelta(hours=1)).tripped  # daily latch
    # Degraded across the UTC-day boundary: the day must not roll (rolling
    # would CLEAR the daily latch on a mixed-basis reading).
    d = b.evaluate(980.0, datetime(2026, 7, 5, 0, 30, tzinfo=UTC), allow_transitions=False)
    assert d.tripped
    # Healthy next-day reading rolls the day and clears normally.
    d2 = b.evaluate(980.0, datetime(2026, 7, 5, 1, 0, tzinfo=UTC))
    assert not d2.tripped


# --- pre-order gate integration --------------------------------------------


class _Handler(SharedEntryHandlerMixin):
    def __init__(self, breaker, unrealized_pnl_provider=None):
        self.configure_exposure_gate(None, None)
        self.configure_circuit_breaker(breaker, unrealized_pnl_provider=unrealized_pnl_provider)


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


# --- pre-order gate equity basis (#986 gap A) --------------------------------


def test_pre_order_gate_trips_on_open_position_loss_where_cash_would_not():
    # Cash balance is flat at 1000 while an open position is down 30 (-3%
    # true-equity daily loss): the breaker must see the mark-to-market loss.
    b = _breaker()
    b.evaluate(1000.0, D0)  # anchor baseline at true equity (flat)
    h = _Handler(b, unrealized_pnl_provider=lambda: -30.0)
    allowed, reason = h.apply_pre_order_gates(
        0.3, regime=None, equity=1000.0, now=D0 + timedelta(hours=1)
    )
    assert allowed == 0.0
    assert reason is not None and "circuit_breaker" in reason


def test_pre_order_gate_degrades_to_balance_when_unrealized_unavailable(caplog):
    # Fault isolation: an unreadable unrealized P&L must degrade the breaker to
    # balance-only with an explicit WARNING — never crash, never spurious-halt.
    def _boom():
        raise RuntimeError("mark price unavailable")

    b = _breaker()
    b.evaluate(1000.0, D0)
    h = _Handler(b, unrealized_pnl_provider=_boom)
    with caplog.at_level("WARNING"):
        allowed, reason = h.apply_pre_order_gates(
            0.3, regime=None, equity=1000.0, now=D0 + timedelta(hours=1)
        )
    assert allowed == 0.3
    assert reason is None
    assert any("balance-only" in r.message for r in caplog.records)


def test_pre_order_gate_degrades_on_non_finite_unrealized(caplog):
    b = _breaker()
    b.evaluate(1000.0, D0)
    h = _Handler(b, unrealized_pnl_provider=lambda: float("nan"))
    with caplog.at_level("WARNING"):
        allowed, reason = h.apply_pre_order_gates(
            0.3, regime=None, equity=1000.0, now=D0 + timedelta(hours=1)
        )
    assert allowed == 0.3
    assert reason is None
    assert any("balance-only" in r.message for r in caplog.records)


def test_pre_order_gate_degraded_reading_does_not_latch_daily_halt():
    # Winning open position (+50) anchors the day at equity 1050; a provider
    # fault then collapses the reading to cash 1000 — an apparent -4.8% "loss"
    # on a mixed basis that must NOT latch the daily halt for the rest of the
    # day. Once the feed recovers near the anchor, entries must still be open.
    class _Feed:
        def __init__(self):
            self.value = 50.0

        def __call__(self):
            if isinstance(self.value, BaseException):
                raise self.value
            return self.value

    feed = _Feed()
    b = _breaker()
    h = _Handler(b, unrealized_pnl_provider=feed)
    allowed, reason = h.apply_pre_order_gates(0.3, regime=None, equity=1000.0, now=D0)
    assert allowed == 0.3 and reason is None  # healthy anchor at 1050

    feed.value = RuntimeError("mark price unavailable")
    allowed, reason = h.apply_pre_order_gates(
        0.3, regime=None, equity=1000.0, now=D0 + timedelta(hours=1)
    )
    assert allowed == 0.3 and reason is None  # degraded: no spurious block

    feed.value = 40.0  # recovered: equity 1040, a real -0.95% vs the 1050 anchor
    allowed, reason = h.apply_pre_order_gates(
        0.3, regime=None, equity=1000.0, now=D0 + timedelta(hours=2)
    )
    assert allowed == 0.3 and reason is None  # no latch stuck from the fault
