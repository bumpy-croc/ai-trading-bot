"""Durable peak source for the dynamic-risk throttle (#845 peak-reset class).

The graduated drawdown throttle read the in-memory PerformanceTracker peak,
which re-anchors to the (possibly depressed) current balance on restart —
silently disarming the 5–20% size reductions exactly when the account is in
drawdown. The max-drawdown hard cap was already hardened to seed from the
durable ``account_history`` session peak; ``_durable_peak_balance`` pins the
throttle to that same baseline so both controls measure drawdown alike.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

pytestmark = pytest.mark.fast


def _tiered_manager():
    """Real DynamicRiskManager with a single 10%-drawdown → ×0.5 tier."""
    from src.position_management.dynamic_risk import DynamicRiskConfig, DynamicRiskManager

    return DynamicRiskManager(
        config=DynamicRiskConfig(
            enabled=True,
            drawdown_thresholds=[0.10],
            risk_reduction_factors=[0.5],
        )
    )


def _enable_dynamic_risk(engine) -> None:
    manager = _tiered_manager()
    engine.dynamic_risk_manager = manager
    engine._dynamic_risk_handler.set_manager(manager)


def test_durable_peak_prefers_guard_seeded_db_peak(make_live_engine):
    """Restart mid-drawdown: the tracker re-anchors to the depressed balance
    but the durable peak keeps the account_history session max."""
    engine = make_live_engine(db_peak=100.0, resumed_balance=85.0)
    engine._check_max_drawdown()  # loop check arms the guard from the DB peak

    assert engine.performance_tracker.peak_balance == pytest.approx(85.0)
    assert engine._durable_peak_balance() == pytest.approx(100.0)


def test_durable_peak_degrades_to_tracker_before_guard_seeds(make_live_engine):
    """Until the guard arms (first loop iteration at boot) the in-memory peak
    is the best available baseline — matching pre-hardening behavior."""
    engine = make_live_engine(db_peak=100.0, resumed_balance=85.0)

    assert engine._durable_peak_balance() == pytest.approx(85.0)


def test_durable_peak_never_below_current_balance(make_live_engine):
    engine = make_live_engine(db_peak=100.0, resumed_balance=85.0)
    engine._check_max_drawdown()
    engine.current_balance = 120.0  # new equity high not yet observed

    assert engine._durable_peak_balance() == pytest.approx(120.0)


def test_throttle_applies_depressed_tier_after_restart(make_live_engine):
    """Restart 15% below the durable peak: the 10%-tier multiplier must apply
    even though the in-memory tracker reads ~0 drawdown."""
    engine = make_live_engine(db_peak=100.0, resumed_balance=85.0)
    _enable_dynamic_risk(engine)
    engine._check_max_drawdown()  # arm the guard from the DB peak

    adjusted = engine._apply_dynamic_risk_adjustment(0.2, datetime.now(UTC))

    assert adjusted == pytest.approx(0.1)  # 15% drawdown ≥ 0.10 tier → ×0.5


def test_throttle_stays_full_size_when_at_the_durable_peak(make_live_engine):
    """No false throttling: at the peak the multiplier stays 1.0."""
    engine = make_live_engine(db_peak=100.0, resumed_balance=100.0)
    _enable_dynamic_risk(engine)
    engine._check_max_drawdown()

    adjusted = engine._apply_dynamic_risk_adjustment(0.2, datetime.now(UTC))

    assert adjusted == pytest.approx(0.2)


def test_risk_param_adjustment_uses_the_durable_peak(make_live_engine):
    """_get_dynamic_risk_adjusted_params (stop widening, daily-risk scaling)
    must measure drawdown from the same durable baseline."""
    engine = make_live_engine(db_peak=100.0, resumed_balance=85.0)
    _enable_dynamic_risk(engine)
    engine._check_max_drawdown()

    adjusted = engine._get_dynamic_risk_adjusted_params()

    assert adjusted.base_risk_per_trade == pytest.approx(
        engine.risk_manager.params.base_risk_per_trade * 0.5
    )
