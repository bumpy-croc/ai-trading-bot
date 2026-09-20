"""Restart recovery must restore trailing state, remaining exposure and MFE/MAE peaks (#742, #993)."""

from __future__ import annotations

import math
from datetime import UTC, datetime
from unittest.mock import Mock

import pytest

from src.engines.live.trading_engine import LiveTradingEngine
from src.position_management.mfe_mae_tracker import MFEMAETracker, MFEMetrics
from src.risk.risk_manager import RiskManager


def _make_engine() -> LiveTradingEngine:
    strategy = Mock()
    strategy.get_risk_overrides.return_value = None
    data_provider = Mock()
    data_provider.get_current_price.return_value = 100.0
    return LiveTradingEngine(
        strategy=strategy,
        data_provider=data_provider,
        initial_balance=1_000.0,
        enable_live_trading=False,
        log_trades=False,
    )


def _row(**overrides) -> dict:
    row = {
        "id": 7,
        "symbol": "ETHUSDT",
        "side": "LONG",
        "size": 0.25,
        "entry_price": 100.0,
        "entry_time": datetime(2025, 1, 1, tzinfo=UTC),
        "entry_order_id": "order-7",
        "quantity": 2.5,
        "entry_balance": 1000.0,
        "original_size": 0.25,
        "current_size": 0.10,
        "trailing_stop_activated": True,
        "trailing_stop_price": 104.0,
        "breakeven_triggered": True,
        "mfe": 0.08,
        "mae": -0.03,
        "mfe_price": 108.0,
        "mae_price": 97.0,
        "mfe_time": datetime(2025, 1, 2, tzinfo=UTC),
        "mae_time": datetime(2025, 1, 3, tzinfo=UTC),
    }
    row.update(overrides)
    return row


def _recover(engine: LiveTradingEngine, row: dict) -> None:
    engine.trading_session_id = 1
    engine.db_manager.get_active_positions = lambda *a, **k: [row]
    engine.db_manager.heal_positions_with_terminal_trades = lambda *a, **k: 0
    engine._recover_active_positions()


def test_recovery_restores_trailing_and_breakeven_state():
    engine = _make_engine()
    _recover(engine, _row())

    pos = next(iter(engine.live_position_tracker.positions.values()))
    assert pos.trailing_stop_activated is True
    assert pos.trailing_stop_price == pytest.approx(104.0)
    assert pos.breakeven_triggered is True


def test_recovery_registers_risk_manager_at_remaining_size():
    engine = _make_engine()
    engine.risk_manager = Mock()
    _recover(engine, _row())

    kwargs = engine.risk_manager.update_position.call_args.kwargs
    assert kwargs["size"] == pytest.approx(0.10)


def test_recovery_seeds_mfe_mae_from_persisted_peaks():
    engine = _make_engine()
    _recover(engine, _row())

    tracker = engine.live_position_tracker.mfe_mae_tracker
    metrics = tracker.get_position_metrics("order-7")
    assert metrics is not None
    assert metrics.mfe == pytest.approx(0.08)
    assert metrics.mae == pytest.approx(-0.03)
    assert metrics.mfe_price == pytest.approx(108.0)

    # A post-restart price below the old peak must not lower the running max.
    engine.live_position_tracker.update_mfe_mae(102.0, persist_to_db=False)
    metrics = tracker.get_position_metrics("order-7")
    assert metrics.mfe == pytest.approx(0.08)
    assert metrics.mae == pytest.approx(-0.03)


def test_tracker_recover_positions_seeds_mfe_mae():
    engine = _make_engine()
    engine.db_manager.get_active_positions = lambda *a, **k: [_row()]
    engine.live_position_tracker.db_manager = engine.db_manager

    engine.live_position_tracker.recover_positions(session_id=1)

    metrics = engine.live_position_tracker.mfe_mae_tracker.get_position_metrics("order-7")
    assert metrics is not None
    assert metrics.mfe == pytest.approx(0.08)


def test_seed_metrics_keeps_larger_in_memory_peaks():
    tracker = MFEMAETracker()
    tracker.update_position_metrics("k", 100.0, 120.0, "long", datetime(2025, 1, 1, tzinfo=UTC))
    tracker.seed_metrics("k", MFEMetrics(mfe=0.05, mae=-0.04, mae_price=96.0))

    metrics = tracker.get_position_metrics("k")
    assert metrics.mfe == pytest.approx(0.20)
    assert metrics.mae == pytest.approx(-0.04)


def test_recovery_with_real_risk_manager_registers_remaining_and_skips_drained():
    engine = _make_engine()
    engine.risk_manager = RiskManager()
    _recover(engine, _row())
    assert engine.risk_manager.positions["ETHUSDT"]["size"] == pytest.approx(0.10)

    drained = _make_engine()
    drained.risk_manager = RiskManager()
    _recover(drained, _row(current_size=0.0))
    # Fully partial-exited row: tracked, but not registered (size>0 validator) and no crash.
    assert "ETHUSDT" not in drained.risk_manager.positions
    assert drained.live_position_tracker.positions


def test_seed_metrics_ignores_non_finite_and_does_not_mutate_argument():
    tracker = MFEMAETracker()
    seed = MFEMetrics(mfe=float("nan"), mae=float("inf"))
    tracker.seed_metrics("k", seed)

    assert math.isnan(seed.mfe)
    metrics = tracker.get_position_metrics("k")
    assert metrics is not seed
    assert metrics.mfe == 0.0 and metrics.mae == 0.0
    tracker.update_position_metrics("k", 100.0, 105.0, "long", datetime(2025, 1, 1, tzinfo=UTC))
    assert tracker.get_position_metrics("k").mfe == pytest.approx(0.05)


def test_seeded_peak_times_are_utc_aware():
    engine = _make_engine()
    _recover(engine, _row(mfe_time=datetime(2025, 1, 2), mae_time=datetime(2025, 1, 3)))

    metrics = engine.live_position_tracker.mfe_mae_tracker.get_position_metrics("order-7")
    assert metrics.mfe_time.tzinfo is UTC
    assert metrics.mae_time.tzinfo is UTC
