"""Unit tests for live engine hot-swap refresh behaviour.

After a regime/manual hot-swap, the live engine must refresh strategy-dependent
state so the new strategy's risk overrides take effect on the very next
decision. Without this, the engine silently diverges from a backtest-validated
strategy until the next restart (P2 issue documented on this branch).

These tests target the live engine's swap-apply pipeline (the same code path
the run loop drives every cycle when ``strategy_manager.has_pending_update()``
is True). The corresponding backtest behaviour lives in ``Backtester._switch_strategy``.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.engines.live.trading_engine import LiveTradingEngine
from src.strategies.ml_basic import create_ml_basic_strategy
from tests.mocks import MockDatabaseManager

pytestmark = pytest.mark.unit


def _build_engine(monkeypatch, *, strategy, **kwargs) -> LiveTradingEngine:
    """Construct a LiveTradingEngine with mocked DB for unit testing."""
    monkeypatch.setattr("src.engines.live.trading_engine.DatabaseManager", MockDatabaseManager)
    kwargs.setdefault("data_provider", object())
    kwargs.setdefault("enable_live_trading", False)
    kwargs.setdefault("log_trades", False)
    kwargs.setdefault("enable_hot_swapping", True)
    return LiveTradingEngine(strategy=strategy, **kwargs)


def _drive_swap(engine: LiveTradingEngine, new_strategy_name: str, new_config: dict) -> bool:
    """Drive a full hot-swap through the production code path.

    Mirrors what the live engine main loop does when a pending update is
    detected: prepares the swap via ``hot_swap_strategy``, applies the pending
    update, and runs the engine-side refresh hook. Returns success bool.
    """
    prepared = engine.strategy_manager.hot_swap_strategy(new_strategy_name, new_config=new_config)
    if not prepared:
        return False
    return engine._apply_pending_strategy_update()


# ---------------------------------------------------------------------------
# Component risk binding
# ---------------------------------------------------------------------------


def test_hot_swap_rebinds_component_risk_to_engine_core_manager(monkeypatch):
    """The new strategy's CoreRiskAdapter must be bound to the engine's
    PortfolioRiskManager after a swap. Without this, the strategy adapter
    writes position-tracking state into a stale risk manager that diverges
    from the engine's, breaking correlation/exposure caps.
    """
    initial = create_ml_basic_strategy(fast_mode=True, stop_loss_pct=0.05)
    engine = _build_engine(monkeypatch, strategy=initial)
    engine_core = engine.risk_manager

    # Sanity: at startup, the initial strategy's adapter is bound to the engine.
    assert initial.risk_manager._core_manager is engine_core

    success = _drive_swap(engine, "ml_basic", {"stop_loss_pct": 0.02})
    assert success, "Hot-swap apply pipeline should succeed"

    new_strategy = engine.strategy
    assert new_strategy is not initial
    # New strategy adapter must point at the engine's portfolio risk manager,
    # not the fresh one its factory created in isolation.
    assert new_strategy.risk_manager._core_manager is engine_core


def test_hot_swap_propagates_new_strategy_overrides_to_component_risk(monkeypatch):
    """After swap, the new component risk adapter must carry the new
    strategy's risk overrides (so SL/TP overrides flow into runtime sizing).
    """
    initial = create_ml_basic_strategy(fast_mode=True, stop_loss_pct=0.05)
    engine = _build_engine(monkeypatch, strategy=initial)

    success = _drive_swap(engine, "ml_basic", {"stop_loss_pct": 0.02})
    assert success

    new_strategy = engine.strategy
    overrides = new_strategy.risk_manager._strategy_overrides
    assert overrides["stop_loss_pct"] == pytest.approx(0.02)


# ---------------------------------------------------------------------------
# Entry handler & SL/TP propagation (the headline regression)
# ---------------------------------------------------------------------------


def test_hot_swap_applies_new_overrides_on_next_entry(monkeypatch):
    """Headline regression: after swap, the live entry handler reads the new
    strategy's SL override on the very next entry. Backtest already does this
    via ``_switch_strategy`` rebinding; live must match.
    """
    initial = create_ml_basic_strategy(fast_mode=True, stop_loss_pct=0.05)
    engine = _build_engine(monkeypatch, strategy=initial)

    # Pre-swap sanity
    assert initial.risk_manager._strategy_overrides["stop_loss_pct"] == pytest.approx(0.05)

    success = _drive_swap(engine, "ml_basic", {"stop_loss_pct": 0.02})
    assert success

    new_strategy = engine.strategy
    # Entry handler points at the new strategy.
    assert engine.live_entry_handler.component_strategy is new_strategy
    # And the new strategy exposes the new SL via the same path entry_utils uses.
    assert new_strategy.risk_manager._strategy_overrides["stop_loss_pct"] == pytest.approx(0.02)


# ---------------------------------------------------------------------------
# Engine-level policy refresh
# ---------------------------------------------------------------------------


def _component_strategy_with_trailing_overrides(*, activation: float, distance_pct: float):
    """Build a component strategy whose ``get_risk_overrides`` returns trailing config."""
    strategy = create_ml_basic_strategy(fast_mode=True)
    # ml_basic does not call set_risk_overrides on the Strategy itself, so do it here
    # to drive the trailing-policy builder through the strategy override branch.
    strategy.set_risk_overrides(
        {
            "stop_loss_pct": 0.02,
            "trailing_stop": {
                "activation_threshold": activation,
                "trailing_distance_pct": distance_pct,
            },
        }
    )
    return strategy


def test_hot_swap_rebuilds_engine_trailing_stop_policy(monkeypatch):
    """Engine-level ``trailing_stop_policy`` must reflect the new strategy's
    trailing overrides after a swap (and the live exit handler must see the
    refreshed reference, not the stale construction-time one).
    """
    initial = _component_strategy_with_trailing_overrides(activation=0.01, distance_pct=0.005)
    engine = _build_engine(monkeypatch, strategy=initial)

    pre_policy = engine.trailing_stop_policy
    assert pre_policy is not None
    assert pre_policy.activation_threshold == pytest.approx(0.01)
    assert engine.live_exit_handler.trailing_stop_policy is pre_policy
    assert engine.live_exit_handler._trailing_stop_manager.policy is pre_policy

    # Swap to a different ml_basic instance and inject distinct trailing overrides
    # via the manager's hot-swap path.
    success = engine.strategy_manager.hot_swap_strategy("ml_basic", new_config={})
    assert success
    new_strategy = engine.strategy_manager.pending_update["data"]["new_strategy"]
    new_strategy.set_risk_overrides(
        {
            "trailing_stop": {
                "activation_threshold": 0.05,
                "trailing_distance_pct": 0.02,
            },
        }
    )
    success = engine._apply_pending_strategy_update()
    assert success

    refreshed = engine.trailing_stop_policy
    assert refreshed is not None
    assert refreshed is not pre_policy
    assert refreshed.activation_threshold == pytest.approx(0.05)
    # Live exit handler must observe the same refreshed instance.
    assert engine.live_exit_handler.trailing_stop_policy is refreshed
    assert engine.live_exit_handler._trailing_stop_manager.policy is refreshed


def test_hot_swap_rebuilds_partial_manager(monkeypatch):
    """Engine-level partial_manager must reflect the new strategy's
    ``partial_operations`` overrides after a swap, including the wrapped
    PartialOperationsManager inside the live exit handler.
    """
    initial = create_ml_basic_strategy(fast_mode=True)
    initial.set_risk_overrides(
        {
            "partial_operations": {
                "exit_targets": [0.01],
                "exit_sizes": [0.5],
                "scale_in_thresholds": [],
                "scale_in_sizes": [],
                "max_scale_ins": 0,
            },
        }
    )
    engine = _build_engine(
        monkeypatch,
        strategy=initial,
        enable_partial_operations=True,
    )
    pre_partial = engine.partial_manager
    assert pre_partial is not None
    assert pre_partial.exit_targets == [0.01]

    success = engine.strategy_manager.hot_swap_strategy("ml_basic", new_config={})
    assert success
    new_strategy = engine.strategy_manager.pending_update["data"]["new_strategy"]
    new_strategy.set_risk_overrides(
        {
            "partial_operations": {
                "exit_targets": [0.05, 0.10],
                "exit_sizes": [0.4, 0.4],
                "scale_in_thresholds": [],
                "scale_in_sizes": [],
                "max_scale_ins": 0,
            },
        }
    )
    success = engine._apply_pending_strategy_update()
    assert success

    refreshed = engine.partial_manager
    assert refreshed is not None
    assert refreshed is not pre_partial
    assert refreshed.exit_targets == [0.05, 0.10]
    # Exit handler's PartialOperationsManager wraps the refreshed policy.
    assert engine.live_exit_handler.partial_manager is not None
    assert engine.live_exit_handler.partial_manager.policy is refreshed


# ---------------------------------------------------------------------------
# Correlation handler refresh
# ---------------------------------------------------------------------------


def test_hot_swap_refreshes_correlation_handler_strategy(monkeypatch):
    """When a correlation_handler is wired on the live entry handler, the
    swap must refresh its strategy reference (mirrors backtest engine.py:817).
    """
    initial = create_ml_basic_strategy(fast_mode=True)
    engine = _build_engine(monkeypatch, strategy=initial)

    correlation_handler = MagicMock()
    engine.live_entry_handler.correlation_handler = correlation_handler

    success = _drive_swap(engine, "ml_basic", {"stop_loss_pct": 0.02})
    assert success

    new_strategy = engine.strategy
    correlation_handler.set_strategy.assert_called_once_with(new_strategy)


# ---------------------------------------------------------------------------
# Sizer-invariant guard
# ---------------------------------------------------------------------------


def test_hot_swap_rejects_invalid_min_confidence_floor(monkeypatch):
    """Attempting to swap to a strategy with ``min_confidence_floor > min_confidence``
    must fail at preparation time and leave the engine on its original strategy.

    The ConfidenceWeightedSizer constructor enforces the invariant; this test
    confirms the regime-driven hot-swap path surfaces that failure cleanly
    instead of leaving the engine in an invalid sizer state.
    """
    initial = create_ml_basic_strategy(fast_mode=False)
    engine = _build_engine(monkeypatch, strategy=initial)

    success = engine.strategy_manager.hot_swap_strategy(
        "ml_basic",
        new_config={"min_confidence": 0.3, "min_confidence_floor": 0.5},
    )
    assert success is False
    # No pending update was queued.
    assert engine.strategy_manager.has_pending_update() is False
    # Engine remains on the original strategy.
    assert engine.strategy is initial


# ---------------------------------------------------------------------------
# No-correlation-handler defensive path
# ---------------------------------------------------------------------------


def test_hot_swap_succeeds_when_no_correlation_handler_wired(monkeypatch):
    """The swap path must not fail when no correlation_handler is wired
    (default for live engine today).
    """
    initial = create_ml_basic_strategy(fast_mode=True)
    engine = _build_engine(monkeypatch, strategy=initial)
    assert engine.live_entry_handler.correlation_handler is None

    success = _drive_swap(engine, "ml_basic", {"stop_loss_pct": 0.02})
    assert success
    assert engine.live_entry_handler.correlation_handler is None
