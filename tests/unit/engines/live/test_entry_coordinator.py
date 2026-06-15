"""Unit tests for LiveEntryCoordinator.execute_entry_locked branches (#486).

The entry pipeline was extracted from LiveTradingEngine into
``LiveEntryCoordinator`` (the engine keeps thin delegating wrappers). These
tests target the real-money execution branches directly on the coordinator,
driving it with a mocked engine-state backref. They also pin the CODE.md
hardening applied after the extraction: the stop-loss gate now keys on
``stop_loss is not None`` (so a misconfigured ``0.0`` stop is no longer
silently skipped), and the SL-calc / risk-override failure paths log.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.engines.live.execution.entry_coordinator import (
    LiveEntryCoordinator,
    LiveEntryEngineState,
)
from src.engines.shared.models import PositionSide

pytestmark = pytest.mark.fast


def _make_position() -> MagicMock:
    position = MagicMock()
    position.order_id = "order-1"
    position.entry_price = 50000.0
    position.quantity = 0.01
    position.entry_balance = 1000.0
    position.metadata = {}
    return position


def _make_result(position: MagicMock, **overrides) -> MagicMock:
    result = MagicMock()
    result.executed = True
    result.position = position
    result.entry_fee = 1.0
    result.slippage_cost = 0.5
    result.ambiguous = False
    result.error = None
    for k, v in overrides.items():
        setattr(result, k, v)
    return result


def _make_state(position: MagicMock, result: MagicMock, **overrides) -> MagicMock:
    """Build a mocked engine-state backref with sane happy-path defaults.

    Spec'd against the coordinator's protocol so method-name drift is caught;
    every attribute the execution path reads is set explicitly (protocol
    attributes are annotation-only, so they must be assigned to be readable).
    """
    state = MagicMock(spec=LiveEntryEngineState)
    state.enable_live_trading = False
    state.current_balance = 1000.0
    state.max_position_size = 1.0
    state.trading_session_id = None
    # Protocol attributes are annotation-only (not in dir()), so a spec'd mock
    # won't auto-create them — assign each object attribute the path reads.
    state.exchange_interface = MagicMock()
    state.order_tracker = MagicMock()
    state.live_position_tracker = MagicMock()
    state.risk_manager = MagicMock()
    state.live_entry_handler = MagicMock()
    state.stop_loss_manager = MagicMock()
    state.db_manager = MagicMock()

    state.live_position_tracker.has_position_for_symbol.return_value = False
    state.live_position_tracker.position_count = 0
    state.risk_manager.get_max_concurrent_positions.return_value = 1
    state.live_entry_handler.execute_entry.return_value = result
    state.stop_loss_manager.place_protection.return_value = "sl-order-1"
    state._strategy_name.return_value = "test_strategy"

    for k, v in overrides.items():
        setattr(state, k, v)
    return state


def _call(state: MagicMock, *, side=PositionSide.LONG, stop_loss=49000.0, take_profit=51000.0):
    coordinator = LiveEntryCoordinator(engine_state=state)
    coordinator.execute_entry_locked(
        symbol="BTCUSDT",
        side=side,
        size=0.1,
        price=50000.0,
        stop_loss=stop_loss,
        take_profit=take_profit,
        signal_strength=0.8,
        signal_confidence=0.7,
    )
    return coordinator


# ---------------------------------------------------------------------------
# Pre-execution guards
# ---------------------------------------------------------------------------


def test_duplicate_position_guard_skips_execution():
    position = _make_position()
    state = _make_state(position, _make_result(position))
    state.live_position_tracker.has_position_for_symbol.return_value = True

    _call(state)

    state.live_entry_handler.execute_entry.assert_not_called()


def test_max_concurrent_positions_guard_skips_execution():
    position = _make_position()
    state = _make_state(position, _make_result(position))
    state.live_position_tracker.position_count = 1
    state.risk_manager.get_max_concurrent_positions.return_value = 1

    _call(state)

    state.live_entry_handler.execute_entry.assert_not_called()


def test_size_capped_at_max_position_size():
    position = _make_position()
    state = _make_state(position, _make_result(position))
    state.max_position_size = 0.05  # below the requested 0.1

    coordinator = LiveEntryCoordinator(engine_state=state)
    coordinator.execute_entry_locked(
        symbol="BTCUSDT",
        side=PositionSide.LONG,
        size=0.1,
        price=50000.0,
        stop_loss=49000.0,
        take_profit=51000.0,
        signal_strength=0.8,
        signal_confidence=0.7,
    )

    # The handler is called with the capped size, not the requested 0.1.
    _, kwargs = state.live_entry_handler.execute_entry.call_args
    assert kwargs["signal"].size_fraction == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# Happy path + tracking
# ---------------------------------------------------------------------------


def test_happy_path_tracks_position_and_registers_risk():
    position = _make_position()
    state = _make_state(position, _make_result(position))

    _call(state)

    state.live_position_tracker.open_position.assert_called_once()
    state.risk_manager.update_position.assert_called_once()
    # No-session path deducts the fee directly.
    assert state.current_balance == pytest.approx(1000.0 - 1.0)


def test_failed_execution_returns_without_tracking():
    position = _make_position()
    result = _make_result(position, executed=False, error="boom")
    result.position = None
    state = _make_state(position, result)

    _call(state)

    state.live_position_tracker.open_position.assert_not_called()


# ---------------------------------------------------------------------------
# Failure / emergency-close paths
# ---------------------------------------------------------------------------


def test_position_tracking_failure_triggers_emergency_close_and_refund():
    position = _make_position()
    state = _make_state(position, _make_result(position))
    state.enable_live_trading = True
    state.live_position_tracker.open_position.side_effect = RuntimeError("tracker down")

    _call(state)

    # Emergency close placed with AUTO_REPAY; fee refunded directly (no session).
    state.exchange_interface.place_order.assert_called_once()
    assert state.current_balance == pytest.approx(1000.0)  # -1.0 fee then +1.0 refund


def test_risk_manager_failure_closes_position():
    position = _make_position()
    state = _make_state(position, _make_result(position))
    state.risk_manager.update_position.side_effect = ValueError("risk sync failed")

    _call(state)

    state._execute_exit.assert_called_once()


def test_balance_update_failure_emergency_closes_with_session():
    position = _make_position()
    state = _make_state(position, _make_result(position))
    state.enable_live_trading = True
    state.trading_session_id = 42
    state.db_manager.atomic_balance_update.side_effect = RuntimeError("db down")

    _call(state)

    # Balance update failed before tracking → emergency close, never tracks.
    state.exchange_interface.place_order.assert_called_once()
    state.live_position_tracker.open_position.assert_not_called()


def test_ambiguous_submission_enters_close_only_mode_without_stop_loss():
    position = _make_position()
    state = _make_state(position, _make_result(position, ambiguous=True))
    state.enable_live_trading = True

    _call(state)

    state._enter_close_only_mode.assert_called_once()
    state.stop_loss_manager.place_protection.assert_not_called()


def test_stop_loss_placement_failure_triggers_emergency_exit():
    position = _make_position()
    state = _make_state(position, _make_result(position))
    state.enable_live_trading = True
    state.stop_loss_manager.place_protection.return_value = None  # placement failed

    _call(state)

    state._record_event.assert_called_once()
    state._execute_exit.assert_called_once()


# ---------------------------------------------------------------------------
# CODE.md hardening: stop-loss gate keys on `is not None` (#813 follow-up)
# ---------------------------------------------------------------------------


def test_zero_stop_loss_is_not_silently_skipped():
    """A 0.0 stop must enter the placement path (and fail there → emergency
    close) rather than being silently treated as "no stop-loss" and leaving the
    position unprotected. Regression guard for the truthy-check fix."""
    position = _make_position()
    state = _make_state(position, _make_result(position))
    state.enable_live_trading = True

    _call(state, stop_loss=0.0)

    state.stop_loss_manager.place_protection.assert_called_once()


def test_none_stop_loss_skips_placement():
    position = _make_position()
    state = _make_state(position, _make_result(position))
    state.enable_live_trading = True

    _call(state, stop_loss=None)

    state.stop_loss_manager.place_protection.assert_not_called()
