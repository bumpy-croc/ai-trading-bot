"""Unit tests for LiveExitCoordinator branches (#486).

The exit pipeline was extracted from LiveTradingEngine into
``LiveExitCoordinator`` (the engine keeps thin delegating wrappers). These
tests target the coordinator directly via a mocked engine-state backref,
covering the routing seam (``check_exit_conditions`` must invoke the close
through ``state._execute_exit`` so engine-level test mocks still intercept),
the base-asset-locked delegation in ``execute_exit``, and the early-return
guards in ``execute_exit_locked``. They also pin the CODE.md hardening applied
after the extraction (notably the ``datetime.now(UTC)`` position-age reason,
which must not raise on an aware ``entry_time``).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.engines.live.execution.exit_coordinator import (
    LiveExitCoordinator,
    LiveExitEngineState,
)
from src.engines.shared.models import PositionSide

pytestmark = pytest.mark.fast

CANDLE_TIME = pd.Timestamp("2024-01-02 00:00:00")


def _make_position(*, entry_price: float = 50000.0) -> MagicMock:
    """Return a minimal open-position mock for use in coordinator tests."""
    position = MagicMock()
    position.order_id = "order-1"
    position.symbol = "BTCUSDT"
    position.side = PositionSide.LONG
    position.entry_price = entry_price
    position.size = 0.1
    position.current_size = None
    # Entered well before the current candle so same-bar protection doesn't skip it.
    position.entry_time = datetime(2024, 1, 1, tzinfo=UTC) - timedelta(days=1)
    position.stop_loss_order_id = None
    position.metadata = {}
    return position


def _make_exit_check(*, should_exit: bool, reason: str = "take_profit") -> MagicMock:
    """Return a mocked exit-condition result from the exit handler."""
    check = MagicMock()
    check.should_exit = should_exit
    check.exit_reason = reason
    check.limit_price = None
    return check


def _make_df() -> pd.DataFrame:
    """Single-row OHLC frame indexed at CANDLE_TIME (provides high/low + candle_time)."""
    return pd.DataFrame(
        {"open": [50000.0], "high": [50500.0], "low": [49500.0], "close": [50100.0]},
        index=pd.DatetimeIndex([CANDLE_TIME]),
    )


def _make_state(position: MagicMock, exit_check: MagicMock, **overrides) -> MagicMock:
    """Build a mocked engine-state backref with happy-path exit defaults.

    Spec'd against the coordinator's protocol so method-name drift is caught;
    protocol attributes are annotation-only, so each one the path reads is
    assigned explicitly.
    """
    state = MagicMock(spec=LiveExitEngineState)
    state.enable_live_trading = False
    state.current_balance = 1000.0
    state.trading_session_id = None
    state.log_trades = False
    state.completed_trades = []
    state._component_strategy = None
    state.live_position_tracker = MagicMock()
    state.live_exit_handler = MagicMock()
    state.exchange_interface = MagicMock()
    state.db_manager = MagicMock()
    state._base_asset_locks = MagicMock()

    state.live_position_tracker.positions = {position.order_id: position}
    state.live_exit_handler.check_exit_conditions.return_value = exit_check
    state._extract_indicators.return_value = {"prediction_confidence": 0.5}
    state._extract_sentiment_data.return_value = {}
    state._extract_ml_predictions.return_value = {}
    state._strategy_name.return_value = "test_strategy"

    for k, v in overrides.items():
        setattr(state, k, v)
    return state


def test_check_exit_conditions_no_positions_returns_early():
    position = _make_position()
    state = _make_state(position, _make_exit_check(should_exit=False))
    state.live_position_tracker.positions = {}

    LiveExitCoordinator(engine_state=state).check_exit_conditions(_make_df(), 0, 50100.0)

    state.live_exit_handler.check_exit_conditions.assert_not_called()
    state._execute_exit.assert_not_called()


def test_check_exit_conditions_routes_close_through_engine_wrapper():
    """should_exit=True must call state._execute_exit (the engine wrapper that
    tests replace with a Mock), NOT self.execute_exit — the mock-routing seam."""
    position = _make_position()
    state = _make_state(position, _make_exit_check(should_exit=True, reason="take_profit"))

    LiveExitCoordinator(engine_state=state).check_exit_conditions(
        _make_df(), 0, 50100.0, candle={"k": "v"}
    )

    state._execute_exit.assert_called_once()
    args = state._execute_exit.call_args[0]
    # position, reason, limit_price, current_price, candle_high, candle_low, candle
    assert args[0] is position
    assert args[1] == "take_profit"
    assert args[2] is None
    assert args[3] == 50100.0
    assert args[4] == 50500.0  # candle_high
    assert args[5] == 49500.0  # candle_low
    assert args[6] == {"k": "v"}


def test_check_exit_conditions_hold_does_not_close():
    position = _make_position()
    state = _make_state(position, _make_exit_check(should_exit=False))

    LiveExitCoordinator(engine_state=state).check_exit_conditions(_make_df(), 0, 50100.0)

    state.live_exit_handler.check_exit_conditions.assert_called_once()
    state._execute_exit.assert_not_called()


def test_check_exit_conditions_skips_same_bar_entry():
    position = _make_position()
    # Entered on/after the candle open -> same-bar protection skips it entirely.
    position.entry_time = CANDLE_TIME.to_pydatetime().replace(tzinfo=UTC)
    state = _make_state(position, _make_exit_check(should_exit=True))

    LiveExitCoordinator(engine_state=state).check_exit_conditions(_make_df(), 0, 50100.0)

    state.live_exit_handler.check_exit_conditions.assert_not_called()
    state._execute_exit.assert_not_called()


def test_check_exit_conditions_position_age_reason_handles_aware_entry_time():
    """Regression for the datetime.utcnow() -> datetime.now(UTC) hardening: the
    position-age log reason must not raise on a timezone-aware entry_time."""
    position = _make_position()
    state = _make_state(position, _make_exit_check(should_exit=False))

    # Must not raise (naive utcnow - naive entry worked; aware now - naive would
    # raise, so the fix strips tzinfo on both sides).
    LiveExitCoordinator(engine_state=state).check_exit_conditions(_make_df(), 0, 50100.0)

    state.db_manager.log_strategy_execution.assert_called_once()


def test_execute_exit_takes_base_asset_lock_then_delegates():
    position = _make_position()
    state = _make_state(position, _make_exit_check(should_exit=False))
    coordinator = LiveExitCoordinator(engine_state=state)
    coordinator.execute_exit_locked = MagicMock()

    coordinator.execute_exit(position, "take_profit", None, 50100.0, None, None, None)

    state._base_asset_locks.lock_for.assert_called_once()
    coordinator.execute_exit_locked.assert_called_once()
    assert coordinator.execute_exit_locked.call_args[0][0] is position


def test_execute_exit_locked_skips_when_position_already_closed():
    position = _make_position()
    state = _make_state(position, _make_exit_check(should_exit=False))
    state.live_position_tracker.has_position.return_value = False

    LiveExitCoordinator(engine_state=state).execute_exit_locked(
        position, "take_profit", None, 50100.0, None, None, None
    )

    state.live_exit_handler.execute_filled_exit.assert_not_called()
    state.live_exit_handler.execute_exit.assert_not_called()


def test_execute_exit_locked_skips_on_invalid_entry_price():
    position = _make_position(entry_price=0.0)
    state = _make_state(position, _make_exit_check(should_exit=False))
    state.live_position_tracker.has_position.return_value = True

    LiveExitCoordinator(engine_state=state).execute_exit_locked(
        position, "take_profit", None, 50100.0, None, None, None
    )

    state.live_exit_handler.execute_filled_exit.assert_not_called()
    state.live_exit_handler.execute_exit.assert_not_called()


def test_execute_exit_locked_returns_on_failed_close():
    """A failed filled-exit logs and returns before any balance/trade accounting."""
    position = _make_position()
    state = _make_state(position, _make_exit_check(should_exit=False))
    state.live_position_tracker.has_position.return_value = True
    failed = MagicMock()
    failed.success = False
    failed.error = "boom"
    state.live_exit_handler.execute_filled_exit.return_value = failed

    LiveExitCoordinator(engine_state=state).execute_exit_locked(
        position, "stop_loss", None, 50100.0, None, None, None, skip_live_close=True
    )

    # skip_live_close -> filled-exit path; failure means no trade recorded.
    state.live_exit_handler.execute_filled_exit.assert_called_once()
    assert state.completed_trades == []
