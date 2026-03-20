"""Tests for _check_exit_conditions to ensure _execute_exit receives all required arguments.

This is a regression test suite for the bug where _execute_exit() was being
called without the required 'candle' parameter, causing TypeError when trailing
stops were triggered.
"""

from unittest.mock import MagicMock, Mock, patch
from datetime import datetime, UTC

import pytest
import pandas as pd

from src.engines.live.trading_engine import LiveTradingEngine
from src.strategies.ml_basic import create_ml_basic_strategy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_engine_with_exit_mock() -> LiveTradingEngine:
    """Create a LiveTradingEngine with mocked dependencies and exit tracking."""
    mock_data_provider = MagicMock()
    strategy = create_ml_basic_strategy()

    with patch("src.engines.live.trading_engine.DatabaseManager"):
        engine = LiveTradingEngine(
            strategy=strategy,
            data_provider=mock_data_provider,
            initial_balance=1000.0,
        )

    engine.db_manager = MagicMock()
    engine._active_symbol = "BTCUSDT"
    return engine


def make_mock_position(symbol: str = "BTCUSDT", entry_price: float = 70000.0):
    """Create a mock position for testing."""
    from src.database.models import PositionSide

    position = MagicMock()
    position.symbol = symbol
    position.side = PositionSide.SHORT
    position.entry_price = entry_price
    position.size = 0.1
    position.order_id = "test_order_1"
    position.entry_time = datetime.now(UTC)
    position.current_size = 0.1
    position.original_size = 0.1
    return position


def make_minimal_df_with_candle() -> pd.DataFrame:
    """Create a minimal DataFrame with candle data for testing."""
    return pd.DataFrame({
        "open": [70000.0],
        "high": [71000.0],
        "low": [69000.0],
        "close": [70500.0],
        "volume": [1000.0],
    })


def make_mock_exit_check(should_exit: bool = True, exit_reason: str = "stop_loss"):
    """Create a mock ExitCheck with desired behavior."""
    exit_check = MagicMock()
    exit_check.should_exit = should_exit
    exit_check.exit_reason = exit_reason
    exit_check.limit_price = None
    return exit_check


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_check_exit_conditions_passes_candle_to_execute_exit():
    """Regression test: _check_exit_conditions must pass candle to _execute_exit.

    This test would have caught the bug where _execute_exit() was called
    without the required 'candle' argument, causing TypeError when trailing
    stops triggered exits.

    The test verifies that when an exit condition is met, _execute_exit is
    called with all 7 positional arguments including candle, plus skip_live_close
    as a keyword argument with its default value of False.
    """
    engine = make_engine_with_exit_mock()
    engine._execute_exit = MagicMock()

    # Create a mock position that should exit
    position = make_mock_position(entry_price=70000.0)
    engine.live_position_tracker.track_recovered_position(position, db_id=None)

    # Create minimal DF with candle data
    df = make_minimal_df_with_candle()
    candle = {"test": "data"}  # The candle object to be passed through

    # Mock the exit checker to return an exit signal
    exit_check = make_mock_exit_check(should_exit=True, exit_reason="trailing_stop")

    with patch.object(engine.live_exit_handler, "check_exit_conditions", return_value=exit_check):
        engine._check_exit_conditions(
            df=df,
            current_index=0,
            current_price=70500.0,
            candle=candle,
        )

    # Verify _execute_exit was called with ALL required arguments including candle
    engine._execute_exit.assert_called_once()
    call_args = engine._execute_exit.call_args

    # Positional arguments (7 total - position, reason, limit_price, current_price,
    # candle_high, candle_low, candle)
    args = call_args[0]
    assert len(args) == 7, f"Expected 7 positional arguments, got {len(args)}: {args}"

    position_arg, reason_arg, limit_arg, price_arg, high_arg, low_arg, candle_arg = args

    assert position_arg is position, "First arg should be the position"
    assert reason_arg == "trailing_stop", "Second arg should be exit reason"
    assert limit_arg is None, "Third arg should be limit_price"
    assert price_arg == 70500.0, "Fourth arg should be current_price"
    assert high_arg == 71000.0, "Fifth arg should be candle_high"
    assert low_arg == 69000.0, "Sixth arg should be candle_low"
    assert candle_arg is candle, "Seventh arg must be the candle object!"


@pytest.mark.fast
def test_check_exit_conditions_without_candle_df():
    """Test that _check_exit_conditions handles missing candle data gracefully.

    When df is None or candle data is missing, candle_high/low should be None
    but the exit should still proceed if conditions are met.
    """
    engine = make_engine_with_exit_mock()
    engine._execute_exit = MagicMock()

    position = make_mock_position(entry_price=70000.0)
    engine.live_position_tracker.track_recovered_position(position, db_id=None)

    # No candle data available
    candle = {"test": "data"}
    exit_check = make_mock_exit_check(should_exit=True, exit_reason="stop_loss")

    # Mock the helper methods that would fail when df is None
    with (
        patch.object(engine.live_exit_handler, "check_exit_conditions", return_value=exit_check),
        patch.object(engine, "_extract_indicators", return_value={}),
        patch.object(engine, "_extract_sentiment_data", return_value={}),
        patch.object(engine, "_extract_ml_predictions", return_value={}),
    ):
        engine._check_exit_conditions(
            df=None,  # No DataFrame
            current_index=0,
            current_price=70500.0,
            candle=candle,
        )

    # Exit should still execute
    engine._execute_exit.assert_called_once()
    call_args = engine._execute_exit.call_args[0]

    # candle_high and candle_low should be None when df is None
    assert call_args[4] is None, "candle_high should be None when df is None"
    assert call_args[5] is None, "candle_low should be None when df is None"
    # But candle must still be passed!
    assert call_args[6] is candle, "candle object must still be passed!"


@pytest.mark.fast
def test_check_exit_conditions_with_trailing_stop_scenario():
    """Regression test: Simulate the exact trailing stop scenario that caused the crash.

    Session #107 crashed because trailing stop activated but _execute_exit()
    was called without the candle argument. This test reproduces that scenario
    and verifies the fix.
    """
    engine = make_engine_with_exit_mock()
    engine._execute_exit = MagicMock()

    # Simulate a short position in profit (trailing stop activated)
    position = make_mock_position(symbol="BTCUSDT", entry_price=74000.0)
    engine.live_position_tracker.track_recovered_position(position, db_id=None)

    # Current price below entry (short is profitable)
    df = make_minimal_df_with_candle()
    candle = {
        "timestamp": datetime.now(UTC),
        "test": "trailing_stop_activation"
    }

    # Trailing stop is triggered
    exit_check = make_mock_exit_check(should_exit=True, exit_reason="trailing_stop")

    with patch.object(engine.live_exit_handler, "check_exit_conditions", return_value=exit_check):
        engine._check_exit_conditions(
            df=df,
            current_index=0,
            current_price=70500.0,  # Price dropped, trailing stop activates
            candle=candle,
        )

    # Critical: _execute_exit MUST be called with candle argument
    engine._execute_exit.assert_called_once()

    # Verify the call signature includes candle
    call_args = engine._execute_exit.call_args[0]
    assert call_args[6] is candle, (
        "BUG: candle argument not passed to _execute_exit! "
        "This would cause TypeError and crash the bot."
    )


@pytest.mark.fast
def test_check_exit_conditions_skips_when_no_positions():
    """Edge case: _check_exit_conditions returns early when no positions exist."""
    engine = make_engine_with_exit_mock()
    engine._execute_exit = MagicMock()

    # Should return immediately without attempting any exits
    result = engine._check_exit_conditions(
        df=None,
        current_index=0,
        current_price=70500.0,
        candle=None,
    )

    assert result is None
    engine._execute_exit.assert_not_called()


@pytest.mark.fast
def test_check_exit_conditions_with_multiple_positions():
    """Test that _check_exit_conditions processes all positions correctly.

    When multiple positions are open, each should be checked independently
    and each exit should receive the correct candle argument.
    """
    engine = make_engine_with_exit_mock()
    engine._execute_exit = MagicMock()

    # Create two mock positions
    position1 = make_mock_position(symbol="BTCUSDT", entry_price=70000.0)
    position1.order_id = "order_1"
    position2 = make_mock_position(symbol="ETHUSDT", entry_price=3000.0)
    position2.order_id = "order_2"

    engine.live_position_tracker.track_recovered_position(position1, db_id=None)
    engine.live_position_tracker.track_recovered_position(position2, db_id=None)

    df = make_minimal_df_with_candle()
    candle = {"test": "data"}

    # First position exits, second doesn't
    def mock_check_exit_conditions(position, **_kwargs):
        if position.symbol == "BTCUSDT":
            return make_mock_exit_check(should_exit=True, exit_reason="take_profit")
        return make_mock_exit_check(should_exit=False)

    with patch.object(engine.live_exit_handler, "check_exit_conditions", side_effect=mock_check_exit_conditions):
        engine._check_exit_conditions(
            df=df,
            current_index=0,
            current_price=71000.0,
            candle=candle,
        )

    # Only one exit should be attempted (for BTCUSDT)
    engine._execute_exit.assert_called_once()

    # And it must have the candle argument
    call_args = engine._execute_exit.call_args[0]
    assert call_args[6] is candle, "candle must be passed even with multiple positions"
