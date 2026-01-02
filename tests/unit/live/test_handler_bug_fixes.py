"""Tests for bug fixes in the live trading handler modules.

Covers regression tests for:
- Exit fee calculation using exit notional (not entry notional)
- Thread safety in LivePositionTracker
- Daily P&L tracking in LiveEventLogger
"""

from __future__ import annotations

import threading
import time
from datetime import date, datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from src.engines.live.execution.entry_handler import LiveEntryHandler, LiveEntrySignal
from src.engines.live.execution.exit_handler import LiveExitHandler
from src.engines.live.execution.position_tracker import (
    LivePosition,
    LivePositionTracker,
    PositionSide,
)
from src.engines.live.logging.event_logger import LiveEventLogger


class TestExitFeeCalculation:
    """Test that exit fees use exit notional, not entry notional."""

    def test_exit_fee_accounts_for_price_change_winning_trade(self) -> None:
        """Exit fee should be higher for winning trades (position worth more at exit)."""
        # Arrange
        position_tracker = LivePositionTracker()
        execution_engine = MagicMock()
        execution_engine.execute_exit.return_value = MagicMock(
            success=True,
            executed_price=55000.0,  # Exit price higher than entry
            order_id="test-exit",
            fill_quantity=1.0,
        )

        exit_handler = LiveExitHandler(
            position_tracker=position_tracker,
            execution_engine=execution_engine,
        )

        # Create winning position: entry at 50000, exit at 55000 (+10%)
        position = LivePosition(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            size=0.1,
            entry_price=50000.0,
            entry_time=datetime.now(timezone.utc),
            entry_balance=1000.0,
            order_id="test-order-123",
        )
        position_tracker.open_position(position)

        # Act
        exit_handler.execute_exit(
            position=position,
            exit_reason="take_profit",
            current_price=55000.0,
            limit_price=55000.0,
            current_balance=1100.0,
        )

        # Assert: position_notional passed to execute_exit should include price adjustment
        execute_exit_call = execution_engine.execute_exit.call_args
        position_notional = execute_exit_call.kwargs["position_notional"]

        # Entry notional = 1000 * 0.1 = 100
        # Price adjustment = 55000 / 50000 = 1.10
        # Exit notional = 100 * 1.10 = 110
        expected_exit_notional = 1000.0 * 0.1 * (55000.0 / 50000.0)
        assert position_notional == pytest.approx(expected_exit_notional, rel=0.01)

    def test_exit_fee_accounts_for_price_change_losing_trade(self) -> None:
        """Exit fee should be lower for losing trades (position worth less at exit)."""
        # Arrange
        position_tracker = LivePositionTracker()
        execution_engine = MagicMock()
        execution_engine.execute_exit.return_value = MagicMock(
            success=True,
            executed_price=45000.0,  # Exit price lower than entry
            order_id="test-exit",
            fill_quantity=1.0,
        )

        exit_handler = LiveExitHandler(
            position_tracker=position_tracker,
            execution_engine=execution_engine,
        )

        # Create losing position: entry at 50000, exit at 45000 (-10%)
        position = LivePosition(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            size=0.1,
            entry_price=50000.0,
            entry_time=datetime.now(timezone.utc),
            entry_balance=1000.0,
            order_id="test-order-456",
        )
        position_tracker.open_position(position)

        # Act
        exit_handler.execute_exit(
            position=position,
            exit_reason="stop_loss",
            current_price=45000.0,
            limit_price=45000.0,
            current_balance=900.0,
        )

        # Assert: position_notional should reflect lower exit value
        execute_exit_call = execution_engine.execute_exit.call_args
        position_notional = execute_exit_call.kwargs["position_notional"]

        # Entry notional = 1000 * 0.1 = 100
        # Price adjustment = 45000 / 50000 = 0.90
        # Exit notional = 100 * 0.90 = 90
        expected_exit_notional = 1000.0 * 0.1 * (45000.0 / 50000.0)
        assert position_notional == pytest.approx(expected_exit_notional, rel=0.01)


class TestTakeProfitLimitPricing:
    """Test take profit exits use limit price instead of favorable candle extremes."""

    def test_take_profit_uses_limit_price_for_long(self) -> None:
        """Long take profit should not exceed the limit price."""
        position_tracker = LivePositionTracker()
        execution_engine = MagicMock()
        execution_engine.execute_exit.return_value = MagicMock(
            success=True,
            executed_price=100.0,
            order_id="tp-exit-long",
            fill_quantity=1.0,
        )

        exit_handler = LiveExitHandler(
            position_tracker=position_tracker,
            execution_engine=execution_engine,
        )

        position = LivePosition(
            symbol="ETHUSDT",
            side=PositionSide.LONG,
            size=0.5,
            entry_price=90.0,
            entry_time=datetime.now(timezone.utc),
            entry_balance=1000.0,
            order_id="tp-order-long",
        )
        position_tracker.open_position(position)

        exit_handler.execute_exit(
            position=position,
            exit_reason="Take profit",
            current_price=110.0,
            limit_price=100.0,
            current_balance=1100.0,
            candle_high=120.0,
            candle_low=95.0,
        )

        execute_exit_call = execution_engine.execute_exit.call_args
        base_price = execute_exit_call.kwargs["base_price"]

        assert base_price == pytest.approx(100.0)

    def test_take_profit_uses_limit_price_for_short(self) -> None:
        """Short take profit should not exceed the limit price."""
        position_tracker = LivePositionTracker()
        execution_engine = MagicMock()
        execution_engine.execute_exit.return_value = MagicMock(
            success=True,
            executed_price=80.0,
            order_id="tp-exit-short",
            fill_quantity=1.0,
        )

        exit_handler = LiveExitHandler(
            position_tracker=position_tracker,
            execution_engine=execution_engine,
        )

        position = LivePosition(
            symbol="ETHUSDT",
            side=PositionSide.SHORT,
            size=0.5,
            entry_price=100.0,
            entry_time=datetime.now(timezone.utc),
            entry_balance=1000.0,
            order_id="tp-order-short",
        )
        position_tracker.open_position(position)

        exit_handler.execute_exit(
            position=position,
            exit_reason="Take profit",
            current_price=70.0,
            limit_price=80.0,
            current_balance=1100.0,
            candle_high=95.0,
            candle_low=65.0,  # More favorable than limit, should not be used
        )

        execute_exit_call = execution_engine.execute_exit.call_args
        base_price = execute_exit_call.kwargs["base_price"]

        # Base price should be the limit price (80.0), not the more favorable candle_low (65.0)
        assert base_price == pytest.approx(80.0)


class TestExitConditionOrdering:
    """Test exit condition evaluation order matches backtest."""

    def test_strategy_exit_checked_before_risk_exits(self) -> None:
        """Strategy exit evaluation runs even when stop loss triggers."""
        position_tracker = LivePositionTracker()
        execution_engine = MagicMock()
        exit_handler = LiveExitHandler(
            position_tracker=position_tracker,
            execution_engine=execution_engine,
        )

        position = LivePosition(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            size=0.1,
            entry_price=100.0,
            entry_time=datetime.now(timezone.utc),
            order_id="order-exit-1",
            stop_loss=95.0,
        )

        runtime_decision = MagicMock()
        component_strategy = MagicMock()

        with patch.object(
            exit_handler, "_check_strategy_exit", return_value=(True, "Strategy signal")
        ) as mock_check:
            result = exit_handler.check_exit_conditions(
                position=position,
                current_price=94.0,
                candle_high=101.0,
                candle_low=94.0,
                runtime_decision=runtime_decision,
                component_strategy=component_strategy,
            )

        mock_check.assert_called_once()
        assert result.should_exit is True
        assert "Stop loss" in result.exit_reason


class TestPositionTrackerThreadSafety:
    """Test thread safety of LivePositionTracker."""

    def test_concurrent_position_access_no_race_condition(self) -> None:
        """Concurrent access to positions should not cause race conditions."""
        # Arrange
        tracker = LivePositionTracker()
        errors: list[Exception] = []
        positions_opened = 0
        positions_closed = 0

        def open_positions() -> None:
            nonlocal positions_opened
            for i in range(50):
                try:
                    position = LivePosition(
                        symbol="BTCUSDT",
                        side=PositionSide.LONG,
                        size=0.1,
                        entry_price=50000.0 + i,
                        entry_time=datetime.now(timezone.utc),
                        order_id=f"open-{i}",
                    )
                    tracker.open_position(position)
                    positions_opened += 1
                except Exception as e:
                    errors.append(e)

        def close_positions() -> None:
            nonlocal positions_closed
            time.sleep(0.01)  # Let some positions open first
            for i in range(50):
                try:
                    order_id = f"open-{i}"
                    if tracker.has_position(order_id):
                        tracker.close_position(
                            order_id=order_id,
                            exit_price=51000.0,
                            exit_reason="test",
                            basis_balance=1000.0,
                        )
                        positions_closed += 1
                except Exception as e:
                    errors.append(e)

        def read_positions() -> None:
            for _ in range(100):
                try:
                    _ = tracker.positions
                    _ = tracker.position_count
                    _ = tracker.position_db_ids
                except Exception as e:
                    errors.append(e)

        # Act: Run concurrent operations
        threads = [
            threading.Thread(target=open_positions),
            threading.Thread(target=close_positions),
            threading.Thread(target=read_positions),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Assert: No errors occurred
        assert len(errors) == 0, f"Race condition errors: {errors}"

    def test_positions_property_returns_copy(self) -> None:
        """The positions property should return a copy to prevent external modification."""
        # Arrange
        tracker = LivePositionTracker()
        position = LivePosition(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            size=0.1,
            entry_price=50000.0,
            entry_time=datetime.now(timezone.utc),
            order_id="test-1",
        )
        tracker.open_position(position)

        # Act: Get positions and try to modify
        positions = tracker.positions
        positions.clear()  # This should not affect the internal state

        # Assert: Internal state is unchanged
        assert tracker.position_count == 1
        assert tracker.has_position("test-1")


class TestDailyPnLTracking:
    """Test daily P&L tracking in LiveEventLogger."""

    def test_daily_pnl_initialized_on_first_snapshot(self) -> None:
        """Daily P&L should be 0 on first snapshot (balance = day start balance)."""
        # Arrange
        db_manager = MagicMock()
        logger = LiveEventLogger(
            db_manager=db_manager,
            log_to_database=True,
            session_id=1,
        )

        # Act
        logger.log_account_snapshot(
            balance=1000.0,
            positions={},
            total_pnl=0.0,
            peak_balance=1000.0,
        )

        # Assert: daily_pnl should be 0 on first call
        call_kwargs = db_manager.log_account_snapshot.call_args.kwargs
        assert call_kwargs["daily_pnl"] == 0.0

    def test_daily_pnl_calculated_from_day_start_balance(self) -> None:
        """Daily P&L should be calculated from day start balance."""
        # Arrange
        db_manager = MagicMock()
        logger = LiveEventLogger(
            db_manager=db_manager,
            log_to_database=True,
            session_id=1,
        )

        # First snapshot sets day start balance
        logger.log_account_snapshot(
            balance=1000.0,
            positions={},
            total_pnl=0.0,
            peak_balance=1000.0,
        )

        # Act: Second snapshot with different balance
        logger.log_account_snapshot(
            balance=1050.0,  # +50 from start
            positions={},
            total_pnl=50.0,
            peak_balance=1050.0,
        )

        # Assert: daily_pnl should be the difference from day start
        call_kwargs = db_manager.log_account_snapshot.call_args.kwargs
        assert call_kwargs["daily_pnl"] == pytest.approx(50.0, rel=0.01)


class TestEntryBalanceBasis:
    """Test entry balance basis uses post-fee balance for parity with backtest."""

    def test_entry_balance_subtracts_entry_fee(self) -> None:
        """Entry balance should be balance minus entry fee for backtest parity."""
        # Arrange
        execution_engine = MagicMock()
        entry_fee = 1.23
        execution_engine.execute_entry.return_value = MagicMock(
            success=True,
            executed_price=100.0,
            order_id="entry-1",
            quantity=0.1,
            entry_fee=entry_fee,
            slippage_cost=0.0,
        )
        entry_handler = LiveEntryHandler(execution_engine=execution_engine)
        signal = LiveEntrySignal(
            should_enter=True,
            side=PositionSide.LONG,
            size_fraction=0.1,
        )
        balance = 1000.0

        # Act
        result = entry_handler.execute_entry(
            signal=signal,
            symbol="BTCUSDT",
            current_price=100.0,
            balance=balance,
        )

        # Assert: entry_balance = balance - entry_fee (matches backtest behavior)
        assert result.executed is True
        assert result.position is not None
        assert result.position.entry_balance == balance - entry_fee

    def test_daily_pnl_resets_on_date_change(self) -> None:
        """Daily P&L should reset when the trading date changes."""
        # Arrange
        db_manager = MagicMock()
        logger = LiveEventLogger(
            db_manager=db_manager,
            log_to_database=True,
            session_id=1,
        )

        # First snapshot
        logger.log_account_snapshot(
            balance=1000.0,
            positions={},
            total_pnl=0.0,
            peak_balance=1000.0,
        )

        # Simulate date change
        tomorrow = date.today()
        with patch("src.engines.live.logging.event_logger.date") as mock_date:
            mock_date.today.return_value = tomorrow
            mock_date.side_effect = lambda *args, **kwargs: date(*args, **kwargs)

            # Act: First snapshot of new day
            logger._current_trading_date = date(2020, 1, 1)  # Force date change
            logger.log_account_snapshot(
                balance=1050.0,  # Previous day ended at 1050
                positions={},
                total_pnl=50.0,
                peak_balance=1050.0,
            )

        # Assert: Daily P&L should be 0 (new day starts fresh at current balance)
        call_kwargs = db_manager.log_account_snapshot.call_args.kwargs
        assert call_kwargs["daily_pnl"] == pytest.approx(0.0, rel=0.01)

    def test_set_day_start_balance_for_recovery(self) -> None:
        """set_day_start_balance should allow recovery of day start from database."""
        # Arrange
        db_manager = MagicMock()
        logger = LiveEventLogger(
            db_manager=db_manager,
            log_to_database=True,
            session_id=1,
        )

        # Simulate recovery: set day start balance from DB
        logger.set_day_start_balance(900.0)

        # Act: Log snapshot with current balance
        logger.log_account_snapshot(
            balance=950.0,  # +50 from recovered day start
            positions={},
            total_pnl=50.0,
            peak_balance=950.0,
        )

        # Assert: Daily P&L calculated from recovered day start
        call_kwargs = db_manager.log_account_snapshot.call_args.kwargs
        assert call_kwargs["daily_pnl"] == pytest.approx(50.0, rel=0.01)
