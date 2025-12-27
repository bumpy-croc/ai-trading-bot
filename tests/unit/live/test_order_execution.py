"""
Tests for Live Trading Engine order execution methods.

This module tests the following methods:
- LiveExecutionEngine._execute_live_order: Executes market orders via exchange interface
- _close_order: Closes positions via opposite-side market orders
- _handle_order_fill: Handles order fill callbacks (including stop-loss detection)
- _handle_partial_fill: Handles partial fill callbacks (with SL warning)
- _handle_order_cancel: Handles order cancellation (with phantom position removal)
- _reconcile_positions_with_exchange: Reconciles local state with exchange on startup
"""

from decimal import Decimal
from unittest.mock import Mock, MagicMock, patch

import pytest

from src.data_providers.exchange_interface import (
    OrderSide,
    OrderType,
    OrderStatus as ExchangeOrderStatus,
)
from src.engines.live.execution.entry_handler import LiveEntrySignal
from src.engines.live.trading_engine import LiveTradingEngine, Position, PositionSide


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def mock_data_provider():
    """Create a mock data provider."""
    provider = Mock()
    provider.get_current_price.return_value = 50000.0
    return provider


@pytest.fixture
def mock_exchange():
    """Create a mock exchange interface."""
    exchange = Mock()
    exchange.get_symbol_info.return_value = {
        "step_size": 0.00001,
        "min_qty": 0.00001,
        "min_notional": 10.0,
    }
    exchange.place_order.return_value = "order123"
    exchange.cancel_order.return_value = True
    return exchange


@pytest.fixture
def mock_order_tracker():
    """Create a mock order tracker."""
    tracker = Mock()
    return tracker


@pytest.fixture
def engine_with_exchange(mock_data_provider, mock_exchange, mock_order_tracker):
    """Create a LiveTradingEngine with mocked exchange and order tracker."""
    with patch("src.engines.live.trading_engine.DatabaseManager"):
        engine = LiveTradingEngine(
            strategy=Mock(),
            data_provider=mock_data_provider,
            initial_balance=10000.0,
            enable_live_trading=False,
        )
        engine.exchange_interface = mock_exchange
        engine.order_tracker = mock_order_tracker
        engine.enable_live_trading = True
        engine.live_execution_engine.exchange_interface = mock_exchange
        engine.live_execution_engine.enable_live_trading = True
        return engine


@pytest.fixture
def sample_position():
    """Create a sample position for testing."""
    return Position(
        symbol="BTCUSDT",
        side=PositionSide.LONG,
        size=0.1,  # 10% of balance
        entry_price=50000.0,
        stop_loss=48000.0,
        take_profit=55000.0,
        entry_time=None,
        order_id="entry_order_123",
        entry_balance=Decimal("10000.0"),
        stop_loss_order_id="sl_order_456",
    )


# ============================================================================
# Tests for LiveExecutionEngine order execution
# ============================================================================


class TestExecuteOrder:
    """Tests for the LiveExecutionEngine _execute_live_order method."""

    def test_execute_order_no_exchange_interface(self, mock_data_provider):
        """Order fails when exchange interface is not initialized."""
        with patch("src.engines.live.trading_engine.DatabaseManager"):
            engine = LiveTradingEngine(
                strategy=Mock(),
                data_provider=mock_data_provider,
                initial_balance=10000.0,
            )
            engine.exchange_interface = None

            result = engine.live_execution_engine._execute_live_order(
                "BTCUSDT", PositionSide.LONG, 1000.0, 50000.0
            )

            assert result is None

    def test_execute_order_invalid_price_zero(self, engine_with_exchange):
        """Order fails with zero price."""
        result = engine_with_exchange.live_execution_engine._execute_live_order(
            "BTCUSDT", PositionSide.LONG, 1000.0, 0.0
        )

        assert result is None

    def test_execute_order_invalid_price_negative(self, engine_with_exchange):
        """Order fails with negative price."""
        result = engine_with_exchange.live_execution_engine._execute_live_order(
            "BTCUSDT", PositionSide.LONG, 1000.0, -100.0
        )

        assert result is None

    def test_execute_order_long_success(self, engine_with_exchange, mock_exchange):
        """Long order is placed correctly."""
        result = engine_with_exchange.live_execution_engine._execute_live_order(
            "BTCUSDT", PositionSide.LONG, 1000.0, 50000.0
        )

        assert result == "order123"
        mock_exchange.place_order.assert_called_once()
        call_kwargs = mock_exchange.place_order.call_args.kwargs
        assert call_kwargs["symbol"] == "BTCUSDT"
        assert call_kwargs["side"] == OrderSide.BUY
        assert call_kwargs["order_type"] == OrderType.MARKET
        # quantity = 1000 / 50000 = 0.02
        assert abs(call_kwargs["quantity"] - 0.02) < 0.0001

    def test_execute_order_short_success(self, engine_with_exchange, mock_exchange):
        """Short order is placed correctly."""
        result = engine_with_exchange.live_execution_engine._execute_live_order(
            "BTCUSDT", PositionSide.SHORT, 1000.0, 50000.0
        )

        assert result == "order123"
        call_kwargs = mock_exchange.place_order.call_args.kwargs
        assert call_kwargs["side"] == OrderSide.SELL

    def test_execute_order_quantity_rounded_to_step_size(
        self, engine_with_exchange, mock_exchange
    ):
        """Quantity is rounded to step size."""
        mock_exchange.get_symbol_info.return_value = {
            "step_size": 0.001,
            "min_qty": 0.001,
            "min_notional": 10.0,
        }

        engine_with_exchange.live_execution_engine._execute_live_order(
            "BTCUSDT", PositionSide.LONG, 1234.56, 50000.0
        )

        call_kwargs = mock_exchange.place_order.call_args.kwargs
        # 1234.56 / 50000 = 0.0246912, rounded to 0.025
        assert call_kwargs["quantity"] == 0.025

    def test_execute_order_below_min_quantity(self, engine_with_exchange, mock_exchange):
        """Order fails if quantity is below minimum."""
        mock_exchange.get_symbol_info.return_value = {
            "step_size": 0.001,
            "min_qty": 1.0,  # High minimum
            "min_notional": 10.0,
        }

        result = engine_with_exchange.live_execution_engine._execute_live_order(
            "BTCUSDT", PositionSide.LONG, 100.0, 50000.0  # Only 0.002 BTC
        )

        assert result is None
        mock_exchange.place_order.assert_not_called()

    def test_execute_order_below_min_notional(self, engine_with_exchange, mock_exchange):
        """Order fails if value is below minimum notional."""
        mock_exchange.get_symbol_info.return_value = {
            "step_size": 0.00001,
            "min_qty": 0.00001,
            "min_notional": 100.0,  # High minimum notional
        }

        result = engine_with_exchange.live_execution_engine._execute_live_order(
            "BTCUSDT", PositionSide.LONG, 50.0, 50000.0  # Only $50 value
        )

        assert result is None
        mock_exchange.place_order.assert_not_called()

    def test_execute_order_tracks_order(
        self, engine_with_exchange, mock_order_tracker
    ):
        """Placed order is tracked via order tracker."""
        entry_signal = LiveEntrySignal(
            should_enter=True,
            side=PositionSide.LONG,
            size_fraction=0.1,
            stop_loss=None,
            take_profit=None,
        )
        engine_with_exchange._execute_entry_signal(
            entry_signal,
            symbol="BTCUSDT",
            current_price=50000.0,
        )

        mock_order_tracker.track_order.assert_called_once_with("order123", "BTCUSDT")

    def test_execute_order_place_fails(self, engine_with_exchange, mock_exchange):
        """Order returns None when exchange place_order fails."""
        mock_exchange.place_order.return_value = None

        result = engine_with_exchange.live_execution_engine._execute_live_order(
            "BTCUSDT", PositionSide.LONG, 1000.0, 50000.0
        )

        assert result is None

    def test_execute_order_no_symbol_info(self, engine_with_exchange, mock_exchange):
        """Order succeeds without symbol info (uses defaults)."""
        mock_exchange.get_symbol_info.return_value = None

        result = engine_with_exchange.live_execution_engine._execute_live_order(
            "BTCUSDT", PositionSide.LONG, 1000.0, 50000.0
        )

        assert result == "order123"


# ============================================================================
# Tests for _close_order
# ============================================================================


class TestCloseOrder:
    """Tests for the _close_order method."""

    def test_close_order_no_exchange_interface(self, mock_data_provider, sample_position):
        """Close fails when exchange interface is not initialized."""
        with patch("src.engines.live.trading_engine.DatabaseManager"):
            engine = LiveTradingEngine(
                strategy=Mock(),
                data_provider=mock_data_provider,
                initial_balance=10000.0,
            )
            engine.exchange_interface = None

            result = engine._close_order(sample_position)

            assert result is False

    def test_close_order_no_current_price(
        self, engine_with_exchange, mock_data_provider, sample_position
    ):
        """Close fails when current price is unavailable."""
        mock_data_provider.get_current_price.return_value = None

        result = engine_with_exchange._close_order(sample_position)

        assert result is False

    def test_close_order_long_position(
        self, engine_with_exchange, mock_exchange, sample_position
    ):
        """Closing a long position places a sell order."""
        sample_position.side = PositionSide.LONG

        result = engine_with_exchange._close_order(sample_position)

        assert result is True
        call_kwargs = mock_exchange.place_order.call_args.kwargs
        assert call_kwargs["side"] == OrderSide.SELL
        assert call_kwargs["order_type"] == OrderType.MARKET

    def test_close_order_short_position(
        self, engine_with_exchange, mock_exchange, sample_position
    ):
        """Closing a short position places a buy order."""
        sample_position.side = PositionSide.SHORT

        result = engine_with_exchange._close_order(sample_position)

        assert result is True
        call_kwargs = mock_exchange.place_order.call_args.kwargs
        assert call_kwargs["side"] == OrderSide.BUY

    def test_close_order_uses_entry_balance(
        self, engine_with_exchange, mock_exchange, sample_position
    ):
        """Close order uses entry_balance for quantity calculation."""
        sample_position.entry_balance = Decimal("20000.0")  # Different from current
        sample_position.size = 0.1  # 10% of balance
        engine_with_exchange.current_balance = 15000.0  # Different current balance

        engine_with_exchange._close_order(sample_position)

        call_kwargs = mock_exchange.place_order.call_args.kwargs
        # quantity = (0.1 * 20000) / 50000 = 0.04
        assert abs(call_kwargs["quantity"] - 0.04) < 0.0001

    def test_close_order_uses_current_size_if_partial(
        self, engine_with_exchange, mock_exchange, sample_position
    ):
        """Close order uses current_size if position was partially exited."""
        sample_position.size = 0.1
        sample_position.current_size = 0.05  # Half already exited
        sample_position.entry_balance = Decimal("10000.0")

        engine_with_exchange._close_order(sample_position)

        call_kwargs = mock_exchange.place_order.call_args.kwargs
        # quantity = (0.05 * 10000) / 50000 = 0.01
        assert abs(call_kwargs["quantity"] - 0.01) < 0.0001

    def test_close_order_place_fails(
        self, engine_with_exchange, mock_exchange, sample_position
    ):
        """Close returns False when exchange place_order fails."""
        mock_exchange.place_order.return_value = None

        result = engine_with_exchange._close_order(sample_position)

        assert result is False

    def test_close_order_fallback_when_entry_balance_none(
        self, engine_with_exchange, mock_exchange, sample_position
    ):
        """Close order falls back to current_balance when entry_balance is None."""
        sample_position.entry_balance = None
        sample_position.size = 0.1  # 10% of balance
        engine_with_exchange.current_balance = 15000.0

        engine_with_exchange._close_order(sample_position)

        call_kwargs = mock_exchange.place_order.call_args.kwargs
        # quantity = (0.1 * 15000) / 50000 = 0.03
        assert abs(call_kwargs["quantity"] - 0.03) < 0.0001

    def test_close_order_fallback_when_entry_balance_zero(
        self, engine_with_exchange, mock_exchange, sample_position
    ):
        """Close order falls back to current_balance when entry_balance is zero."""
        sample_position.entry_balance = Decimal("0.0")
        sample_position.size = 0.1  # 10% of balance
        engine_with_exchange.current_balance = 12000.0

        engine_with_exchange._close_order(sample_position)

        call_kwargs = mock_exchange.place_order.call_args.kwargs
        # quantity = (0.1 * 12000) / 50000 = 0.024
        assert abs(call_kwargs["quantity"] - 0.024) < 0.0001

    def test_close_order_fallback_when_entry_balance_negative(
        self, engine_with_exchange, mock_exchange, sample_position
    ):
        """Close order falls back to current_balance when entry_balance is negative."""
        sample_position.entry_balance = Decimal("-100.0")
        sample_position.size = 0.1  # 10% of balance
        engine_with_exchange.current_balance = 18000.0

        engine_with_exchange._close_order(sample_position)

        call_kwargs = mock_exchange.place_order.call_args.kwargs
        # quantity = (0.1 * 18000) / 50000 = 0.036
        assert abs(call_kwargs["quantity"] - 0.036) < 0.0001


# ============================================================================
# Tests for _handle_order_fill
# ============================================================================


class TestHandleOrderFill:
    """Tests for the _handle_order_fill method."""

    def test_handle_fill_logs_event(self, engine_with_exchange):
        """Fill callback logs the event."""
        with patch("src.engines.live.trading_engine.log_order_event") as mock_log:
            engine_with_exchange._handle_order_fill(
                "order123", "BTCUSDT", 1.5, 50000.0
            )

            mock_log.assert_called_once()
            assert mock_log.call_args.args[0] == "order_filled"

    def test_handle_fill_detects_stop_loss_fill(
        self, engine_with_exchange, sample_position
    ):
        """Fill callback detects and handles stop-loss fills."""
        sample_position.stop_loss_order_id = "sl_order_456"
        engine_with_exchange.positions["entry_order_123"] = sample_position

        with patch.object(
            engine_with_exchange, "_close_position"
        ) as mock_close:
            engine_with_exchange._handle_order_fill(
                "sl_order_456", "BTCUSDT", 0.02, 48000.0
            )

            mock_close.assert_called_once()
            call_args = mock_close.call_args
            assert call_args.args[0] == sample_position
            assert call_args.kwargs["reason"] == "stop_loss"
            assert call_args.kwargs["limit_price"] == 48000.0

    def test_handle_fill_ignores_entry_orders(
        self, engine_with_exchange, sample_position
    ):
        """Fill callback does not trigger close for entry order fills."""
        engine_with_exchange.positions["entry_order_123"] = sample_position

        with patch.object(
            engine_with_exchange, "_close_position"
        ) as mock_close:
            engine_with_exchange._handle_order_fill(
                "entry_order_123", "BTCUSDT", 0.02, 50000.0
            )

            mock_close.assert_not_called()

    def test_handle_fill_no_matching_position(self, engine_with_exchange):
        """Fill callback handles fill with no matching position."""
        # Should not raise, just log
        engine_with_exchange._handle_order_fill(
            "unknown_order", "BTCUSDT", 1.0, 50000.0
        )


# ============================================================================
# Tests for _handle_partial_fill
# ============================================================================


class TestHandlePartialFill:
    """Tests for the _handle_partial_fill method."""

    def test_handle_partial_fill_logs_event(self, engine_with_exchange):
        """Partial fill callback logs the event."""
        with patch("src.engines.live.trading_engine.log_order_event") as mock_log:
            engine_with_exchange._handle_partial_fill(
                "order123", "BTCUSDT", 0.5, 50000.0
            )

            mock_log.assert_called_once()
            assert mock_log.call_args.args[0] == "partial_fill"

    def test_handle_partial_fill_logs_critical_for_sl(
        self, engine_with_exchange, sample_position, caplog
    ):
        """Partial fill of stop-loss logs critical warning."""
        sample_position.stop_loss_order_id = "sl_order_456"
        engine_with_exchange.positions["entry_order_123"] = sample_position

        import logging

        with caplog.at_level(logging.CRITICAL):
            engine_with_exchange._handle_partial_fill(
                "sl_order_456", "BTCUSDT", 0.5, 48000.0
            )

            assert "PARTIAL STOP-LOSS FILL" in caplog.text

    def test_handle_partial_fill_logs_sl_warning_event(
        self, engine_with_exchange, sample_position
    ):
        """Partial SL fill logs specialized event."""
        sample_position.stop_loss_order_id = "sl_order_456"
        engine_with_exchange.positions["entry_order_123"] = sample_position

        with patch("src.engines.live.trading_engine.log_order_event") as mock_log:
            engine_with_exchange._handle_partial_fill(
                "sl_order_456", "BTCUSDT", 0.5, 48000.0
            )

            # Should have two calls: partial_fill and partial_sl_fill_warning
            assert mock_log.call_count == 2
            event_types = [call.args[0] for call in mock_log.call_args_list]
            assert "partial_sl_fill_warning" in event_types

    def test_handle_partial_fill_no_matching_position(self, engine_with_exchange):
        """Partial fill callback handles fill with no matching position."""
        # Should not raise
        engine_with_exchange._handle_partial_fill(
            "unknown_order", "BTCUSDT", 0.5, 50000.0
        )


# ============================================================================
# Tests for _handle_order_cancel
# ============================================================================


class TestHandleOrderCancel:
    """Tests for the _handle_order_cancel method."""

    def test_handle_cancel_logs_event(self, engine_with_exchange):
        """Cancel callback logs the event."""
        with patch("src.engines.live.trading_engine.log_order_event") as mock_log:
            engine_with_exchange._handle_order_cancel("order123", "BTCUSDT")

            mock_log.assert_called_once()
            assert mock_log.call_args.args[0] == "order_cancelled"

    def test_handle_cancel_removes_phantom_position(
        self, engine_with_exchange, sample_position
    ):
        """Cancel callback removes phantom position if entry order cancelled."""
        engine_with_exchange.positions["entry_order_123"] = sample_position

        engine_with_exchange._handle_order_cancel("entry_order_123", "BTCUSDT")

        assert "entry_order_123" not in engine_with_exchange.positions

    def test_handle_cancel_thread_safe_with_pop(
        self, engine_with_exchange, sample_position
    ):
        """Cancel uses atomic pop() for thread safety."""
        engine_with_exchange.positions["entry_order_123"] = sample_position

        # Calling twice should not raise - pop returns None on second call
        engine_with_exchange._handle_order_cancel("entry_order_123", "BTCUSDT")
        engine_with_exchange._handle_order_cancel("entry_order_123", "BTCUSDT")

        assert "entry_order_123" not in engine_with_exchange.positions

    def test_handle_cancel_no_phantom_position(self, engine_with_exchange):
        """Cancel callback handles cancel with no matching position."""
        # Should not raise
        engine_with_exchange._handle_order_cancel("unknown_order", "BTCUSDT")


# ============================================================================
# Tests for _reconcile_positions_with_exchange
# ============================================================================


class TestReconcilePositionsWithExchange:
    """Tests for the _reconcile_positions_with_exchange method."""

    def test_reconcile_no_exchange_skipped(self, mock_data_provider):
        """Reconciliation skipped when no exchange interface."""
        with patch("src.engines.live.trading_engine.DatabaseManager"):
            engine = LiveTradingEngine(
                strategy=Mock(),
                data_provider=mock_data_provider,
                initial_balance=10000.0,
            )
            engine.exchange_interface = None

            # Should not raise
            engine._reconcile_positions_with_exchange()

    def test_reconcile_no_live_trading_skipped(self, engine_with_exchange):
        """Reconciliation skipped when not live trading."""
        engine_with_exchange.enable_live_trading = False

        # Should not raise
        engine_with_exchange._reconcile_positions_with_exchange()

    def test_reconcile_no_positions_skipped(self, engine_with_exchange):
        """Reconciliation skipped when no local positions."""
        engine_with_exchange.positions = {}

        # Should not raise
        engine_with_exchange._reconcile_positions_with_exchange()

    def test_reconcile_detects_filled_stop_loss(
        self, engine_with_exchange, mock_exchange, sample_position
    ):
        """Reconciliation detects stop-loss orders that filled offline."""
        engine_with_exchange.positions["entry_order_123"] = sample_position
        sample_position.stop_loss_order_id = "sl_order_456"
        engine_with_exchange.current_balance = 10000.0

        # SL order is not in open orders (was filled)
        mock_exchange.get_open_orders.return_value = []

        # SL order status shows FILLED
        mock_sl_order = Mock()
        mock_sl_order.status = ExchangeOrderStatus.FILLED
        mock_sl_order.average_price = 48000.0
        mock_exchange.get_order.return_value = mock_sl_order

        engine_with_exchange._reconcile_positions_with_exchange()

        # Position should be removed
        assert "entry_order_123" not in engine_with_exchange.positions

    def test_reconcile_updates_balance_for_sl(
        self, engine_with_exchange, mock_exchange, sample_position
    ):
        """Reconciliation updates balance based on SL exit price, including fees."""
        sample_position.entry_balance = Decimal("10000.0")
        sample_position.size = 0.1  # 10%
        sample_position.entry_price = 50000.0
        sample_position.side = PositionSide.LONG
        engine_with_exchange.positions["entry_order_123"] = sample_position
        engine_with_exchange.current_balance = 10000.0

        mock_exchange.get_open_orders.return_value = []

        mock_sl_order = Mock()
        mock_sl_order.status = ExchangeOrderStatus.FILLED
        mock_sl_order.average_price = 48000.0  # 4% loss
        mock_exchange.get_order.return_value = mock_sl_order

        engine_with_exchange._reconcile_positions_with_exchange()

        # PnL = ((48000 - 50000) / 50000) * 0.1 * 10000 = -40
        # Exit notional = 10000 * 0.1 * (48000/50000) = 960
        # Exit fee = 960 * 0.001 = 0.96
        # Realized PnL = -40 - 0.96 = -40.96
        expected_balance = 10000.0 - 40.96
        assert abs(engine_with_exchange.current_balance - expected_balance) < 0.01

    def test_reconcile_uses_entry_balance_not_current(
        self, engine_with_exchange, mock_exchange, sample_position
    ):
        """Reconciliation uses entry_balance for PnL, not current_balance."""
        sample_position.entry_balance = Decimal("8000.0")  # Different from current
        sample_position.size = 0.1
        sample_position.entry_price = 50000.0
        sample_position.side = PositionSide.LONG
        engine_with_exchange.positions["entry_order_123"] = sample_position
        engine_with_exchange.current_balance = 12000.0  # Different

        mock_exchange.get_open_orders.return_value = []

        mock_sl_order = Mock()
        mock_sl_order.status = ExchangeOrderStatus.FILLED
        mock_sl_order.average_price = 48000.0
        mock_exchange.get_order.return_value = mock_sl_order

        engine_with_exchange._reconcile_positions_with_exchange()

        # PnL = ((48000 - 50000) / 50000) * 0.1 * 8000 = -32
        # Exit notional = 8000 * 0.1 * (48000/50000) = 768
        # Exit fee = 768 * 0.001 = 0.768
        # Realized PnL = -32 - 0.768 = -32.768
        expected_balance = 12000.0 - 32.768
        assert abs(engine_with_exchange.current_balance - expected_balance) < 0.01

    def test_reconcile_handles_short_position_pnl(
        self, engine_with_exchange, mock_exchange, sample_position
    ):
        """Reconciliation calculates PnL correctly for short positions."""
        sample_position.entry_balance = Decimal("10000.0")
        sample_position.size = 0.1
        sample_position.entry_price = 50000.0
        sample_position.side = PositionSide.SHORT
        engine_with_exchange.positions["entry_order_123"] = sample_position
        engine_with_exchange.current_balance = 10000.0

        mock_exchange.get_open_orders.return_value = []

        mock_sl_order = Mock()
        mock_sl_order.status = ExchangeOrderStatus.FILLED
        mock_sl_order.average_price = 52000.0  # 4% loss for short
        mock_exchange.get_order.return_value = mock_sl_order

        engine_with_exchange._reconcile_positions_with_exchange()

        # PnL = ((50000 - 52000) / 50000) * 0.1 * 10000 = -40
        # Exit notional = 10000 * 0.1 * (52000/50000) = 1040
        # Exit fee = 1040 * 0.001 = 1.04
        # Realized PnL = -40 - 1.04 = -41.04
        expected_balance = 10000.0 - 41.04
        assert abs(engine_with_exchange.current_balance - expected_balance) < 0.01

    def test_reconcile_sl_still_active_no_action(
        self, engine_with_exchange, mock_exchange, sample_position
    ):
        """Reconciliation takes no action if SL order is still active."""
        engine_with_exchange.positions["entry_order_123"] = sample_position
        sample_position.stop_loss_order_id = "sl_order_456"
        initial_balance = engine_with_exchange.current_balance

        # SL order is still in open orders
        mock_sl_order = Mock()
        mock_sl_order.order_id = "sl_order_456"
        mock_exchange.get_open_orders.return_value = [mock_sl_order]

        engine_with_exchange._reconcile_positions_with_exchange()

        # Position should remain
        assert "entry_order_123" in engine_with_exchange.positions
        assert engine_with_exchange.current_balance == initial_balance

    def test_reconcile_handles_api_error(
        self, engine_with_exchange, mock_exchange, sample_position
    ):
        """Reconciliation handles API errors gracefully."""
        engine_with_exchange.positions["entry_order_123"] = sample_position
        mock_exchange.get_open_orders.side_effect = Exception("API error")

        # Should not raise
        engine_with_exchange._reconcile_positions_with_exchange()

        # Position should remain (couldn't verify)
        assert "entry_order_123" in engine_with_exchange.positions

    def test_reconcile_fallback_when_entry_balance_none(
        self, engine_with_exchange, mock_exchange, sample_position
    ):
        """Reconciliation falls back to current_balance when entry_balance is None."""
        sample_position.entry_balance = None
        sample_position.size = 0.1
        sample_position.entry_price = 50000.0
        sample_position.side = PositionSide.LONG
        engine_with_exchange.positions["entry_order_123"] = sample_position
        engine_with_exchange.current_balance = 8000.0  # Current balance used as fallback

        mock_exchange.get_open_orders.return_value = []

        mock_sl_order = Mock()
        mock_sl_order.status = ExchangeOrderStatus.FILLED
        mock_sl_order.average_price = 48000.0  # 4% loss
        mock_exchange.get_order.return_value = mock_sl_order

        engine_with_exchange._reconcile_positions_with_exchange()

        # PnL = ((48000 - 50000) / 50000) * 0.1 * 8000 = -32
        # Exit notional = 8000 * 0.1 * (48000/50000) = 768
        # Exit fee = 768 * 0.001 = 0.768
        # Realized PnL = -32 - 0.768 = -32.768
        expected_balance = 8000.0 - 32.768
        assert abs(engine_with_exchange.current_balance - expected_balance) < 0.01

    def test_reconcile_fallback_when_entry_balance_zero(
        self, engine_with_exchange, mock_exchange, sample_position
    ):
        """Reconciliation falls back to current_balance when entry_balance is zero."""
        sample_position.entry_balance = Decimal("0.0")
        sample_position.size = 0.1
        sample_position.entry_price = 50000.0
        sample_position.side = PositionSide.LONG
        engine_with_exchange.positions["entry_order_123"] = sample_position
        engine_with_exchange.current_balance = 6000.0

        mock_exchange.get_open_orders.return_value = []

        mock_sl_order = Mock()
        mock_sl_order.status = ExchangeOrderStatus.FILLED
        mock_sl_order.average_price = 48000.0  # 4% loss
        mock_exchange.get_order.return_value = mock_sl_order

        engine_with_exchange._reconcile_positions_with_exchange()

        # PnL = ((48000 - 50000) / 50000) * 0.1 * 6000 = -24
        # Exit notional = 6000 * 0.1 * (48000/50000) = 576
        # Exit fee = 576 * 0.001 = 0.576
        # Realized PnL = -24 - 0.576 = -24.576
        expected_balance = 6000.0 - 24.576
        assert abs(engine_with_exchange.current_balance - expected_balance) < 0.01

    def test_reconcile_fallback_when_entry_balance_negative(
        self, engine_with_exchange, mock_exchange, sample_position
    ):
        """Reconciliation falls back to current_balance when entry_balance is negative."""
        sample_position.entry_balance = Decimal("-500.0")
        sample_position.size = 0.1
        sample_position.entry_price = 50000.0
        sample_position.side = PositionSide.SHORT
        engine_with_exchange.positions["entry_order_123"] = sample_position
        engine_with_exchange.current_balance = 9000.0

        mock_exchange.get_open_orders.return_value = []

        mock_sl_order = Mock()
        mock_sl_order.status = ExchangeOrderStatus.FILLED
        mock_sl_order.average_price = 52000.0  # 4% loss for short
        mock_exchange.get_order.return_value = mock_sl_order

        engine_with_exchange._reconcile_positions_with_exchange()

        # PnL = ((50000 - 52000) / 50000) * 0.1 * 9000 = -36
        # Exit notional = 9000 * 0.1 * (52000/50000) = 936
        # Exit fee = 936 * 0.001 = 0.936
        # Realized PnL = -36 - 0.936 = -36.936
        expected_balance = 9000.0 - 36.936
        assert abs(engine_with_exchange.current_balance - expected_balance) < 0.01


# ============================================================================
# Integration Tests
# ============================================================================


class TestOrderExecutionIntegration:
    """Integration tests for order execution workflow."""

    def test_full_order_lifecycle_entry_to_sl_fill(
        self, engine_with_exchange, mock_exchange, mock_order_tracker
    ):
        """Test complete order lifecycle from entry to stop-loss fill."""
        # 1. Execute entry order
        entry_signal = LiveEntrySignal(
            should_enter=True,
            side=PositionSide.LONG,
            size_fraction=0.1,
            stop_loss=None,
            take_profit=None,
        )
        engine_with_exchange._execute_entry_signal(
            entry_signal,
            symbol="BTCUSDT",
            current_price=50000.0,
        )
        entry_order_id = next(iter(engine_with_exchange.positions))
        assert entry_order_id == "order123"
        mock_order_tracker.track_order.assert_called_with("order123", "BTCUSDT")

        # 2. Attach SL order ID to existing position
        position = engine_with_exchange.positions[entry_order_id]
        position.stop_loss_order_id = "sl_order_789"

        # 3. Simulate SL fill callback
        with patch.object(engine_with_exchange, "_close_position") as mock_close:
            engine_with_exchange._handle_order_fill(
                "sl_order_789", "BTCUSDT", 0.02, 48000.0
            )

            mock_close.assert_called_once()
            assert mock_close.call_args.kwargs["reason"] == "stop_loss"

    def test_order_cancelled_removes_phantom_position_thread_safe(
        self, engine_with_exchange
    ):
        """Test that cancelled order removes phantom position atomically."""
        import threading

        # Create position
        position = Position(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            size=0.1,
            entry_price=50000.0,
            stop_loss=48000.0,
            take_profit=55000.0,
            entry_time=None,
            order_id="order123",
            entry_balance=Decimal("10000.0"),
        )
        engine_with_exchange.positions["order123"] = position

        errors = []

        def cancel_handler():
            try:
                for _ in range(10):
                    engine_with_exchange._handle_order_cancel("order123", "BTCUSDT")
            except Exception as e:
                errors.append(e)

        # Run multiple threads trying to cancel same order
        threads = [threading.Thread(target=cancel_handler) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # No errors should occur
        assert len(errors) == 0
        # Position should be removed
        assert "order123" not in engine_with_exchange.positions
