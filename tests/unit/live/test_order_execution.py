"""
Tests for live trading order execution methods.

This module tests the following components:
- LiveExecutionEngine entry/exit order placement
- _handle_order_fill: Handles order fill callbacks (including stop-loss detection)
- _handle_partial_fill: Handles partial fill callbacks (with SL warning)
- _handle_order_cancel: Handles order cancellation (with phantom position removal)
- _reconcile_positions_with_exchange: Reconciles local state with exchange on startup
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import MagicMock, Mock, patch

import pytest

from src.data_providers.exchange_interface import (
    OrderSide,
    OrderType,
)
from src.data_providers.exchange_interface import (
    OrderStatus as ExchangeOrderStatus,
)
from src.engines.live.execution.execution_engine import LiveExecutionEngine
from src.engines.live.execution.exit_handler import LiveExitHandler
from src.engines.live.execution.position_tracker import LivePosition, LivePositionTracker
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
    exchange.get_open_orders.return_value = []
    exchange.get_order.return_value = None
    return exchange


@pytest.fixture
def mock_order_tracker():
    """Create a mock order tracker."""
    return Mock()


@pytest.fixture
def execution_engine_with_exchange(mock_exchange):
    """Create a LiveExecutionEngine with mocked exchange."""
    return LiveExecutionEngine(
        fee_rate=0.0,
        slippage_rate=0.0,
        enable_live_trading=True,
        exchange_interface=mock_exchange,
    )


@pytest.fixture
def live_position_tracker():
    """Create a LivePositionTracker for handler tests."""
    return LivePositionTracker()


@pytest.fixture
def live_exit_handler(execution_engine_with_exchange, live_position_tracker):
    """Create a LiveExitHandler for handler tests."""
    return LiveExitHandler(
        execution_engine=execution_engine_with_exchange,
        position_tracker=live_position_tracker,
        risk_manager=None,
    )


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
        entry_time=datetime.now(UTC),
        order_id="entry_order_123",
        entry_balance=Decimal("10000.0"),
        stop_loss_order_id="sl_order_456",
    )


# ============================================================================
# Tests for LiveExecutionEngine entry execution
# ============================================================================


class TestExecuteEntry:
    """Tests for LiveExecutionEngine entry execution."""

    def test_execute_entry_invalid_price_zero(self, execution_engine_with_exchange):
        """Order fails with zero price."""
        result = execution_engine_with_exchange.execute_entry(
            "BTCUSDT", PositionSide.LONG, 0.1, 0.0, 10000.0
        )

        assert result.success is False

    def test_execute_entry_invalid_price_negative(self, execution_engine_with_exchange):
        """Order fails with negative price."""
        result = execution_engine_with_exchange.execute_entry(
            "BTCUSDT", PositionSide.LONG, 0.1, -100.0, 10000.0
        )

        assert result.success is False

    def test_execute_entry_long_success(self, execution_engine_with_exchange, mock_exchange):
        """Long order is placed correctly."""
        result = execution_engine_with_exchange.execute_entry(
            "BTCUSDT", PositionSide.LONG, 0.1, 50000.0, 10000.0
        )

        assert result.success is True
        assert result.order_id == "order123"
        call_kwargs = mock_exchange.place_order.call_args.kwargs
        assert call_kwargs["symbol"] == "BTCUSDT"
        assert call_kwargs["side"] == OrderSide.BUY
        assert call_kwargs["order_type"] == OrderType.MARKET
        # quantity = (0.1 * 10000) / 50000 = 0.02
        assert abs(call_kwargs["quantity"] - 0.02) < 0.0001

    def test_execute_entry_short_success(self, execution_engine_with_exchange, mock_exchange):
        """Short order is placed correctly."""
        result = execution_engine_with_exchange.execute_entry(
            "BTCUSDT", PositionSide.SHORT, 0.1, 50000.0, 10000.0
        )

        assert result.order_id == "order123"
        call_kwargs = mock_exchange.place_order.call_args.kwargs
        assert call_kwargs["side"] == OrderSide.SELL

    def test_execute_entry_quantity_rounded_to_step_size(
        self, execution_engine_with_exchange, mock_exchange
    ):
        """Quantity is rounded to step size."""
        mock_exchange.get_symbol_info.return_value = {
            "step_size": 0.001,
            "min_qty": 0.001,
            "min_notional": 10.0,
        }

        execution_engine_with_exchange.execute_entry(
            "BTCUSDT", PositionSide.LONG, 0.123456, 50000.0, 10000.0
        )

        call_kwargs = mock_exchange.place_order.call_args.kwargs
        # (0.123456 * 10000) / 50000 = 0.0246912, rounded to 0.025
        assert call_kwargs["quantity"] == 0.025

    def test_execute_entry_below_min_quantity(
        self, execution_engine_with_exchange, mock_exchange
    ):
        """Order fails if quantity is below minimum."""
        mock_exchange.get_symbol_info.return_value = {
            "step_size": 0.001,
            "min_qty": 1.0,  # High minimum
            "min_notional": 10.0,
        }

        result = execution_engine_with_exchange.execute_entry(
            "BTCUSDT", PositionSide.LONG, 0.01, 50000.0, 10000.0
        )

        assert result.success is False
        mock_exchange.place_order.assert_not_called()

    def test_execute_entry_below_min_notional(
        self, execution_engine_with_exchange, mock_exchange
    ):
        """Order fails if value is below minimum notional."""
        mock_exchange.get_symbol_info.return_value = {
            "step_size": 0.00001,
            "min_qty": 0.00001,
            "min_notional": 100.0,  # High minimum notional
        }

        result = execution_engine_with_exchange.execute_entry(
            "BTCUSDT", PositionSide.LONG, 0.005, 50000.0, 10000.0
        )

        assert result.success is False
        mock_exchange.place_order.assert_not_called()

    def test_execute_entry_place_fails(
        self, execution_engine_with_exchange, mock_exchange
    ):
        """Order returns failure when exchange place_order fails."""
        mock_exchange.place_order.return_value = None

        result = execution_engine_with_exchange.execute_entry(
            "BTCUSDT", PositionSide.LONG, 0.1, 50000.0, 10000.0
        )

        assert result.success is False

    def test_execute_entry_no_symbol_info(
        self, execution_engine_with_exchange, mock_exchange
    ):
        """Order succeeds without symbol info (uses defaults)."""
        mock_exchange.get_symbol_info.return_value = None

        result = execution_engine_with_exchange.execute_entry(
            "BTCUSDT", PositionSide.LONG, 0.1, 50000.0, 10000.0
        )

        assert result.order_id == "order123"


# ============================================================================
# Tests for LiveExecutionEngine exit execution
# ============================================================================


class TestExecuteExit:
    """Tests for LiveExecutionEngine exit execution."""

    def test_execute_exit_invalid_price(self, execution_engine_with_exchange):
        """Exit fails with invalid price."""
        result = execution_engine_with_exchange.execute_exit(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            order_id="order123",
            base_price=0.0,
            position_notional=1000.0,
        )

        assert result.success is False

    def test_execute_exit_success(self, execution_engine_with_exchange, mock_exchange):
        """Exit order places the correct side."""
        result = execution_engine_with_exchange.execute_exit(
            symbol="BTCUSDT",
            side=PositionSide.SHORT,
            order_id="order123",
            base_price=50000.0,
            position_notional=1000.0,
        )

        assert result.success is True
        call_kwargs = mock_exchange.place_order.call_args.kwargs
        assert call_kwargs["side"] == OrderSide.BUY


# ============================================================================
# Tests for LiveExitHandler filled exits
# ============================================================================


class TestExecuteFilledExit:
    """Tests for LiveExitHandler execute_filled_exit."""

    def test_execute_filled_exit_closes_position(
        self, live_exit_handler, live_position_tracker
    ):
        """Filled exits close the position and return results."""
        position = LivePosition(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            size=0.1,
            entry_price=50000.0,
            entry_time=datetime.now(UTC),
            order_id="entry_order_123",
            entry_balance=10000.0,
        )
        live_position_tracker.track_recovered_position(position, db_id=None)

        result = live_exit_handler.execute_filled_exit(
            position=position,
            exit_reason="stop_loss",
            filled_price=48000.0,
            current_balance=10000.0,
        )

        assert result.success is True
        assert not live_position_tracker.has_position("entry_order_123")


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
        engine_with_exchange.live_position_tracker.track_recovered_position(
            sample_position, db_id=None
        )

        with patch.object(engine_with_exchange, "_execute_exit") as mock_close:
            engine_with_exchange._handle_order_fill(
                "sl_order_456", "BTCUSDT", 0.02, 48000.0
            )

            mock_close.assert_called_once()
            call_args = mock_close.call_args
            assert call_args.args[0] == sample_position
            assert call_args.kwargs["reason"] == "stop_loss"
            assert call_args.kwargs["limit_price"] == 48000.0

    def test_handle_fill_skips_if_position_already_closed(
        self, engine_with_exchange, sample_position
    ):
        """Fill callback skips exit when position is already closed."""
        sample_position.stop_loss_order_id = "sl_order_456"
        engine_with_exchange.live_position_tracker.track_recovered_position(
            sample_position, db_id=None
        )

        with patch.object(engine_with_exchange.live_position_tracker, "has_position") as mock_has:
            mock_has.return_value = False
            with patch.object(engine_with_exchange, "_execute_exit") as mock_close:
                engine_with_exchange._handle_order_fill(
                    "sl_order_456", "BTCUSDT", 0.02, 48000.0
                )

                mock_close.assert_not_called()

    def test_handle_fill_ignores_entry_orders(
        self, engine_with_exchange, sample_position
    ):
        """Fill callback does not trigger close for entry order fills."""
        engine_with_exchange.live_position_tracker.track_recovered_position(
            sample_position, db_id=None
        )

        with patch.object(engine_with_exchange, "_execute_exit") as mock_close:
            engine_with_exchange._handle_order_fill(
                "entry_order_123", "BTCUSDT", 0.02, 50000.0
            )

            mock_close.assert_not_called()

    def test_handle_fill_no_matching_position(self, engine_with_exchange):
        """Fill callback handles fill with no matching position."""
        engine_with_exchange._handle_order_fill("unknown_order", "BTCUSDT", 1.0, 50000.0)


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
        engine_with_exchange.live_position_tracker.track_recovered_position(
            sample_position, db_id=None
        )

        import logging

        with caplog.at_level(logging.CRITICAL):
            engine_with_exchange._handle_partial_fill(
                "sl_order_456", "BTCUSDT", 0.01, 48000.0
            )

        assert "PARTIAL STOP-LOSS FILL" in caplog.text

    def test_handle_partial_fill_logs_sl_warning_event(
        self, engine_with_exchange, sample_position
    ):
        """Partial fill warning logs event details."""
        sample_position.stop_loss_order_id = "sl_order_456"
        engine_with_exchange.live_position_tracker.track_recovered_position(
            sample_position, db_id=None
        )

        with patch("src.engines.live.trading_engine.log_order_event") as mock_log:
            engine_with_exchange._handle_partial_fill(
                "sl_order_456", "BTCUSDT", 0.01, 48000.0
            )

            mock_log.assert_any_call(
                "partial_sl_fill_warning",
                order_id="sl_order_456",
                position_order_id="entry_order_123",
                symbol="BTCUSDT",
                filled_quantity=0.01,
                average_price=48000.0,
            )

    def test_handle_partial_fill_no_matching_position(self, engine_with_exchange):
        """Partial fill with no matching position should not error."""
        engine_with_exchange._handle_partial_fill("unknown", "BTCUSDT", 0.5, 50000.0)


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
        """Cancel callback removes phantom position."""
        engine_with_exchange.live_position_tracker.track_recovered_position(
            sample_position, db_id=None
        )

        engine_with_exchange._handle_order_cancel("entry_order_123", "BTCUSDT")

        assert not engine_with_exchange.live_position_tracker.has_position("entry_order_123")

    def test_handle_cancel_thread_safe_with_pop(
        self, engine_with_exchange, sample_position
    ):
        """Cancel callback is thread-safe when removing positions."""
        engine_with_exchange.live_position_tracker.track_recovered_position(
            sample_position, db_id=None
        )

        engine_with_exchange._handle_order_cancel("entry_order_123", "BTCUSDT")

        assert not engine_with_exchange.live_position_tracker.has_position("entry_order_123")

    def test_handle_cancel_no_phantom_position(self, engine_with_exchange):
        """Cancel callback handles missing position gracefully."""
        engine_with_exchange._handle_order_cancel("unknown", "BTCUSDT")


# ============================================================================
# Tests for _reconcile_positions_with_exchange
# ============================================================================


class TestReconcilePositionsWithExchange:
    """Tests for the _reconcile_positions_with_exchange method."""

    def test_reconcile_no_exchange_skipped(self, engine_with_exchange):
        """Reconciliation skipped when no exchange interface."""
        engine_with_exchange.exchange_interface = None
        engine_with_exchange._reconcile_positions_with_exchange()

    def test_reconcile_no_live_trading_skipped(self, engine_with_exchange):
        """Reconciliation skipped when not live trading."""
        engine_with_exchange.enable_live_trading = False
        engine_with_exchange._reconcile_positions_with_exchange()

    def test_reconcile_no_positions_skipped(self, engine_with_exchange):
        """Reconciliation skipped when no local positions."""
        engine_with_exchange.live_position_tracker.reset()
        engine_with_exchange._reconcile_positions_with_exchange()

    def test_reconcile_sl_still_active_no_action(
        self, engine_with_exchange, sample_position
    ):
        """Positions remain when stop-loss order is still active."""
        engine_with_exchange.live_position_tracker.track_recovered_position(
            sample_position, db_id=None
        )
        engine_with_exchange.exchange_interface.get_open_orders.return_value = [
            MagicMock(order_id="sl_order_456")
        ]

        engine_with_exchange._reconcile_positions_with_exchange()

        assert engine_with_exchange.live_position_tracker.has_position("entry_order_123")

    def test_reconcile_detects_filled_stop_loss(
        self, engine_with_exchange, sample_position
    ):
        """Reconciliation detects filled stop-loss and removes position."""
        engine_with_exchange.live_position_tracker.track_recovered_position(
            sample_position, db_id=None
        )
        engine_with_exchange.exchange_interface.get_open_orders.return_value = []
        sl_order = MagicMock()
        sl_order.status = ExchangeOrderStatus.FILLED
        sl_order.average_price = 48000.0
        engine_with_exchange.exchange_interface.get_order.return_value = sl_order

        engine_with_exchange._reconcile_positions_with_exchange()

        assert not engine_with_exchange.live_position_tracker.has_position("entry_order_123")

    def test_reconcile_handles_api_error(self, engine_with_exchange, sample_position):
        """Reconciliation handles API errors gracefully."""
        engine_with_exchange.live_position_tracker.track_recovered_position(
            sample_position, db_id=None
        )
        engine_with_exchange.exchange_interface.get_open_orders.side_effect = Exception(
            "API error"
        )

        engine_with_exchange._reconcile_positions_with_exchange()

    def test_reconcile_uses_entry_balance_not_current(
        self, engine_with_exchange, sample_position
    ):
        """Reconciliation uses entry balance for PnL calculations."""
        engine_with_exchange.live_position_tracker.track_recovered_position(
            sample_position, db_id=None
        )
        engine_with_exchange.exchange_interface.get_open_orders.return_value = []
        sl_order = MagicMock()
        sl_order.status = ExchangeOrderStatus.FILLED
        sl_order.average_price = 48000.0
        engine_with_exchange.exchange_interface.get_order.return_value = sl_order
        engine_with_exchange.current_balance = 20000.0

        engine_with_exchange._reconcile_positions_with_exchange()

        assert engine_with_exchange.current_balance < 20000.0
