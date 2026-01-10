"""Unit tests for OrderTracker class."""

from unittest.mock import MagicMock, Mock

import pytest

from src.data_providers.exchange_interface import OrderStatus
from src.engines.live.order_tracker import OrderTracker


@pytest.fixture
def mock_exchange():
    """Create a mock exchange interface."""
    exchange = Mock()
    return exchange


@pytest.fixture
def order_tracker(mock_exchange):
    """Create an OrderTracker instance with mocked callbacks."""
    on_fill = Mock()
    on_partial_fill = Mock()
    on_cancel = Mock()

    tracker = OrderTracker(
        exchange=mock_exchange,
        poll_interval=0.1,  # Short interval for testing
        on_fill=on_fill,
        on_partial_fill=on_partial_fill,
        on_cancel=on_cancel,
    )
    return tracker


def test_order_tracker_initialization(mock_exchange):
    """Test OrderTracker initialization with default and custom parameters."""
    # Default parameters
    tracker = OrderTracker(exchange=mock_exchange)
    assert tracker.exchange == mock_exchange
    assert tracker.poll_interval == 5
    assert tracker.on_fill is None
    assert tracker.on_partial_fill is None
    assert tracker.on_cancel is None
    assert tracker.get_tracked_count() == 0

    # Custom parameters
    on_fill = Mock()
    on_partial_fill = Mock()
    on_cancel = Mock()
    tracker = OrderTracker(
        exchange=mock_exchange,
        poll_interval=10,
        on_fill=on_fill,
        on_partial_fill=on_partial_fill,
        on_cancel=on_cancel,
    )
    assert tracker.poll_interval == 10
    assert tracker.on_fill == on_fill
    assert tracker.on_partial_fill == on_partial_fill
    assert tracker.on_cancel == on_cancel


def test_track_order(order_tracker):
    """Test tracking an order."""
    order_tracker.track_order("order123", "BTCUSDT")

    assert order_tracker.get_tracked_count() == 1


def test_track_multiple_orders(order_tracker):
    """Test tracking multiple orders."""
    order_tracker.track_order("order1", "BTCUSDT")
    order_tracker.track_order("order2", "ETHUSDT")
    order_tracker.track_order("order3", "BTCUSDT")

    assert order_tracker.get_tracked_count() == 3


def test_stop_tracking(order_tracker):
    """Test stopping tracking of an order."""
    order_tracker.track_order("order123", "BTCUSDT")
    assert order_tracker.get_tracked_count() == 1

    order_tracker.stop_tracking("order123")
    assert order_tracker.get_tracked_count() == 0


def test_stop_tracking_nonexistent_order(order_tracker):
    """Test stopping tracking of an order that doesn't exist."""
    # Should not raise an error
    order_tracker.stop_tracking("nonexistent")
    assert order_tracker.get_tracked_count() == 0


def test_filled_order_triggers_callback(order_tracker, mock_exchange):
    """Test that a filled order triggers the on_fill callback and stops tracking."""
    # Mock order status
    mock_order = MagicMock()
    mock_order.status = OrderStatus.FILLED
    mock_order.filled_quantity = 1.5
    mock_order.average_price = 50000.0
    mock_exchange.get_order.return_value = mock_order

    # Track order and check it
    order_tracker.track_order("order123", "BTCUSDT")
    order_tracker._check_orders()

    # Verify callback was called
    order_tracker.on_fill.assert_called_once_with("order123", "BTCUSDT", 1.5, 50000.0)

    # Verify order is no longer tracked
    assert order_tracker.get_tracked_count() == 0


def test_partially_filled_order_triggers_callback(order_tracker, mock_exchange):
    """Test that a partially filled order triggers the on_partial_fill callback."""
    # Mock order status
    mock_order = MagicMock()
    mock_order.status = OrderStatus.PARTIALLY_FILLED
    mock_order.filled_quantity = 0.5
    mock_order.average_price = 50000.0
    mock_exchange.get_order.return_value = mock_order

    # Track order and check it
    order_tracker.track_order("order123", "BTCUSDT")
    order_tracker._check_orders()

    # Verify partial fill callback was called with new filled quantity
    order_tracker.on_partial_fill.assert_called_once_with("order123", "BTCUSDT", 0.5, 50000.0)

    # Verify order is still being tracked
    assert order_tracker.get_tracked_count() == 1


def test_multiple_partial_fills_only_notify_new_fills(order_tracker, mock_exchange):
    """Test that multiple partial fills only trigger callbacks for new fills."""
    # First partial fill
    mock_order = MagicMock()
    mock_order.status = OrderStatus.PARTIALLY_FILLED
    mock_order.filled_quantity = 0.5
    mock_order.average_price = 50000.0
    mock_exchange.get_order.return_value = mock_order

    order_tracker.track_order("order123", "BTCUSDT")
    order_tracker._check_orders()

    # Verify first partial fill
    order_tracker.on_partial_fill.assert_called_once_with("order123", "BTCUSDT", 0.5, 50000.0)
    order_tracker.on_partial_fill.reset_mock()

    # Second partial fill - more quantity filled
    mock_order.filled_quantity = 1.0
    order_tracker._check_orders()

    # Verify only the new fill amount was reported
    order_tracker.on_partial_fill.assert_called_once_with("order123", "BTCUSDT", 0.5, 50000.0)


def test_cancelled_order_triggers_callback(order_tracker, mock_exchange):
    """Test that a cancelled order triggers the on_cancel callback."""
    # Mock order status
    mock_order = MagicMock()
    mock_order.status = OrderStatus.CANCELLED
    mock_exchange.get_order.return_value = mock_order

    # Track order and check it
    order_tracker.track_order("order123", "BTCUSDT")
    order_tracker._check_orders()

    # Verify callback was called
    order_tracker.on_cancel.assert_called_once_with("order123", "BTCUSDT")

    # Verify order is no longer tracked
    assert order_tracker.get_tracked_count() == 0


def test_rejected_order_triggers_cancel_callback(order_tracker, mock_exchange):
    """Test that a rejected order triggers the on_cancel callback."""
    mock_order = MagicMock()
    mock_order.status = OrderStatus.REJECTED
    mock_exchange.get_order.return_value = mock_order

    order_tracker.track_order("order123", "BTCUSDT")
    order_tracker._check_orders()

    order_tracker.on_cancel.assert_called_once_with("order123", "BTCUSDT")
    assert order_tracker.get_tracked_count() == 0


def test_expired_order_triggers_cancel_callback(order_tracker, mock_exchange):
    """Test that an expired order triggers the on_cancel callback."""
    mock_order = MagicMock()
    mock_order.status = OrderStatus.EXPIRED
    mock_exchange.get_order.return_value = mock_order

    order_tracker.track_order("order123", "BTCUSDT")
    order_tracker._check_orders()

    order_tracker.on_cancel.assert_called_once_with("order123", "BTCUSDT")
    assert order_tracker.get_tracked_count() == 0


def test_order_not_found_continues_tracking(order_tracker, mock_exchange):
    """Test that order not found doesn't crash and continues tracking."""
    # Exchange returns None when order can't be fetched
    mock_exchange.get_order.return_value = None

    order_tracker.track_order("order123", "BTCUSDT")
    order_tracker._check_orders()

    # No callbacks should be triggered
    order_tracker.on_fill.assert_not_called()
    order_tracker.on_cancel.assert_not_called()

    # Order should still be tracked
    assert order_tracker.get_tracked_count() == 1


def test_exchange_error_continues_tracking(order_tracker, mock_exchange):
    """Test that exchange errors don't crash the tracker."""
    # Exchange raises an exception
    mock_exchange.get_order.side_effect = Exception("API error")

    order_tracker.track_order("order123", "BTCUSDT")
    order_tracker._check_orders()

    # No callbacks should be triggered
    order_tracker.on_fill.assert_not_called()
    order_tracker.on_cancel.assert_not_called()

    # Order should still be tracked
    assert order_tracker.get_tracked_count() == 1


def test_start_stop_polling_thread(order_tracker):
    """Test starting and stopping the polling thread."""
    assert not order_tracker._running

    order_tracker.start()
    assert order_tracker._running
    assert order_tracker._thread is not None
    assert order_tracker._thread.is_alive()

    order_tracker.stop()
    assert not order_tracker._running
    # Thread should be set to None after stopping
    assert order_tracker._thread is None


def test_start_already_running_warning(order_tracker):
    """Test that starting an already running tracker logs a warning."""
    order_tracker.start()
    assert order_tracker._running

    # Starting again should log warning but not crash
    order_tracker.start()
    assert order_tracker._running

    order_tracker.stop()


def test_callbacks_optional(mock_exchange):
    """Test that OrderTracker works without callbacks."""
    tracker = OrderTracker(exchange=mock_exchange, poll_interval=0.1)

    mock_order = MagicMock()
    mock_order.status = OrderStatus.FILLED
    mock_order.filled_quantity = 1.0
    mock_order.average_price = 50000.0
    mock_exchange.get_order.return_value = mock_order

    tracker.track_order("order123", "BTCUSDT")
    tracker._check_orders()

    # Should not crash without callbacks
    assert tracker.get_tracked_count() == 0


def test_thread_safety_concurrent_tracking(order_tracker):
    """Test that tracking/untracking from multiple threads is safe."""
    import threading

    def add_orders():
        for i in range(100):
            order_tracker.track_order(f"order{i}", "BTCUSDT")

    def remove_orders():
        for i in range(100):
            order_tracker.stop_tracking(f"order{i}")

    # Start threads
    t1 = threading.Thread(target=add_orders)
    t2 = threading.Thread(target=remove_orders)

    t1.start()
    t2.start()

    t1.join()
    t2.join()

    # Should complete without errors
    # Final count may vary due to race conditions, but should be valid
    count = order_tracker.get_tracked_count()
    assert 0 <= count <= 100


def test_invalid_average_price_skips_fill_callback(order_tracker, mock_exchange):
    """Test that invalid average_price skips the fill callback to prevent corrupt P&L."""
    mock_order = MagicMock()
    mock_order.status = OrderStatus.FILLED
    mock_order.filled_quantity = 1.0
    mock_order.average_price = None  # Exchange may return None
    mock_exchange.get_order.return_value = mock_order

    order_tracker.track_order("order123", "BTCUSDT")
    order_tracker._check_orders()

    # Should NOT call fill callback with invalid price - prevents corrupt P&L
    order_tracker.on_fill.assert_not_called()

    # Order should still be tracked - keep polling until valid price
    assert order_tracker.get_tracked_count() == 1


def test_zero_average_price_skips_fill_callback(order_tracker, mock_exchange):
    """Test that zero average_price skips the fill callback."""
    mock_order = MagicMock()
    mock_order.status = OrderStatus.FILLED
    mock_order.filled_quantity = 1.0
    mock_order.average_price = 0.0  # Invalid price
    mock_exchange.get_order.return_value = mock_order

    order_tracker.track_order("order123", "BTCUSDT")
    order_tracker._check_orders()

    # Should NOT call fill callback with zero price
    order_tracker.on_fill.assert_not_called()
    assert order_tracker.get_tracked_count() == 1
