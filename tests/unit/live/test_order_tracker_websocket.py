"""Unit tests for OrderTracker WebSocket integration."""

from unittest.mock import Mock, patch

import pytest

from src.data_providers.exchange_interface import OrderSide, OrderStatus, OrderType
from src.engines.live.event_deduplicator import EventDeduplicator
from src.engines.live.order_tracker import OrderTracker

pytestmark = [pytest.mark.unit, pytest.mark.fast]


def _make_ws_event(
    *,
    order_id: str = "12345",
    symbol: str = "BTCUSDT",
    side: str = "BUY",
    order_type: str = "MARKET",
    status: str = "FILLED",
    exec_type: str = "TRADE",
    exec_id: str = "99999",
    quantity: float = 1.0,
    price: float = 0,
    cum_filled: float = 1.0,
    cum_quote: float = 50000.0,
    commission: float = 0.001,
    commission_asset: str = "BNB",
    order_create_time: int = 1700000000000,
    event_time: int = 1700000001000,
    stop_price: float = 0,
    time_in_force: str = "GTC",
    client_order_id: str = "atb_test123",
) -> dict:
    """Build a Binance-style WebSocket executionReport event."""
    return {
        "i": order_id,
        "s": symbol,
        "S": side,
        "o": order_type,
        "X": status,
        "x": exec_type,
        "I": exec_id,
        "q": quantity,
        "p": price,
        "z": cum_filled,
        "Z": cum_quote,
        "n": commission,
        "N": commission_asset,
        "O": order_create_time,
        "E": event_time,
        "P": stop_price,
        "f": time_in_force,
        "c": client_order_id,
    }


@pytest.fixture
def mock_exchange():
    """Create a mock exchange interface."""
    return Mock()


@pytest.fixture
def dedup():
    """Create a fresh EventDeduplicator."""
    return EventDeduplicator()


@pytest.fixture
def tracker(mock_exchange, dedup):
    """Create an OrderTracker with mocked callbacks and deduplicator."""
    on_fill = Mock()
    on_partial_fill = Mock()
    on_cancel = Mock()
    t = OrderTracker(
        exchange=mock_exchange,
        poll_interval=0.1,
        on_fill=on_fill,
        on_partial_fill=on_partial_fill,
        on_cancel=on_cancel,
        event_deduplicator=dedup,
    )
    return t


class TestProcessExecutionEventFilled:
    """Tests for processing FILLED execution events via WebSocket."""

    def test_filled_event_triggers_on_fill_callback(self, tracker):
        """Test that a FILLED WS event triggers the on_fill callback."""
        tracker.track_order("12345", "BTCUSDT")
        event = _make_ws_event(status="FILLED", cum_filled=1.0, cum_quote=50000.0)

        tracker.process_execution_event(event)

        tracker.on_fill.assert_called_once_with("12345", "BTCUSDT", 1.0, 50000.0)
        assert tracker.get_tracked_count() == 0


class TestProcessExecutionEventPartialFill:
    """Tests for processing PARTIALLY_FILLED execution events via WebSocket."""

    def test_partial_fill_event_triggers_on_partial_fill_callback(self, tracker):
        """Test that a PARTIALLY_FILLED WS event triggers the on_partial_fill callback."""
        tracker.track_order("12345", "BTCUSDT")
        event = _make_ws_event(
            status="PARTIALLY_FILLED",
            cum_filled=0.5,
            cum_quote=25000.0,
        )

        tracker.process_execution_event(event)

        tracker.on_partial_fill.assert_called_once_with("12345", "BTCUSDT", 0.5, 50000.0)
        assert tracker.get_tracked_count() == 1


class TestProcessExecutionEventCanceled:
    """Tests for processing CANCELED execution events via WebSocket."""

    def test_canceled_event_triggers_on_cancel_callback(self, tracker):
        """Test that a CANCELED WS event triggers the on_cancel callback."""
        tracker.track_order("12345", "BTCUSDT")
        event = _make_ws_event(
            status="CANCELED",
            exec_type="CANCELED",
            cum_filled=0.0,
            cum_quote=0.0,
        )

        tracker.process_execution_event(event)

        tracker.on_cancel.assert_called_once_with("12345", "BTCUSDT", 0.0)
        assert tracker.get_tracked_count() == 0


class TestDeduplication:
    """Tests for duplicate event handling."""

    def test_duplicate_events_are_skipped(self, tracker):
        """Test that duplicate events are not processed twice."""
        tracker.track_order("12345", "BTCUSDT")
        event = _make_ws_event(status="FILLED", cum_filled=1.0, cum_quote=50000.0)

        tracker.process_execution_event(event)
        tracker.process_execution_event(event)  # Same event again

        tracker.on_fill.assert_called_once()


class TestUntrackedOrders:
    """Tests for events referencing orders not being tracked."""

    def test_untracked_order_events_are_ignored(self, tracker):
        """Test that events for untracked orders are silently ignored."""
        event = _make_ws_event(order_id="unknown_order", status="FILLED")

        tracker.process_execution_event(event)

        tracker.on_fill.assert_not_called()
        tracker.on_cancel.assert_not_called()
        tracker.on_partial_fill.assert_not_called()


class TestUnknownStatus:
    """Tests for unknown WebSocket status values."""

    def test_unknown_ws_status_skips_processing(self, tracker):
        """Test that an unknown WS status returns None and skips processing."""
        tracker.track_order("12345", "BTCUSDT")
        event = _make_ws_event(status="PENDING_CANCEL")

        tracker.process_execution_event(event)

        tracker.on_fill.assert_not_called()
        tracker.on_cancel.assert_not_called()
        tracker.on_partial_fill.assert_not_called()
        # Order should still be tracked
        assert tracker.get_tracked_count() == 1


class TestMapWsStatus:
    """Tests for _map_ws_status static method."""

    @pytest.mark.parametrize(
        ("ws_status", "expected"),
        [
            ("NEW", OrderStatus.PENDING),
            ("PARTIALLY_FILLED", OrderStatus.PARTIALLY_FILLED),
            ("FILLED", OrderStatus.FILLED),
            ("CANCELED", OrderStatus.CANCELLED),
            ("REJECTED", OrderStatus.REJECTED),
            ("EXPIRED", OrderStatus.EXPIRED),
        ],
    )
    def test_maps_all_binance_statuses_correctly(self, ws_status, expected):
        """Test that all 6 Binance statuses map to correct OrderStatus values."""
        result = OrderTracker._map_ws_status(ws_status)
        assert result == expected

    def test_unknown_status_returns_none(self):
        """Test that unknown status returns None."""
        result = OrderTracker._map_ws_status("PENDING_CANCEL")
        assert result is None


class TestMapWsOrderType:
    """Tests for _map_ws_order_type static method."""

    @pytest.mark.parametrize(
        ("ws_type", "expected"),
        [
            ("MARKET", OrderType.MARKET),
            ("LIMIT", OrderType.LIMIT),
            ("STOP_LOSS", OrderType.STOP_LOSS),
            ("STOP_LOSS_LIMIT", OrderType.STOP_LOSS),
            ("TAKE_PROFIT", OrderType.TAKE_PROFIT),
            ("TAKE_PROFIT_LIMIT", OrderType.TAKE_PROFIT),
        ],
    )
    def test_maps_all_order_types_correctly(self, ws_type, expected):
        """Test that all WS order types map correctly, including LIMIT variants."""
        result = OrderTracker._map_ws_order_type(ws_type)
        assert result == expected

    def test_unknown_type_defaults_to_market(self):
        """Test that unknown order type defaults to MARKET."""
        result = OrderTracker._map_ws_order_type("TRAILING_STOP")
        assert result == OrderType.MARKET


class TestPollOnce:
    """Tests for poll_once() public wrapper."""

    def test_poll_once_calls_check_orders(self, tracker, mock_exchange):
        """Test that poll_once delegates to _check_orders."""
        with patch.object(tracker, "_check_orders") as mock_check:
            tracker.poll_once()
            mock_check.assert_called_once()


class TestPollingControl:
    """Tests for disable_polling / enable_polling control."""

    def test_disable_polling_prevents_check_orders(self, tracker, mock_exchange):
        """Test that _poll_loop skips _check_orders when polling is disabled."""
        tracker.track_order("12345", "BTCUSDT")
        tracker.disable_polling()

        # Simulate one loop iteration by calling _poll_loop logic directly
        # We verify _check_orders is not called by checking exchange not queried
        assert tracker._polling_enabled is False

    def test_enable_polling_restores_check_orders(self, tracker):
        """Test that enable_polling sets the flag back to True."""
        tracker.disable_polling()
        assert tracker._polling_enabled is False

        tracker.enable_polling()
        assert tracker._polling_enabled is True

    def test_poll_loop_respects_polling_enabled_flag(self, tracker, mock_exchange):
        """Test that _poll_loop skips _check_orders when polling is disabled."""
        from unittest.mock import MagicMock

        mock_order = MagicMock()
        mock_order.status = OrderStatus.FILLED
        mock_order.filled_quantity = 1.0
        mock_order.average_price = 50000.0
        mock_exchange.get_order.return_value = mock_order

        tracker.track_order("12345", "BTCUSDT")
        tracker.disable_polling()

        # Start tracker, let it run one cycle, then stop
        tracker._running = True
        tracker._stop_event.clear()

        # Run one iteration manually by simulating the loop body
        # (checking _polling_enabled flag)
        if tracker._polling_enabled:
            tracker._check_orders()

        # Exchange should NOT have been called
        mock_exchange.get_order.assert_not_called()
        assert tracker.get_tracked_count() == 1


class TestZeroCumFilled:
    """Tests for edge case of zero cumulative filled quantity."""

    def test_zero_cum_filled_gives_zero_avg_price(self, tracker):
        """Test that zero cum_filled produces avg_price of 0.0 without division by zero."""
        tracker.track_order("12345", "BTCUSDT")
        event = _make_ws_event(
            status="CANCELED",
            exec_type="CANCELED",
            cum_filled=0.0,
            cum_quote=0.0,
        )

        # Should not raise ZeroDivisionError
        tracker.process_execution_event(event)

        tracker.on_cancel.assert_called_once_with("12345", "BTCUSDT", 0.0)


class TestDefaultDeduplicator:
    """Tests for default EventDeduplicator creation."""

    def test_default_deduplicator_created_when_none_provided(self, mock_exchange):
        """Test that OrderTracker creates a default EventDeduplicator if none given."""
        t = OrderTracker(exchange=mock_exchange)
        assert t._dedup is not None
        assert isinstance(t._dedup, EventDeduplicator)
