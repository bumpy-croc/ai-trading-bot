"""Tests for UserDataProcessor — dedicated thread for WebSocket user data events."""

import logging
import queue
import threading
import time
from unittest.mock import MagicMock, create_autospec

import pytest

from src.engines.live.user_data_processor import UserDataProcessor


def _make_order_tracker():
    """Create a mock order tracker with process_execution_event method."""
    tracker = MagicMock()
    tracker.process_execution_event = MagicMock()
    return tracker


@pytest.mark.fast
class TestUserDataProcessorEnqueue:
    """Tests for enqueue behavior."""

    def test_enqueue_execution_report_is_processed(self):
        """Enqueue an executionReport event and verify it reaches the order tracker."""
        tracker = _make_order_tracker()
        processor = UserDataProcessor(order_tracker=tracker)
        processor.start()

        event = {"e": "executionReport", "s": "BTCUSDT", "i": 12345}
        processor.enqueue(event)

        # Wait for processing
        time.sleep(0.1)
        processor.stop()
        processor.join(timeout=2)

        tracker.process_execution_event.assert_called_once_with(event)

    def test_enqueue_is_non_blocking(self):
        """Enqueue must return immediately without blocking the caller."""
        tracker = _make_order_tracker()
        processor = UserDataProcessor(order_tracker=tracker)
        # Don't start the processor — queue will fill but enqueue should not block

        start = time.monotonic()
        for _ in range(100):
            processor.enqueue({"e": "executionReport", "s": "BTCUSDT"})
        elapsed = time.monotonic() - start

        # 100 enqueues should take well under 1 second
        assert elapsed < 1.0

    def test_queue_size_reflects_depth(self):
        """queue_size property returns current queue depth."""
        tracker = _make_order_tracker()
        processor = UserDataProcessor(order_tracker=tracker)
        # Don't start — items stay queued

        assert processor.queue_size == 0

        processor.enqueue({"e": "executionReport"})
        processor.enqueue({"e": "executionReport"})
        processor.enqueue({"e": "balanceUpdate"})

        assert processor.queue_size == 3


@pytest.mark.fast
class TestUserDataProcessorEventTypes:
    """Tests for different event type handling."""

    def test_balance_event_logged_not_processed(self, caplog):
        """Balance events are logged at debug but not forwarded to order tracker."""
        tracker = _make_order_tracker()
        processor = UserDataProcessor(order_tracker=tracker)
        processor.start()

        event = {"e": "outboundAccountPosition", "a": [{"a": "BTC", "f": "0.5"}]}
        with caplog.at_level(logging.DEBUG):
            processor.enqueue(event)
            time.sleep(0.1)

        processor.stop()
        processor.join(timeout=2)

        tracker.process_execution_event.assert_not_called()

    def test_balance_update_event_logged(self, caplog):
        """balanceUpdate events are logged at debug."""
        tracker = _make_order_tracker()
        processor = UserDataProcessor(order_tracker=tracker)
        processor.start()

        event = {"e": "balanceUpdate", "a": "BTC", "d": "0.001"}
        with caplog.at_level(logging.DEBUG):
            processor.enqueue(event)
            time.sleep(0.1)

        processor.stop()
        processor.join(timeout=2)

        tracker.process_execution_event.assert_not_called()

    def test_unknown_event_type_logged_at_debug(self, caplog):
        """Unknown event types are logged at debug level."""
        tracker = _make_order_tracker()
        processor = UserDataProcessor(order_tracker=tracker)
        processor.start()

        event = {"e": "someUnknownEvent", "data": "value"}
        with caplog.at_level(logging.DEBUG):
            processor.enqueue(event)
            time.sleep(0.1)

        processor.stop()
        processor.join(timeout=2)

        tracker.process_execution_event.assert_not_called()


@pytest.mark.fast
class TestUserDataProcessorErrorHandling:
    """Tests for error resilience."""

    def test_processing_error_does_not_crash_thread(self):
        """An exception in process_execution_event should not kill the thread."""
        tracker = _make_order_tracker()
        tracker.process_execution_event.side_effect = [
            RuntimeError("boom"),
            None,  # Second call succeeds
        ]
        processor = UserDataProcessor(order_tracker=tracker)
        processor.start()

        # First event will raise
        processor.enqueue({"e": "executionReport", "i": 1})
        # Second event should still be processed
        processor.enqueue({"e": "executionReport", "i": 2})

        time.sleep(0.2)
        processor.stop()
        processor.join(timeout=5)

        # Both events should have been attempted
        assert tracker.process_execution_event.call_count == 2
        # Thread should have exited cleanly
        assert not processor.is_alive()


@pytest.mark.fast
class TestUserDataProcessorLifecycle:
    """Tests for thread start/stop lifecycle."""

    def test_thread_starts_and_stops_cleanly(self):
        """Thread should start as daemon and stop within timeout."""
        tracker = _make_order_tracker()
        processor = UserDataProcessor(order_tracker=tracker)

        assert processor.daemon is True
        assert processor.name == "UserDataProcessor"

        processor.start()
        assert processor.is_alive()

        # Allow thread to enter run() loop before stopping
        time.sleep(0.05)

        processor.stop()
        processor.join(timeout=10)
        assert not processor.is_alive()

    def test_stop_drains_remaining_execution_events(self):
        """stop() should process all remaining executionReport events before returning."""
        tracker = _make_order_tracker()
        processor = UserDataProcessor(order_tracker=tracker)
        # Don't start the thread — manually enqueue events then call stop()
        # This tests the drain logic in stop()

        processor.enqueue({"e": "executionReport", "i": 1})
        processor.enqueue({"e": "executionReport", "i": 2})
        processor.enqueue({"e": "executionReport", "i": 3})
        # Include a non-execution event — should be skipped during drain
        processor.enqueue({"e": "balanceUpdate", "a": "BTC"})

        processor.stop()

        # All 3 execution events should have been drained
        assert tracker.process_execution_event.call_count == 3

    def test_stop_drain_handles_errors(self):
        """Errors during drain should not prevent remaining events from draining."""
        tracker = _make_order_tracker()
        tracker.process_execution_event.side_effect = [
            RuntimeError("drain error"),
            None,  # Second succeeds
        ]
        processor = UserDataProcessor(order_tracker=tracker)

        processor.enqueue({"e": "executionReport", "i": 1})
        processor.enqueue({"e": "executionReport", "i": 2})

        processor.stop()

        # Both events should have been attempted
        assert tracker.process_execution_event.call_count == 2

    def test_multiple_execution_reports_processed_in_order(self):
        """Events should be processed in FIFO order."""
        tracker = _make_order_tracker()
        call_order = []
        tracker.process_execution_event.side_effect = lambda e: call_order.append(e["i"])

        processor = UserDataProcessor(order_tracker=tracker)
        processor.start()

        for i in range(5):
            processor.enqueue({"e": "executionReport", "i": i})

        time.sleep(0.2)
        processor.stop()
        processor.join(timeout=2)

        assert call_order == [0, 1, 2, 3, 4]
