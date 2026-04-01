"""Dedicated thread for processing WebSocket user data stream events.

Sits between the WebSocket user data stream and the OrderTracker,
running on a dedicated thread to prevent fill processing from blocking
the trading engine's heartbeat loop.
"""

import logging
import queue
import threading

logger = logging.getLogger(__name__)


class UserDataProcessor(threading.Thread):
    """Process executionReport and balance events with minimal latency.

    Runs on a dedicated thread separate from the trading engine heartbeat
    to prevent head-of-line blocking of fill processing.
    """

    def __init__(self, order_tracker) -> None:
        """Initialize the user data processor.

        Args:
            order_tracker: OrderTracker instance with process_execution_event() method.
        """
        super().__init__(daemon=True, name="UserDataProcessor")
        self._order_tracker = order_tracker
        self._queue: queue.Queue = queue.Queue()
        self._running = False
        self._closed = False  # Gate to reject events after stop()
        self._close_lock = threading.Lock()  # Synchronises enqueue vs drain
        self._stop_event = threading.Event()

    def enqueue(self, event: dict) -> None:
        """Enqueue a raw WebSocket user data event for processing.

        Called from the WebSocket callback thread. Must be non-blocking.
        Rejects events after stop() to prevent post-drain accumulation.
        """
        with self._close_lock:
            if self._closed:
                return
            self._queue.put(event)

    def run(self) -> None:
        """Process events from the queue until stopped."""
        self._running = True
        logger.info("UserDataProcessor started")
        while self._running and not self._stop_event.is_set():
            try:
                event = self._queue.get(timeout=1)
            except queue.Empty:
                continue

            if event is None:
                break  # Sentinel from stop()

            self._handle_event(event)

        logger.info("UserDataProcessor stopped")

    def _handle_event(self, event: dict) -> None:
        """Dispatch a single user data event by type.

        Args:
            event: Raw WebSocket user data event dict.
        """
        event_type = event.get("e", "")
        try:
            if event_type == "executionReport":
                self._order_tracker.process_execution_event(event)
            elif event_type in ("outboundAccountPosition", "balanceUpdate"):
                # Balance events logged for future BalanceCache integration
                logger.debug(
                    "Balance event received: %s",
                    event.get("a", event.get("e", "")),
                )
            else:
                logger.debug("Unhandled user data event type: %s", event_type)
        except Exception as e:
            logger.error("Error processing user data event: %s", e, exc_info=True)

    def stop(self) -> bool:
        """Stop the processor and drain remaining execution events.

        Processes all remaining queued executionReport events before returning.
        This is critical for the WS-to-REST handoff to prevent missed fills.

        Returns:
            True if the thread stopped cleanly, False if it timed out.
        """
        # Signal the run() loop to exit (keep accepting enqueues until drain completes
        # so in-flight callbacks from the socket aren't dropped)
        self._running = False
        self._stop_event.set()
        # Put sentinel to unblock the queue.get() call in run()
        self._queue.put(None)
        # Wait for run() thread to finish its current event before draining
        if self.is_alive():
            self.join(timeout=5)
        if self.is_alive():
            logger.critical(
                "UserDataProcessor thread did not exit within timeout — "
                "skipping drain to avoid concurrent event processing"
            )
            self._closed = True
            return False

        # Drain remaining events to prevent missed fills during WS->REST handoff
        drained = 0
        while True:
            try:
                event = self._queue.get_nowait()
                if event is None:
                    continue  # Skip sentinel
                event_type = event.get("e", "")
                if event_type == "executionReport":
                    try:
                        self._order_tracker.process_execution_event(event)
                        drained += 1
                    except Exception as e:
                        logger.error(
                            "Error draining execution event: %s", e, exc_info=True
                        )
            except queue.Empty:
                break

        # Atomically reject further enqueues and do final drain under lock
        with self._close_lock:
            self._closed = True
            # Final drain under lock — catches any enqueue that raced before _closed
            while True:
                try:
                    event = self._queue.get_nowait()
                    if event is None:
                        continue
                    if event.get("e") == "executionReport":
                        try:
                            self._order_tracker.process_execution_event(event)
                            drained += 1
                        except Exception as e:
                            logger.error("Error in final drain: %s", e, exc_info=True)
                except queue.Empty:
                    break

        if drained > 0:
            logger.info("Drained %d execution events during shutdown", drained)
        return True

    @property
    def queue_size(self) -> int:
        """Return current queue depth for monitoring."""
        return self._queue.qsize()
