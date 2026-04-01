"""Thread-safe tracker for processed events to prevent duplicate state mutations."""

import threading
from collections import OrderedDict
from datetime import UTC, datetime


class EventDeduplicator:
    """Thread-safe tracker for processed events to prevent duplicate state mutations.

    Prevents the same WebSocket execution event from being processed twice.
    Both WebSocket and REST paths can process the same order status update
    during transitions, so deduplication is essential.
    """

    def __init__(self, max_size: int = 10000) -> None:
        """Initialize the deduplicator.

        Args:
            max_size: Maximum number of events to track. Oldest entries are
                evicted when this limit is exceeded.
        """
        self._max_size = max_size
        self._lock = threading.Lock()
        self._seen: OrderedDict[tuple[str, str, str], datetime] = OrderedDict()

    def is_duplicate(self, order_id: str, exec_type: str, exec_id: str) -> bool:
        """Check if event was already processed and mark it as seen.

        Key is (orderId, executionType, executionId) -- Binance fields (i, x, I).
        Uses I (execution ID) not t (trade ID) because t is -1 for non-trade
        events like cancels, rejects, and expires.

        Args:
            order_id: Binance order ID (field 'i').
            exec_type: Execution type e.g. NEW, TRADE, CANCELED (field 'x').
            exec_id: Execution ID unique per event (field 'I').

        Returns:
            True if this event was already seen, False if it is new.
        """
        key = (order_id, exec_type, exec_id)
        with self._lock:
            if key in self._seen:
                return True
            self._seen[key] = datetime.now(UTC)
            # Evict oldest entries when capacity exceeded
            while len(self._seen) > self._max_size:
                self._seen.popitem(last=False)
            return False

    def is_seen(self, order_id: str, exec_type: str, exec_id: str) -> bool:
        """Check if event was already processed WITHOUT marking it.

        Use this when you want to defer marking until after successful processing.
        Call mark_seen() after the event is successfully handled.
        """
        key = (order_id, exec_type, exec_id)
        with self._lock:
            return key in self._seen

    def mark_seen(self, order_id: str, exec_type: str, exec_id: str) -> None:
        """Mark an event as processed after successful handling."""
        key = (order_id, exec_type, exec_id)
        with self._lock:
            self._seen[key] = datetime.now(UTC)
            while len(self._seen) > self._max_size:
                self._seen.popitem(last=False)
