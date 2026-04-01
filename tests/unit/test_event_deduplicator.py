"""Tests for EventDeduplicator — thread-safe duplicate event tracker."""

import threading
from collections import OrderedDict
from datetime import UTC, datetime

import pytest

from src.engines.live.event_deduplicator import EventDeduplicator


@pytest.mark.fast
class TestEventDeduplicator:
    """Tests for EventDeduplicator."""

    def test_first_event_is_not_duplicate(self) -> None:
        """First occurrence of an event returns False (not a duplicate)."""
        dedup = EventDeduplicator()
        assert dedup.is_duplicate("order1", "TRADE", "exec1") is False

    def test_same_event_is_duplicate(self) -> None:
        """Second occurrence of the same event returns True (duplicate)."""
        dedup = EventDeduplicator()
        dedup.is_duplicate("order1", "TRADE", "exec1")
        assert dedup.is_duplicate("order1", "TRADE", "exec1") is True

    def test_different_order_id_is_not_duplicate(self) -> None:
        """Events with different order_id are distinct."""
        dedup = EventDeduplicator()
        dedup.is_duplicate("order1", "TRADE", "exec1")
        assert dedup.is_duplicate("order2", "TRADE", "exec1") is False

    def test_different_exec_type_is_not_duplicate(self) -> None:
        """Events with different exec_type are distinct."""
        dedup = EventDeduplicator()
        dedup.is_duplicate("order1", "TRADE", "exec1")
        assert dedup.is_duplicate("order1", "NEW", "exec1") is False

    def test_different_exec_id_is_not_duplicate(self) -> None:
        """Events with different exec_id are distinct."""
        dedup = EventDeduplicator()
        dedup.is_duplicate("order1", "TRADE", "exec1")
        assert dedup.is_duplicate("order1", "TRADE", "exec2") is False

    def test_eviction_removes_oldest_entries(self) -> None:
        """When max_size is exceeded, oldest entries are evicted first."""
        dedup = EventDeduplicator(max_size=3)

        # Fill to capacity
        dedup.is_duplicate("order1", "TRADE", "exec1")
        dedup.is_duplicate("order2", "TRADE", "exec2")
        dedup.is_duplicate("order3", "TRADE", "exec3")

        # Add one more — should evict order1
        dedup.is_duplicate("order4", "TRADE", "exec4")

        # order1 was evicted, so it's no longer considered a duplicate
        assert dedup.is_duplicate("order1", "TRADE", "exec1") is False
        # order2 should still be tracked (or evicted by the re-add of order1)
        # After adding order1 again, we have: order3, order4, order1 (size=3)
        assert dedup.is_duplicate("order3", "TRADE", "exec3") is True
        assert dedup.is_duplicate("order4", "TRADE", "exec4") is True

    def test_thread_safety_concurrent_calls(self) -> None:
        """Concurrent is_duplicate calls don't corrupt internal state."""
        dedup = EventDeduplicator(max_size=5000)
        results: list[bool] = []
        lock = threading.Lock()

        def call_is_duplicate(order_id: str, exec_id: str) -> None:
            """Call is_duplicate and record the result."""
            result = dedup.is_duplicate(order_id, "TRADE", exec_id)
            with lock:
                results.append(result)

        threads = []
        # 100 threads each inserting a unique event
        for i in range(100):
            t = threading.Thread(
                target=call_is_duplicate, args=(f"order{i}", f"exec{i}")
            )
            threads.append(t)

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All 100 events are unique, so all should return False
        assert len(results) == 100
        assert all(r is False for r in results)

    def test_thread_safety_duplicate_detection(self) -> None:
        """When multiple threads submit the same event, exactly one sees False."""
        dedup = EventDeduplicator()
        results: list[bool] = []
        lock = threading.Lock()

        def call_is_duplicate() -> None:
            """Call is_duplicate for the same event and record result."""
            result = dedup.is_duplicate("orderX", "TRADE", "execX")
            with lock:
                results.append(result)

        threads = [threading.Thread(target=call_is_duplicate) for _ in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Exactly one thread should see False (first to acquire lock)
        assert results.count(False) == 1
        assert results.count(True) == 49

    def test_non_trade_events_cancel_and_reject(self) -> None:
        """Cancel and reject events get unique dedup keys via different exec_type.

        Binance sets trade ID (t) to -1 for non-trade events, but we use
        execution ID (I) which is unique per event. Different exec_types
        for the same order are distinct events.
        """
        dedup = EventDeduplicator()

        # Same order, different lifecycle events — all distinct
        assert dedup.is_duplicate("order1", "NEW", "exec100") is False
        assert dedup.is_duplicate("order1", "CANCELED", "exec101") is False
        assert dedup.is_duplicate("order1", "REJECTED", "exec102") is False
        assert dedup.is_duplicate("order1", "EXPIRED", "exec103") is False

        # Replaying any of them is a duplicate
        assert dedup.is_duplicate("order1", "NEW", "exec100") is True
        assert dedup.is_duplicate("order1", "CANCELED", "exec101") is True
