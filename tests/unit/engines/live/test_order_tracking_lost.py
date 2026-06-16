"""OrderTracker polling give-up must NOT be treated as an order cancellation.

Previously the tracker fired ``on_cancel`` after MAX_API_ERROR_RETRIES
consecutive failed/None polls, and ``_handle_order_cancel`` then popped the
(possibly live) position from the tracker and refunded its entry fee — i.e. a
~50 s exchange API degradation vaporized a real position from the books:
untracked exposure, corrupted balance, and room for a double entry on the
next signal (LESSONS §1.8 fail-open class).

The give-up now routes to ``on_tracking_lost`` → ``_handle_order_tracking_lost``,
which keeps the position, leaves the balance untouched, and escalates via a
critical system event + alert so the reconciler resolves exchange truth.
"""

from __future__ import annotations

from unittest.mock import MagicMock, create_autospec, patch

import pytest

from src.database.manager import DatabaseManager
from src.database.models import EventType
from src.engines.live.execution.position_tracker import LivePositionTracker
from src.engines.live.trading_engine import LiveTradingEngine
from src.strategies.ml_basic import create_ml_basic_strategy

pytestmark = pytest.mark.fast


def make_engine() -> LiveTradingEngine:
    with patch("src.engines.live.trading_engine.DatabaseManager"):
        engine = LiveTradingEngine(
            strategy=create_ml_basic_strategy(),
            data_provider=MagicMock(),
            initial_balance=1000.0,
        )
    # autospec catches signature mismatches on every db_manager call
    engine.db_manager = create_autospec(DatabaseManager, instance=True)
    return engine


def _track_fake_position(engine: LiveTradingEngine, order_id: str = "entry_1") -> MagicMock:
    position = MagicMock()
    position.symbol = "BTCUSDT"
    position.order_id = order_id
    # autospec'd tracker: get_position/pop_position calls are signature-checked
    tracker = create_autospec(LivePositionTracker, instance=True)
    tracker.get_position.return_value = position
    tracker.pop_position.return_value = position
    engine.live_position_tracker = tracker
    return position


class TestHandleOrderTrackingLost:
    def test_keeps_position_and_balance(self):
        engine = make_engine()
        _track_fake_position(engine)
        balance_before = engine.current_balance

        engine._handle_order_tracking_lost("entry_1", "BTCUSDT", 10)

        engine.live_position_tracker.pop_position.assert_not_called()
        assert engine.current_balance == balance_before
        engine.db_manager.atomic_balance_update.assert_not_called()

    def test_records_critical_event_with_alert(self):
        engine = make_engine()
        _track_fake_position(engine)

        with patch.object(engine, "_record_event") as record:
            engine._handle_order_tracking_lost("entry_1", "BTCUSDT", 10)

        record.assert_called_once()
        args, kwargs = record.call_args
        assert args[0] == EventType.ERROR
        assert "UNKNOWN" in args[1]
        assert kwargs["severity"] == "critical"
        assert kwargs["error_code"] == "ORDER_TRACKING_LOST"
        assert kwargs["alert"] is True

    def test_no_position_still_escalates_without_error(self):
        engine = make_engine()
        tracker = create_autospec(LivePositionTracker, instance=True)
        tracker.get_position.return_value = None
        engine.live_position_tracker = tracker

        with patch.object(engine, "_record_event") as record:
            engine._handle_order_tracking_lost("orphan_1", "ETHUSDT", 10)

        record.assert_called_once()

    def test_order_tracker_wiring_routes_give_up_to_tracking_lost(self):
        """End-to-end: a tracker constructed like the engine's must call the
        fail-closed handler, never _handle_order_cancel, on polling give-up."""
        from src.engines.live.order_tracker import MAX_API_ERROR_RETRIES, OrderTracker

        engine = make_engine()
        exchange = MagicMock()
        exchange.get_order.return_value = None
        tracker = OrderTracker(
            exchange=exchange,
            poll_interval=0.01,
            on_fill=engine._handle_order_fill,
            on_partial_fill=engine._handle_partial_fill,
            on_cancel=engine._handle_order_cancel,
            on_tracking_lost=engine._handle_order_tracking_lost,
        )
        tracker.track_order("entry_42", "BTCUSDT")
        _track_fake_position(engine, "entry_42")

        with (
            patch.object(engine, "_handle_order_cancel") as cancel,
            patch.object(engine, "_record_event") as record,
        ):
            tracker.on_cancel = engine._handle_order_cancel
            for _ in range(MAX_API_ERROR_RETRIES):
                tracker._check_orders()

        cancel.assert_not_called()
        record.assert_called_once()
        assert tracker.get_tracked_count() == 0
