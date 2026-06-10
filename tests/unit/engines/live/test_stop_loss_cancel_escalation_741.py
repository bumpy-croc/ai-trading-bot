"""#741: a tracked position's stop-loss terminating unexpectedly must escalate.

Previously _handle_order_cancel only popped positions by ENTRY order id, so a
stop-loss order's terminal CANCELED/REJECTED/EXPIRED notification was a silent
no-op — the position stayed believed-protected with no re-protection and no
alert. Now the engine clears the stale stop_loss_order_id (so the periodic
reconciler's missing-stop path re-places protection) and emits a critical
system event + alert.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.database.models import EventType
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
    engine.db_manager = MagicMock()
    return engine


def _with_tracked_position(engine: LiveTradingEngine, sl_id: str = "sl_42") -> MagicMock:
    position = MagicMock()
    position.symbol = "BTCUSDT"
    position.order_id = "entry_42"
    position.stop_loss_order_id = sl_id
    engine.live_position_tracker = MagicMock()
    engine.live_position_tracker.positions = {"entry_42": position}
    return position


class TestStopLossCancelEscalation:
    def test_sl_cancel_clears_id_and_escalates(self):
        engine = make_engine()
        _with_tracked_position(engine, "sl_42")

        with patch.object(engine, "_record_event") as record:
            engine._handle_order_cancel("sl_42", "BTCUSDT")

        engine.live_position_tracker.set_stop_loss_order_id.assert_called_once_with(
            "entry_42", None
        )
        record.assert_called_once()
        args, kwargs = record.call_args
        assert args[0] == EventType.ERROR
        assert "UNPROTECTED" in args[1]
        assert kwargs["severity"] == "critical"
        assert kwargs["error_code"] == "STOP_LOSS_CANCELLED"
        assert kwargs["alert"] is True
        # The entry-cancel branch must not run for an SL order id
        engine.live_position_tracker.pop_position.assert_not_called()

    def test_unrelated_order_id_is_not_escalated(self):
        engine = make_engine()
        _with_tracked_position(engine, "sl_42")
        engine.live_position_tracker.pop_position.return_value = None

        with patch.object(engine, "_record_event") as record:
            engine._handle_order_cancel("some_other_order", "BTCUSDT")

        record.assert_not_called()
        engine.live_position_tracker.set_stop_loss_order_id.assert_not_called()
        # falls through to the normal entry-cancel handling
        engine.live_position_tracker.pop_position.assert_called_once_with("some_other_order")

    def test_entry_cancel_path_unchanged(self):
        engine = make_engine()
        position = MagicMock()
        position.symbol = "BTCUSDT"
        position.metadata = {}
        position.stop_loss_order_id = None
        engine.live_position_tracker = MagicMock()
        engine.live_position_tracker.positions = {"entry_9": position}
        engine.live_position_tracker.pop_position.return_value = position

        engine._handle_order_cancel("entry_9", "BTCUSDT")

        engine.live_position_tracker.pop_position.assert_called_once_with("entry_9")
