"""Unit tests for LiveStopLossManager (#486 handler extraction).

Behavioral coverage of the cancel/fill/reprotect paths already exists in
test_close_cancel_stop_710.py and test_stop_loss_cancel_escalation_741.py via
the engine wrappers; these tests cover the manager's own contract — dynamic
engine-state reads, placement retry/registration, and offline-fill detection.
"""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from src.data_providers.exchange_interface import OrderSide, SideEffectType
from src.data_providers.exchange_interface import OrderStatus as ExchangeOrderStatus
from src.engines.live.execution.stop_loss_manager import LiveStopLossManager
from src.engines.shared.models import PositionSide

pytestmark = pytest.mark.fast


def make_state(**overrides):
    """Engine-state stand-in with the attributes the manager reads at call time."""
    state = SimpleNamespace(
        enable_live_trading=True,
        exchange_interface=Mock(),
        order_tracker=Mock(),
        live_position_tracker=Mock(),
    )
    for key, value in overrides.items():
        setattr(state, key, value)
    return state


def make_position(**overrides):
    position = SimpleNamespace(
        symbol="BTCUSDT",
        side=PositionSide.LONG,
        order_id="entry-1",
        stop_loss_order_id="sl-1",
        stop_loss=48000.0,
        quantity=0.5,
        current_size=0.02,
        original_size=0.02,
    )
    for key, value in overrides.items():
        setattr(position, key, value)
    return position


class TestDynamicStateReads:
    def test_reads_exchange_interface_assigned_after_construction(self):
        # Arrange: manager built before the exchange exists (engine startup order)
        state = make_state(exchange_interface=None, enable_live_trading=False)
        manager = LiveStopLossManager(engine_state=state, send_alert=Mock())

        # Act: engine assigns exchange + flips live flag later (start() / tests)
        state.exchange_interface = Mock()
        state.exchange_interface.cancel_order.return_value = True
        state.enable_live_trading = True
        result = manager.cancel(make_position())

        # Assert: the late-bound exchange was used
        assert result is True
        state.exchange_interface.cancel_order.assert_called_once_with("sl-1", "BTCUSDT")


class TestPlaceProtection:
    def test_success_registers_stop_with_tracker_and_order_tracker(self):
        state = make_state()
        state.exchange_interface.place_stop_loss_order.return_value = "sl-99"
        manager = LiveStopLossManager(engine_state=state, send_alert=Mock())
        position = make_position(stop_loss_order_id=None)

        sl_order_id = manager.place_protection(
            position=position,
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            quantity=0.5,
            stop_price=48000.0,
        )

        assert sl_order_id == "sl-99"
        state.exchange_interface.place_stop_loss_order.assert_called_once_with(
            symbol="BTCUSDT",
            side=OrderSide.SELL,
            quantity=0.5,
            stop_price=48000.0,
            side_effect_type=SideEffectType.AUTO_REPAY,
        )
        state.live_position_tracker.set_stop_loss_order_id.assert_called_once_with(
            "entry-1", "sl-99"
        )
        state.order_tracker.track_order.assert_called_once_with("sl-99", "BTCUSDT")

    def test_short_position_uses_buy_side(self):
        state = make_state()
        state.exchange_interface.place_stop_loss_order.return_value = "sl-2"
        manager = LiveStopLossManager(engine_state=state, send_alert=Mock())

        manager.place_protection(
            position=make_position(side=PositionSide.SHORT),
            symbol="BTCUSDT",
            side=PositionSide.SHORT,
            quantity=0.5,
            stop_price=52000.0,
        )

        call = state.exchange_interface.place_stop_loss_order.call_args
        assert call.kwargs["side"] == OrderSide.BUY

    @patch("src.engines.live.execution.stop_loss_manager.time.sleep")
    def test_retries_on_exception_then_succeeds(self, mock_sleep):
        state = make_state()
        state.exchange_interface.place_stop_loss_order.side_effect = [
            ConnectionError("boom"),
            "sl-after-retry",
        ]
        manager = LiveStopLossManager(engine_state=state, send_alert=Mock())

        sl_order_id = manager.place_protection(
            position=make_position(),
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            quantity=0.5,
            stop_price=48000.0,
        )

        assert sl_order_id == "sl-after-retry"
        assert state.exchange_interface.place_stop_loss_order.call_count == 2

    @patch("src.engines.live.execution.stop_loss_manager.time.sleep")
    def test_returns_none_after_exhausting_retries_without_registration(self, mock_sleep):
        state = make_state()
        state.exchange_interface.place_stop_loss_order.return_value = None
        manager = LiveStopLossManager(engine_state=state, send_alert=Mock())

        sl_order_id = manager.place_protection(
            position=make_position(),
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            quantity=0.5,
            stop_price=48000.0,
        )

        assert sl_order_id is None
        assert state.exchange_interface.place_stop_loss_order.call_count == 3
        state.live_position_tracker.set_stop_loss_order_id.assert_not_called()
        state.order_tracker.track_order.assert_not_called()


class TestCheckFilled:
    def test_filled_order_returns_fill_price(self):
        state = make_state()
        state.exchange_interface.get_order.return_value = SimpleNamespace(
            status=ExchangeOrderStatus.FILLED, average_price=47950.0
        )
        manager = LiveStopLossManager(engine_state=state, send_alert=Mock())

        filled, price = manager.check_filled(make_position())

        assert filled is True
        assert price == 47950.0

    def test_paper_mode_short_circuits_without_exchange_call(self):
        state = make_state(enable_live_trading=False)
        manager = LiveStopLossManager(engine_state=state, send_alert=Mock())

        filled, price = manager.check_filled(make_position())

        assert filled is False
        assert price is None
        state.exchange_interface.get_order.assert_not_called()


class TestFindOfflineFilledStops:
    def test_detects_filled_stop_missing_from_open_orders(self):
        state = make_state()
        state.exchange_interface.get_open_orders.return_value = [
            SimpleNamespace(order_id="other-order")
        ]
        state.exchange_interface.get_order.return_value = SimpleNamespace(
            status=ExchangeOrderStatus.FILLED, average_price=47900.0
        )
        manager = LiveStopLossManager(engine_state=state, send_alert=Mock())
        position = make_position()

        result = manager.find_offline_filled_stops({"entry-1": position})

        assert result == [(position, 47900.0)]

    def test_resting_stop_is_not_flagged(self):
        state = make_state()
        state.exchange_interface.get_open_orders.return_value = [SimpleNamespace(order_id="sl-1")]
        manager = LiveStopLossManager(engine_state=state, send_alert=Mock())

        result = manager.find_offline_filled_stops({"entry-1": make_position()})

        assert result == []
        state.exchange_interface.get_order.assert_not_called()

    def test_unverifiable_order_is_skipped_not_closed(self):
        state = make_state()
        state.exchange_interface.get_open_orders.return_value = []
        state.exchange_interface.get_order.side_effect = ConnectionError("api down")
        manager = LiveStopLossManager(engine_state=state, send_alert=Mock())

        result = manager.find_offline_filled_stops({"entry-1": make_position()})

        assert result == []

    def test_open_orders_failure_propagates_to_caller(self):
        state = make_state()
        state.exchange_interface.get_open_orders.side_effect = ConnectionError("api down")
        manager = LiveStopLossManager(engine_state=state, send_alert=Mock())

        with pytest.raises(ConnectionError):
            manager.find_offline_filled_stops({"entry-1": make_position()})
