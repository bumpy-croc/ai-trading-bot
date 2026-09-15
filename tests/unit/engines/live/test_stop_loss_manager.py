"""Unit tests for LiveStopLossManager (#486 handler extraction).

Behavioral coverage of the cancel/fill/reprotect paths already exists in
test_close_cancel_stop_710.py and test_stop_loss_cancel_escalation_741.py via
the engine wrappers; these tests cover the manager's own contract — dynamic
engine-state reads, placement retry/registration, and offline-fill detection.
"""

import threading
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from src.data_providers.exchange_interface import OrderSide, SideEffectType
from src.data_providers.exchange_interface import OrderStatus as ExchangeOrderStatus
from src.engines.live.execution.stop_loss_manager import LiveStopLossManager
from src.engines.live.reconciliation import BaseAssetLockRegistry
from src.engines.shared.models import PositionSide

pytestmark = pytest.mark.fast


def make_exchange(**overrides):
    """Exchange stand-in that, by default, confirms no resting stop for any
    symbol (#1112's guard_stop_placement PROCEED case) so existing placement
    tests keep exercising the actual placement path unless a test overrides
    ``get_open_orders_checked`` to probe the guard's ADOPT/REFUSE branches.
    """
    exchange = Mock()
    exchange.get_open_orders_checked.return_value = []
    for key, value in overrides.items():
        setattr(exchange, key, value)
    return exchange


def make_state(**overrides):
    """Engine-state stand-in with the attributes the manager reads at call time.

    ``_base_asset_locks`` is a real (not mocked) ``BaseAssetLockRegistry`` by
    default so ``move()``'s lock-for-the-duration-of-the-mutation behavior
    (#1167 P0) is exercised for real rather than through a mock that would
    silently accept any usage, including a missing lock.
    """
    state = SimpleNamespace(
        enable_live_trading=True,
        exchange_interface=make_exchange(),
        order_tracker=Mock(),
        live_position_tracker=Mock(),
        db_manager=Mock(),
        trading_session_id=1,
        _base_asset_locks=BaseAssetLockRegistry(),
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


class TestPlaceProtectionRestingStopGuard1112:
    """#1112: place_protection must consult the resting-stop guard before
    calling place_stop_loss_order, so a duplicate can never stack."""

    @staticmethod
    def _resting_order(side, order_id="already_resting"):
        return SimpleNamespace(order_id=order_id, side=side, stop_price=48000.0)

    def test_adopts_matching_untracked_resting_stop_without_placing(self):
        state = make_state()
        state.exchange_interface.get_open_orders_checked.return_value = [
            self._resting_order(OrderSide.SELL)
        ]
        manager = LiveStopLossManager(engine_state=state, send_alert=Mock())
        position = make_position(stop_loss_order_id=None)

        sl_order_id = manager.place_protection(
            position=position,
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            quantity=0.5,
            stop_price=48000.0,
        )

        assert sl_order_id == "already_resting"
        state.exchange_interface.place_stop_loss_order.assert_not_called()
        state.live_position_tracker.set_stop_loss_order_id.assert_called_once_with(
            "entry-1", "already_resting"
        )
        state.order_tracker.track_order.assert_called_once_with("already_resting", "BTCUSDT")

    def test_refuses_when_resting_order_is_wrong_side(self):
        state = make_state()
        state.exchange_interface.get_open_orders_checked.return_value = [
            self._resting_order(OrderSide.BUY)
        ]
        manager = LiveStopLossManager(engine_state=state, send_alert=Mock())

        sl_order_id = manager.place_protection(
            position=make_position(stop_loss_order_id=None),
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            quantity=0.5,
            stop_price=48000.0,
        )

        assert sl_order_id is None
        state.exchange_interface.place_stop_loss_order.assert_not_called()
        state.order_tracker.track_order.assert_not_called()

    def test_refuses_when_open_orders_lookup_is_unconfirmed(self):
        state = make_state()
        state.exchange_interface.get_open_orders_checked.return_value = None
        manager = LiveStopLossManager(engine_state=state, send_alert=Mock())

        sl_order_id = manager.place_protection(
            position=make_position(stop_loss_order_id=None),
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            quantity=0.5,
            stop_price=48000.0,
        )

        assert sl_order_id is None
        state.exchange_interface.place_stop_loss_order.assert_not_called()


class TestMove:
    """#1167: a trailing-stop ratchet must move the resting exchange order
    (cancel + re-place), not just the in-memory/DB stop_loss value."""

    @staticmethod
    def _held_exchange(**overrides):
        """Spot exchange stand-in confirmed as holding the position's base asset."""
        exchange = make_exchange(**overrides)
        exchange.is_margin_mode = False
        exchange.get_balance.return_value = SimpleNamespace(free=0.5, locked=0.0)
        exchange.cancel_order.return_value = True
        return exchange

    def test_cancels_old_order_and_places_new_one_at_new_price(self):
        exchange = self._held_exchange()
        exchange.place_stop_loss_order.return_value = "sl-new"
        state = make_state(exchange_interface=exchange)
        manager = LiveStopLossManager(engine_state=state, send_alert=Mock())

        moved = manager.move(make_position(), 49000.0)

        assert moved is True
        exchange.cancel_order.assert_called_once_with("sl-1", "BTCUSDT")
        exchange.place_stop_loss_order.assert_called_once_with(
            symbol="BTCUSDT",
            side=OrderSide.SELL,
            quantity=0.5,
            stop_price=49000.0,
            side_effect_type=SideEffectType.AUTO_REPAY,
        )
        state.live_position_tracker.set_stop_loss_order_id.assert_called_once_with(
            "entry-1", "sl-new"
        )
        state.order_tracker.track_order.assert_called_once_with("sl-new", "BTCUSDT")

    def test_short_position_uses_buy_side(self):
        exchange = self._held_exchange()
        exchange.place_stop_loss_order.return_value = "sl-new"
        state = make_state(exchange_interface=exchange)
        manager = LiveStopLossManager(engine_state=state, send_alert=Mock())

        manager.move(make_position(side=PositionSide.SHORT), 53000.0)

        call = exchange.place_stop_loss_order.call_args
        assert call.kwargs["side"] == OrderSide.BUY

    def test_paper_mode_short_circuits_without_exchange_calls(self):
        state = make_state(enable_live_trading=False)
        manager = LiveStopLossManager(engine_state=state, send_alert=Mock())

        moved = manager.move(make_position(), 49000.0)

        assert moved is False
        state.exchange_interface.cancel_order.assert_not_called()

    def test_no_resting_order_is_a_no_op(self):
        state = make_state()
        manager = LiveStopLossManager(engine_state=state, send_alert=Mock())

        moved = manager.move(make_position(stop_loss_order_id=None), 49000.0)

        assert moved is False
        state.exchange_interface.cancel_order.assert_not_called()
        state.exchange_interface.place_stop_loss_order.assert_not_called()

    def test_skips_when_position_no_longer_held(self):
        exchange = make_exchange()
        exchange.is_margin_mode = False
        exchange.get_balance.return_value = SimpleNamespace(free=0.0, locked=0.0)
        state = make_state(exchange_interface=exchange)
        manager = LiveStopLossManager(engine_state=state, send_alert=Mock())

        moved = manager.move(make_position(), 49000.0)

        assert moved is False
        exchange.cancel_order.assert_not_called()

    def test_does_not_place_when_cancel_fails(self):
        exchange = self._held_exchange()
        exchange.cancel_order.return_value = False
        state = make_state(exchange_interface=exchange)
        manager = LiveStopLossManager(engine_state=state, send_alert=Mock())

        moved = manager.move(make_position(), 49000.0)

        assert moved is False
        exchange.place_stop_loss_order.assert_not_called()

    def test_refuses_and_alerts_when_guard_detects_ambiguous_resting_stop(self):
        exchange = self._held_exchange()
        # A resting order that survives the exclude_order_id filter (different
        # id) on the correct side but at an unrelated price -> REFUSE (#1112).
        exchange.get_open_orders_checked.return_value = [
            SimpleNamespace(order_id="mystery", side=OrderSide.SELL, stop_price=47000.0)
        ]
        send_alert = Mock()
        state = make_state(exchange_interface=exchange)
        manager = LiveStopLossManager(engine_state=state, send_alert=send_alert)

        moved = manager.move(make_position(), 49000.0)

        assert moved is False
        exchange.place_stop_loss_order.assert_not_called()
        send_alert.assert_called_once()

    @patch("src.engines.live.execution.stop_loss_manager.time.sleep")
    def test_escalates_when_replacement_fails_after_cancel(self, mock_sleep):
        exchange = self._held_exchange()
        exchange.place_stop_loss_order.return_value = None
        send_alert = Mock()
        state = make_state(exchange_interface=exchange)
        manager = LiveStopLossManager(engine_state=state, send_alert=send_alert)

        moved = manager.move(make_position(), 49000.0)

        assert moved is False
        assert exchange.cancel_order.call_count == 1
        assert exchange.place_stop_loss_order.call_count == 3
        send_alert.assert_called_once()
        # #1167 P1: cancel-succeeded/re-place-failed must escalate with a
        # persisted CRITICAL audit row, not just a log line + alert — matching
        # the periodic reconciler's identical scenario (_audit_unprotected).
        state.db_manager.log_audit_event.assert_called_once()
        audit_call = state.db_manager.log_audit_event.call_args.kwargs
        assert audit_call["severity"] == "CRITICAL"
        assert audit_call["field"] == "stop_loss_order_id"

    def test_rejects_invalid_new_stop_price_before_touching_exchange(self):
        """#1167 P2: validate new_stop_price BEFORE cancelling, like reprotect()
        does for its own stop_price — once cancelled it's too late to bail
        safely."""
        exchange = self._held_exchange()
        state = make_state(exchange_interface=exchange)
        manager = LiveStopLossManager(engine_state=state, send_alert=Mock())

        for bad_price in (0.0, -1.0, float("nan"), float("inf")):
            moved = manager.move(make_position(), bad_price)
            assert moved is False

        exchange.cancel_order.assert_not_called()
        exchange.place_stop_loss_order.assert_not_called()


class TestMoveBaseAssetLock:
    """#1167 P0: move()'s cancel-guard-place sequence must serialise on the
    position's base-asset lock exactly like execute_entry/execute_exit and the
    periodic reconciler's own re-placement — otherwise a concurrent
    reconciliation cycle can observe the naked post-cancel window and place a
    second resting stop on the same held quantity (#1104/#1108 class)."""

    def test_lock_is_held_for_the_whole_cancel_and_place_round_trip(self):
        registry = BaseAssetLockRegistry()
        exchange = TestMove._held_exchange()
        exchange.place_stop_loss_order.return_value = "sl-new"

        entered_cancel = threading.Event()
        release_cancel = threading.Event()

        def blocking_cancel(order_id, symbol):
            entered_cancel.set()
            release_cancel.wait(timeout=2)
            return True

        exchange.cancel_order.side_effect = blocking_cancel
        state = make_state(exchange_interface=exchange, _base_asset_locks=registry)
        manager = LiveStopLossManager(engine_state=state, send_alert=Mock())

        mover = threading.Thread(target=manager.move, args=(make_position(), 49000.0))
        mover.start()
        try:
            assert entered_cancel.wait(timeout=2), "move() never reached cancel()"

            # While move() is inside its cancel-guard-place section, a second
            # thread (standing in for the periodic reconciler's re-placement)
            # must NOT be able to acquire the same base-asset lock.
            lock = registry.lock_for("BTC")
            acquired_by_other_thread = lock.acquire(blocking=False)
            try:
                assert not acquired_by_other_thread, (
                    "base-asset lock was not held during move()'s cancel/place — "
                    "the reconciler could race it and stack a duplicate stop"
                )
            finally:
                if acquired_by_other_thread:
                    lock.release()
        finally:
            release_cancel.set()
            mover.join(timeout=2)

        # Released once move() completes.
        lock = registry.lock_for("BTC")
        assert lock.acquire(blocking=False)
        lock.release()


class TestMoveAdoptBranch:
    """#1167/#1112: move()'s ADOPT branch (guard_stop_placement finds an
    untracked resting stop within tolerance) must track the ACHIEVED price,
    not the ratchet's intended price — guard_stop_placement only guarantees
    the adopted order is within tolerance, not equal to what was asked for."""

    @staticmethod
    def _resting_order(side, stop_price, order_id="already_resting"):
        return SimpleNamespace(order_id=order_id, side=side, stop_price=stop_price)

    def test_adopts_and_persists_the_actual_resting_price_not_the_intent(self):
        exchange = TestMove._held_exchange()
        # Adopted order rests at 48990, not the intended 49000 -- within
        # guard_stop_placement's tolerance but not identical.
        exchange.get_open_orders_checked.return_value = [
            self._resting_order(OrderSide.SELL, 48990.0)
        ]
        state = make_state(exchange_interface=exchange)
        manager = LiveStopLossManager(engine_state=state, send_alert=Mock())
        position = make_position()

        moved = manager.move(position, 49000.0)

        assert moved is True
        exchange.place_stop_loss_order.assert_not_called()
        state.live_position_tracker.set_stop_loss_order_id.assert_called_once_with(
            "entry-1", "already_resting"
        )
        # The tracked stop_loss price must reflect what's actually resting on
        # the exchange (48990), not the ratchet's intent (49000) -- otherwise
        # the engine's own exit check trusts a price the exchange will never
        # trigger at, the exact #1167 divergence in a different guise.
        state.live_position_tracker.set_stop_loss_price.assert_called_once_with("entry-1", 48990.0)

    def test_adopting_at_exactly_the_intended_price_does_not_rewrite_it(self):
        exchange = TestMove._held_exchange()
        exchange.get_open_orders_checked.return_value = [
            self._resting_order(OrderSide.SELL, 49000.0)
        ]
        state = make_state(exchange_interface=exchange)
        manager = LiveStopLossManager(engine_state=state, send_alert=Mock())

        moved = manager.move(make_position(), 49000.0)

        assert moved is True
        state.live_position_tracker.set_stop_loss_price.assert_not_called()


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
