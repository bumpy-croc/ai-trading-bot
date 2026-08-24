"""#1104: the ETHUSDT stop-loss place -> die -> replace loop of 2026-08-19.

Two independent defects combined into a self-sustaining loop that left a live
position unprotected for ~66s at a time, once per trading-loop iteration:

1. ``LiveStopLossManager.cancel`` issued the exchange cancel BEFORE untracking
   the order. Binance emits the CANCELED executionReport on the already-open
   user socket before the DELETE response returns, so the OrderTracker fired the
   unexpected-cancel escalation for a cancel WE issued — a false UNPROTECTED
   page — and that handler nulls ``position.stop_loss_order_id``. With the id
   gone, the reconciler's missing-stop path stacked a DUPLICATE stop on top of
   one still resting on the exchange, orphaning it.
2. The orphaned stop locked the position's entire base inventory, so the
   free-base cap in ``ExecutionEngine._close_live_position`` silently shrank
   every close to the leftover dust (prod: 0.00009419 of 0.0087 ETH). Had that
   dust order cleared MIN_NOTIONAL it would have filled, and the caller books a
   FULL close on success — abandoning ~99% of the inventory untracked and
   unprotected.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.data_providers.exchange_interface import OrderSide
from src.engines.live.execution.execution_engine import LiveExecutionEngine
from src.engines.live.execution.stop_loss_manager import LiveStopLossManager
from src.engines.shared.models import PositionSide

pytestmark = pytest.mark.fast


def _position(**overrides):
    position = SimpleNamespace(
        symbol="ETHUSDT",
        side=PositionSide.LONG,
        order_id="entry-48553879892",
        stop_loss_order_id="49075673082",
        stop_loss=2072.3612,
        quantity=0.0087,
        current_size=0.20,
        original_size=0.20,
    )
    for key, value in overrides.items():
        setattr(position, key, value)
    return position


class _RacingExchange:
    """Exchange whose cancel_order fires the tracker callback mid-round-trip.

    Models the real Binance behaviour: the CANCELED executionReport lands on the
    open user socket before the DELETE HTTP response returns.
    """

    def __init__(self, tracker, position, *, confirm: bool = True):
        self._tracker = tracker
        self._position = position
        self._confirm = confirm
        self.cancelled: list[str] = []

    def cancel_order(self, order_id: str, symbol: str) -> bool:
        self.cancelled.append(order_id)
        # The WS/poll callback arrives here, before cancel_order has returned.
        self._tracker.deliver_terminal_status(order_id, symbol)
        return self._confirm


class _Tracker:
    """OrderTracker stand-in that only escalates ids it is still tracking.

    Mirrors OrderTracker.process_execution_event / _process_order_status, which
    early-return when the id is absent from ``_pending_orders``.
    """

    def __init__(self, on_cancel):
        self.tracked: set[str] = set()
        self.calls: list[tuple[str, str]] = []
        self._on_cancel = on_cancel

    def track_order(self, order_id: str, symbol: str) -> None:
        self.calls.append(("track", order_id))
        self.tracked.add(order_id)

    def stop_tracking(self, order_id: str) -> None:
        self.calls.append(("stop", order_id))
        self.tracked.discard(order_id)

    def deliver_terminal_status(self, order_id: str, symbol: str) -> None:
        if order_id in self.tracked:
            self._on_cancel(order_id, symbol)


class TestDeliberateCancelDoesNotSelfEscalate:
    """Defect 1: our own pre-close cancel must never fire the UNPROTECTED path."""

    def _build(self, *, confirm: bool = True):
        position = _position()
        escalations: list[str] = []

        def on_cancel(order_id: str, symbol: str) -> None:
            # What LiveOrderFillCoordinator.handle_stop_loss_cancelled does:
            # emit the critical STOP_LOSS_CANCELLED event and null the shared
            # field on the tracked position object.
            if position.stop_loss_order_id == order_id:
                escalations.append(order_id)
                position.stop_loss_order_id = None

        tracker = _Tracker(on_cancel)
        tracker.track_order(position.stop_loss_order_id, position.symbol)
        tracker.calls.clear()  # setup noise; only the cancel path is under test
        exchange = _RacingExchange(tracker, position, confirm=confirm)
        state = SimpleNamespace(
            enable_live_trading=True,
            exchange_interface=exchange,
            order_tracker=tracker,
            live_position_tracker=Mock(),
        )
        manager = LiveStopLossManager(engine_state=state, send_alert=Mock())
        return manager, position, tracker, exchange, escalations

    def test_confirmed_cancel_emits_no_unprotected_escalation(self):
        manager, position, tracker, exchange, escalations = self._build()

        assert manager.cancel(position) is True

        # The cancel reached the exchange with the real id...
        assert exchange.cancelled == ["49075673082"]
        # ...and untracking happened FIRST, so the terminal-status callback
        # found nothing to escalate.
        assert tracker.calls[0] == ("stop", "49075673082")
        assert escalations == []
        # The shared field survives: the reconciler will not see "no stop-loss"
        # and stack a duplicate on a still-resting order.
        assert position.stop_loss_order_id == "49075673082"
        assert "49075673082" not in tracker.tracked

    def test_unconfirmed_cancel_restores_tracking(self):
        """A cancel we could not confirm may still be resting — keep watching it."""
        manager, position, tracker, _exchange, escalations = self._build(confirm=False)

        assert manager.cancel(position) is False

        assert "49075673082" in tracker.tracked
        assert escalations == []
        assert position.stop_loss_order_id == "49075673082"

    def test_id_is_read_once_even_if_cleared_mid_cancel(self):
        """The cancel must not re-read a field another thread can null."""
        position = _position()
        tracker = _Tracker(lambda *_: None)
        tracker.track_order("49075673082", "ETHUSDT")
        tracker.calls.clear()

        def cancel_order(order_id: str, symbol: str) -> bool:
            # Concurrent escalation clears the shared field mid-round-trip.
            position.stop_loss_order_id = None
            return True

        state = SimpleNamespace(
            enable_live_trading=True,
            exchange_interface=SimpleNamespace(cancel_order=cancel_order),
            order_tracker=tracker,
            live_position_tracker=Mock(),
        )
        manager = LiveStopLossManager(engine_state=state, send_alert=Mock())

        assert manager.cancel(position) is True
        # stop_tracking was called with the real id, never with None.
        assert ("stop", "49075673082") in tracker.calls
        assert ("stop", None) not in tracker.calls
        assert "49075673082" not in tracker.tracked


class TestCloseIsNeverSilentlyShrunkToDust:
    """Defect 2: locked inventory must abort the close, not shrink it."""

    def _engine(self, free_base: float) -> LiveExecutionEngine:
        engine = LiveExecutionEngine(
            enable_live_trading=True,
            exchange_interface=Mock(),
        )
        engine._free_base_for_close = lambda symbol: free_base  # type: ignore[method-assign]
        engine._normalize_quantity = (  # type: ignore[method-assign]
            lambda symbol, quantity, notional, floor=False: quantity
        )
        return engine

    def test_orphaned_stop_locking_inventory_aborts_the_close(self):
        """The exact prod numbers: 0.0087 intended, 0.00009419 free."""
        engine = self._engine(free_base=0.00009419)

        order_id = engine._close_live_order(
            symbol="ETHUSDT",
            side=PositionSide.LONG,
            quantity=0.0087,
            position_notional=18.0,
        )

        assert order_id is None
        # No order was ever submitted — the dangerous branch is not reached.
        engine.exchange_interface.place_order.assert_not_called()

    def test_fee_rounding_sliver_still_caps_and_proceeds(self):
        """The cap's legitimate purpose (a sub-1% shortfall) is preserved."""
        engine = self._engine(free_base=0.0087 * 0.999)
        engine.exchange_interface.place_order.return_value = SimpleNamespace(
            order_id="ok", filled_quantity=0.0087
        )

        engine._close_live_order(
            symbol="ETHUSDT",
            side=PositionSide.LONG,
            quantity=0.0087,
            position_notional=18.0,
        )

        assert engine.exchange_interface.place_order.called
        submitted = engine.exchange_interface.place_order.call_args.kwargs
        assert submitted["side"] == OrderSide.SELL
        assert submitted["quantity"] == pytest.approx(0.0087 * 0.999)
