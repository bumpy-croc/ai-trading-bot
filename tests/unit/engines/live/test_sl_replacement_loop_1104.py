"""#1104: the ETHUSDT stop-loss place -> die -> replace loop of 2026-08-19.

Three defects combined into a self-sustaining loop that left a live position
unprotected for ~66s at a time, once per trading-loop iteration:

1. ``LiveStopLossManager.cancel`` gave the OrderTracker no way to tell OUR
   cancel from an unexpected one. Binance emits the terminal executionReport on
   the already-open user socket before the DELETE response returns, so the
   tracker escalated our own cancel as an unexpected termination — a false
   UNPROTECTED page — and that handler nulls ``position.stop_loss_order_id``.
   With the id gone, the reconciler's missing-stop path stacked a DUPLICATE stop
   on one still resting, orphaning it. The fix must NOT untrack across the
   cancel: a stop is cancelled exactly when price is touching it, so a genuine
   FILL in that window is the likely case and must still be processed.
2. The orphaned stop locked the position's base inventory, so the free-base cap
   in ``LiveExecutionEngine._close_live_order`` silently shrank the close to the
   leftover dust (prod: 0.00009419 of 0.0087 ETH). The caller books a FULL close
   on success, abandoning the rest untracked and unprotected.
3. Under that same condition the re-protect path placed a stop covering only the
   dust and recorded it as full protection — the reconciler's SL audit checks
   order status, never quantity (#1109).
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.config.constants import CLOSE_ABORT_CLOSE_ONLY_STREAK, HOLDINGS_CAP_MIN_RATIO
from src.data_providers.exchange_interface import Order, OrderSide, OrderStatus, OrderType
from src.engines.live.execution.execution_engine import LiveExecutionEngine
from src.engines.live.execution.stop_loss_manager import LiveStopLossManager
from src.engines.live.order_tracker import (
    SELF_CANCEL_SUPPRESSION_TTL_SECONDS,
    OrderTracker,
)
from src.engines.shared.models import PositionSide

pytestmark = pytest.mark.fast

SL_ID = "49075673082"


def _position(**overrides):
    position = SimpleNamespace(
        symbol="ETHUSDT",
        side=PositionSide.LONG,
        order_id="entry-48553879892",
        stop_loss_order_id=SL_ID,
        stop_loss=2072.3612,
        quantity=0.0087,
        current_size=0.20,
        original_size=0.20,
    )
    for key, value in overrides.items():
        setattr(position, key, value)
    return position


def _order(status: OrderStatus, filled: float = 0.0) -> Order:
    return Order(
        order_id=SL_ID,
        symbol="ETHUSDT",
        side=OrderSide.SELL,
        order_type=OrderType.STOP_LOSS,
        quantity=0.0087,
        price=2072.3612,
        status=status,
        filled_quantity=filled,
        average_price=2072.3612 if filled else 0.0,
        commission=0.0,
        commission_asset="USDT",
        create_time=datetime.now(UTC),
        update_time=datetime.now(UTC),
    )


class _Harness:
    """A REAL OrderTracker wired to recording callbacks.

    The exchange's ``cancel_order`` delivers the terminal executionReport
    mid-round-trip, exactly as Binance does over the open user socket.
    """

    def __init__(self, *, terminal: Order, confirm_cancel: bool = True):
        self.position = _position()
        self.escalations: list[str] = []
        self.fills: list[tuple[str, float]] = []
        self.terminal = terminal
        self.confirm_cancel = confirm_cancel

        def on_cancel(order_id: str, symbol: str, filled_qty: float = 0.0) -> None:
            # What LiveOrderFillCoordinator.handle_stop_loss_cancelled does: page,
            # and null the shared field on the tracked position object.
            if self.position.stop_loss_order_id == order_id:
                self.escalations.append(order_id)
                self.position.stop_loss_order_id = None

        def on_fill(order_id: str, symbol: str, filled_qty: float, avg_price: float) -> None:
            self.fills.append((order_id, filled_qty))

        self.tracker = OrderTracker(
            exchange=Mock(),
            on_fill=on_fill,
            on_cancel=on_cancel,
        )
        self.tracker.track_order(SL_ID, "ETHUSDT")

        def cancel_order(order_id: str, symbol: str) -> bool:
            # The terminal report lands before the DELETE response returns.
            self.tracker._process_order_status(
                order_id, self.tracker._pending_orders[order_id], self.terminal
            )
            return self.confirm_cancel

        self.state = SimpleNamespace(
            enable_live_trading=True,
            exchange_interface=SimpleNamespace(cancel_order=cancel_order),
            order_tracker=self.tracker,
            live_position_tracker=Mock(),
        )
        self.manager = LiveStopLossManager(engine_state=self.state, send_alert=Mock())


class TestDeliberateCancelIsNotEscalated:
    """Defect 1: our own pre-close cancel must never fire the UNPROTECTED path."""

    def test_confirmed_cancel_emits_no_unprotected_escalation(self):
        h = _Harness(terminal=_order(OrderStatus.CANCELLED))

        assert h.manager.cancel(h.position) is True

        assert h.escalations == []
        # The shared field survives, so the reconciler will not see "no stop-loss"
        # and stack a duplicate on a still-resting order.
        assert h.position.stop_loss_order_id == SL_ID
        assert SL_ID not in h.tracker._pending_orders

    def test_stop_that_fills_during_the_cancel_is_still_processed(self):
        """The regression the fix must not introduce.

        A stop is cancelled at the exact moment price is touching it, so a FILL
        landing in the cancel window is the likely case. Untracking before the
        cancel would discard it — process_execution_event early-returns on
        unknown ids — and with the WS primary, polling is disabled and nothing
        re-delivers it.
        """
        h = _Harness(terminal=_order(OrderStatus.FILLED, filled=0.0087))

        h.manager.cancel(h.position)

        assert h.fills == [(SL_ID, 0.0087)]
        assert h.escalations == []

    def test_partial_fill_carried_by_the_terminal_status_is_still_booked(self):
        partials: list[float] = []
        h = _Harness(terminal=_order(OrderStatus.CANCELLED, filled=0.004))
        h.tracker.on_partial_fill = lambda oid, sym, qty, price: partials.append(qty)

        h.manager.cancel(h.position)

        # Suppression applies to the escalation only, never to fill accounting.
        assert partials == [0.004]
        assert h.escalations == []

    def test_unconfirmed_cancel_keeps_tracking_and_re_arms_escalation(self):
        h = _Harness(terminal=_order(OrderStatus.CANCELLED), confirm_cancel=False)
        h.tracker.track_order(SL_ID, "ETHUSDT")  # still resting as far as we know

        assert h.manager.cancel(h.position) is False

        # The mark is cleared, so a genuine later cancellation still escalates.
        assert h.tracker._consume_self_cancelled(SL_ID) is False
        assert h.position.stop_loss_order_id == SL_ID

    def test_a_genuine_unexpected_cancel_still_escalates(self):
        """The #741 behaviour must be untouched for cancels we did not issue."""
        h = _Harness(terminal=_order(OrderStatus.CANCELLED))

        h.tracker._process_order_status(
            SL_ID, h.tracker._pending_orders[SL_ID], _order(OrderStatus.CANCELLED)
        )

        assert h.escalations == [SL_ID]
        assert h.position.stop_loss_order_id is None

    def test_stale_self_cancel_mark_expires(self):
        """An abandoned mark must not silence a real cancellation minutes later."""
        h = _Harness(terminal=_order(OrderStatus.CANCELLED))
        h.tracker.mark_self_cancelled(SL_ID)
        # Backdate the mark past its TTL, as a cancel that crashed mid-flight would.
        h.tracker._self_cancelled[SL_ID] -= SELF_CANCEL_SUPPRESSION_TTL_SECONDS + 1

        assert h.tracker._consume_self_cancelled(SL_ID) is False

    def test_id_is_read_once_even_if_cleared_mid_cancel(self):
        position = _position()
        tracker = OrderTracker(exchange=Mock())
        tracker.track_order(SL_ID, "ETHUSDT")

        def cancel_order(order_id: str, symbol: str) -> bool:
            assert order_id == SL_ID  # never None
            position.stop_loss_order_id = None  # concurrent escalation
            return True

        state = SimpleNamespace(
            enable_live_trading=True,
            exchange_interface=SimpleNamespace(cancel_order=cancel_order),
            order_tracker=tracker,
            live_position_tracker=Mock(),
        )

        assert LiveStopLossManager(engine_state=state, send_alert=Mock()).cancel(position) is True
        # stop_tracking ran with the real id, so the order is genuinely untracked.
        assert SL_ID not in tracker._pending_orders


class _CloseHarness:
    """LiveExecutionEngine with a REAL _normalize_quantity over a stub exchange."""

    def __init__(self, *, free_base: float, step_size: float = 0.00001):
        self.exchange = Mock()
        self.exchange.get_symbol_info.return_value = {
            "step_size": step_size,
            "min_qty": 0.0,
            "min_notional": 0.0,
        }
        balance = Mock()
        balance.free = free_base
        self.exchange.get_balance.return_value = balance
        self.engine = LiveExecutionEngine(
            enable_live_trading=True,
            exchange_interface=self.exchange,
        )
        self.events: list[tuple[str, dict]] = []
        self.alerts: list[str] = []
        self.criticals: list[str] = []
        self.engine._log_execution_event = (  # type: ignore[method-assign]
            lambda event_type, message, error_code, *, severity="error", details=None: (
                self.events.append((error_code, {"message": message, **(details or {})}))
            )
        )
        self.engine.alert_dispatcher = self.alerts.append
        self.engine.on_critical = self.criticals.append

    def close(self, quantity: float, notional: float = 18.0):
        return self.engine._close_live_order(
            symbol="ETHUSDT",
            side=PositionSide.LONG,
            quantity=quantity,
            position_notional=notional,
        )


class TestCloseIsNeverSilentlyShrunk:
    """Defect 2: an unsellable close aborts loudly instead of selling a fraction."""

    def test_orphaned_stop_locking_inventory_aborts_the_close(self):
        """The exact prod numbers: 0.0087 intended, 0.00009419 free."""
        h = _CloseHarness(free_base=0.00009419)

        assert h.close(0.0087) is None
        h.exchange.place_order.assert_not_called()

    def test_lot_floor_alone_can_trigger_the_abort(self):
        """The gate must see the SUBMITTED quantity, not just free_base.

        free_base is 0.9999 of intended, so a free_base-only gate passes — but the
        lot snap then floors 4 lots down to 3, submitting 75% of the position while
        the caller books a FULL close. This is the case the first fix missed.
        """
        h = _CloseHarness(free_base=0.00039996, step_size=0.0001)

        assert h.close(0.0004, notional=40.0) is None
        h.exchange.place_order.assert_not_called()

    def test_records_and_pages_the_close_inventory_locked_event(self):
        h = _CloseHarness(free_base=0.00009419)

        h.close(0.0087)

        assert [code for code, _ in h.events] == ["CLOSE_INVENTORY_LOCKED"]
        _code, details = h.events[0]
        assert details["intended_quantity"] == pytest.approx(0.0087)
        assert details["free_base_balance"] == pytest.approx(0.00009419)
        assert details["consecutive_aborts"] == 1
        # A DB row nobody reads is the #1096 shape — this must page too.
        assert len(h.alerts) == 1
        assert "ETHUSDT" in h.alerts[0]

    def test_repeated_aborts_latch_close_only_instead_of_paging_forever(self):
        """The exit signal re-fires every ~66s; an un-latched abort recreates #1104."""
        h = _CloseHarness(free_base=0.00009419)

        for _ in range(CLOSE_ABORT_CLOSE_ONLY_STREAK):
            h.close(0.0087)

        assert len(h.criticals) == 1
        assert "close aborted" in h.criticals[0]
        assert h.events[-1][1]["consecutive_aborts"] == CLOSE_ABORT_CLOSE_ONLY_STREAK

    def test_a_successful_close_resets_the_streak(self):
        h = _CloseHarness(free_base=0.00009419)
        h.close(0.0087)
        h.exchange.get_balance.return_value.free = 0.0087

        h.close(0.0087)  # succeeds — clears the latch counter
        h.exchange.get_balance.return_value.free = 0.00009419
        h.close(0.0087)

        assert h.events[-1][1]["consecutive_aborts"] == 1

    @pytest.mark.parametrize(
        ("free_base", "aborts"),
        [
            (0.0087 * (HOLDINGS_CAP_MIN_RATIO - 0.001), True),  # just under -> abort
            (0.0087 * (HOLDINGS_CAP_MIN_RATIO + 0.005), False),  # just over -> proceed
        ],
    )
    def test_ratio_boundary(self, free_base: float, aborts: bool):
        """Strict `<`: a submitted quantity exactly at the ratio is allowed."""
        h = _CloseHarness(free_base=free_base)

        result = h.close(0.0087)

        assert (result is None) is aborts
        assert h.exchange.place_order.called is not aborts


class TestUndersizedStopIsRefused:
    """Defect 3 (#1109): never record a dust stop as full protection."""

    def _provider(self, free_base: float):
        from src.data_providers.binance_provider import BinanceProvider

        provider = BinanceProvider.__new__(BinanceProvider)
        provider._client = Mock()
        provider.order_error_sink = None
        provider.get_symbol_info = lambda symbol: {  # type: ignore[method-assign]
            "tick_size": 0.01,
            "step_size": 0.00001,
            "base_asset": "ETH",
        }
        provider._free_base_balance = lambda asset: free_base  # type: ignore[method-assign]
        provider._call_create_order = Mock(return_value={"orderId": "new-sl"})
        return provider

    def test_stop_covering_only_the_dust_is_not_placed(self):
        provider = self._provider(free_base=0.00009419)

        result = provider.place_stop_loss_order(
            symbol="ETHUSDT",
            side=OrderSide.SELL,
            quantity=0.0087,
            stop_price=2072.3612,
        )

        # No order id to record, so the honest UNPROTECTED escalation fires instead
        # of the reconciler seeing a NEW order and believing the position is covered.
        assert result is None
        provider._call_create_order.assert_not_called()

    def test_fee_rounding_sliver_still_places_the_stop(self):
        provider = self._provider(free_base=0.0087 * 0.999)

        result = provider.place_stop_loss_order(
            symbol="ETHUSDT",
            side=OrderSide.SELL,
            quantity=0.0087,
            stop_price=2072.3612,
        )

        assert result == "new-sl"
        assert provider._call_create_order.called
