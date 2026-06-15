"""Live order-fill callback coordinator (#486).

Owns the callbacks the live engine registers with ``OrderTracker``:
full fills (``handle_order_fill``), partial fills (``handle_partial_fill``),
cancellations/rejections (``handle_order_cancel``, with its stop-loss-cancel
escalation helper ``handle_stop_loss_cancelled``), and abandoned-tracking
(``handle_order_tracking_lost``). These were moved verbatim out of
``LiveTradingEngine`` (a mechanical ``self.`` -> ``state.`` rewrite against an
engine backref); the engine keeps thin delegating wrappers and still registers
those wrappers with ``OrderTracker``.

Threading (unchanged from the engine):
- These callbacks fire on the **OrderTracker poll thread**, not the trading
  loop. They mutate no coordinator-local state — everything is read/written
  through the ``state`` backref exactly as before, so the existing
  single-writer / thread-safe-handoff discipline is preserved:
  * A stop-loss full fill does NOT close inline; it enqueues
    ``(position_order_id, avg_price)`` on the thread-safe
    ``state._pending_fill_exits`` SimpleQueue, which the trading loop drains
    and closes with ``skip_live_close=True`` (#631).
  * Position reads use ``LivePositionTracker.positions`` (thread-safe copy) and
    mutations use its atomic ``pop_position`` / ``set_stop_loss_order_id``.
  * The cancel-refund balance update goes through
    ``db_manager.atomic_balance_update`` just as before.
- ``_record_event`` and ``_send_alert`` stay on the engine and are invoked via
  ``state`` so engine-level test mocks still intercept; ``handle_order_cancel``
  -> ``handle_stop_loss_cancelled`` is a coordinator-internal call.
"""

from __future__ import annotations

import logging
import queue
from typing import TYPE_CHECKING, Any, Protocol

from src.database.models import EventType
from src.infrastructure.logging.events import log_order_event

if TYPE_CHECKING:
    from src.engines.live.execution.position_tracker import LivePositionTracker

logger = logging.getLogger(__name__)


class LiveOrderFillEngineState(Protocol):
    """Engine state the order-fill coordinator reads and mutates at call time.

    Accessed dynamically through this backref because the trackers/session are
    wired during construction and ``start()``. The callbacks run on the
    OrderTracker poll thread; see the module docstring for the threading model.
    """

    current_balance: float
    trading_session_id: int | None
    live_position_tracker: LivePositionTracker
    db_manager: Any
    _pending_fill_exits: queue.SimpleQueue

    # Engine helpers that stay on the engine; invoked via this backref so
    # subclass/test overrides on the engine still apply.
    def _send_alert(self, message: str) -> bool: ...

    def _record_event(
        self,
        event_type: EventType,
        message: str,
        *,
        severity: str = ...,
        component: str | None = ...,
        error_code: str | None = ...,
        exc: BaseException | None = ...,
        alert: bool = ...,
    ) -> None: ...


class LiveOrderFillCoordinator:
    """Owns the OrderTracker fill/cancel/tracking-lost callbacks."""

    def __init__(self, engine_state: LiveOrderFillEngineState) -> None:
        """Bind to the engine's live state (see protocol for the surface)."""
        self._state = engine_state

    def handle_order_fill(
        self, order_id: str, symbol: str, filled_qty: float, avg_price: float
    ) -> None:
        """
        Handle a fully filled order notification from OrderTracker.

        This handles both entry order fills and stop-loss order fills.
        For stop-loss fills, it closes the associated position.

        Args:
            order_id: The filled order ID
            symbol: Trading symbol
            filled_qty: Total quantity filled
            avg_price: Average fill price
        """
        state = self._state
        logger.info(
            f"Order fill confirmed: {order_id} {symbol} qty={filled_qty} @ ${avg_price:.2f}"
        )
        log_order_event(
            "order_filled",
            order_id=order_id,
            symbol=symbol,
            filled_quantity=filled_qty,
            average_price=avg_price,
        )

        # If this is a stop-loss order fill, the position must be closed — but the
        # close (DB writes, P&L bookkeeping) is DEFERRED to the trading loop via a
        # queue rather than run here on the OrderTracker poll thread. Running it
        # inline blocked all order polling on a slow close and, on failure, drove
        # the order toward force-removal (orphaning the position). The stop-loss
        # has already executed on-exchange, so the loop drains and runs the exit
        # with skip_live_close=True; a drain failure is backstopped by the
        # periodic/startup reconcilers (#631). positions returns a thread-safe copy.
        for pos_order_id, position in state.live_position_tracker.positions.items():
            if position.stop_loss_order_id == order_id:
                logger.warning(
                    "Stop-loss order %s filled for position %s at $%.2f - "
                    "queuing position close for the trading loop",
                    order_id,
                    pos_order_id,
                    avg_price,
                )
                state._pending_fill_exits.put((pos_order_id, float(avg_price)))
                break

    def handle_partial_fill(
        self, order_id: str, symbol: str, new_filled_qty: float, avg_price: float
    ) -> None:
        """
        Handle a partial fill notification from OrderTracker.

        For stop-loss partial fills, logs a critical warning since position
        remains exposed. Full handling of partial SL fills would require
        placing a new SL order for the remaining quantity.

        Args:
            order_id: The partially filled order ID
            symbol: Trading symbol
            new_filled_qty: Additional quantity filled since last check
            avg_price: Average fill price
        """
        state = self._state
        logger.info("Partial fill: %s %s +%s @ $%.2f", order_id, symbol, new_filled_qty, avg_price)
        log_order_event(
            "partial_fill",
            order_id=order_id,
            symbol=symbol,
            new_filled_quantity=new_filled_qty,
            average_price=avg_price,
        )

        # Check if this is a stop-loss order partial fill - log critical warning
        # Partial SL fills leave the position partially exposed without protection
        for pos_order_id, position in state.live_position_tracker.positions.items():
            if position.stop_loss_order_id == order_id:
                logger.critical(
                    "PARTIAL STOP-LOSS FILL: Position %s SL order %s partially filled "
                    "(%.4f @ $%.2f). Remaining SL order is still active on exchange. "
                    "MANUAL MONITORING REQUIRED.",
                    pos_order_id,
                    order_id,
                    new_filled_qty,
                    avg_price,
                )
                log_order_event(
                    "partial_sl_fill_warning",
                    order_id=order_id,
                    position_order_id=pos_order_id,
                    symbol=symbol,
                    filled_quantity=new_filled_qty,
                    average_price=avg_price,
                )
                # Do NOT auto-close the remaining position here. The partial SL
                # fill already sold part of the position, and the remaining SL
                # order is still active on the exchange protecting the unfilled
                # portion. Calling _execute_exit(skip_live_close=False) would
                # place a market order for the FULL position.quantity, over-selling
                # by the already-filled amount and creating unintended exposure.
                # If the remaining SL order expires/cancels, the cancel callback
                # will fire and can be handled there.
                state._send_alert(
                    f"⚠️ PARTIAL SL FILL: {symbol} position {pos_order_id} "
                    f"partially filled ({new_filled_qty} @ ${avg_price:.2f}). "
                    f"Remaining SL order still active. MANUAL MONITORING REQUIRED."
                )
                return

    def handle_stop_loss_cancelled(self, order_id: str, symbol: str) -> bool:
        """Escalate when a tracked position's stop-loss order terminates unexpectedly.

        Clears the stale in-memory ``stop_loss_order_id`` so the periodic
        reconciler's missing-stop path re-places protection on its next cycle
        (which also persists the NEW id over the stale DB value), and emits a
        critical ``system_events`` row + webhook alert. Deliberate cancels from
        the close path don't reach here — that path stops tracking the SL order
        before the callback can fire.

        Returns True when ``order_id`` matched a tracked position's stop-loss.
        """
        state = self._state
        matched_key: str | None = None
        for pos_key, position in state.live_position_tracker.positions.items():
            if getattr(position, "stop_loss_order_id", None) == order_id:
                matched_key = pos_key
                break
        if matched_key is None:
            return False

        state.live_position_tracker.set_stop_loss_order_id(matched_key, None)
        message = (
            f"Stop-loss order {order_id} for OPEN {symbol} position {matched_key} was "
            f"cancelled/rejected/expired on the exchange — position is UNPROTECTED until "
            f"the reconciler re-places it (next cycle)."
        )
        logger.critical("CRITICAL: %s", message)
        state._record_event(
            EventType.ERROR,
            message,
            severity="critical",
            component="order_tracker",
            error_code="STOP_LOSS_CANCELLED",
            alert=True,
        )
        return True

    def handle_order_cancel(self, order_id: str, symbol: str, filled_qty: float = 0.0) -> None:
        """Handle an order cancellation/rejection notification from OrderTracker.

        Refunds only the entry fee for the unfilled portion of the order.
        When an entry limit order partially fills before cancellation, part of
        the fee was legitimately incurred on the exchange; only the unfilled
        fraction is refunded to prevent over-crediting the balance.

        Args:
            order_id: The cancelled/rejected order ID.
            symbol: Trading symbol.
            filled_qty: Cumulative quantity filled before cancellation (0.0 if fully unfilled).
        """
        state = self._state
        logger.warning("Order cancelled/rejected: %s %s", order_id, symbol)
        log_order_event(
            "order_cancelled",
            order_id=order_id,
            symbol=symbol,
        )
        # Not an entry order? Check whether a tracked position's STOP-LOSS was
        # cancelled/rejected/expired out from under it (#741). Must run before
        # pop_position: popping by an SL order id can never match (positions
        # are keyed by entry order id), but the position it protects must not
        # be left silently unprotected.
        if self.handle_stop_loss_cancelled(order_id, symbol):
            return

        # Check if this was an entry order for a position we thought we had.
        # Use atomic pop_position() for thread safety - combines get + remove
        # in a single lock acquisition (called from OrderTracker background thread).
        removed_position = state.live_position_tracker.pop_position(order_id)
        if removed_position is not None:
            logger.error("Entry order %s was cancelled - removing phantom position", order_id)
            # Refund only the fee for the unfilled portion. If the order partially filled
            # before cancellation, the exchange kept the fee for those fills; refunding the
            # full entry_fee would over-credit the balance and corrupt P&L accounting.
            entry_fee = float(removed_position.metadata.get("entry_fee", 0.0))
            if entry_fee > 0:
                original_qty = removed_position.quantity or 0.0
                if original_qty > 0 and filled_qty > 0:
                    # Compute the fraction of the order that was NOT filled.
                    unfilled_fraction = max(0.0, (original_qty - filled_qty) / original_qty)
                    refund_amount = entry_fee * unfilled_fraction
                    if unfilled_fraction < 1.0:
                        logger.info(
                            "Order %s partially filled (%.6f / %.6f qty); "
                            "refunding %.6f of %.6f entry fee (unfilled fraction: %.4f)",
                            order_id,
                            filled_qty,
                            original_qty,
                            refund_amount,
                            entry_fee,
                            unfilled_fraction,
                        )
                else:
                    # Order was fully unfilled - refund the entire fee
                    refund_amount = entry_fee

                if refund_amount > 0:
                    if state.trading_session_id is not None:
                        try:
                            with state.db_manager.atomic_balance_update(
                                balance_change=refund_amount,
                                reason=f"refund_entry_fee_{symbol}_order_cancelled",
                                updated_by="live_engine",
                                correlation_id=order_id,
                            ) as balance_result:
                                state.current_balance = balance_result["new_balance"]
                            logger.info(
                                "Refunded entry fee %.6f for cancelled order %s on %s",
                                refund_amount,
                                order_id,
                                symbol,
                            )
                        except Exception as refund_err:
                            logger.critical(
                                "CRITICAL: Failed to refund entry fee %.6f for cancelled "
                                "order %s on %s. MANUAL RECONCILIATION REQUIRED. Error: %s",
                                refund_amount,
                                order_id,
                                symbol,
                                refund_err,
                            )
                    else:
                        state.current_balance += refund_amount

    def handle_order_tracking_lost(self, order_id: str, symbol: str, failures: int) -> None:
        """Handle the OrderTracker giving up on an order whose state is UNKNOWN.

        Fail-closed counterpart to :meth:`handle_order_cancel`: after
        ``failures`` consecutive failed/None polls the order's exchange state
        could not be confirmed, so the position (if any) is deliberately KEPT
        tracked — removing it and refunding its entry fee here would vaporize a
        possibly-live position from the books (untracked exposure, corrupted
        balance, double-entry on the next signal). The periodic reconciler
        resolves the true state from the exchange: it removes ghosts whose
        entry order is confirmed cancelled and books offline stop-loss fills.
        """
        state = self._state
        position = state.live_position_tracker.get_position(order_id)
        message = (
            f"Order {order_id} on {symbol} state UNKNOWN after {failures} consecutive "
            f"failed polls — tracking abandoned (NOT treated as cancelled). "
            f"{'Position kept tracked' if position is not None else 'No tracked position'}; "
            f"reconciler will resolve from exchange truth."
        )
        logger.critical("CRITICAL: %s", message)
        log_order_event(
            "order_tracking_lost",
            order_id=order_id,
            symbol=symbol,
        )
        state._record_event(
            EventType.ERROR,
            message,
            severity="critical",
            component="order_tracker",
            error_code="ORDER_TRACKING_LOST",
            alert=True,
        )
