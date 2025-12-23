"""
Order Tracker - Monitors order status for live trading.

This module polls the exchange for order status updates and notifies
the trading engine when orders fill, partially fill, or get cancelled.
"""

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

from src.data_providers.exchange_interface import ExchangeInterface, OrderStatus

logger = logging.getLogger(__name__)


@dataclass
class TrackedOrder:
    """Represents an order being tracked."""

    order_id: str
    symbol: str
    last_filled_qty: float
    added_at: datetime


class OrderTracker:
    """
    Tracks pending orders and notifies on status changes.

    Runs a background thread that polls the exchange for order status
    and calls registered callbacks when orders fill or cancel.
    """

    def __init__(
        self,
        exchange: ExchangeInterface,
        poll_interval: int = 5,
        on_fill: Callable[[str, str, float, float], None] | None = None,
        on_partial_fill: Callable[[str, str, float, float], None] | None = None,
        on_cancel: Callable[[str, str], None] | None = None,
    ):
        """
        Initialize the order tracker.

        Args:
            exchange: Exchange interface for querying order status
            poll_interval: Seconds between status checks (default 5)
            on_fill: Callback(order_id, symbol, filled_qty, avg_price) for filled orders
            on_partial_fill: Callback(order_id, symbol, new_filled_qty, avg_price) for partial fills
            on_cancel: Callback(order_id, symbol) for cancelled/rejected orders
        """
        self.exchange = exchange
        self.poll_interval = poll_interval
        self.on_fill = on_fill
        self.on_partial_fill = on_partial_fill
        self.on_cancel = on_cancel

        self._pending_orders: dict[str, TrackedOrder] = {}
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None

    def track_order(self, order_id: str, symbol: str) -> None:
        """
        Add an order to tracking.

        Args:
            order_id: Exchange order ID to track
            symbol: Trading symbol for the order
        """
        with self._lock:
            self._pending_orders[order_id] = TrackedOrder(
                order_id=order_id,
                symbol=symbol,
                last_filled_qty=0.0,
                added_at=datetime.utcnow(),
            )
        logger.debug(f"Now tracking order {order_id} for {symbol}")

    def stop_tracking(self, order_id: str) -> None:
        """
        Remove an order from tracking.

        Args:
            order_id: Order ID to stop tracking
        """
        with self._lock:
            if order_id in self._pending_orders:
                del self._pending_orders[order_id]
                logger.debug(f"Stopped tracking order {order_id}")

    def get_tracked_count(self) -> int:
        """Return the number of orders currently being tracked."""
        with self._lock:
            return len(self._pending_orders)

    def start(self) -> None:
        """Start the background polling thread."""
        if self._running:
            logger.warning("OrderTracker already running")
            return

        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        logger.info(f"OrderTracker started (poll interval: {self.poll_interval}s)")

    def stop(self) -> None:
        """Stop the background polling thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)
            self._thread = None
        logger.info("OrderTracker stopped")

    def _poll_loop(self) -> None:
        """Main polling loop - runs in background thread."""
        while self._running:
            try:
                self._check_orders()
            except Exception as e:
                logger.error(f"Order tracking error: {e}")
            time.sleep(self.poll_interval)

    def _check_orders(self) -> None:
        """Check status of all pending orders."""
        # Get snapshot of orders to check (avoid holding lock during API calls)
        with self._lock:
            orders_to_check = list(self._pending_orders.items())

        for order_id, tracked in orders_to_check:
            try:
                order = self.exchange.get_order(order_id, tracked.symbol)
                if not order:
                    logger.warning(f"Could not fetch order {order_id} - may have expired")
                    continue

                self._process_order_status(order_id, tracked, order)

            except Exception as e:
                logger.warning(f"Failed to check order {order_id}: {e}")

    def _process_order_status(
        self, order_id: str, tracked: TrackedOrder, order: object
    ) -> None:
        """
        Process order status and trigger appropriate callbacks.

        Args:
            order_id: The order ID being checked
            tracked: The tracked order info
            order: Order object from exchange
        """
        status = order.status
        filled_qty = order.filled_quantity
        avg_price = order.average_price or 0.0

        if status == OrderStatus.FILLED:
            logger.info(
                f"Order filled: {order_id} {tracked.symbol} "
                f"qty={filled_qty} @ {avg_price}"
            )
            if self.on_fill:
                self.on_fill(order_id, tracked.symbol, filled_qty, avg_price)
            self.stop_tracking(order_id)

        elif status == OrderStatus.PARTIALLY_FILLED:
            new_filled = filled_qty - tracked.last_filled_qty
            if new_filled > 0:
                logger.info(
                    f"Partial fill: {order_id} {tracked.symbol} "
                    f"+{new_filled} @ {avg_price}"
                )
                if self.on_partial_fill:
                    self.on_partial_fill(order_id, tracked.symbol, new_filled, avg_price)
                # Update tracked quantity
                with self._lock:
                    if order_id in self._pending_orders:
                        self._pending_orders[order_id].last_filled_qty = filled_qty

        elif status in (
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED,
        ):
            logger.warning(f"Order {status.value}: {order_id} {tracked.symbol}")
            if self.on_cancel:
                self.on_cancel(order_id, tracked.symbol)
            self.stop_tracking(order_id)
