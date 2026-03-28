"""
Order Tracker - Monitors order status for live trading.

This module polls the exchange for order status updates and notifies
the trading engine when orders fill, partially fill, or get cancelled.
"""

import logging
import math
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

from src.config.constants import DEFAULT_ORDER_POLL_INTERVAL, DEFAULT_ORDER_TRACKER_TIMEOUT
from src.data_providers.exchange_interface import ExchangeInterface, Order, OrderStatus
from src.infrastructure.circuit_breaker import CircuitBreaker

logger = logging.getLogger(__name__)


@dataclass
class TrackedOrder:
    """Represents an order being tracked."""

    order_id: str
    symbol: str
    last_filled_qty: float
    added_at: datetime
    # Counts consecutive invalid-data poll responses (NaN, negative prices).
    # After MAX_INVALID_DATA_RETRIES, the order is force-removed to prevent
    # permanent tracking when the exchange returns persistently corrupt data.
    invalid_data_count: int = 0
    # Counts consecutive fill callback failures (on_fill raises).
    # After MAX_CALLBACK_RETRIES the order is force-removed to prevent an
    # unbounded retry loop when the callback fails deterministically.
    callback_failure_count: int = 0
    # Counts consecutive API errors when polling get_order() for this order.
    # After MAX_API_ERROR_RETRIES, the order is force-removed to prevent an
    # infinite error loop (e.g. Binance -1100 for invalid orderId format).
    api_error_count: int = 0


# Maximum polls with invalid data before force-removing a tracked order.
# Set to 10 to tolerate transient exchange issues (typically 1-3 polls) while
# preventing permanent ghost-order tracking from persistent corrupt data.
# At a typical 5-second poll interval, this gives ~50 seconds of tolerance.
MAX_INVALID_DATA_RETRIES = 10

# Maximum fill-callback failures before force-removing a tracked order.
# Set to 5 (fewer than data retries) because callback failures are typically
# deterministic bugs rather than transient issues. At a 5-second poll interval
# this gives ~25 seconds before the order is force-removed with a CRITICAL alert.
MAX_CALLBACK_RETRIES = 5

# Maximum consecutive API errors before force-removing a tracked order.
# Prevents infinite error loops when the exchange persistently rejects requests
# for a specific order (e.g. Binance -1100 for invalid orderId format).
# Set to 10 to tolerate transient network/API issues (typically 1-3 polls) while
# preventing permanent polling of unreachable orders. At a 5-second poll interval
# this gives ~50 seconds of tolerance before force-removal with a CRITICAL alert.
MAX_API_ERROR_RETRIES = 10


class OrderTracker:
    """
    Tracks pending orders and notifies on status changes.

    Runs a background thread that polls the exchange for order status
    and calls registered callbacks when orders fill or cancel.
    """

    def __init__(
        self,
        exchange: ExchangeInterface,
        poll_interval: int = DEFAULT_ORDER_POLL_INTERVAL,
        on_fill: Callable[[str, str, float, float], None] | None = None,
        on_partial_fill: Callable[[str, str, float, float], None] | None = None,
        on_cancel: Callable[[str, str, float], None] | None = None,
    ):
        """
        Initialize the order tracker.

        Args:
            exchange: Exchange interface for querying order status
            poll_interval: Seconds between status checks
            on_fill: Callback(order_id, symbol, filled_qty, avg_price) for filled orders
            on_partial_fill: Callback(order_id, symbol, new_filled_qty, avg_price) for partial fills
            on_cancel: Callback(order_id, symbol, filled_qty) for cancelled/rejected orders.
                filled_qty is the cumulative quantity filled before cancellation (0.0 if unfilled).
        """
        self.exchange = exchange
        self.poll_interval = poll_interval
        self.on_fill = on_fill
        self.on_partial_fill = on_partial_fill
        self.on_cancel = on_cancel

        self._pending_orders: dict[str, TrackedOrder] = {}
        self._lock = threading.Lock()
        self._running = False
        self._stop_event = threading.Event()  # For clean, interruptible shutdown
        self._thread: threading.Thread | None = None
        # Circuit breaker to handle exchange API failures gracefully
        # Prevents resource exhaustion from repeated failing API calls
        self._circuit_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60.0)

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
                added_at=datetime.now(UTC),
            )
        logger.debug("Now tracking order %s for %s", order_id, symbol)

    def stop_tracking(self, order_id: str) -> None:
        """
        Remove an order from tracking.

        Args:
            order_id: Order ID to stop tracking
        """
        with self._lock:
            if order_id in self._pending_orders:
                del self._pending_orders[order_id]
                logger.debug("Stopped tracking order %s", order_id)

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
        self._stop_event.clear()  # Clear stop signal for new run
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        logger.info("OrderTracker started (poll interval: %ss)", self.poll_interval)

    def stop(self) -> None:
        """Stop the background polling thread."""
        self._running = False
        self._stop_event.set()  # Signal thread to wake up and exit
        if self._thread:
            self._thread.join(timeout=DEFAULT_ORDER_TRACKER_TIMEOUT)
            # Verify thread actually stopped after timeout
            if self._thread.is_alive():
                logger.critical(
                    "OrderTracker thread did not stop after timeout - thread may be stuck! "
                    "This indicates a blocking call in _poll_loop. "
                    "Tracker will be marked as stopped but thread continues running."
                )
                # Mark as None anyway to prevent double-start, but thread is leaked
                self._thread = None
                return
            self._thread = None
        logger.info("OrderTracker stopped")

    def _poll_loop(self) -> None:
        """Main polling loop - runs in background thread."""
        while self._running:
            try:
                self._check_orders()
            except Exception as e:
                logger.error("Order tracking error: %s", e)
            # Use Event.wait() instead of time.sleep() for interruptible sleep
            # This allows stop() to immediately wake up the thread
            self._stop_event.wait(self.poll_interval)

    def _check_orders(self) -> None:
        """Check status of all pending orders."""
        # Get snapshot of orders to check (avoid holding lock during API calls)
        with self._lock:
            orders_to_check = list(self._pending_orders.items())

        for order_id, tracked in orders_to_check:
            try:
                # Use circuit breaker to prevent resource exhaustion during exchange outages
                # If circuit is OPEN (too many failures), skip API call and log warning
                order = self._circuit_breaker.call(
                    self.exchange.get_order, order_id, tracked.symbol
                )
                if not order:
                    # If order disappeared from exchange AND callback previously failed,
                    # untrack to prevent ghost-order memory leak (matches CANCELLED path logic).
                    if tracked.callback_failure_count > 0:
                        logger.critical(
                            "CRITICAL: Order %s on %s no longer returned by exchange after "
                            "%d failed callback attempts. Force-removing to prevent permanent "
                            "tracking. MANUAL RECONCILIATION REQUIRED.",
                            order_id,
                            tracked.symbol,
                            tracked.callback_failure_count,
                        )
                        self.stop_tracking(order_id)
                    else:
                        logger.warning("Could not fetch order %s - may have expired", order_id)
                    continue

                # Reset API error counter on any successful response
                tracked.api_error_count = 0
                self._process_order_status(order_id, tracked, order)

            except Exception as e:
                tracked.api_error_count += 1
                if tracked.api_error_count >= MAX_API_ERROR_RETRIES:
                    logger.critical(
                        "CRITICAL: Order %s on %s failed %d consecutive API calls: %s. "
                        "Force-removing to prevent infinite error loop. "
                        "MANUAL RECONCILIATION REQUIRED.",
                        order_id,
                        tracked.symbol,
                        tracked.api_error_count,
                        e,
                    )
                    # Call cancel callback BEFORE stop_tracking so the callback
                    # can still access order metadata if needed
                    if self.on_cancel:
                        try:
                            self.on_cancel(order_id, tracked.symbol, tracked.last_filled_qty)
                        except Exception as cb_err:
                            logger.error(
                                "Cancel callback failed for force-removed order %s: %s",
                                order_id,
                                cb_err,
                            )
                    self.stop_tracking(order_id)
                else:
                    logger.warning(
                        "Failed to check order %s (attempt %d/%d): %s",
                        order_id,
                        tracked.api_error_count,
                        MAX_API_ERROR_RETRIES,
                        e,
                    )

    def _process_order_status(self, order_id: str, tracked: TrackedOrder, order: Order) -> None:
        """
        Process order status and trigger appropriate callbacks.

        Args:
            order_id: The order ID being checked
            tracked: The tracked order info
            order: Order object from exchange
        """
        status = order.status
        filled_qty = order.filled_quantity or 0.0
        avg_price = order.average_price or 0.0

        if status == OrderStatus.FILLED:
            # Validate avg_price for fills to prevent corrupt P&L calculations
            # Check for NaN explicitly since NaN passes isinstance but corrupts calculations
            if (
                not isinstance(avg_price, int | float)
                or math.isnan(float(avg_price))
                or avg_price <= 0
            ):
                tracked.invalid_data_count += 1
                if tracked.invalid_data_count >= MAX_INVALID_DATA_RETRIES:
                    logger.critical(
                        "CRITICAL: Order %s on %s returned invalid avg_price %s for %d "
                        "consecutive polls. Force-removing to prevent permanent tracking. "
                        "MANUAL RECONCILIATION REQUIRED.",
                        order_id,
                        tracked.symbol,
                        avg_price,
                        tracked.invalid_data_count,
                    )
                    self.stop_tracking(order_id)
                    return
                logger.error(
                    "Invalid average price %s (NaN or <= 0) for filled order %s "
                    "(attempt %d/%d) - retrying on next poll",
                    avg_price,
                    order_id,
                    tracked.invalid_data_count,
                    MAX_INVALID_DATA_RETRIES,
                )
                return

            # Validate filled_qty to prevent division by zero and corrupt position tracking
            # Check for NaN explicitly since NaN passes isinstance but corrupts calculations
            if (
                not isinstance(filled_qty, int | float)
                or math.isnan(float(filled_qty))
                or filled_qty <= 0
            ):
                tracked.invalid_data_count += 1
                if tracked.invalid_data_count >= MAX_INVALID_DATA_RETRIES:
                    logger.critical(
                        "CRITICAL: Order %s on %s returned invalid filled_qty %s for %d "
                        "consecutive polls. Force-removing to prevent permanent tracking. "
                        "MANUAL RECONCILIATION REQUIRED.",
                        order_id,
                        tracked.symbol,
                        filled_qty,
                        tracked.invalid_data_count,
                    )
                    self.stop_tracking(order_id)
                    return
                logger.error(
                    "Invalid filled quantity %s (NaN or <= 0) for order %s "
                    "(attempt %d/%d) - retrying on next poll",
                    filled_qty,
                    order_id,
                    tracked.invalid_data_count,
                    MAX_INVALID_DATA_RETRIES,
                )
                # Don't stop tracking - keep polling until we get valid quantity
                return

            # Reset invalid data counter on successful validation
            tracked.invalid_data_count = 0
            logger.info(
                "Order filled: %s %s qty=%s @ %s", order_id, tracked.symbol, filled_qty, avg_price
            )
            # Call callback outside any lock to prevent deadlock.
            # Only stop tracking after a SUCCESSFUL callback. If the callback
            # fails, the engine never processes the fill, leaving an orphaned
            # position on the exchange. Keeping the order tracked lets the
            # next poll cycle retry the callback.
            callback_succeeded = False
            if self.on_fill:
                try:
                    self.on_fill(order_id, tracked.symbol, filled_qty, avg_price)
                    callback_succeeded = True
                except Exception as e:
                    tracked.callback_failure_count += 1
                    if tracked.callback_failure_count >= MAX_CALLBACK_RETRIES:
                        logger.critical(
                            "CRITICAL: Fill callback failed %d times for order %s on %s: %s. "
                            "Force-removing to prevent unbounded retry loop. "
                            "POSITION IS ORPHANED ON EXCHANGE - MANUAL RECONCILIATION REQUIRED.",
                            tracked.callback_failure_count,
                            order_id,
                            tracked.symbol,
                            e,
                            exc_info=True,
                        )
                        self.stop_tracking(order_id)
                        return
                    logger.critical(
                        "CRITICAL: Fill callback failed for order %s on %s (attempt %d/%d): %s. "
                        "Order remains tracked for retry on next poll cycle.",
                        order_id,
                        tracked.symbol,
                        tracked.callback_failure_count,
                        MAX_CALLBACK_RETRIES,
                        e,
                        exc_info=True,
                    )
            else:
                callback_succeeded = True

            if callback_succeeded:
                self.stop_tracking(order_id)

        elif status == OrderStatus.PARTIALLY_FILLED:
            # Validate avg_price for partial fills to prevent corrupt P&L calculations.
            # Increment invalid_data_count so persistent bad data is force-removed
            # (matching the FILLED path), preventing infinite ghost-order polling.
            if (
                not isinstance(avg_price, int | float)
                or math.isnan(float(avg_price))
                or avg_price <= 0
            ):
                tracked.invalid_data_count += 1
                if tracked.invalid_data_count >= MAX_INVALID_DATA_RETRIES:
                    logger.critical(
                        "CRITICAL: Partial fill %s on %s returned invalid avg_price %s "
                        "for %d consecutive polls. Force-removing to prevent permanent "
                        "tracking. MANUAL RECONCILIATION REQUIRED.",
                        order_id,
                        tracked.symbol,
                        avg_price,
                        tracked.invalid_data_count,
                    )
                    self.stop_tracking(order_id)
                    return
                logger.error(
                    "Invalid average price %s (NaN or <= 0) for partial fill order %s "
                    "(attempt %d/%d) - skipping callback, retrying on next poll",
                    avg_price,
                    order_id,
                    tracked.invalid_data_count,
                    MAX_INVALID_DATA_RETRIES,
                )
                return

            # Validate filled_qty for partial fills to prevent corrupt position tracking.
            # Same counter logic as avg_price validation above.
            if (
                not isinstance(filled_qty, int | float)
                or math.isnan(float(filled_qty))
                or filled_qty <= 0
            ):
                tracked.invalid_data_count += 1
                if tracked.invalid_data_count >= MAX_INVALID_DATA_RETRIES:
                    logger.critical(
                        "CRITICAL: Partial fill %s on %s returned invalid filled_qty %s "
                        "for %d consecutive polls. Force-removing to prevent permanent "
                        "tracking. MANUAL RECONCILIATION REQUIRED.",
                        order_id,
                        tracked.symbol,
                        filled_qty,
                        tracked.invalid_data_count,
                    )
                    self.stop_tracking(order_id)
                    return
                logger.error(
                    "Invalid filled quantity %s (NaN or <= 0) for partial fill %s "
                    "(attempt %d/%d) - skipping callback, retrying on next poll",
                    filled_qty,
                    order_id,
                    tracked.invalid_data_count,
                    MAX_INVALID_DATA_RETRIES,
                )
                return

            # Reset counter on successful validation (matching FILLED path)
            tracked.invalid_data_count = 0

            new_filled = filled_qty - tracked.last_filled_qty

            # CRITICAL: Always update last_filled_qty, even if delta is non-positive
            # This prevents infinite loops if exchange reports decreasing fills
            # Validate and prepare callback parameters inside lock, but call callback outside
            should_call_callback = False
            with self._lock:
                if order_id not in self._pending_orders:
                    logger.warning(
                        "Order %s no longer tracked during partial fill processing", order_id
                    )
                    return

                # Detect anomalous fill quantity changes
                if new_filled < 0:
                    logger.critical(
                        "ANOMALY: Filled quantity decreased for order %s: %.8f -> %.8f (delta: %.8f). "
                        "This indicates exchange API inconsistency. Updating tracker to prevent divergence.",
                        order_id,
                        tracked.last_filled_qty,
                        filled_qty,
                        new_filled,
                    )
                    # Update to prevent infinite loop, but don't trigger callback
                    self._pending_orders[order_id].last_filled_qty = filled_qty
                    return

                if new_filled == 0:
                    logger.debug("Partial fill status with no quantity change for %s", order_id)
                    return

                # Normal case: positive fill delta
                logger.info(
                    "Partial fill: %s %s +%s @ %s", order_id, tracked.symbol, new_filled, avg_price
                )
                should_call_callback = True

            # Call callback OUTSIDE lock to prevent deadlock if callback accesses tracker
            if should_call_callback and self.on_partial_fill:
                try:
                    self.on_partial_fill(order_id, tracked.symbol, new_filled, avg_price)
                except Exception as e:
                    logger.error("Partial fill callback failed for %s: %s", order_id, e)

            # Update tracker state in separate lock scope to ensure it happens even if callback fails
            with self._lock:
                if order_id in self._pending_orders:
                    self._pending_orders[order_id].last_filled_qty = filled_qty

        elif status in (
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED,
        ):
            logger.warning("Order %s: %s %s", status.value, order_id, tracked.symbol)
            # Call callback outside any lock to prevent deadlock.
            # Pass last_filled_qty so the caller can compute a proportional fee refund
            # when the order was partially filled before cancellation.
            if self.on_cancel:
                try:
                    self.on_cancel(order_id, tracked.symbol, tracked.last_filled_qty)
                except Exception as e:
                    # Escalate to CRITICAL: the position may still exist in the
                    # tracker with no exchange order backing it. The order won't
                    # reappear, so we must stop tracking, but a phantom position
                    # remains until the next reconciliation cycle.
                    logger.critical(
                        "CRITICAL: Cancel callback failed for order %s on %s: %s. "
                        "Order will be untracked; position may be orphaned in tracker. "
                        "MANUAL RECONCILIATION REQUIRED.",
                        order_id,
                        tracked.symbol,
                        e,
                    )
            # Stop tracking even if callback fails - cancelled orders won't
            # re-appear on exchange so keeping them tracked is a memory leak.
            self.stop_tracking(order_id)
