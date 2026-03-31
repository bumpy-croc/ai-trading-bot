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
from src.data_providers.exchange_interface import (
    ExchangeInterface,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
)
from src.engines.live.event_deduplicator import EventDeduplicator
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
# At a 10-second poll interval this gives ~100 seconds of tolerance.
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
        event_deduplicator: EventDeduplicator | None = None,
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
            event_deduplicator: Optional deduplicator for WebSocket events. A default
                instance is created if not provided.
        """
        self.exchange = exchange
        self.poll_interval = poll_interval
        self.on_fill = on_fill
        self.on_partial_fill = on_partial_fill
        self.on_cancel = on_cancel
        self._dedup = event_deduplicator or EventDeduplicator()
        self._polling_enabled = True  # Controls whether _poll_loop runs checks

        self._pending_orders: dict[str, TrackedOrder] = {}
        self._lock = threading.Lock()
        self._order_locks: dict[str, threading.Lock] = {}
        self._running = False
        self._stop_event = threading.Event()  # For clean, interruptible shutdown
        self._thread: threading.Thread | None = None
        # Circuit breaker to handle exchange API failures gracefully
        # Prevents resource exhaustion from repeated failing API calls
        self._circuit_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60.0)

    def _get_order_lock(self, order_id: str) -> threading.Lock:
        """Return the per-order lock, creating one if needed.

        Must be called without ``self._lock`` held (it acquires it internally).
        """
        with self._lock:
            if order_id not in self._order_locks:
                self._order_locks[order_id] = threading.Lock()
            return self._order_locks[order_id]

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
                self._order_locks.pop(order_id, None)
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
                if self._polling_enabled:
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
            # Per-order lock prevents concurrent WS processing of the same order
            order_lock = self._get_order_lock(order_id)
            with order_lock:
                # Re-check under lock — order may have been removed by WS terminal event
                with self._lock:
                    if order_id not in self._pending_orders:
                        continue
                try:
                    # Use circuit breaker to prevent resource exhaustion during exchange outages
                    # If circuit is OPEN (too many failures), skip API call and log warning
                    order = self._circuit_breaker.call(
                        self.exchange.get_order, order_id, tracked.symbol
                    )
                    if not order:
                        # None return counts as an API error — get_order_by_client_id()
                        # swallows exceptions and returns None, so the except block below
                        # is never reached for client order ID failures.
                        tracked.api_error_count += 1
                        if tracked.api_error_count >= MAX_API_ERROR_RETRIES:
                            logger.critical(
                                "CRITICAL: Order %s on %s returned None for %d consecutive "
                                "polls. Force-removing to prevent infinite polling. "
                                "MANUAL RECONCILIATION REQUIRED.",
                                order_id,
                                tracked.symbol,
                                tracked.api_error_count,
                            )
                            if self.on_cancel:
                                try:
                                    self.on_cancel(order_id, tracked.symbol, tracked.last_filled_qty)
                                except Exception as cb_err:
                                    logger.error(
                                        "Cancel callback failed for force-removed order %s: %s",
                                        order_id,
                                        cb_err,
                                        exc_info=True,
                                    )
                            self.stop_tracking(order_id)
                        elif tracked.callback_failure_count > 0:
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
                            logger.warning(
                                "Could not fetch order %s (attempt %d/%d) - may have expired",
                                order_id,
                                tracked.api_error_count,
                                MAX_API_ERROR_RETRIES,
                            )
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
                            exc_info=True,
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
                                    exc_info=True,
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
            # Reconcile fill delta: if terminal status carries more fill qty than
            # we've seen (e.g. partial fill event was missed), process the delta
            # before handling the cancel to avoid under-counting fills.
            actual_filled = order.filled_quantity if order.filled_quantity else 0.0
            if actual_filled > tracked.last_filled_qty:
                fill_delta = actual_filled - tracked.last_filled_qty
                logger.warning(
                    "Order %s: reconciling missed fill delta %.8f before %s",
                    order_id, fill_delta, status.value,
                )
                if self.on_partial_fill:
                    try:
                        self.on_partial_fill(order_id, tracked.symbol, fill_delta, avg_price)
                    except Exception as e:
                        logger.error("Fill reconciliation callback failed for %s: %s", order_id, e)
                with self._lock:
                    if order_id in self._pending_orders:
                        self._pending_orders[order_id].last_filled_qty = actual_filled

            logger.warning("Order %s: %s %s", status.value, order_id, tracked.symbol)
            # Call callback outside any lock to prevent deadlock.
            # Pass actual filled qty so the caller uses the reconciled value.
            if self.on_cancel:
                try:
                    self.on_cancel(order_id, tracked.symbol, actual_filled)
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

    def process_execution_event(self, event: dict) -> None:
        """Process a WebSocket executionReport event.

        Extracts order data from Binance WS fields, deduplicates, and delegates
        to the existing _process_order_status() for lifecycle handling.

        Args:
            event: Raw Binance executionReport payload with single-letter keys.
        """
        order_id = str(event.get("i", ""))
        exec_type = str(event.get("x", ""))
        exec_id = str(event.get("I", ""))

        # Per-order lock serialises WS and REST processing of the same order,
        # preventing double fills when both paths race on the same update.
        order_lock = self._get_order_lock(order_id)
        with order_lock:
            with self._lock:
                tracked = self._pending_orders.get(order_id)
            if tracked is None:
                return

            # Dedup after confirming order is tracked — events arriving before
            # track_order() should not be permanently marked as seen.
            if self._dedup.is_duplicate(order_id, exec_type, exec_id):
                return

            cum_filled = float(event.get("z", 0))
            cum_quote = float(event.get("Z", 0))
            avg_price = cum_quote / cum_filled if cum_filled > 0 else 0.0

            status = self._map_ws_status(str(event.get("X", "")))

            order = Order(
                order_id=order_id,
                symbol=str(event.get("s", tracked.symbol)),
                side=OrderSide(str(event.get("S", "BUY"))),
                order_type=self._map_ws_order_type(str(event.get("o", "MARKET"))),
                quantity=float(event.get("q", 0)),
                price=float(event.get("p", 0)) or None,
                status=status,
                filled_quantity=cum_filled,
                average_price=avg_price,
                commission=float(event.get("n") or 0),
                commission_asset=str(event.get("N") or ""),
                create_time=datetime.fromtimestamp(event.get("O", 0) / 1000, tz=UTC),
                update_time=datetime.fromtimestamp(event.get("E", 0) / 1000, tz=UTC),
                stop_price=float(event.get("P", 0)) or None,
                time_in_force=str(event.get("f", "GTC")),
                client_order_id=str(event.get("c", "")),
            )

            self._process_order_status(order_id, tracked, order)

    @staticmethod
    def _map_ws_status(ws_status: str) -> OrderStatus:
        """Map a Binance WebSocket order status to our OrderStatus enum.

        Args:
            ws_status: Binance WS status string (e.g. "FILLED", "CANCELED").

        Returns:
            Mapped OrderStatus. Unknown statuses are treated as EXPIRED
            to ensure terminal states are never silently ignored.
        """
        mapping = {
            "NEW": OrderStatus.PENDING,
            "PARTIALLY_FILLED": OrderStatus.PARTIALLY_FILLED,
            "FILLED": OrderStatus.FILLED,
            "CANCELED": OrderStatus.CANCELLED,  # Binance 1 L, our enum 2 Ls
            "REJECTED": OrderStatus.REJECTED,
            "EXPIRED": OrderStatus.EXPIRED,
            "EXPIRED_IN_MATCH": OrderStatus.EXPIRED,  # STP / self-trade prevention
            "PENDING_CANCEL": OrderStatus.CANCELLED,
        }
        status = mapping.get(ws_status)
        if status is None:
            logger.warning("Unknown WS order status: %s — treating as EXPIRED", ws_status)
            return OrderStatus.EXPIRED
        return status

    @staticmethod
    def _map_ws_order_type(ws_type: str) -> OrderType:
        """Map a Binance WebSocket order type to our OrderType enum.

        Args:
            ws_type: Binance WS order type string (e.g. "MARKET", "STOP_LOSS_LIMIT").

        Returns:
            Mapped OrderType, defaulting to MARKET for unknown types.
        """
        mapping = {
            "MARKET": OrderType.MARKET,
            "LIMIT": OrderType.LIMIT,
            "STOP_LOSS": OrderType.STOP_LOSS,
            "STOP_LOSS_LIMIT": OrderType.STOP_LOSS,
            "TAKE_PROFIT": OrderType.TAKE_PROFIT,
            "TAKE_PROFIT_LIMIT": OrderType.TAKE_PROFIT,
        }
        return mapping.get(ws_type, OrderType.MARKET)

    def poll_once(self) -> None:
        """Execute a single poll cycle. Used during WS to REST transitions."""
        self._check_orders()

    def disable_polling(self) -> None:
        """Disable REST polling. Used when WebSocket is active."""
        self._polling_enabled = False
        logger.info("Order polling disabled — WebSocket active")

    def enable_polling(self) -> None:
        """Enable REST polling. Used when WebSocket fails."""
        self._polling_enabled = True
        logger.info("Order polling enabled — REST fallback active")
