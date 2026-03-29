"""Binance reconciliation module for position, order, and balance verification.

Provides startup reconciliation (resolve pending orders, verify positions),
periodic runtime reconciliation, and discrepancy handling with severity-based
responses including close-only mode for critical issues.

All corrections are recorded as immutable audit events with before/after values.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from src.config.constants import (
    DEFAULT_RECONCILIATION_BALANCE_THRESHOLD_PCT,
    DEFAULT_RECONCILIATION_DUST_THRESHOLD,
    DEFAULT_RECONCILIATION_INTERVAL_SECONDS,
    DEFAULT_RECONCILIATION_ORDER_MATCH_TIME_WINDOW_MIN,
    DEFAULT_RECONCILIATION_ORDER_MATCH_TOLERANCE_PCT,
    DEFAULT_STOP_LOSS_PCT,
)
from src.engines.shared.models import PositionSide

if TYPE_CHECKING:
    from src.data_providers.exchange_interface import ExchangeInterface
    from src.database.manager import DatabaseManager
    from src.engines.live.execution.position_tracker import LivePositionTracker

logger = logging.getLogger(__name__)


@runtime_checkable
class _HasDbPositionId(Protocol):
    """Structural type for position objects that carry a DB position ID."""

    db_position_id: int | None
    symbol: str
    entry_price: float


# ---------- Data Models ----------


class Severity(str, Enum):
    """Discrepancy severity levels.

    Numeric ordering: LOW < MEDIUM < HIGH < CRITICAL.
    String values are preserved for serialization/logging.
    """

    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"

    @staticmethod
    def _numeric_rank(member: Severity) -> int:
        """Return numeric rank for ordering comparisons."""
        return {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}[member.value]

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Severity):
            return NotImplemented
        return self._numeric_rank(self) < self._numeric_rank(other)

    def __le__(self, other: object) -> bool:
        if not isinstance(other, Severity):
            return NotImplemented
        return self._numeric_rank(self) <= self._numeric_rank(other)

    def __gt__(self, other: object) -> bool:
        if not isinstance(other, Severity):
            return NotImplemented
        return self._numeric_rank(self) > self._numeric_rank(other)

    def __ge__(self, other: object) -> bool:
        if not isinstance(other, Severity):
            return NotImplemented
        return self._numeric_rank(self) >= self._numeric_rank(other)


@dataclass
class AuditEvent:
    """In-memory audit event before persistence."""

    entity_type: str  # position, order, balance
    entity_id: int | None
    field: str
    old_value: str | None
    new_value: str | None
    reason: str
    severity: Severity


@dataclass
class ReconciliationResult:
    """Result of reconciling a single entity."""

    entity_type: str
    entity_id: int | str | None
    status: str  # resolved, corrected, unresolved, skipped
    corrections: list[AuditEvent] = field(default_factory=list)
    severity: Severity = Severity.LOW


# ---------- Startup Reconciliation ----------


class PositionReconciler:
    """Verifies recovered positions and resolves pending orders on startup.

    Uses Binance as the source of truth for execution state. The DB is the
    audit trail — when they disagree, Binance wins.
    """

    def __init__(
        self,
        exchange_interface: ExchangeInterface,
        position_tracker: LivePositionTracker,
        db_manager: DatabaseManager,
        session_id: int,
        max_position_size: float = 0.1,
        use_margin: bool = False,
    ) -> None:
        self.exchange = exchange_interface
        self.position_tracker = position_tracker
        self.db_manager = db_manager
        self.session_id = session_id
        self.max_position_size = max_position_size
        self._use_margin = use_margin

    def reconcile_startup(self, positions: dict[str, Any]) -> list[ReconciliationResult]:
        """Run full startup reconciliation.

        Args:
            positions: Current position_tracker.positions snapshot.

        Returns:
            List of reconciliation results.
        """
        results: list[ReconciliationResult] = []

        # Step A: Resolve pending orders from crash recovery
        results.extend(self.resolve_pending_orders())

        # Step B: Verify each recovered position
        for order_id, position in positions.items():
            result = self.reconcile_position(position)
            results.append(result)

        # Step C: Verify balance consistency
        results.append(self._reconcile_balance())

        return results

    def resolve_pending_orders(self) -> list[ReconciliationResult]:
        """Resolve orders stuck in PENDING_SUBMIT/SUBMITTED/UNKNOWN."""
        results: list[ReconciliationResult] = []
        unresolved = self.db_manager.get_unresolved_orders(self.session_id)

        if not unresolved:
            logger.info("No pending orders to resolve")
            return results

        logger.info("Resolving %d pending orders...", len(unresolved))

        for order_data in unresolved:
            result = self._resolve_single_order(order_data)
            results.append(result)

        return results

    def _resolve_single_order(self, order_data: dict) -> ReconciliationResult:
        """Resolve a single pending order against Binance."""
        client_order_id = order_data.get("client_order_id")
        symbol = order_data["symbol"]
        status = order_data["status"]
        result = ReconciliationResult(
            entity_type="order",
            entity_id=order_data["id"],
            status="unresolved",
        )

        # 1. Query by client_order_id
        exchange_order = None
        if client_order_id:
            try:
                exchange_order = self.exchange.get_order_by_client_id(client_order_id, symbol)
            except Exception as e:
                logger.warning("Failed to query order by client_id %s: %s", client_order_id, e)

        if exchange_order:
            return self._handle_found_order(order_data, exchange_order, result)

        # 2. Not found by client_id — bounded fallback
        return self._handle_not_found_order(order_data, status, result)

    def _handle_found_order(
        self, order_data: dict, exchange_order: Any, result: ReconciliationResult
    ) -> ReconciliationResult:
        """Handle an order found on exchange."""
        from src.data_providers.exchange_interface import OrderStatus as ExOrderStatus

        eo_status = exchange_order.status

        if eo_status == ExOrderStatus.FILLED:
            # First, persist exchange data as SUBMITTED so we have the fill
            # info recorded. We defer CONFIRMED until position repair succeeds
            # so a failed repair is retried on next restart.
            self.db_manager.update_order_journal(
                client_order_id=order_data["client_order_id"],
                status="SUBMITTED",
                exchange_order_id=exchange_order.order_id,
                fill_price=exchange_order.average_price,
                fill_quantity=exchange_order.filled_quantity,
                commission=exchange_order.commission,
            )

            # Reconcile position state for filled orders. Only mark
            # CONFIRMED after successful position repair — if repair fails
            # the order stays as SUBMITTED so the next restart retries.
            try:
                self._reconcile_filled_order_position(order_data, exchange_order)
            except Exception as e:
                logger.warning(
                    "Position repair failed for order %s, leaving as SUBMITTED: %s",
                    order_data["client_order_id"],
                    e,
                )
                result.status = "unresolved"
                result.severity = Severity.CRITICAL
                return result

            self.db_manager.update_order_journal(
                client_order_id=order_data["client_order_id"],
                status="CONFIRMED",
                exchange_order_id=exchange_order.order_id,
            )
            audit = AuditEvent(
                entity_type="order",
                entity_id=order_data["id"],
                field="status",
                old_value=order_data["status"],
                new_value="CONFIRMED",
                reason=f"Order filled on exchange (price={exchange_order.average_price})",
                severity=Severity.MEDIUM,
            )
            self._persist_audit(audit)
            result.status = "resolved"
            result.severity = Severity.MEDIUM
            result.corrections.append(audit)
            logger.info(
                "Resolved order %s: FILLED @ %s",
                order_data["client_order_id"],
                exchange_order.average_price,
            )

        elif eo_status in (ExOrderStatus.CANCELLED, ExOrderStatus.REJECTED, ExOrderStatus.EXPIRED):
            self.db_manager.update_order_journal(
                client_order_id=order_data["client_order_id"],
                status="CANCELLED",
            )
            result.status = "resolved"
            result.severity = Severity.MEDIUM
            logger.info(
                "Resolved order %s: %s",
                order_data["client_order_id"],
                eo_status.value,
            )

        elif eo_status == ExOrderStatus.PARTIALLY_FILLED:
            order_type = order_data.get("order_type", "")

            if order_type == "ENTRY":
                # Create a position for the filled portion so acquired
                # inventory is tracked with a stop-loss. Cancel the remaining
                # order on exchange to avoid untracked fills — the strategy
                # can open a new position on the next signal if needed.
                fill_price = exchange_order.average_price or 0.0
                fill_qty = exchange_order.filled_quantity or 0.0
                if fill_price > 0 and fill_qty > 0:
                    symbol = order_data.get("symbol", "")
                    side = order_data.get("side", "LONG")
                    self._reconcile_filled_entry(
                        order_data, exchange_order, symbol, side, fill_price, fill_qty
                    )

                # Cancel the unfilled remainder on exchange so subsequent
                # fills do not become untracked inventory without a stop-loss.
                cancel_success = False
                try:
                    cancel_symbol = order_data.get("symbol", "")
                    cancel_success = self.exchange.cancel_order(
                        exchange_order.order_id, cancel_symbol
                    )
                except Exception as e:
                    logger.warning("Failed to cancel partial entry remainder: %s", e)

                if cancel_success:
                    # Safe to mark CONFIRMED — all fills accounted for,
                    # remainder cancelled.
                    logger.warning(
                        "Partially filled ENTRY %s: created position for "
                        "%.6f @ %.2f and CANCELLED remaining order on "
                        "exchange to prevent untracked fills",
                        order_data["client_order_id"],
                        fill_qty,
                        fill_price,
                    )
                    self.db_manager.update_order_journal(
                        client_order_id=order_data["client_order_id"],
                        status="CONFIRMED",
                        exchange_order_id=exchange_order.order_id,
                        fill_price=fill_price if fill_price > 0 else None,
                        fill_quantity=fill_qty if fill_qty > 0 else None,
                        commission=exchange_order.commission,
                    )
                    result.status = "resolved"
                    result.severity = Severity.MEDIUM
                else:
                    # Remainder may still be live — leave as SUBMITTED so
                    # the order stays in the unresolved queue for retry.
                    logger.warning(
                        "Could not cancel partial entry %s — remainder "
                        "may still fill; keeping journal as SUBMITTED",
                        order_data["client_order_id"],
                    )
                    self.db_manager.update_order_journal(
                        client_order_id=order_data["client_order_id"],
                        status="SUBMITTED",
                        exchange_order_id=exchange_order.order_id,
                        fill_price=fill_price if fill_price > 0 else None,
                        fill_quantity=fill_qty if fill_qty > 0 else None,
                        commission=exchange_order.commission,
                    )
                    result.status = "resolved"
                    result.severity = Severity.CRITICAL
            else:
                # Non-entry partial fills: keep journal as SUBMITTED since
                # the order is still active on exchange.
                self.db_manager.update_order_journal(
                    client_order_id=order_data["client_order_id"],
                    status="SUBMITTED",
                    exchange_order_id=exchange_order.order_id,
                )
                result.status = "resolved"
                result.severity = Severity.MEDIUM

        else:
            # Still pending on exchange (NEW, etc.)
            self.db_manager.update_order_journal(
                client_order_id=order_data["client_order_id"],
                status="SUBMITTED",
                exchange_order_id=exchange_order.order_id,
            )
            result.status = "resolved"
            result.severity = Severity.LOW

        return result

    def _reconcile_filled_order_position(self, order_data: dict, exchange_order: Any) -> None:
        """Reconcile position state after discovering a filled order on exchange.

        Handles two crash scenarios:
        1. ENTRY filled but position never persisted — creates the position.
        2. FULL_EXIT or PARTIAL_EXIT filled but position still tracked — closes it.

        Defensive: catches all exceptions since the position may already be
        in the correct state.
        """
        from src.engines.live.execution.position_tracker import LivePosition

        order_type = order_data.get("order_type", "")
        symbol = order_data.get("symbol", "")
        side = order_data.get("side", "LONG")
        fill_price = exchange_order.average_price or 0.0
        fill_qty = exchange_order.filled_quantity or 0.0

        if order_type == "ENTRY":
            self._reconcile_filled_entry(
                order_data, exchange_order, symbol, side, fill_price, fill_qty
            )
        elif order_type == "FULL_EXIT":
            self._reconcile_filled_exit(order_data, fill_price)
        elif order_type == "PARTIAL_EXIT":
            self._reconcile_filled_partial_exit(order_data, fill_price)

    def _reconcile_filled_entry(
        self,
        order_data: dict,
        exchange_order: Any,
        symbol: str,
        side: str,
        fill_price: float,
        fill_qty: float,
    ) -> None:
        """Create position if an ENTRY order filled but was never persisted."""
        from datetime import UTC, datetime

        from src.engines.live.execution.position_tracker import LivePosition

        client_order_id = order_data.get("client_order_id", "")
        # Derive tracker key from the verified exchange order object, not from
        # order_data which may still hold the stale client_order_id on first
        # recovery. This prevents duplicate positions on restart-twice.
        order_id = exchange_order.order_id or order_data.get("exchange_order_id") or client_order_id

        # Skip if position already exists under either the exchange_order_id
        # or the client_order_id (covers first-recovery key mismatch).
        with self.position_tracker._positions_lock:
            if (
                order_id in self.position_tracker._positions
                or client_order_id in self.position_tracker._positions
            ):
                logger.debug(
                    "Entry order %s: position already tracked (checked both "
                    "exchange_order_id=%s and client_order_id=%s), skipping.",
                    client_order_id,
                    order_id,
                    client_order_id,
                )
                return

        try:
            # Calculate size as balance fraction from entry_balance if available,
            # otherwise fetch current DB balance. size/original_size/current_size
            # are balance-fraction fields (0.0-1.0), NOT asset quantities.
            # Exits use size * entry_balance to compute sell quantity, so both
            # size_fraction and entry_balance must be accurate.
            entry_balance = order_data.get("entry_balance")
            if not entry_balance or entry_balance <= 0:
                try:
                    entry_balance = self.db_manager.get_current_balance(self.session_id)
                except Exception as e:
                    logger.warning("Failed to fetch DB balance for recovered entry sizing: %s", e)
                    entry_balance = None

            if entry_balance and entry_balance > 0 and fill_price > 0:
                size_fraction = min((fill_qty * fill_price) / entry_balance, 1.0)
            else:
                size_fraction = self.max_position_size  # Respect configured limit
                entry_balance = None  # Do not store an invalid balance

            position = LivePosition(
                symbol=symbol,
                side=side,
                entry_price=fill_price,
                entry_time=datetime.now(UTC),
                size=size_fraction,
                quantity=fill_qty,
                original_size=size_fraction,
                current_size=size_fraction,
                order_id=order_id,
                exchange_order_id=exchange_order.order_id,
                client_order_id=order_data.get("client_order_id"),
                entry_balance=entry_balance,
            )

            # Persist to DB first
            db_id = None
            try:
                db_id = self.db_manager.log_position(
                    symbol=symbol,
                    side=side,
                    entry_price=fill_price,
                    size=size_fraction,
                    strategy_name="recovered",
                    entry_order_id=order_id,
                    quantity=fill_qty,
                    session_id=self.session_id,
                    client_order_id=client_order_id,
                )
            except Exception as e:
                logger.warning("Failed to persist recovered entry position to DB: %s", e)

            # Track in memory
            self.position_tracker.track_recovered_position(position, db_id)

            # Apply a conservative default stop-loss so the recovered position
            # is never unprotected. The strategy may tighten this later.
            side_is_short = side == PositionSide.SHORT or str(side).lower() == "short"
            if side_is_short:
                default_stop = fill_price * (1.0 + DEFAULT_STOP_LOSS_PCT)
            else:
                default_stop = fill_price * (1.0 - DEFAULT_STOP_LOSS_PCT)
            position.stop_loss = default_stop
            logger.warning(
                "Recovered position %s has no stop-loss; applied default "
                "%.2f%% stop at %.4f (entry=%.4f, side=%s)",
                order_id,
                DEFAULT_STOP_LOSS_PCT * 100,
                default_stop,
                fill_price,
                side,
            )

            # Persist the default stop-loss to the database
            if db_id is not None:
                try:
                    self.db_manager.update_position(db_id, stop_loss=default_stop)
                except Exception as e:
                    logger.warning(
                        "Failed to persist default stop-loss for recovered " "position %s: %s",
                        order_id,
                        e,
                    )

            # Place server-side stop-loss on exchange for protection.
            # If SL placement fails, emergency-close the position to match
            # the normal entry path behavior (never leave unprotected).
            side_lower = side.lower()
            sl_placed = False
            if position.stop_loss and hasattr(self.exchange, "place_stop_loss_order"):
                try:
                    from src.data_providers.exchange_interface import OrderSide

                    sl_side = OrderSide.SELL if side_lower == "long" else OrderSide.BUY
                    sl_order_id = self.exchange.place_stop_loss_order(
                        symbol=symbol,
                        side=sl_side,
                        quantity=fill_qty,
                        stop_price=position.stop_loss,
                        side_effect_type="AUTO_REPAY",
                    )
                    if sl_order_id:
                        position.stop_loss_order_id = sl_order_id
                        sl_placed = True
                        logger.info(
                            "Placed recovery stop-loss for %s: %s @ %.2f",
                            symbol,
                            sl_order_id,
                            position.stop_loss,
                        )
                        # Persist the SL order ID to DB so it survives restarts
                        if db_id is not None:
                            try:
                                self.db_manager.update_position(
                                    position_id=db_id,
                                    stop_loss_order_id=sl_order_id,
                                )
                            except Exception as e:
                                logger.warning(
                                    "Failed to persist SL order ID for " "position %s: %s",
                                    order_id,
                                    e,
                                )
                except Exception as e:
                    logger.critical(
                        "Failed to place recovery stop-loss for %s: %s — "
                        "emergency-closing unprotected position",
                        symbol,
                        e,
                    )

                if not sl_placed:
                    # Emergency-close: sell on exchange, remove from tracker,
                    # and close in DB. The position can be re-entered on the
                    # next signal.
                    logger.critical(
                        "Recovery SL placement failed for %s (order_id=%s) — "
                        "emergency-closing on exchange and removing from "
                        "tracker and DB",
                        symbol,
                        order_id,
                    )

                    # Attempt to actually sell the asset on exchange
                    sell_result = None
                    try:
                        from src.data_providers.exchange_interface import (
                            OrderSide,
                            OrderType,
                        )

                        sell_side = OrderSide.SELL if side_lower == "long" else OrderSide.BUY
                        sell_result = self.exchange.place_order(
                            symbol=symbol,
                            side=sell_side,
                            order_type=OrderType.MARKET,
                            quantity=fill_qty,
                            side_effect_type="AUTO_REPAY",
                        )
                        if sell_result is not None:
                            logger.critical(
                                "Emergency-closed recovered %s position on "
                                "exchange (qty=%.8f, side=%s)",
                                symbol,
                                fill_qty,
                                sell_side,
                            )
                        else:
                            logger.critical(
                                "Emergency sell returned None for %s "
                                "(qty=%.8f) — keeping position tracked",
                                symbol,
                                fill_qty,
                            )
                    except Exception as sell_err:
                        logger.critical(
                            "CRITICAL: Emergency sell FAILED for %s "
                            "(qty=%.8f). MANUAL INTERVENTION REQUIRED. "
                            "Error: %s",
                            symbol,
                            fill_qty,
                            sell_err,
                        )

                    if sell_result is not None:
                        # Sell confirmed — safe to remove position
                        self.position_tracker.remove_position(order_id)
                        if db_id is not None:
                            try:
                                self.db_manager.close_position(db_id)
                            except Exception as db_err:
                                logger.critical(
                                    "Failed to close DB position %s after SL "
                                    "failure: %s — manual intervention required",
                                    db_id,
                                    db_err,
                                )
                        return
                    else:
                        # Sell ambiguous — keep position tracked, trigger
                        # close-only mode in the caller via RuntimeError
                        raise RuntimeError(
                            f"Emergency sell ambiguous for {symbol} — "
                            f"position kept tracked, entering close-only mode"
                        )

            logger.info(
                "Reconciled filled ENTRY %s: created position %s (db_id=%s)",
                client_order_id,
                order_id,
                db_id,
            )
        except Exception as e:
            logger.warning(
                "Failed to reconcile filled entry order %s: %s",
                client_order_id,
                e,
            )
            raise

    def _reconcile_filled_exit(self, order_data: dict, fill_price: float) -> None:
        """Close position if an exit order filled but position is still tracked."""
        position_id = order_data.get("position_id")
        client_order_id = order_data.get("client_order_id", "")

        # Try to find and remove from in-memory tracker by position_id mapping
        try:
            if position_id is not None:
                # Search tracker for position with matching db_position_id
                order_id_to_remove = None
                matched_position = None
                with self.position_tracker._positions_lock:
                    for oid, pos in self.position_tracker._positions.items():
                        if getattr(pos, "db_position_id", None) == position_id:
                            order_id_to_remove = oid
                            matched_position = pos
                            break

                if order_id_to_remove is not None:
                    self.position_tracker.remove_position(order_id_to_remove)
                    logger.info(
                        "Reconciled filled exit %s: removed position %s from tracker",
                        client_order_id,
                        order_id_to_remove,
                    )

                # Close in DB
                try:
                    self.db_manager.close_position(
                        position_id, exit_price=fill_price if fill_price > 0 else None
                    )
                    logger.info(
                        "Reconciled filled exit %s: closed DB position %s",
                        client_order_id,
                        position_id,
                    )
                except Exception as e:
                    logger.warning(
                        "Failed to close DB position %s for exit order %s: %s",
                        position_id,
                        client_order_id,
                        e,
                    )

                # Realize P&L so session balance stays correct
                if matched_position is not None and fill_price > 0:
                    self._realize_pnl_on_close(matched_position, fill_price, "exit_order_recovery")
        except Exception as e:
            logger.warning(
                "Failed to reconcile filled exit order %s: %s",
                client_order_id,
                e,
            )
            raise

    def _reconcile_filled_partial_exit(self, order_data: dict, fill_price: float) -> None:
        """Reduce position size for a filled partial exit, keeping it open.

        Unlike _reconcile_filled_exit which removes the position entirely,
        this method only reduces current_size by the exit's size_fraction.
        """
        position_id = order_data.get("position_id")
        client_order_id = order_data.get("client_order_id", "")
        size_fraction = order_data.get("size_fraction", 0.0)

        try:
            if position_id is not None:
                # Find the position in the tracker by db_position_id
                target_pos = None
                with self.position_tracker._positions_lock:
                    for _oid, pos in self.position_tracker._positions.items():
                        if getattr(pos, "db_position_id", None) == position_id:
                            target_pos = pos
                            break

                if target_pos is not None and size_fraction > 0:
                    new_size = max(target_pos.current_size - size_fraction, 0.0)
                    target_pos.current_size = new_size
                    target_pos.partial_exits_taken = (
                        getattr(target_pos, "partial_exits_taken", 0) + 1
                    )
                    target_pos.last_partial_exit_price = fill_price
                    logger.info(
                        "Reconciled filled PARTIAL_EXIT %s: reduced position size "
                        "by %.4f to %.4f",
                        client_order_id,
                        size_fraction,
                        new_size,
                    )

                    # Cancel and re-place stop-loss with correct remaining quantity
                    self._resize_stop_loss_after_partial_exit(target_pos)

                # Update DB position with reduced size
                try:
                    if size_fraction > 0:
                        self.db_manager.update_position(
                            position_id,
                            current_size=(target_pos.current_size if target_pos else None),
                            partial_exits_taken=(
                                target_pos.partial_exits_taken if target_pos else None
                            ),
                            last_partial_exit_price=fill_price if fill_price > 0 else None,
                        )
                    logger.info(
                        "Reconciled filled PARTIAL_EXIT %s: updated DB position %s",
                        client_order_id,
                        position_id,
                    )
                except Exception as e:
                    logger.warning(
                        "Failed to update DB position %s for partial exit %s: %s",
                        position_id,
                        client_order_id,
                        e,
                    )
        except Exception as e:
            logger.warning(
                "Failed to reconcile filled partial exit order %s: %s",
                client_order_id,
                e,
            )
            raise

    def _resize_stop_loss_after_partial_exit(self, position: object) -> None:
        """Cancel and re-place stop-loss with correct remaining quantity.

        After a partial exit reduces current_size, the existing stop-loss
        order still has the old (larger) quantity. This cancels the stale
        SL and places a new one sized to the remaining position.
        """
        sl_order_id = getattr(position, "stop_loss_order_id", None)
        stop_loss = getattr(position, "stop_loss", None)
        if not sl_order_id or not stop_loss:
            return
        if not hasattr(self.exchange, "place_stop_loss_order"):
            return

        symbol = getattr(position, "symbol", "")
        db_pos_id = getattr(position, "db_position_id", None)

        # Cancel the old stop-loss order
        try:
            self.exchange.cancel_order(sl_order_id, symbol)
            logger.info(
                "Cancelled stale stop-loss %s for %s after partial exit",
                sl_order_id,
                symbol,
            )
        except Exception as e:
            logger.warning(
                "Failed to cancel stale stop-loss %s for %s: %s",
                sl_order_id,
                symbol,
                e,
            )
            # If cancel fails, keep the old SL — better oversized than none
            return

        position.stop_loss_order_id = None  # type: ignore[attr-defined]

        # Compute remaining quantity based on current_size / original_size
        qty = getattr(position, "quantity", 0) or 0.0
        current = getattr(position, "current_size", None)
        original = getattr(position, "original_size", None)
        if current is not None and original is not None and original > 0:
            qty = qty * (current / original)

        if qty <= 0:
            return

        # Place new stop-loss with correct remaining quantity
        try:
            from src.data_providers.exchange_interface import OrderSide

            side = getattr(position, "side", "long")
            side_is_long = side == PositionSide.LONG or str(side).lower() == "long"
            sl_side = OrderSide.SELL if side_is_long else OrderSide.BUY
            new_sl_id = self.exchange.place_stop_loss_order(
                symbol=symbol,
                side=sl_side,
                quantity=qty,
                stop_price=stop_loss,
                side_effect_type="AUTO_REPAY",
            )
            if new_sl_id:
                position.stop_loss_order_id = new_sl_id  # type: ignore[attr-defined]
                logger.info(
                    "Replaced stop-loss for %s after partial exit: %s @ %.2f " "(qty=%.6f)",
                    symbol,
                    new_sl_id,
                    stop_loss,
                    qty,
                )
                # Persist the new SL order ID to DB
                if db_pos_id is not None:
                    try:
                        self.db_manager.update_position(
                            position_id=db_pos_id,
                            stop_loss_order_id=new_sl_id,
                        )
                    except Exception as e:
                        logger.warning(
                            "Failed to persist resized SL order ID " "for %s: %s",
                            symbol,
                            e,
                        )
            else:
                logger.warning(
                    "Failed to place resized stop-loss for %s — no order ID " "returned",
                    symbol,
                )
        except Exception as e:
            logger.warning(
                "Failed to place resized stop-loss for %s: %s",
                symbol,
                e,
            )

    def _handle_not_found_order(
        self, order_data: dict, status: str, result: ReconciliationResult
    ) -> ReconciliationResult:
        """Handle an order not found on exchange."""
        client_order_id = order_data.get("client_order_id", "")

        # Try bounded fallback search for ALL statuses (including PENDING_SUBMIT).
        # A PENDING_SUBMIT order may actually exist on exchange if place_order
        # raised ConnectionError/TimeoutError after the request was sent.
        try:
            created_at = order_data.get("created_at")
            if created_at:
                start_time = created_at - timedelta(minutes=5)
                all_orders = self.exchange.get_all_orders(
                    order_data["symbol"], start_time=start_time, limit=100
                )
                match = self._find_matching_order(order_data, all_orders)
                if match:
                    return self._handle_found_order(order_data, match, result)
        except Exception as e:
            logger.warning("Bounded fallback search failed: %s", e)

        # No match found on exchange after fallback search
        if status == "PENDING_SUBMIT":
            # Order was PENDING_SUBMIT and not found via client_id or fallback
            # — safe to assume it never reached the exchange
            self.db_manager.update_order_journal(
                client_order_id=client_order_id,
                status="CANCELLED",
            )
            audit = AuditEvent(
                entity_type="order",
                entity_id=order_data["id"],
                field="status",
                old_value=status,
                new_value="CANCELLED",
                reason="Order was PENDING_SUBMIT and not found on exchange after fallback search — never sent",
                severity=Severity.LOW,
            )
            self._persist_audit(audit)
            result.status = "resolved"
            result.severity = Severity.LOW
            result.corrections.append(audit)
            logger.info("Cancelled unsent order: %s", client_order_id)
            return result

        # SUBMITTED or UNKNOWN — mark UNRESOLVED for manual intervention
        self.db_manager.update_order_journal(
            client_order_id=client_order_id,
            status="UNRESOLVED",
        )
        audit = AuditEvent(
            entity_type="order",
            entity_id=order_data["id"],
            field="status",
            old_value=status,
            new_value="UNRESOLVED",
            reason=f"Order {status} but not found on exchange — manual intervention required",
            severity=Severity.CRITICAL,
        )
        self._persist_audit(audit)
        result.status = "unresolved"
        result.severity = Severity.CRITICAL
        result.corrections.append(audit)
        logger.critical(
            "UNRESOLVED order %s: was %s but not found on exchange",
            client_order_id,
            status,
        )
        return result

    def _find_matching_order(self, order_data: dict, exchange_orders: list[Any]) -> Any | None:
        """Find a matching order using strict correlation criteria."""
        target_qty = order_data["quantity"]
        target_side = order_data["side"]
        created_at = order_data.get("created_at")
        client_prefix = "atb"
        tolerance = DEFAULT_RECONCILIATION_ORDER_MATCH_TOLERANCE_PCT
        time_window = timedelta(minutes=DEFAULT_RECONCILIATION_ORDER_MATCH_TIME_WINDOW_MIN)

        # Determine expected exchange side based on order type.
        # For entries, LONG→BUY, SHORT→SELL.
        # For exits (FULL_EXIT, PARTIAL_EXIT), the side is inverted because
        # closing a LONG requires a SELL and closing a SHORT requires a BUY.
        order_type = order_data.get("order_type", "ENTRY")
        is_exit_order = order_type in ("FULL_EXIT", "PARTIAL_EXIT")
        if is_exit_order:
            expected_exchange_side = "SELL" if target_side == "LONG" else "BUY"
        else:
            expected_exchange_side = "BUY" if target_side == "LONG" else "SELL"

        candidates = []
        for order in exchange_orders:
            # Filter: must have our prefix (skip manually placed orders too)
            # Accepts both "atb_" (entry) and "atbx_" (exit) prefixed orders.
            client_id = getattr(order, "client_order_id", None)
            if not client_id or not client_id.startswith(client_prefix):
                continue

            # Filter: matching side
            order_side = order.side.value if hasattr(order.side, "value") else str(order.side)
            if order_side != expected_exchange_side:
                continue

            # Filter: quantity within tolerance
            qty_diff = abs(order.quantity - target_qty) / max(target_qty, 1e-9)
            if qty_diff > tolerance:
                continue

            # Filter: timestamp within window
            # Normalize created_at to UTC-aware before comparison
            if created_at and created_at.tzinfo is None:
                created_at = created_at.replace(tzinfo=UTC)
            if created_at and hasattr(order, "create_time") and order.create_time:
                time_diff = abs((order.create_time - created_at).total_seconds())
                if time_diff > time_window.total_seconds():
                    continue

            candidates.append(order)

        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            logger.warning(
                "Ambiguous match: %d candidates for order %s",
                len(candidates),
                order_data.get("client_order_id"),
            )
        return None

    def reconcile_position(self, position: Any) -> ReconciliationResult:
        """Verify a single recovered position against exchange data."""
        result = ReconciliationResult(
            entity_type="position",
            entity_id=getattr(position, "db_position_id", None),
            status="verified",
        )

        exchange_order_id = getattr(position, "exchange_order_id", None)
        client_order_id = getattr(position, "client_order_id", None)
        symbol = position.symbol

        # 1. Verify entry order
        exchange_order = None
        if exchange_order_id:
            try:
                exchange_order = self.exchange.get_order(exchange_order_id, symbol)
            except Exception as e:
                logger.warning("Failed to verify entry order %s: %s", exchange_order_id, e)
            # If exchange_order_id is actually a client_order_id (phantom from timeout),
            # fall back to client_id lookup
            if exchange_order is None and exchange_order_id.startswith("atb"):
                try:
                    exchange_order = self.exchange.get_order_by_client_id(exchange_order_id, symbol)
                    if exchange_order:
                        # Update to real exchange order ID
                        position.exchange_order_id = exchange_order.order_id
                        logger.info(
                            "Resolved phantom exchange_order_id %s to real order %s",
                            exchange_order_id,
                            exchange_order.order_id,
                        )
                except Exception as e:
                    logger.warning(
                        "Failed to resolve phantom order %s by client_id: %s",
                        exchange_order_id,
                        e,
                    )
        elif client_order_id:
            try:
                exchange_order = self.exchange.get_order_by_client_id(client_order_id, symbol)
            except Exception as e:
                logger.warning("Failed to verify entry by client_id %s: %s", client_order_id, e)

        if exchange_order:
            result = self._verify_entry_order(position, exchange_order, result)
        elif not exchange_order_id and not client_order_id:
            # Legacy position — no exchange identifiers
            logger.warning(
                "Legacy position %s has no exchange identifiers — skipping verification",
                symbol,
            )
            result.status = "skipped"
            result.severity = Severity.LOW

        # 2. Verify stop-loss order if present
        sl_order_id = getattr(position, "stop_loss_order_id", None)
        if sl_order_id:
            self._verify_stop_loss(position, sl_order_id, result)

        # 3. Place missing exchange stop-loss for positions that have a SL price
        # but no exchange SL order (e.g. phantom positions from timeout, or any
        # case where SL placement was missed before shutdown).
        if result.status != "corrected" and not getattr(position, "stop_loss_order_id", None):
            sl_price = getattr(position, "stop_loss", None)
            if not sl_price:
                # No stop_loss price either — compute a default one
                entry_price = getattr(position, "entry_price", None) or 0.0
                if entry_price > 0:
                    side = getattr(position, "side", "long")
                    side_is_short = side == PositionSide.SHORT or str(side).lower() == "short"
                    if side_is_short:
                        sl_price = entry_price * (1.0 + DEFAULT_STOP_LOSS_PCT)
                    else:
                        sl_price = entry_price * (1.0 - DEFAULT_STOP_LOSS_PCT)
                    position.stop_loss = sl_price
                    logger.warning(
                        "Position %s has no stop-loss price; applied default "
                        "%.2f%% stop at %.4f (entry=%.4f, side=%s)",
                        position.symbol,
                        DEFAULT_STOP_LOSS_PCT * 100,
                        sl_price,
                        entry_price,
                        side,
                    )

            if sl_price and hasattr(self.exchange, "place_stop_loss_order"):
                try:
                    from src.data_providers.exchange_interface import OrderSide

                    side = getattr(position, "side", "long")
                    side_is_long = side == PositionSide.LONG or str(side).lower() == "long"
                    sl_side = OrderSide.SELL if side_is_long else OrderSide.BUY
                    # Scale quantity by remaining size after partial exits
                    qty = getattr(position, "quantity", 0) or 0.0
                    current = getattr(position, "current_size", None)
                    original = getattr(position, "original_size", None)
                    if current is not None and original is not None and original > 0:
                        qty = qty * (current / original)
                    new_sl_id = self.exchange.place_stop_loss_order(
                        symbol=position.symbol,
                        side=sl_side,
                        quantity=qty,
                        stop_price=sl_price,
                        side_effect_type="AUTO_REPAY",
                    )
                    if new_sl_id:
                        position.stop_loss_order_id = new_sl_id
                        logger.info(
                            "Placed missing stop-loss for %s: %s @ %.2f",
                            position.symbol,
                            new_sl_id,
                            sl_price,
                        )
                        # Persist to DB
                        db_pos_id = getattr(position, "db_position_id", None)
                        if db_pos_id is not None:
                            try:
                                self.db_manager.update_position(
                                    position_id=db_pos_id,
                                    stop_loss_order_id=new_sl_id,
                                    stop_loss=sl_price,
                                )
                            except Exception as e:
                                logger.warning(
                                    "Failed to persist SL order ID for %s: %s",
                                    position.symbol,
                                    e,
                                )
                    else:
                        logger.critical(
                            "Failed to place missing stop-loss for %s — " "position is unprotected",
                            position.symbol,
                        )
                        result.severity = Severity.CRITICAL
                except Exception as e:
                    logger.critical(
                        "Exception placing missing stop-loss for %s: %s — "
                        "position is unprotected",
                        position.symbol,
                        e,
                    )
                    result.severity = Severity.CRITICAL

        # 4. Verify asset holdings — detect external closes
        if result.status != "corrected":
            self._verify_asset_holdings(position, result)

        return result

    def _verify_entry_order(
        self, position: Any, exchange_order: Any, result: ReconciliationResult
    ) -> ReconciliationResult:
        """Verify entry order fill data matches position."""
        from src.data_providers.exchange_interface import OrderStatus as ExOrderStatus

        if exchange_order.status == ExOrderStatus.FILLED:
            # Check fill price matches entry_price
            if exchange_order.average_price and position.entry_price > 0:
                price_diff_pct = (
                    abs(exchange_order.average_price - position.entry_price) / position.entry_price
                )
                if price_diff_pct > 0.001:  # >0.1% difference
                    audit = AuditEvent(
                        entity_type="position",
                        entity_id=getattr(position, "db_position_id", None),
                        field="entry_price",
                        old_value=str(position.entry_price),
                        new_value=str(exchange_order.average_price),
                        reason=f"Entry price mismatch ({price_diff_pct:.4%}) — exchange is source of truth",
                        severity=Severity.MEDIUM,
                    )
                    self._persist_audit(audit)
                    result.corrections.append(audit)
                    result.severity = Severity.MEDIUM
                    logger.info(
                        "Entry price corrected for %s: %.8f → %.8f",
                        position.symbol,
                        position.entry_price,
                        exchange_order.average_price,
                    )
                    previous_entry_price = position.entry_price
                    position.entry_price = exchange_order.average_price
                    # Persist corrected entry_price to DB
                    try:
                        self._persist_position_correction(
                            position, entry_price=exchange_order.average_price
                        )
                    except Exception as e:
                        position.entry_price = previous_entry_price
                        logger.critical(
                            "DB/memory state diverged for position %s entry_price — rolled back in-memory: %s",
                            position.symbol,
                            e,
                            exc_info=True,
                        )
                        raise

            # Check fill quantity matches position quantity
            filled_qty = getattr(exchange_order, "filled_quantity", None)
            position_qty = getattr(position, "quantity", None) or 0.0
            if filled_qty and filled_qty > 0 and position_qty > 0:
                qty_diff_pct = abs(filled_qty - position_qty) / position_qty
                # Correct if >1% difference — significant enough to affect P&L
                if qty_diff_pct > 0.01:
                    audit = AuditEvent(
                        entity_type="position",
                        entity_id=getattr(position, "db_position_id", None),
                        field="quantity",
                        old_value=str(position_qty),
                        new_value=str(filled_qty),
                        reason=(
                            f"Quantity mismatch ({qty_diff_pct:.4%}) "
                            "— exchange is source of truth"
                        ),
                        severity=Severity.MEDIUM,
                    )
                    self._persist_audit(audit)
                    result.corrections.append(audit)
                    if result.severity < Severity.MEDIUM:
                        result.severity = Severity.MEDIUM
                    logger.info(
                        "Quantity corrected for %s: %.8f → %.8f",
                        position.symbol,
                        position_qty,
                        filled_qty,
                    )
                    # Snapshot in-memory state for rollback on DB failure
                    prev_quantity = position_qty
                    prev_size = getattr(position, "size", None)
                    prev_current_size = getattr(position, "current_size", None)
                    prev_original_size = getattr(position, "original_size", None)

                    position.quantity = filled_qty

                    # Recalculate size (balance fraction) if entry_balance is available
                    entry_price = position.entry_price
                    entry_balance = getattr(position, "entry_balance", None)
                    if entry_price and entry_price > 0 and entry_balance and entry_balance > 0:
                        new_size = (filled_qty * entry_price) / entry_balance
                        position.size = new_size
                        if hasattr(position, "current_size"):
                            position.current_size = new_size
                        if hasattr(position, "original_size"):
                            position.original_size = new_size
                        logger.info(
                            "Size recalculated for %s: %s → %.6f",
                            position.symbol,
                            prev_size,
                            new_size,
                        )

                    # Persist corrected quantity/size to DB
                    # Only include size fields if they were actually recalculated
                    correction_kwargs: dict[str, float | None] = {"quantity": filled_qty}
                    if entry_price and entry_price > 0 and entry_balance and entry_balance > 0:
                        correction_kwargs["size"] = getattr(position, "size", None)
                        correction_kwargs["current_size"] = getattr(position, "current_size", None)
                        correction_kwargs["original_size"] = getattr(
                            position, "original_size", None
                        )
                    try:
                        self._persist_position_correction(
                            position,
                            **correction_kwargs,
                        )
                    except Exception as e:
                        # Rollback in-memory state to prevent DB/memory divergence
                        position.quantity = prev_quantity
                        if prev_size is not None:
                            position.size = prev_size
                        if prev_current_size is not None and hasattr(position, "current_size"):
                            position.current_size = prev_current_size
                        if prev_original_size is not None and hasattr(position, "original_size"):
                            position.original_size = prev_original_size
                        logger.critical(
                            "DB/memory state diverged for position %s quantity/size — rolled back in-memory: %s",
                            position.symbol,
                            e,
                            exc_info=True,
                        )
                        raise

        elif exchange_order.status in (
            ExOrderStatus.CANCELLED,
            ExOrderStatus.REJECTED,
        ):
            # Position never actually opened
            audit = AuditEvent(
                entity_type="position",
                entity_id=getattr(position, "db_position_id", None),
                field="status",
                old_value="OPEN",
                new_value="CANCELLED",
                reason=f"Entry order was {exchange_order.status.value} — position never opened",
                severity=Severity.HIGH,
            )
            self._persist_audit(audit)
            result.corrections.append(audit)
            result.status = "corrected"
            result.severity = Severity.HIGH
            logger.warning(
                "Position %s entry order was %s — removing from tracker",
                position.symbol,
                exchange_order.status.value,
            )
            # Remove phantom position from in-memory tracker
            self.position_tracker.remove_position(position.order_id)
            # Close the DB position if it exists
            db_pos_id = getattr(position, "db_position_id", None)
            if db_pos_id:
                try:
                    self.db_manager.close_position(db_pos_id)
                except Exception as e:
                    logger.warning("Failed to close DB position %s: %s", db_pos_id, e)

        return result

    def _verify_stop_loss(
        self, position: Any, sl_order_id: str, result: ReconciliationResult
    ) -> None:
        """Verify stop-loss order status."""
        from src.data_providers.exchange_interface import OrderStatus as ExOrderStatus

        try:
            sl_order = self.exchange.get_order(sl_order_id, position.symbol)
            if not sl_order:
                # SL order not found — may need re-placement
                audit = AuditEvent(
                    entity_type="position",
                    entity_id=getattr(position, "db_position_id", None),
                    field="stop_loss_order_id",
                    old_value=sl_order_id,
                    new_value="MISSING",
                    reason="Stop-loss order not found on exchange — needs re-placement",
                    severity=Severity.MEDIUM,
                )
                self._persist_audit(audit)
                result.corrections.append(audit)
                if result.severity < Severity.MEDIUM:
                    result.severity = Severity.MEDIUM
                position.stop_loss_order_id = None

                # Attempt to re-place the stop-loss so the position is protected
                if position.stop_loss and hasattr(self.exchange, "place_stop_loss_order"):
                    try:
                        from src.data_providers.exchange_interface import OrderSide

                        side = getattr(position, "side", "long")
                        side_is_long = side == PositionSide.LONG or str(side).lower() == "long"
                        sl_side = OrderSide.SELL if side_is_long else OrderSide.BUY
                        # Scale quantity by remaining size after partial exits
                        qty = getattr(position, "quantity", 0) or 0.0
                        current = getattr(position, "current_size", None)
                        original = getattr(position, "original_size", None)
                        if current is not None and original is not None and original > 0:
                            qty = qty * (current / original)
                        new_sl_id = self.exchange.place_stop_loss_order(
                            symbol=position.symbol,
                            side=sl_side,
                            quantity=qty,
                            stop_price=position.stop_loss,
                            side_effect_type="AUTO_REPAY",
                        )
                        if new_sl_id:
                            position.stop_loss_order_id = new_sl_id
                            logger.info(
                                "Re-placed missing stop-loss for %s: %s @ %.2f",
                                position.symbol,
                                new_sl_id,
                                position.stop_loss,
                            )
                            # Persist the new SL order ID to DB
                            db_pos_id = getattr(position, "db_position_id", None)
                            if db_pos_id is not None:
                                try:
                                    self.db_manager.update_position(
                                        position_id=db_pos_id,
                                        stop_loss_order_id=new_sl_id,
                                    )
                                except Exception as e:
                                    logger.warning(
                                        "Failed to persist re-placed SL order ID " "for %s: %s",
                                        position.symbol,
                                        e,
                                    )
                        else:
                            # SL re-placement failed — escalate to CRITICAL
                            logger.critical(
                                "Failed to re-place missing stop-loss for %s — "
                                "position is unprotected, entering close-only mode",
                                position.symbol,
                            )
                            result.severity = Severity.CRITICAL
                    except Exception as e:
                        logger.critical(
                            "Exception re-placing missing stop-loss for %s: %s — "
                            "position is unprotected",
                            position.symbol,
                            e,
                        )
                        result.severity = Severity.CRITICAL

                return

            if sl_order.status in (ExOrderStatus.CANCELLED, ExOrderStatus.EXPIRED):
                # SL was cancelled or expired — position is unprotected
                filled_qty = getattr(sl_order, "filled_quantity", None) or 0.0

                # If the SL partially executed before cancellation/expiry,
                # compute the remaining held quantity and update position.quantity
                # to match. After partial exits, the bot holds
                # `quantity * (current_size / original_size)` of the asset. When
                # the SL fills some of that, the remaining is held_qty - filled_qty.
                if filled_qty > 0 and hasattr(position, "quantity") and position.quantity > 0:
                    old_quantity = position.quantity
                    # Compute what we actually hold before the SL fill
                    held_qty = position.quantity
                    current = getattr(position, "current_size", None)
                    original = getattr(position, "original_size", None)
                    if current is not None and original is not None and original > 0:
                        held_qty = position.quantity * (current / original)
                    # After SL fill, what remains
                    remaining_qty = max(held_qty - filled_qty, 0.0)

                    # Reduce current_size proportionally instead of mutating quantity.
                    # Other code paths scale `quantity * (current_size / original_size)`
                    # to compute held amount, so mutating quantity directly would cause
                    # double-reduction in P&L calculations, notional estimates, and the
                    # periodic asset-holdings check.
                    if original is not None and original > 0 and position.quantity > 0:
                        remaining_fraction = remaining_qty / max(position.quantity, 1e-9)
                        if hasattr(position, "current_size"):
                            position.current_size = original * remaining_fraction
                    else:
                        # Fallback: no size tracking, update quantity directly
                        position.quantity = remaining_qty

                    partial_audit = AuditEvent(
                        entity_type="position",
                        entity_id=getattr(position, "db_position_id", None),
                        field="current_size",
                        old_value=str(current or old_quantity),
                        new_value=str(getattr(position, "current_size", remaining_qty)),
                        reason=(
                            f"SL order {sl_order.status.value.lower()} after partial fill "
                            f"of {filled_qty} — reduced position size"
                        ),
                        severity=Severity.MEDIUM,
                    )
                    self._persist_audit(partial_audit)
                    result.corrections.append(partial_audit)
                    logger.warning(
                        "Stop-loss %s for %s was %s with partial fill of %s — "
                        "reduced position size (remaining_qty=%.8f)",
                        sl_order_id,
                        position.symbol,
                        sl_order.status.value,
                        filled_qty,
                        remaining_qty,
                    )

                audit = AuditEvent(
                    entity_type="position",
                    entity_id=getattr(position, "db_position_id", None),
                    field="stop_loss_order_id",
                    old_value=sl_order_id,
                    new_value=None,
                    reason=(
                        f"Stop-loss order {sl_order.status.value.lower()} on exchange "
                        "— position is unprotected, needs SL re-placement"
                    ),
                    severity=Severity.MEDIUM,
                )
                self._persist_audit(audit)
                result.corrections.append(audit)
                if result.severity < Severity.MEDIUM:
                    result.severity = Severity.MEDIUM
                position.stop_loss_order_id = None
                logger.warning(
                    "Stop-loss %s for %s was %s — cleared stale reference",
                    sl_order_id,
                    position.symbol,
                    sl_order.status.value,
                )

                # Attempt to re-place the stop-loss so the position is protected
                if position.stop_loss and hasattr(self.exchange, "place_stop_loss_order"):
                    try:
                        from src.data_providers.exchange_interface import OrderSide

                        side = getattr(position, "side", "long")
                        side_is_long = side == PositionSide.LONG or str(side).lower() == "long"
                        sl_side = OrderSide.SELL if side_is_long else OrderSide.BUY
                        # Scale quantity by current_size/original_size to get
                        # actual held amount after partial exits and SL fills.
                        qty = getattr(position, "quantity", 0) or 0.0
                        cs = getattr(position, "current_size", None)
                        os_ = getattr(position, "original_size", None)
                        if cs is not None and os_ is not None and os_ > 0:
                            qty = qty * (cs / os_)
                        # Position is flat — no SL needed
                        if qty <= 0:
                            logger.info(
                                "Position %s is flat after partial SL fill — skipping SL re-placement",
                                position.symbol,
                            )
                            return
                        new_sl_id = self.exchange.place_stop_loss_order(
                            symbol=position.symbol,
                            side=sl_side,
                            quantity=qty,
                            stop_price=position.stop_loss,
                            side_effect_type="AUTO_REPAY",
                        )
                        if new_sl_id:
                            position.stop_loss_order_id = new_sl_id
                            logger.info(
                                "Re-placed stop-loss for %s: %s @ %.2f",
                                position.symbol,
                                new_sl_id,
                                position.stop_loss,
                            )
                            # Persist new SL order ID and updated current_size
                            # so restart doesn't reload stale values.
                            db_pos_id = getattr(position, "db_position_id", None)
                            if db_pos_id is not None:
                                try:
                                    update_kwargs: dict[str, Any] = {
                                        "stop_loss_order_id": new_sl_id,
                                    }
                                    _cs = getattr(position, "current_size", None)
                                    if _cs is not None:
                                        update_kwargs["current_size"] = _cs
                                    self.db_manager.update_position(
                                        position_id=db_pos_id,
                                        **update_kwargs,
                                    )
                                except Exception as e:
                                    logger.warning(
                                        "Failed to persist re-placed SL / current_size "
                                        "for %s: %s",
                                        position.symbol,
                                        e,
                                    )
                        else:
                            # SL re-placement failed — escalate to CRITICAL
                            logger.critical(
                                "Failed to re-place stop-loss for %s — "
                                "position is unprotected, entering close-only mode",
                                position.symbol,
                            )
                            result.severity = Severity.CRITICAL
                    except Exception as e:
                        logger.critical(
                            "Exception re-placing stop-loss for %s: %s — "
                            "position is unprotected",
                            position.symbol,
                            e,
                        )
                        result.severity = Severity.CRITICAL

                return

            if sl_order.status == ExOrderStatus.FILLED:
                # SL triggered while offline — position should be closed
                audit = AuditEvent(
                    entity_type="position",
                    entity_id=getattr(position, "db_position_id", None),
                    field="status",
                    old_value="OPEN",
                    new_value="CLOSED_BY_SL",
                    reason=f"Stop-loss filled @ {sl_order.average_price} while offline",
                    severity=Severity.HIGH,
                )
                self._persist_audit(audit)
                result.corrections.append(audit)
                result.status = "corrected"
                result.severity = Severity.HIGH
                logger.warning(
                    "Stop-loss triggered offline for %s @ %s",
                    position.symbol,
                    sl_order.average_price,
                )
                # Remove position from in-memory tracker
                self.position_tracker.remove_position(position.order_id)
                # Close the DB position with SL fill details
                db_pos_id = getattr(position, "db_position_id", None)
                exit_price = float(sl_order.average_price) if sl_order.average_price else None
                if db_pos_id:
                    try:
                        self.db_manager.close_position(db_pos_id, exit_price=exit_price)
                    except Exception as e:
                        logger.warning("Failed to close DB position %s: %s", db_pos_id, e)

                # Realize P&L so session balance stays correct
                self._realize_pnl_on_close(position, exit_price, "stop_loss_filled_offline")

        except Exception as e:
            logger.warning("Failed to verify stop-loss order %s: %s", sl_order_id, e)

    @staticmethod
    def _extract_base_asset(symbol: str) -> str:
        """Extract the base asset from a trading pair symbol.

        Strips common quote currencies (USDT, BUSD, USD) from the end
        of the symbol to get the base asset (e.g., "BTC" from "BTCUSDT").
        """
        for quote in ("USDT", "BUSD", "USD"):
            if symbol.endswith(quote) and len(symbol) > len(quote):
                return symbol[: -len(quote)]
        return symbol

    def _verify_asset_holdings(self, position: Any, result: ReconciliationResult) -> None:
        """Verify the bot holds the expected asset on exchange.

        Detects positions closed externally (e.g., manual sell on Binance UI)
        by checking if the actual asset balance is significantly below the
        tracked position quantity. Uses 50% threshold to catch full external
        closes while tolerating partial exits.
        """
        # In margin mode, use borrowed-amount check instead of spot balance.
        if self._use_margin:
            self._verify_margin_position_exists(position, result)
            return

        # Skip short positions — shorts don't hold the base asset on spot,
        # so checking spot balance would falsely flag them as externally closed.
        side = getattr(position, "side", None)
        if side == PositionSide.SHORT or str(side).lower() == "short":
            return

        symbol = position.symbol
        base_asset = self._extract_base_asset(symbol)
        # Use quantity (actual asset amount), not size (balance fraction).
        # Scale by current_size/original_size to account for partial exits
        # that reduce current_size but leave quantity unchanged.
        qty = getattr(position, "quantity", None) or 0.0
        if qty <= 0:
            return

        current_size = getattr(position, "current_size", None)
        original_size = getattr(position, "original_size", None)
        if current_size is not None and original_size is not None and original_size > 0:
            position_qty = qty * (current_size / original_size)
        else:
            position_qty = qty

        try:
            balance = self.exchange.get_balance(base_asset)
            if balance is None:
                logger.warning(
                    "get_balance returned None for %s — skipping asset check "
                    "(transient API error)",
                    base_asset,
                )
                return
            held_qty = balance.total

            # If held quantity exceeds 150% of tracked quantity, there
            # may be untracked fills (e.g., a partial entry remainder
            # kept filling after the bot entered close-only mode).
            if position_qty > 0 and held_qty > position_qty * 1.5:
                excess_audit = AuditEvent(
                    entity_type="position",
                    entity_id=getattr(position, "db_position_id", None),
                    field="quantity",
                    old_value=str(position_qty),
                    new_value=str(held_qty),
                    reason=(
                        f"More asset held than tracked — possible untracked fills "
                        f"(held={held_qty:.8f}, tracked={position_qty:.8f}, "
                        f"asset={base_asset})"
                    ),
                    severity=Severity.HIGH,
                )
                self._persist_audit(excess_audit)
                result.corrections.append(excess_audit)
                result.severity = Severity.HIGH
                logger.warning(
                    "Position %s: more %s held than tracked "
                    "(held=%.8f, tracked=%.8f) — possible untracked fills",
                    symbol,
                    base_asset,
                    held_qty,
                    position_qty,
                )

            # If held quantity is less than 50% of tracked quantity,
            # the position was likely closed externally
            if held_qty < position_qty * 0.5:
                audit = AuditEvent(
                    entity_type="position",
                    entity_id=getattr(position, "db_position_id", None),
                    field="status",
                    old_value="OPEN",
                    new_value="CLOSED_EXTERNALLY",
                    reason=(
                        f"Position asset not found on exchange — likely closed externally "
                        f"(held={held_qty:.8f}, tracked={position_qty:.8f}, asset={base_asset})"
                    ),
                    severity=Severity.HIGH,
                )
                self._persist_audit(audit)
                result.corrections.append(audit)
                result.status = "corrected"
                result.severity = Severity.HIGH
                logger.warning(
                    "Position %s externally closed: held %s=%s, tracked=%s",
                    symbol,
                    base_asset,
                    held_qty,
                    position_qty,
                )
                # Remove from in-memory tracker
                self.position_tracker.remove_position(position.order_id)
                # Close the DB position
                db_pos_id = getattr(position, "db_position_id", None)
                if db_pos_id:
                    try:
                        self.db_manager.close_position(db_pos_id)
                    except Exception as e:
                        logger.warning("Failed to close DB position %s: %s", db_pos_id, e)

                # Realize P&L — no exit price available for external closes,
                # so skip balance update (P&L is unknown)
                # External closes have no fill price; we cannot compute P&L
                logger.info(
                    "External close for %s — P&L not realized (no exit price)",
                    symbol,
                )
        except Exception as e:
            logger.warning("Asset holdings check failed for %s: %s", base_asset, e)

    def _verify_margin_position_exists(
        self, position: Any, result: ReconciliationResult
    ) -> None:
        """Verify a margin position still exists by checking borrowed balance.

        For short positions, checks if the base asset has borrowed > 0.
        For long positions, checks if the base asset has netAsset > 0.
        Detects externally closed or liquidated positions in margin mode.
        """
        symbol = position.symbol
        base_asset = self._extract_base_asset(symbol)
        side = getattr(position, "side", None)
        is_short = side == PositionSide.SHORT or str(side).lower() == "short"

        try:
            # get_balance returns AccountBalance with total=netAsset in margin mode
            balance = self.exchange.get_balance(base_asset)

            if is_short:
                # Short positions create debt — check if borrowed amount exists.
                # Use the raw balances to get the 'borrowed' field.
                balances = self.exchange.get_balances()
                borrowed = 0.0
                for b in balances:
                    if b.asset == base_asset:
                        # AccountBalance doesn't have 'borrowed' field,
                        # but if balance.total (netAsset) is negative, there's debt
                        borrowed = abs(min(b.total, 0))
                        break

                if borrowed == 0 and (balance is None or balance.total >= 0):
                    logger.warning(
                        "Margin short for %s appears externally closed "
                        "(no borrowed %s). Removing tracked position.",
                        symbol,
                        base_asset,
                    )
                    self.position_tracker.remove_position(position.order_id)
                    db_pos_id = getattr(position, "db_position_id", None)
                    if db_pos_id:
                        try:
                            self.db_manager.close_position(db_pos_id)
                        except Exception as e:
                            logger.warning("Failed to close DB position %s: %s", db_pos_id, e)
                    result.status = "corrected"
                    result.severity = Severity.HIGH
            else:
                # Long positions hold the asset — check netAsset > 0
                if balance is None or balance.total <= 0:
                    logger.warning(
                        "Margin long for %s appears externally closed "
                        "(no %s holdings). Removing tracked position.",
                        symbol,
                        base_asset,
                    )
                    self.position_tracker.remove_position(position.order_id)
                    db_pos_id = getattr(position, "db_position_id", None)
                    if db_pos_id:
                        try:
                            self.db_manager.close_position(db_pos_id)
                        except Exception as e:
                            logger.warning("Failed to close DB position %s: %s", db_pos_id, e)
                    result.status = "corrected"
                    result.severity = Severity.HIGH

        except Exception as e:
            logger.warning(
                "Margin position check failed for %s: %s — position retained", symbol, e
            )

    def _estimate_position_notional(self) -> float:
        """Estimate total notional value of open positions using entry prices.

        In spot trading, buying an asset reduces USDT by the purchase amount.
        The DB balance only changes for fees/realized PnL, so comparing raw
        DB balance to exchange USDT would show a false discrepancy equal to
        the notional value of open positions. This method calculates that
        notional so it can be subtracted from the DB balance.
        """
        total = 0.0
        positions = self.position_tracker.positions
        for position in positions.values():
            # Use quantity (actual asset amount), not size (balance fraction)
            qty = getattr(position, "quantity", None) or 0.0
            price = getattr(position, "entry_price", 0)
            if qty > 0 and price > 0:
                # Scale by current_size/original_size to account for partial exits
                current = getattr(position, "current_size", None)
                original = getattr(position, "original_size", None)
                if current is not None and original is not None and original > 0:
                    qty = qty * (current / original)
                total += qty * price
        return total

    def _reconcile_balance(self) -> ReconciliationResult:
        """Verify account balance consistency.

        Compares exchange USDT against the expected USDT after accounting for
        position notional values. In spot trading, buying BTC reduces USDT by
        the purchase amount, so raw DB balance minus position notional gives
        the expected USDT on exchange.

        Skipped in margin mode — cross-margin USDT balance includes short sale
        proceeds and doesn't reflect true equity without subtracting liabilities.
        """
        if self._use_margin:
            logger.info("Skipping startup balance reconciliation in margin mode")
            return ReconciliationResult(
                entity_type="balance",
                entity_id=None,
                status="skipped",
            )

        result = ReconciliationResult(
            entity_type="balance",
            entity_id=None,
            status="verified",
        )

        try:
            usdt_balance = self.exchange.get_balance("USDT")
            if not usdt_balance:
                return result

            exchange_total = usdt_balance.total
            db_balance = self.db_manager.get_current_balance(self.session_id)

            if db_balance > 0:
                # Subtract position notional to get expected USDT on exchange
                position_notional = self._estimate_position_notional()
                expected_usdt = db_balance - position_notional
                # Use abs(expected_usdt) to avoid false CRITICAL when
                # expected_usdt is negative (e.g. position notional > db_balance
                # due to price appreciation)
                comparison_base = max(abs(expected_usdt), db_balance * 0.01)
                diff_pct = abs(exchange_total - expected_usdt) / comparison_base
                if diff_pct > DEFAULT_RECONCILIATION_BALANCE_THRESHOLD_PCT:
                    audit = AuditEvent(
                        entity_type="balance",
                        entity_id=None,
                        field="total_balance",
                        old_value=str(expected_usdt),
                        new_value=str(exchange_total),
                        reason=(
                            f"Balance discrepancy {diff_pct:.2%} exceeds threshold "
                            f"(db={db_balance:.2f}, position_notional={position_notional:.2f}, "
                            f"expected_usdt={expected_usdt:.2f})"
                        ),
                        severity=Severity.CRITICAL,
                    )
                    self._persist_audit(audit)
                    result.corrections.append(audit)
                    result.severity = Severity.CRITICAL
                    result.status = "corrected"
                    logger.critical(
                        "Balance discrepancy: expected=$%.2f vs Exchange=$%.2f (%.2f%%) "
                        "[db=$%.2f, position_notional=$%.2f]",
                        expected_usdt,
                        exchange_total,
                        diff_pct * 100,
                        db_balance,
                        position_notional,
                    )
                    # Correct DB balance to match actual total capital.
                    # DB balance represents total capital (USDT + position notional),
                    # not just free USDT on exchange.
                    corrected_balance = exchange_total + position_notional
                    self.db_manager.update_balance(
                        corrected_balance,
                        "reconciliation_balance_correction",
                        "system",
                        self.session_id,
                    )
                elif diff_pct > 0.01:  # >1% warning
                    result.severity = Severity.LOW
                    logger.info(
                        "Minor balance difference: expected=$%.2f vs Exchange=$%.2f (%.2f%%) "
                        "[db=$%.2f, position_notional=$%.2f]",
                        expected_usdt,
                        exchange_total,
                        diff_pct * 100,
                        db_balance,
                        position_notional,
                    )

        except Exception as e:
            logger.warning("Balance reconciliation failed: %s", e)

        return result

    def _realize_pnl_on_close(
        self,
        position: Any,
        exit_price: float | None,
        reason: str,
    ) -> None:
        """Update session balance with realized P&L after reconciliation close.

        Calculates P&L from position entry_price/quantity and exit_price, then
        adjusts the DB balance so capital and performance stats stay correct.
        Skips silently when exit_price is unavailable or position data is missing.

        Args:
            position: Position object with entry_price, quantity, and side attrs.
            exit_price: Fill price at which the position was closed.
            reason: Human-readable reason for the audit trail.
        """
        if exit_price is None or exit_price <= 0:
            return

        entry_price = getattr(position, "entry_price", 0)
        qty = getattr(position, "quantity", 0) or 0.0
        if entry_price <= 0 or qty <= 0:
            return

        # Scale quantity by current_size/original_size to account for partial
        # exits. Without this, closing after a 50% partial exit would calculate
        # P&L on the full original quantity, doubling the realized amount.
        current = getattr(position, "current_size", None)
        original = getattr(position, "original_size", None)
        if current is not None and original is not None and original > 0:
            qty = qty * (current / original)

        # Calculate realized P&L (long: sell higher = profit)
        side = getattr(position, "side", "long")
        side_is_short = side == PositionSide.SHORT or str(side).lower() == "short"
        if side_is_short:
            pnl = (entry_price - exit_price) * qty
        else:
            pnl = (exit_price - entry_price) * qty

        try:
            current_balance = self.db_manager.get_current_balance(self.session_id)
            if current_balance is None or current_balance < 0:
                logger.warning(
                    "Cannot realize P&L — unable to read current balance " "(session_id=%s)",
                    self.session_id,
                )
                return

            new_balance = current_balance + pnl
            self.db_manager.update_balance(
                new_balance,
                f"reconciliation_close: {reason}",
                "system",
                self.session_id,
            )

            # Audit the P&L correction
            audit = AuditEvent(
                entity_type="balance",
                entity_id=getattr(position, "db_position_id", None),
                field="realized_pnl",
                old_value=f"{current_balance:.2f}",
                new_value=f"{new_balance:.2f}",
                reason=(
                    f"Reconciliation P&L: {pnl:+.2f} "
                    f"(entry={entry_price:.2f}, exit={exit_price:.2f}, "
                    f"qty={qty:.8f}, {reason})"
                ),
                severity=Severity.MEDIUM,
            )
            self._persist_audit(audit)
            logger.info(
                "Realized reconciliation P&L %+.2f for %s (%s)",
                pnl,
                position.symbol,
                reason,
            )
        except Exception as e:
            logger.warning(
                "Failed to realize P&L for position %s: %s",
                getattr(position, "symbol", "unknown"),
                e,
            )

    def _persist_position_correction(
        self,
        position: _HasDbPositionId,
        entry_price: float | None = None,
        quantity: float | None = None,
        size: float | None = None,
        current_size: float | None = None,
        original_size: float | None = None,
    ) -> None:
        """Write corrected position fields back to the database.

        All corrections are applied in a single transaction to prevent
        partial writes (e.g. new size with old quantity) on commit failure.

        Raises:
            ValueError: If any financial value is non-positive or non-finite.
            Exception: Re-raised from DB commit failures.
        """
        db_pos_id = position.db_position_id
        if db_pos_id is None:
            return

        from decimal import Decimal

        from src.database.models import Position as DBPosition

        # Validate financial inputs before persisting
        for name, val in [
            ("entry_price", entry_price),
            ("quantity", quantity),
            ("size", size),
            ("current_size", current_size),
            ("original_size", original_size),
        ]:
            if val is not None and (not math.isfinite(val) or val <= 0):
                raise ValueError(f"{name} must be positive and finite, got {val}")

        corrections: dict[str, Decimal] = {}
        if entry_price is not None:
            corrections["entry_price"] = Decimal(str(entry_price))
        if quantity is not None:
            corrections["quantity"] = Decimal(str(quantity))
        if size is not None:
            corrections["size"] = Decimal(str(size))
        if current_size is not None:
            corrections["current_size"] = Decimal(str(current_size))
        if original_size is not None:
            corrections["original_size"] = Decimal(str(original_size))

        if not corrections:
            return

        try:
            with self.db_manager.get_session() as session:
                db_pos = session.query(DBPosition).filter_by(id=db_pos_id).first()
                if not db_pos:
                    raise RuntimeError(
                        f"Position {db_pos_id} not found in DB; cannot persist correction"
                    )
                for field_name, value in corrections.items():
                    setattr(db_pos, field_name, value)
                session.commit()

            logger.info(
                "Persisted position corrections to DB for position %d: %s",
                db_pos_id,
                list(corrections.keys()),
            )
        except Exception as e:
            logger.error("Failed to persist position corrections for %d: %s", db_pos_id, e)
            raise

    def _persist_audit(self, audit: AuditEvent) -> None:
        """Persist an audit event to the database."""
        try:
            self.db_manager.log_audit_event(
                session_id=self.session_id,
                entity_type=audit.entity_type,
                entity_id=audit.entity_id,
                field=audit.field,
                old_value=audit.old_value,
                new_value=audit.new_value,
                reason=audit.reason,
                severity=audit.severity.value,
            )
        except Exception as e:
            logger.error("Failed to persist audit event: %s", e)


# ---------- Periodic Reconciliation ----------


class PeriodicReconciler:
    """Background daemon that periodically verifies positions/orders/balance.

    Runs in a separate daemon thread. Configurable interval (default 60s).
    Uses per-position mutation locks to prevent double-close races.
    NOT instantiated in paper mode.
    """

    def __init__(
        self,
        exchange_interface: ExchangeInterface,
        position_tracker: LivePositionTracker,
        db_manager: DatabaseManager,
        session_id: int,
        interval: float = DEFAULT_RECONCILIATION_INTERVAL_SECONDS,
        on_critical: Any = None,
        use_margin: bool = False,
    ) -> None:
        """Initialize periodic reconciler.

        Args:
            exchange_interface: Exchange provider.
            position_tracker: Live position tracker.
            db_manager: Database manager.
            session_id: Current trading session ID.
            interval: Seconds between reconciliation cycles.
            on_critical: Callback invoked on CRITICAL severity (e.g., enter close-only mode).
            use_margin: Whether margin trading mode is active. When True,
                skip spot-specific checks (asset holdings) that don't apply
                to cross-margin accounts.
        """
        self.exchange = exchange_interface
        self.position_tracker = position_tracker
        self.db_manager = db_manager
        self.session_id = session_id
        self._use_margin = use_margin
        self.interval = interval
        self.on_critical = on_critical
        self._use_margin = use_margin

        self._running = False
        self._thread: threading.Thread | None = None
        # Per-position mutation locks to serialize reconciler + OrderTracker + exit
        self._position_mutation_locks: dict[str, threading.Lock] = {}
        self._locks_lock = threading.Lock()  # Protects _position_mutation_locks dict

    def start(self) -> None:
        """Start the periodic reconciliation daemon thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._run_loop,
            name="PeriodicReconciler",
            daemon=True,
        )
        self._thread.start()
        logger.info("Periodic reconciler started (interval=%ds)", self.interval)

    def stop(self) -> None:
        """Stop the periodic reconciliation daemon."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=10)
        logger.info("Periodic reconciler stopped")

    def get_position_lock(self, position_key: str) -> threading.Lock:
        """Get or create a per-position mutation lock.

        All mutators (OrderTracker callbacks, reconciler corrections, normal exits)
        should acquire this lock before mutating a position.
        """
        with self._locks_lock:
            if position_key not in self._position_mutation_locks:
                self._position_mutation_locks[position_key] = threading.Lock()
            return self._position_mutation_locks[position_key]

    def _run_loop(self) -> None:
        """Main reconciliation loop running in daemon thread."""
        while self._running:
            try:
                self._reconcile_cycle()
            except Exception as e:
                logger.error("Reconciliation cycle failed: %s", e, exc_info=True)

            # Sleep in small increments to allow quick shutdown
            for _ in range(int(self.interval)):
                if not self._running:
                    break
                time.sleep(1)

    def _reconcile_cycle(self) -> None:
        """Execute one reconciliation cycle."""
        # Snapshot positions (release lock before API calls)
        positions_snapshot = self.position_tracker.positions
        if not positions_snapshot:
            return

        max_severity = Severity.LOW

        # 1. Verify each position's entry order
        for order_key, position in positions_snapshot.items():
            exchange_order_id = getattr(position, "exchange_order_id", None)
            if not exchange_order_id:
                continue

            try:
                exchange_order = self.exchange.get_order(exchange_order_id, position.symbol)
                if not exchange_order:
                    continue

                from src.data_providers.exchange_interface import (
                    OrderStatus as ExOrderStatus,
                )

                # Check if position was closed externally
                if exchange_order.status in (
                    ExOrderStatus.CANCELLED,
                    ExOrderStatus.REJECTED,
                ):
                    severity = Severity.HIGH
                    self.db_manager.log_audit_event(
                        session_id=self.session_id,
                        entity_type="position",
                        entity_id=getattr(position, "db_position_id", None),
                        field="status",
                        old_value="OPEN",
                        new_value=exchange_order.status.value,
                        reason="Entry order cancelled/rejected on exchange",
                        severity=severity.value,
                    )

                    # Remove ghost position from tracker and close in DB.
                    # Thread safety: LivePositionTracker._positions_lock
                    # serializes all mutations, no per-position lock needed.
                    self.position_tracker.remove_position(order_key)
                    db_pos_id = getattr(position, "db_position_id", None)
                    if db_pos_id is not None:
                        self.db_manager.close_position(db_pos_id)
                    logger.warning(
                        "Removed ghost position %s — entry order %s on exchange",
                        order_key,
                        exchange_order.status.value,
                    )

                    if severity > max_severity:
                        max_severity = severity

            except Exception as e:
                logger.warning("Failed to verify position %s: %s", order_key, e)

        # 1b. Verify asset holdings for each position — detect external closes.
        # Skip in margin mode — spot balance checks don't apply to cross-margin
        # where asset balances include borrowed amounts and don't reflect positions.
        # SAFETY: External close detection uses _verify_margin_position_exists()
        # which checks borrowed balance for shorts and asset holdings for longs.
        # Additional protection: margin error codes (-3027, -3028, -3041, -3067)
        # are treated as definitive rejects, so stale SLs that fire after
        # external close are rejected by Binance, preventing naked positions.
        # NOTE: This per-position check compares held asset balance against the
        # individual position's quantity. For a single-symbol bot (at most 1
        # position per asset), this is correct. If multi-position-per-asset
        # support is added, the check must aggregate tracked quantities before
        # comparing against the exchange balance.
        if self._use_margin:
            # Margin mode: check borrowed balance for shorts, netAsset for longs
            for order_key, position in list(positions_snapshot.items()):
                try:
                    symbol = position.symbol
                    base_asset = PositionReconciler._extract_base_asset(symbol)
                    side = getattr(position, "side", None)
                    is_short = (
                        side == PositionSide.SHORT or str(side).lower() == "short"
                    )

                    balance = self.exchange.get_balance(base_asset)
                    position_gone = False

                    if is_short:
                        # Short gone if no negative netAsset (no borrowed debt)
                        if balance is None or balance.total >= 0:
                            position_gone = True
                    else:
                        # Long gone if no positive netAsset
                        if balance is None or balance.total <= 0:
                            position_gone = True

                    if position_gone:
                        logger.warning(
                            "Margin position %s appears externally closed — "
                            "removing from tracker",
                            symbol,
                        )
                        self.position_tracker.remove_position(order_key)
                        db_pos_id = getattr(position, "db_position_id", None)
                        if db_pos_id is not None:
                            try:
                                self.db_manager.close_position(db_pos_id)
                            except Exception as e:
                                logger.warning(
                                    "Failed to close DB position %s: %s", db_pos_id, e
                                )
                except Exception as e:
                    logger.warning(
                        "Margin position check failed for %s: %s",
                        getattr(position, "symbol", "?"),
                        e,
                    )
        else:
            for order_key, position in list(positions_snapshot.items()):
                # Skip short positions — shorts don't hold the base asset on spot
                side = getattr(position, "side", None)
                if side == PositionSide.SHORT or str(side).lower() == "short":
                    continue

                # Use quantity (actual asset amount), not size (balance fraction).
                # Scale by current_size/original_size to account for partial exits
                # that reduce current_size but leave quantity unchanged.
                qty = getattr(position, "quantity", None) or 0.0
                if qty <= 0:
                    continue

                current_size = getattr(position, "current_size", None)
                original_size = getattr(position, "original_size", None)
                if (
                    current_size is not None
                    and original_size is not None
                    and original_size > 0
                ):
                    position_qty = qty * (current_size / original_size)
                else:
                    position_qty = qty

                try:
                    base_asset = PositionReconciler._extract_base_asset(position.symbol)
                    balance = self.exchange.get_balance(base_asset)
                    if balance is None:
                        logger.warning(
                            "get_balance returned None for %s — skipping asset "
                            "check (transient API error)",
                            base_asset,
                        )
                        continue
                    held_qty = balance.total

                    if held_qty < position_qty * 0.5:
                        severity = Severity.HIGH
                        self.db_manager.log_audit_event(
                            session_id=self.session_id,
                            entity_type="position",
                            entity_id=getattr(position, "db_position_id", None),
                            field="status",
                            old_value="OPEN",
                            new_value="CLOSED_EXTERNALLY",
                            reason=(
                                f"Position asset not found on exchange — likely closed externally "
                                f"(held={held_qty:.8f}, tracked={position_qty:.8f}, "
                                f"asset={base_asset})"
                            ),
                            severity=severity.value,
                        )

                        # Remove ghost position from tracker and close in DB.
                        # Thread safety: LivePositionTracker._positions_lock
                        # serializes all mutations, no per-position lock needed.
                        self.position_tracker.remove_position(order_key)
                        db_pos_id = getattr(position, "db_position_id", None)
                        if db_pos_id is not None:
                            self.db_manager.close_position(db_pos_id)
                        logger.warning(
                            "Removed ghost position %s — externally closed "
                            "(held=%.8f, tracked=%.8f)",
                            order_key,
                            held_qty,
                            position_qty,
                        )

                        if severity > max_severity:
                            max_severity = severity
                except Exception as e:
                    logger.warning("Asset holdings check failed for %s: %s", order_key, e)

        # 2. Verify stop-loss orders are still active on exchange
        for order_key, position in list(positions_snapshot.items()):
            sl_order_id = getattr(position, "stop_loss_order_id", None)
            if not sl_order_id:
                # Position has no SL order (e.g. phantom from timeout) —
                # attempt to place one so it is protected at runtime.
                self._place_missing_stop_loss(position, order_key)
                if Severity.HIGH > max_severity:
                    max_severity = Severity.HIGH
                continue

            try:
                from src.data_providers.exchange_interface import (
                    OrderStatus as ExOrderStatus,
                )

                sl_order = self.exchange.get_order(sl_order_id, position.symbol)

                if sl_order and sl_order.status == ExOrderStatus.FILLED:
                    # SL triggered — remove position from tracker + close in DB
                    exit_price = float(sl_order.average_price) if sl_order.average_price else None
                    self.db_manager.log_audit_event(
                        session_id=self.session_id,
                        entity_type="position",
                        entity_id=getattr(position, "db_position_id", None),
                        field="status",
                        old_value="OPEN",
                        new_value="CLOSED_BY_SL",
                        reason=f"Stop-loss filled @ {exit_price} (periodic check)",
                        severity=Severity.HIGH.value,
                    )
                    self.position_tracker.remove_position(order_key)
                    db_pos_id = getattr(position, "db_position_id", None)
                    if db_pos_id is not None:
                        self.db_manager.close_position(db_pos_id, exit_price=exit_price)
                    logger.warning(
                        "Stop-loss filled for %s @ %s (periodic check) — " "removed position %s",
                        position.symbol,
                        exit_price,
                        order_key,
                    )
                    if Severity.HIGH > max_severity:
                        max_severity = Severity.HIGH

                elif (
                    sl_order and sl_order.status in (ExOrderStatus.CANCELLED, ExOrderStatus.EXPIRED)
                ) or sl_order is None:
                    # SL cancelled/expired/missing — attempt re-placement
                    status_desc = sl_order.status.value if sl_order else "MISSING"
                    logger.warning(
                        "Stop-loss %s for %s is %s — attempting re-placement",
                        sl_order_id,
                        position.symbol,
                        status_desc,
                    )
                    position.stop_loss_order_id = None

                    # Account for partial SL fills before re-placement.
                    # If the SL partially executed, reduce current_size
                    # so the replacement uses the correct held quantity.
                    if sl_order is not None:
                        partial_fill = getattr(sl_order, "filled_quantity", None) or 0.0
                        if partial_fill > 0:
                            pos_qty = getattr(position, "quantity", 0) or 0.0
                            current = getattr(position, "current_size", None)
                            original = getattr(position, "original_size", None)
                            if current is not None and original is not None and original > 0 and pos_qty > 0:
                                held = pos_qty * (current / original)
                                remaining = max(held - partial_fill, 0.0)
                                position.current_size = original * (remaining / max(pos_qty, 1e-9))
                            else:
                                if pos_qty > 0:
                                    position.quantity = max(pos_qty - partial_fill, 0.0)
                            logger.info(
                                "Periodic: partial SL fill %.8f for %s — adjusted position size",
                                partial_fill,
                                position.symbol,
                            )

                    stop_price = getattr(position, "stop_loss", None)
                    if stop_price and hasattr(self.exchange, "place_stop_loss_order"):
                        try:
                            from src.data_providers.exchange_interface import (
                                OrderSide,
                            )

                            side = getattr(position, "side", "long")
                            side_is_long = side == PositionSide.LONG or str(side).lower() == "long"
                            sl_side = OrderSide.SELL if side_is_long else OrderSide.BUY
                            # Compute held qty accounting for partial exits
                            qty = getattr(position, "quantity", 0) or 0.0
                            current = getattr(position, "current_size", None)
                            original = getattr(position, "original_size", None)
                            if current is not None and original is not None and original > 0:
                                qty = qty * (current / original)
                            # Position is flat — skip SL, remove from tracker
                            if qty <= 0:
                                logger.info(
                                    "Position %s is flat after partial SL fill "
                                    "— skipping SL re-placement (periodic)",
                                    position.symbol,
                                )
                                continue
                            new_sl_id = self.exchange.place_stop_loss_order(
                                symbol=position.symbol,
                                side=sl_side,
                                quantity=qty,
                                stop_price=stop_price,
                                side_effect_type="AUTO_REPAY",
                            )
                            if new_sl_id:
                                position.stop_loss_order_id = new_sl_id
                                logger.info(
                                    "Re-placed stop-loss for %s: %s @ %.2f " "(periodic check)",
                                    position.symbol,
                                    new_sl_id,
                                    stop_price,
                                )
                                db_pos_id = getattr(position, "db_position_id", None)
                                if db_pos_id is not None:
                                    try:
                                        _update_kw: dict[str, Any] = {
                                            "stop_loss_order_id": new_sl_id,
                                        }
                                        _cs = getattr(position, "current_size", None)
                                        if _cs is not None:
                                            _update_kw["current_size"] = _cs
                                        self.db_manager.update_position(
                                            position_id=db_pos_id,
                                            **_update_kw,
                                        )
                                    except Exception as e:
                                        logger.warning(
                                            "Failed to persist re-placed SL " "order ID for %s: %s",
                                            position.symbol,
                                            e,
                                        )
                            else:
                                logger.critical(
                                    "Failed to re-place stop-loss for %s — "
                                    "position is unprotected (periodic check)",
                                    position.symbol,
                                )
                                max_severity = Severity.CRITICAL
                        except Exception as e:
                            logger.critical(
                                "Exception re-placing stop-loss for %s: %s — "
                                "position is unprotected (periodic check)",
                                position.symbol,
                                e,
                            )
                            max_severity = Severity.CRITICAL
                    else:
                        logger.critical(
                            "Cannot re-place stop-loss for %s — no stop_price "
                            "or exchange does not support SL orders",
                            position.symbol,
                        )
                        max_severity = Severity.CRITICAL

            except Exception as e:
                logger.warning(
                    "Failed to verify stop-loss %s for %s: %s",
                    sl_order_id,
                    position.symbol,
                    e,
                )

        # 3. Check for orphaned orders with our prefix.
        # Rebuild tracked IDs from a fresh position snapshot so that any
        # positions closed during steps 1-2 are excluded, preventing
        # cancellation of valid stop-loss orders on newly opened positions.
        try:
            fresh_snapshot = self.position_tracker.positions
            if fresh_snapshot:
                tracked_exchange_ids = set()
                for pos in fresh_snapshot.values():
                    eid = getattr(pos, "exchange_order_id", None)
                    if eid:
                        tracked_exchange_ids.add(eid)
                    sl_id = getattr(pos, "stop_loss_order_id", None)
                    if sl_id:
                        tracked_exchange_ids.add(sl_id)

                # Query open orders for every symbol with active positions
                symbols = set(pos.symbol for pos in fresh_snapshot.values())
                for symbol in symbols:
                    open_orders = self.exchange.get_open_orders(symbol)
                    for order in open_orders:
                        if order.order_id not in tracked_exchange_ids:
                            client_id = getattr(order, "client_order_id", "") or ""
                            if client_id.startswith("atb"):  # Catches atb_ (entry) and atbx_ (exit)
                                logger.warning(
                                    "Orphaned order found: %s (%s) on %s — cancelling",
                                    order.order_id,
                                    client_id,
                                    symbol,
                                )
                                try:
                                    self.exchange.cancel_order(order.order_id, symbol)
                                    logger.info(
                                        "Cancelled orphaned order %s on %s",
                                        order.order_id,
                                        symbol,
                                    )
                                except Exception as cancel_err:
                                    logger.warning(
                                        "Failed to cancel orphaned order %s: %s",
                                        order.order_id,
                                        cancel_err,
                                    )
                                max_severity = Severity.HIGH
        except Exception as e:
            logger.warning("Orphaned order check failed: %s", e)

        # 4. Verify balance (accounting for position notional values)
        # Skip in margin mode — spot balance doesn't reflect margin account state.
        if not self._use_margin:
            try:
                usdt_balance = self.exchange.get_balance("USDT")
                if usdt_balance:
                    db_balance = self.db_manager.get_current_balance(self.session_id)
                    if db_balance > 0:
                        # Subtract position notional to get expected USDT
                        position_notional = 0.0
                        for position in positions_snapshot.values():
                            # Use quantity (actual asset amount), not size (balance fraction)
                            qty = getattr(position, "quantity", None) or 0.0
                            price = getattr(position, "entry_price", 0)
                            if qty > 0 and price > 0:
                                # Scale by current_size/original_size for partial exits
                                current = getattr(position, "current_size", None)
                                original = getattr(position, "original_size", None)
                                if current is not None and original is not None and original > 0:
                                    qty = qty * (current / original)
                                position_notional += qty * price
                        expected_usdt = db_balance - position_notional
                        # Use abs(expected_usdt) to avoid false CRITICAL when
                        # expected_usdt is negative (e.g. position notional >
                        # db_balance due to price appreciation)
                        comparison_base = max(abs(expected_usdt), db_balance * 0.01)
                        diff_pct = abs(usdt_balance.total - expected_usdt) / comparison_base
                        if diff_pct > DEFAULT_RECONCILIATION_BALANCE_THRESHOLD_PCT:
                            max_severity = Severity.CRITICAL
                            logger.critical(
                                "Balance discrepancy: expected=$%.2f vs Exchange=$%.2f (%.2f%%) "
                                "[db=$%.2f, position_notional=$%.2f]",
                                expected_usdt,
                                usdt_balance.total,
                                diff_pct * 100,
                                db_balance,
                                position_notional,
                            )
                            # Correct DB balance to match actual total capital.
                            # DB balance represents total capital (USDT + position
                            # notional), not just free USDT on exchange.
                            corrected_balance = usdt_balance.total + position_notional
                            self.db_manager.update_balance(
                                corrected_balance,
                                "reconciliation_balance_correction",
                                "system",
                                self.session_id,
                            )
            except Exception as e:
                logger.warning("Balance check failed: %s", e)
        else:
            logger.debug("Skipping balance verification in margin mode")

        # 5. Trigger close-only mode on CRITICAL
        if max_severity == Severity.CRITICAL and self.on_critical:
            try:
                self.on_critical()
            except Exception as e:
                logger.error("on_critical callback failed: %s", e)

    def _place_missing_stop_loss(self, position: Any, order_key: str) -> None:
        """Place a stop-loss for a position that has none (e.g. phantom from timeout).

        Computes a default stop price from the position's stop_loss attribute
        or falls back to DEFAULT_STOP_LOSS_PCT from entry_price.
        """
        stop_price = getattr(position, "stop_loss", None)
        entry_price = getattr(position, "entry_price", None)
        side = getattr(position, "side", "long")
        side_is_long = side == PositionSide.LONG or str(side).lower() == "long"

        # Compute default stop price if none stored on the position
        if not stop_price and entry_price and entry_price > 0:
            if side_is_long:
                stop_price = entry_price * (1.0 - DEFAULT_STOP_LOSS_PCT)
            else:
                stop_price = entry_price * (1.0 + DEFAULT_STOP_LOSS_PCT)
            position.stop_loss = stop_price
            logger.warning(
                "Position %s has no stop_loss; computed default %.4f "
                "(entry=%.4f, side=%s, pct=%.2f%%)",
                order_key,
                stop_price,
                entry_price,
                side,
                DEFAULT_STOP_LOSS_PCT * 100,
            )

        if not stop_price or not hasattr(self.exchange, "place_stop_loss_order"):
            logger.critical(
                "Cannot place missing SL for %s — no stop_price or exchange "
                "does not support SL orders",
                order_key,
            )
            return

        try:
            from src.data_providers.exchange_interface import OrderSide

            sl_side = OrderSide.SELL if side_is_long else OrderSide.BUY

            # Compute held qty accounting for partial exits
            qty = getattr(position, "quantity", 0) or 0.0
            current = getattr(position, "current_size", None)
            original = getattr(position, "original_size", None)
            if current is not None and original is not None and original > 0:
                qty = qty * (current / original)

            new_sl_id = self.exchange.place_stop_loss_order(
                symbol=position.symbol,
                side=sl_side,
                quantity=qty,
                stop_price=stop_price,
                side_effect_type="AUTO_REPAY",
            )
            if new_sl_id:
                position.stop_loss_order_id = new_sl_id
                logger.info(
                    "Placed missing stop-loss for %s: %s @ %.2f (periodic check)",
                    position.symbol,
                    new_sl_id,
                    stop_price,
                )
                db_pos_id = getattr(position, "db_position_id", None)
                if db_pos_id is not None:
                    try:
                        self.db_manager.update_position(
                            position_id=db_pos_id,
                            stop_loss_order_id=new_sl_id,
                            stop_loss=stop_price,
                        )
                    except Exception as e:
                        logger.warning(
                            "Failed to persist SL order ID for %s: %s",
                            position.symbol,
                            e,
                        )
            else:
                logger.critical(
                    "Failed to place missing stop-loss for %s — position "
                    "is unprotected (periodic check)",
                    position.symbol,
                )
        except Exception as e:
            logger.critical(
                "Exception placing missing stop-loss for %s: %s — position "
                "is unprotected (periodic check)",
                position.symbol,
                e,
            )


# ---------- Discrepancy Handling ----------


def classify_severity(
    entity_type: str,
    issue: str,
    value_diff_pct: float | None = None,
) -> Severity:
    """Classify a discrepancy into a severity level.

    Severity taxonomy:
    - CRITICAL: Position in tracker but order never filled; balance >5%; unknown orders
    - HIGH: Position closed externally; SL filled but missed; quantity mismatch
    - MEDIUM: Entry price slippage unrecorded; missing SL order
    - LOW: Minor balance rounding (<1%); timing discrepancy
    """
    if entity_type == "balance" and value_diff_pct is not None:
        if value_diff_pct > DEFAULT_RECONCILIATION_BALANCE_THRESHOLD_PCT:
            return Severity.CRITICAL
        return Severity.LOW

    if entity_type == "order":
        if issue in ("not_found_submitted", "not_found_unknown"):
            return Severity.CRITICAL
        if issue == "not_found_pending_submit":
            return Severity.LOW
        if issue == "filled":
            return Severity.MEDIUM

    if entity_type == "position":
        if issue in ("entry_cancelled", "entry_rejected", "sl_filled_offline"):
            return Severity.HIGH
        if issue in ("price_mismatch", "sl_missing"):
            return Severity.MEDIUM
        if issue == "legacy_no_ids":
            return Severity.LOW

    return Severity.LOW
