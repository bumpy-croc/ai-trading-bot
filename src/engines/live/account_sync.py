"""
Account Synchronization Service

This module provides robust synchronization between the exchange and the bot's database,
ensuring data integrity and handling scenarios where the bot loses track of positions
or trades due to shutdowns or errors.
"""

import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from src.config.constants import (
    DEFAULT_ACCOUNT_SYNC_MIN_INTERVAL_MINUTES,
    DEFAULT_BALANCE_DISCREPANCY_THRESHOLD_PCT,
    DEFAULT_POSITION_SIZE_COMPARISON_TOLERANCE,
    DEFAULT_RECONCILIATION_BALANCE_THRESHOLD_PCT,
)
from src.data_providers.exchange_interface import (
    AccountBalance,
    ExchangeInterface,
    Order,
    Position,
)
from src.data_providers.exchange_interface import OrderStatus as ExchangeOrderStatus
from src.database.manager import DatabaseManager
from src.database.models import EventType, PositionSide, TradeSource
from src.engines.live.reconciliation import Severity

logger = logging.getLogger(__name__)


@dataclass
class SyncResult:
    """Result of account synchronization"""

    success: bool
    message: str
    data: dict[str, Any]
    timestamp: datetime


class AccountSynchronizer:
    """
    Account synchronization service that ensures data integrity between
    the exchange and the bot's database.
    """

    def __init__(
        self,
        exchange: ExchangeInterface,
        db_manager: DatabaseManager,
        session_id: int | None = None,
        use_margin: bool = False,
    ):
        """
        Initialize the account synchronizer.

        Args:
            exchange: Exchange interface for API calls
            db_manager: Database manager for local data
            session_id: Current trading session ID
            use_margin: Whether margin trading mode is active. When True,
                balance/position sync is skipped since margin account balances
                include borrowed amounts and don't reflect true equity.
        """
        self.exchange = exchange
        self.db_manager = db_manager
        self.session_id = session_id
        self._use_margin = use_margin
        self.last_sync_time: datetime | None = None

    def sync_account_data(self, force: bool = False, symbol: str | None = None) -> SyncResult:
        """
        Synchronize all account data from the exchange.

        Args:
            force: Force sync even if recently synced
            symbol: If provided, only sync positions/orders for this symbol
                    to conserve API weight

        Returns:
            SyncResult with synchronization status and data
        """
        try:
            logger.info("Starting account data synchronization...")

            # Check if we should sync (avoid too frequent syncs)
            if not force and self.last_sync_time:
                time_since_last_sync = datetime.now(UTC) - self.last_sync_time
                if time_since_last_sync < timedelta(
                    minutes=DEFAULT_ACCOUNT_SYNC_MIN_INTERVAL_MINUTES
                ):
                    logger.info("Skipping sync - too recent")
                    return SyncResult(
                        success=True,
                        message="Sync skipped - too recent",
                        data={},
                        timestamp=datetime.now(UTC),
                    )

            # Get data from exchange (filter by symbol to save API weight)
            exchange_data = self.exchange.sync_account_data(symbol=symbol)

            if not exchange_data.get("sync_successful", False):
                error_msg = exchange_data.get("error", "Unknown error")
                logger.error("Exchange sync failed: %s", error_msg)
                return SyncResult(
                    success=False,
                    message=f"Exchange sync failed: {error_msg}",
                    data=exchange_data,
                    timestamp=datetime.now(UTC),
                )

            # Skip balance and position sync in margin mode. USDT netAsset
            # doesn't reflect true equity when shorts are open — sale proceeds
            # inflate USDT while borrowed ETH liability is on a separate asset
            # row. Syncing USDT alone would oversize the next trade.
            # Proper margin equity requires account-level totalNetAssetOfBtc,
            # which is a future enhancement. Internal tracking is authoritative.
            # Margin interest tracking is implemented in MarginInterestTracker
            # (src/engines/live/margin_interest_tracker.py). Interest is deducted
            # from realized PnL on position close and logged during reconciliation.
            # Balance/position sync remains skipped in margin mode because USDT
            # netAsset doesn't reflect true equity when shorts are open.
            if not self._use_margin:
                balance_sync_result = self._sync_balances(exchange_data.get("balances", []))
                position_sync_result = self._sync_positions(exchange_data.get("positions", []))
            else:
                # Margin: reconcile the tracked balance against true net equity
                # (assets minus liabilities), not USDT alone. Position sync stays
                # skipped; only the balance safety-net is added here.
                balance_sync_result = self._sync_margin_equity()
                position_sync_result = {"synced": False, "reason": "skipped in margin mode"}

            # Sync orders
            order_sync_result = self._sync_orders(exchange_data.get("open_orders", []))

            # Update last sync time
            self.last_sync_time = datetime.now(UTC)

            sync_data = {
                "exchange_data": exchange_data,
                "balance_sync": balance_sync_result,
                "position_sync": position_sync_result,
                "order_sync": order_sync_result,
                "sync_timestamp": self.last_sync_time.isoformat(),
            }

            logger.info("Account synchronization completed successfully")

            return SyncResult(
                success=True,
                message="Account synchronization completed",
                data=sync_data,
                timestamp=self.last_sync_time,
            )

        except Exception as e:
            logger.error("Account synchronization failed: %s", e)
            return SyncResult(
                success=False,
                message=f"Sync failed: {str(e)}",
                data={},
                timestamp=datetime.now(UTC),
            )

    def _sync_margin_equity(self) -> dict[str, Any]:
        """Reconcile the tracked balance against true cross-margin equity.

        Margin USDT alone overstates equity (sale proceeds inflate USDT while a
        borrowed-asset liability sits on a separate asset row), so the bot
        historically skipped balance sync in margin mode. That let realized
        losses which were never booked to the tracked balance accumulate
        silently, and the bot then over-sized on a phantom balance. Here we use
        the exchange's account-level net equity (``get_account_equity``).

        Only *corrected* while FLAT — and flatness is checked against the
        EXCHANGE, not the DB: net equity equals USDT cash only when no base-asset
        position (long) or borrowed liability (short) is held, so comparing
        equity to the live USDT balance detects an open — or not-yet-reconciled —
        position regardless of DB state. This prevents a phantom/unpersisted
        position's market value from leaking into cash (which would over-size).
        A material divergence is always logged.
        """
        try:
            equity = self.exchange.get_account_equity()
        except Exception as e:
            logger.warning("Margin equity read failed: %s", e)
            return {"synced": False, "reason": f"equity unavailable: {e}"}
        if equity is None or equity <= 0:
            return {"synced": False, "reason": "equity unavailable"}

        # Resolve the session id exactly as update_balance does (it falls back to the
        # DB manager's _current_session_id). During the INITIAL startup sync this
        # synchronizer's own session_id is still None — trading_engine assigns it only
        # AFTER sync_account_data() returns — yet the session already exists on the DB
        # manager. Without this fallback a real startup correction would persist (via
        # update_balance's own fallback) while the audit/event were silently skipped,
        # re-opening the exact gap this path exists to close.
        effective_session_id = (
            self.session_id
            if self.session_id is not None
            else getattr(self.db_manager, "_current_session_id", None)
        )

        current_db_balance = self.db_manager.get_current_balance(effective_session_id)
        diff_pct = (
            abs(equity - current_db_balance) / current_db_balance * 100
            if current_db_balance > 0
            else 0.0
        )

        # Flat iff net equity matches USDT cash (no position value or liability).
        usdt_bal = self.exchange.get_balance("USDT")
        usdt_total = (
            float(usdt_bal.total) if usdt_bal is not None and usdt_bal.total is not None else None
        )
        flat_tolerance = max(1.0, equity * 0.01)
        if usdt_total is None or abs(equity - usdt_total) > flat_tolerance:
            # A position is held (equity diverges from USDT cash) — correcting
            # would fold position value into cash. Surface divergence, defer.
            if diff_pct > DEFAULT_BALANCE_DISCREPANCY_THRESHOLD_PCT:
                logger.warning(
                    "Margin equity divergence: tracked $%.2f vs true equity $%.2f "
                    "(%.2f%%), but a position is held (equity $%.2f vs USDT $%s) — "
                    "deferring correction until flat",
                    current_db_balance,
                    equity,
                    diff_pct,
                    equity,
                    usdt_total,
                )
            return {"synced": False, "reason": "position held"}

        if diff_pct <= DEFAULT_BALANCE_DISCREPANCY_THRESHOLD_PCT:
            return {"synced": True, "corrected": False, "balance": current_db_balance}

        logger.warning(
            "Margin equity reconcile: tracked balance $%.2f vs true equity $%.2f "
            "(%.2f%%) — account is flat (all-USDT), correcting to true equity",
            current_db_balance,
            equity,
            diff_pct,
        )
        balance_updated = self.db_manager.update_balance(
            equity, "margin_equity_sync_correction", "system", effective_session_id
        )
        if not balance_updated:
            # update_balance swallows its own errors and returns False; don't emit
            # an audit trail or alert claiming a correction that never persisted.
            logger.error(
                "Margin equity correction did NOT persist (update_balance returned "
                "False): tracked $%.2f vs true equity $%.2f — skipping audit/event",
                current_db_balance,
                equity,
            )
            return {
                "synced": False,
                "corrected": False,
                "reason": "balance update failed",
                "old_balance": current_db_balance,
                "new_balance": equity,
            }

        # This book-down is the largest capital event a margin session can produce
        # and was historically invisible to auditing/alerting (the prod gap behind
        # this fix). Record the immutable audit trail + an operator alert; the 1%
        # correction gate above guarantees the move is already material. Pass the
        # resolved session id so the audit binds to the same session the balance
        # write used (critical for the startup sync where self.session_id is None).
        self._record_equity_correction_audit(
            current_db_balance, equity, diff_pct, effective_session_id
        )

        return {
            "synced": True,
            "corrected": True,
            "old_balance": current_db_balance,
            "new_balance": equity,
        }

    def _record_equity_correction_audit(
        self,
        old_balance: float,
        new_equity: float,
        diff_pct: float,
        session_id: int | None,
    ) -> None:
        """Persist the audit trail and operator alert for a margin-equity book-down.

        A ``margin_equity_sync_correction`` can be the single largest capital event
        of a session, so every correction MUST leave a ``reconciliation_audit_events``
        row (before/after values) and — because the 1% correction gate guarantees the
        move is already material — a warning+ ``system_events`` row so monitoring and
        alerting can see large book-downs.

        Both writes are best-effort and independently guarded: per CODE.md ("wrap
        callback invocations in try/except so failures don't block state updates"),
        an observability failure must neither raise into the sync loop nor unwind the
        already-persisted balance correction.

        Args:
            old_balance: Tracked balance before the correction.
            new_equity: True cross-margin equity the balance was corrected to.
            diff_pct: Absolute divergence as a percentage of the old balance.
            session_id: The session the balance write was bound to (the caller's
                resolved effective id, which falls back to the DB manager's current
                session). Must be the same id update_balance used.
        """
        if session_id is None:
            # Defensive only: the caller resolves session_id the same way
            # update_balance does, so a persisted correction always has a non-None
            # id here. Kept so a degenerate call logs rather than crashes.
            logger.error(
                "Cannot audit margin-equity correction (tracked $%.2f -> $%.2f): "
                "no active session_id",
                old_balance,
                new_equity,
            )
            return

        delta = new_equity - old_balance
        # ``diff_pct`` is a percentage (0-100); the shared CRITICAL threshold
        # constant is a fraction (0.05 == 5%), so scale it to percent to compare.
        is_critical = diff_pct >= DEFAULT_RECONCILIATION_BALANCE_THRESHOLD_PCT * 100
        # Reuse the reconciler's severity vocabulary so this audit row matches every
        # other log_audit_event call site. SystemEvent uses its own lowercase scale.
        audit_severity = (Severity.CRITICAL if is_critical else Severity.HIGH).value
        event_severity = "critical" if is_critical else "warning"
        reason = (
            f"Margin equity sync correction: tracked balance ${old_balance:.2f} vs "
            f"true cross-margin equity ${new_equity:.2f} "
            f"(Δ ${delta:+.2f}, {diff_pct:.2f}%) while flat"
        )

        try:
            self.db_manager.log_audit_event(
                session_id=session_id,
                entity_type="balance",
                entity_id=None,
                field="total_balance",
                old_value=f"{old_balance:.8f}",
                new_value=f"{new_equity:.8f}",
                reason=reason,
                severity=audit_severity,
            )
        except Exception as e:
            # Best-effort: degraded observability must not break the sync loop.
            logger.error(
                "Failed to record margin-equity reconciliation audit event: %s",
                e,
                exc_info=True,
            )

        try:
            self.db_manager.log_event(
                event_type=EventType.BALANCE_ADJUSTMENT,
                message=reason,
                severity=event_severity,
                component="account_sync.margin_equity",
                details={
                    "old_balance": old_balance,
                    "new_balance": new_equity,
                    "delta": delta,
                    "diff_pct": diff_pct,
                    "update_reason": "margin_equity_sync_correction",
                },
                session_id=session_id,
            )
        except Exception as e:
            # Best-effort: degraded observability must not break the sync loop.
            logger.error(
                "Failed to record margin-equity reconciliation system event: %s",
                e,
                exc_info=True,
            )

    def _sync_balances(self, exchange_balances: list[AccountBalance]) -> dict[str, Any]:
        """Synchronize account balances"""
        try:
            logger.info("Syncing %d balances from exchange", len(exchange_balances))

            # Get current balance from database
            current_db_balance = self.db_manager.get_current_balance(self.session_id)

            # Find USDT balance (our primary currency)
            usdt_balance = None
            for balance in exchange_balances:
                # Validate balance object before accessing attributes
                if balance is None:
                    logger.warning("Skipping None balance object from exchange")
                    continue
                if not hasattr(balance, "asset") or not hasattr(balance, "total"):
                    logger.warning("Skipping malformed balance object: %s", balance)
                    continue
                if balance.asset == "USDT":
                    usdt_balance = balance
                    break

            if usdt_balance:
                # Validate total is numeric
                if usdt_balance.total is None or not isinstance(usdt_balance.total, (int, float)):
                    logger.error(
                        "Invalid USDT balance total: %s (type=%s) - skipping sync",
                        usdt_balance.total,
                        type(usdt_balance.total).__name__,
                    )
                    return SyncResult(
                        success=False,
                        message=f"Invalid balance data from exchange: total={usdt_balance.total}",
                    )
                exchange_balance = float(usdt_balance.total)

                # Check for significant discrepancy
                balance_diff = abs(exchange_balance - current_db_balance)
                balance_diff_pct = (
                    (balance_diff / current_db_balance * 100) if current_db_balance > 0 else 0
                )

                if balance_diff_pct > DEFAULT_BALANCE_DISCREPANCY_THRESHOLD_PCT:
                    logger.warning(
                        f"Balance discrepancy detected: DB=${current_db_balance:.2f} vs Exchange=${exchange_balance:.2f} (diff: {balance_diff_pct:.2f}%)"
                    )

                    # Update database with exchange balance
                    self.db_manager.update_balance(
                        exchange_balance, "exchange_sync_correction", "system", self.session_id
                    )

                    return {
                        "synced": True,
                        "corrected": True,
                        "old_balance": current_db_balance,
                        "new_balance": exchange_balance,
                        "difference": balance_diff,
                        "difference_percent": balance_diff_pct,
                    }
                else:
                    logger.info(
                        f"Balance in sync: DB=${current_db_balance:.2f} vs Exchange=${exchange_balance:.2f}"
                    )
                    return {"synced": True, "corrected": False, "balance": exchange_balance}
            else:
                logger.warning("No USDT balance found in exchange data")
                return {"synced": False, "error": "No USDT balance found"}

        except Exception as e:
            logger.error("Balance sync failed: %s", e)
            return {"synced": False, "error": str(e)}

    def _sync_positions(self, exchange_positions: list[Position]) -> dict[str, Any]:
        """Synchronize open positions"""
        try:
            logger.info("Syncing %d positions from exchange", len(exchange_positions))

            # Get current positions from database
            db_positions = self.db_manager.get_active_positions(self.session_id)

            synced_positions = []
            new_positions = []
            closed_positions = []

            # Check for positions that exist in exchange but not in database
            for exchange_pos in exchange_positions:
                # Find matching position in database
                db_pos = None
                for pos in db_positions:
                    if pos["symbol"] == exchange_pos.symbol and pos["side"] == exchange_pos.side:
                        db_pos = pos
                        break

                if db_pos:
                    # Position exists in both - check for updates
                    # Validate sizes are numeric before comparison to prevent TypeError
                    exchange_size = exchange_pos.size
                    db_size = db_pos["size"]

                    if not isinstance(exchange_size, (int, float)) or not isinstance(
                        db_size, (int, float)
                    ):
                        logger.warning(
                            "Skipping position sync with non-numeric size: "
                            "exchange_size=%s (type=%s), db_size=%s (type=%s)",
                            exchange_size,
                            type(exchange_size).__name__,
                            db_size,
                            type(db_size).__name__,
                        )
                    elif abs(exchange_size - db_size) > DEFAULT_POSITION_SIZE_COMPARISON_TOLERANCE:
                        logger.info(
                            f"Position size updated: {exchange_pos.symbol} {exchange_pos.side} - {db_size} -> {exchange_size}"
                        )
                        # Update position in database
                        self.db_manager.update_position(
                            db_pos["id"],
                            size=exchange_pos.size,
                            current_price=exchange_pos.current_price,
                            unrealized_pnl=exchange_pos.unrealized_pnl,
                        )

                    synced_positions.append(
                        {
                            "symbol": exchange_pos.symbol,
                            "side": exchange_pos.side,
                            "size": exchange_pos.size,
                        }
                    )
                else:
                    # New position found on exchange
                    logger.warning(
                        f"New position found on exchange: {exchange_pos.symbol} {exchange_pos.side} {exchange_pos.size}"
                    )

                    # Add to database
                    position_id = self.db_manager.log_position(
                        symbol=exchange_pos.symbol,
                        side=(
                            PositionSide.LONG if exchange_pos.side == "long" else PositionSide.SHORT
                        ),
                        entry_price=exchange_pos.entry_price,
                        size=exchange_pos.size,
                        strategy_name="exchange_sync",
                        entry_order_id=exchange_pos.order_id
                        or f"sync_{int(datetime.now(UTC).timestamp())}",
                        session_id=self.session_id,
                    )

                    new_positions.append(
                        {
                            "symbol": exchange_pos.symbol,
                            "side": exchange_pos.side,
                            "size": exchange_pos.size,
                            "db_id": position_id,
                        }
                    )

            # Check for positions that exist in database but not in exchange
            for db_pos in db_positions:
                exchange_pos = None
                for pos in exchange_positions:
                    if pos.symbol == db_pos["symbol"] and pos.side == db_pos["side"]:
                        exchange_pos = pos
                        break

                if not exchange_pos:
                    # Position exists in database but not on exchange
                    logger.warning(
                        f"Position closed on exchange: {db_pos['symbol']} {db_pos['side']}"
                    )

                    # Close position in database
                    self.db_manager.close_position(db_pos["id"])

                    closed_positions.append(
                        {"symbol": db_pos["symbol"], "side": db_pos["side"], "size": db_pos["size"]}
                    )

            return {
                "synced": True,
                "total_exchange_positions": len(exchange_positions),
                "total_db_positions": len(db_positions),
                "synced_positions": len(synced_positions),
                "new_positions": len(new_positions),
                "closed_positions": len(closed_positions),
                "details": {
                    "synced": synced_positions,
                    "new": new_positions,
                    "closed": closed_positions,
                },
            }

        except Exception as e:
            logger.error("Position sync failed: %s", e)
            return {"synced": False, "error": str(e)}

    def _sync_orders(self, exchange_orders: list[Order]) -> dict[str, Any]:
        """Synchronize open orders"""
        try:
            logger.info("Syncing %d orders from exchange", len(exchange_orders))

            # Get current open orders from database (using new Order table)
            db_orders = self.db_manager.get_pending_orders_new(self.session_id)

            synced_orders = []
            new_orders = []
            cancelled_orders = []

            # Check for orders that exist in exchange but not in database
            for exchange_order in exchange_orders:
                # Find matching order in database
                db_order = None
                for order in db_orders:
                    order_id = order["exchange_order_id"] or order["internal_order_id"]
                    if order_id == exchange_order.order_id:
                        db_order = order
                        break

                if db_order:
                    # Order exists in both - check for updates
                    if exchange_order.status != ExchangeOrderStatus.PENDING:
                        logger.info(
                            f"Order status changed: {exchange_order.order_id} - {exchange_order.status.value}"
                        )

                        # Update order status using new methods
                        if exchange_order.status == ExchangeOrderStatus.FILLED:
                            self.db_manager.update_order_status_new(
                                order_id=db_order["id"],
                                status="FILLED",
                                filled_quantity=getattr(exchange_order, "filled_quantity", None),
                                filled_price=getattr(exchange_order, "average_price", None),
                                exchange_order_id=exchange_order.order_id,
                            )
                        else:
                            self.db_manager.update_order_status_new(
                                order_id=db_order["id"],
                                status=exchange_order.status.value.upper(),
                                exchange_order_id=exchange_order.order_id,
                            )

                    synced_orders.append(
                        {
                            "order_id": exchange_order.order_id,
                            "symbol": exchange_order.symbol,
                            "status": exchange_order.status.value,
                        }
                    )
                else:
                    # New order found on exchange
                    logger.warning(
                        f"New order found on exchange: {exchange_order.order_id} {exchange_order.symbol}"
                    )

                    # Persist new order to the database
                    # For new orders from exchange, we need to find/create the position
                    # This is simplified - in a real implementation we'd need more logic
                    # For now, we'll skip creating new orders from sync
                    logger.info(
                        f"Skipping creation of new order {exchange_order.order_id} from sync"
                    )

                    # Add to the new_orders list for reporting
                    new_orders.append(
                        {
                            "order_id": exchange_order.order_id,
                            "symbol": exchange_order.symbol,
                            "side": exchange_order.side.value,
                            "quantity": exchange_order.quantity,
                            "price": exchange_order.price,
                        }
                    )

            # Check for orders that exist in database but not in exchange
            for db_order in db_orders:
                exchange_order = None
                order_id = db_order["exchange_order_id"] or db_order["internal_order_id"]
                for order in exchange_orders:
                    if order.order_id == order_id:
                        exchange_order = order
                        break

                if not exchange_order:
                    # Order exists in database but not on exchange
                    logger.warning("Order cancelled on exchange: %s", order_id)

                    # Mark as cancelled using new methods
                    self.db_manager.update_order_status_new(
                        order_id=db_order["id"], status="CANCELLED"
                    )

                    cancelled_orders.append({"order_id": order_id, "symbol": db_order["symbol"]})

            return {
                "synced": True,
                "total_exchange_orders": len(exchange_orders),
                "total_db_orders": len(db_orders),
                "synced_orders": len(synced_orders),
                "new_orders": len(new_orders),
                "cancelled_orders": len(cancelled_orders),
                "details": {
                    "synced": synced_orders,
                    "new": new_orders,
                    "cancelled": cancelled_orders,
                },
            }

        except Exception as e:
            logger.error("Order sync failed: %s", e)
            return {"synced": False, "error": str(e)}

    def recover_missing_trades(self, symbol: str, days_back: int = 7) -> dict[str, Any]:
        """
        Recover missing trades by comparing exchange trade history with database.

        Args:
            symbol: Trading symbol to check
            days_back: Number of days to look back

        Returns:
            Dictionary with recovery results
        """
        try:
            logger.info("Recovering missing trades for %s (last %d days)", symbol, days_back)

            # Get recent trades from exchange
            exchange_trades = self.exchange.get_recent_trades(symbol, limit=1000)

            # Filter by date
            cutoff_date = datetime.now(UTC) - timedelta(days=days_back)
            recent_exchange_trades = [
                trade for trade in exchange_trades if trade.time >= cutoff_date
            ]

            # Get trades from database for the same period
            db_trades = self.db_manager.get_trades_by_symbol_and_date(
                symbol, cutoff_date, self.session_id
            )

            # Find missing trades
            missing_trades = []
            db_trade_ids = {trade["trade_id"] for trade in db_trades if trade.get("trade_id")}

            for trade in recent_exchange_trades:
                if trade.trade_id not in db_trade_ids:
                    missing_trades.append(trade)

            if missing_trades:
                logger.warning("Found %d missing trades", len(missing_trades))

                # Add missing trades to database
                recovered_trades = []
                for trade in missing_trades:
                    try:
                        _trade_id = self.db_manager.log_trade(
                            symbol=trade.symbol,
                            side=(
                                trade.side.value
                                if hasattr(trade.side, "value")
                                else str(trade.side)
                            ),
                            entry_price=trade.price,  # Simplified - we don't have entry/exit prices
                            exit_price=trade.price,
                            size=trade.quantity,
                            entry_time=trade.time,  # Simplified - using same time for entry/exit
                            exit_time=trade.time,
                            pnl=0.0,  # Cannot calculate without entry price
                            exit_reason="recovered_from_exchange",
                            strategy_name="exchange_recovery",
                            source=TradeSource.LIVE,
                            order_id=trade.order_id,
                            session_id=self.session_id,
                        )

                        recovered_trades.append(
                            {
                                "trade_id": trade.trade_id,
                                "symbol": trade.symbol,
                                "side": trade.side.value,
                                "quantity": trade.quantity,
                                "price": trade.price,
                                "time": trade.time.isoformat(),
                            }
                        )

                    except Exception as e:
                        logger.error("Failed to recover trade %s: %s", trade.trade_id, e)

                return {
                    "recovered": True,
                    "total_exchange_trades": len(recent_exchange_trades),
                    "total_db_trades": len(db_trades),
                    "missing_trades": len(missing_trades),
                    "recovered_trades": len(recovered_trades),
                    "details": recovered_trades,
                }
            else:
                logger.info("No missing trades found")
                return {
                    "recovered": True,
                    "total_exchange_trades": len(recent_exchange_trades),
                    "total_db_trades": len(db_trades),
                    "missing_trades": 0,
                    "recovered_trades": 0,
                }

        except Exception as e:
            logger.error("Trade recovery failed: %s", e)
            return {"recovered": False, "error": str(e)}

    def emergency_sync(self) -> SyncResult:
        """
        Emergency synchronization - force sync all data and handle discrepancies.
        Use this when the bot has been down for a while or data integrity is suspected.
        """
        logger.warning("Starting emergency account synchronization")

        # Force sync with exchange
        sync_result = self.sync_account_data(force=True)

        if not sync_result.success:
            return sync_result

        # Recover missing trades for common symbols
        common_symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "AVAXUSDT"]
        trade_recovery_results = {}

        for symbol in common_symbols:
            try:
                result = self.recover_missing_trades(symbol, days_back=30)  # Look back 30 days
                trade_recovery_results[symbol] = result
            except Exception as e:
                logger.error("Failed to recover trades for %s: %s", symbol, e)
                trade_recovery_results[symbol] = {"error": str(e)}

        # Add trade recovery results to sync data
        sync_result.data["emergency_trade_recovery"] = trade_recovery_results

        logger.info("Emergency synchronization completed")
        return sync_result
