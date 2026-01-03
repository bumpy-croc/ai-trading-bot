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
)
from src.data_providers.exchange_interface import (
    AccountBalance,
    ExchangeInterface,
    Order,
    Position,
)
from src.data_providers.exchange_interface import OrderStatus as ExchangeOrderStatus
from src.database.manager import DatabaseManager
from src.database.models import PositionSide, TradeSource

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
    ):
        """
        Initialize the account synchronizer.

        Args:
            exchange: Exchange interface for API calls
            db_manager: Database manager for local data
            session_id: Current trading session ID
        """
        self.exchange = exchange
        self.db_manager = db_manager
        self.session_id = session_id
        self.last_sync_time: datetime | None = None

    def sync_account_data(self, force: bool = False) -> SyncResult:
        """
        Synchronize all account data from the exchange.

        Args:
            force: Force sync even if recently synced

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

            # Get data from exchange
            exchange_data = self.exchange.sync_account_data()

            if not exchange_data.get("sync_successful", False):
                error_msg = exchange_data.get("error", "Unknown error")
                logger.error(f"Exchange sync failed: {error_msg}")
                return SyncResult(
                    success=False,
                    message=f"Exchange sync failed: {error_msg}",
                    data=exchange_data,
                    timestamp=datetime.now(UTC),
                )

            # Sync balances
            balance_sync_result = self._sync_balances(exchange_data.get("balances", []))

            # Sync positions
            position_sync_result = self._sync_positions(exchange_data.get("positions", []))

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
            logger.error(f"Account synchronization failed: {e}")
            return SyncResult(
                success=False,
                message=f"Sync failed: {str(e)}",
                data={},
                timestamp=datetime.now(UTC),
            )

    def _sync_balances(self, exchange_balances: list[AccountBalance]) -> dict[str, Any]:
        """Synchronize account balances"""
        try:
            logger.info(f"Syncing {len(exchange_balances)} balances from exchange")

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
            logger.error(f"Balance sync failed: {e}")
            return {"synced": False, "error": str(e)}

    def _sync_positions(self, exchange_positions: list[Position]) -> dict[str, Any]:
        """Synchronize open positions"""
        try:
            logger.info(f"Syncing {len(exchange_positions)} positions from exchange")

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
            logger.error(f"Position sync failed: {e}")
            return {"synced": False, "error": str(e)}

    def _sync_orders(self, exchange_orders: list[Order]) -> dict[str, Any]:
        """Synchronize open orders"""
        try:
            logger.info(f"Syncing {len(exchange_orders)} orders from exchange")

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
                    logger.warning(f"Order cancelled on exchange: {order_id}")

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
            logger.error(f"Order sync failed: {e}")
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
            logger.info(f"Recovering missing trades for {symbol} (last {days_back} days)")

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
                logger.warning(f"Found {len(missing_trades)} missing trades")

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
                        logger.error(f"Failed to recover trade {trade.trade_id}: {e}")

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
            logger.error(f"Trade recovery failed: {e}")
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
                logger.error(f"Failed to recover trades for {symbol}: {e}")
                trade_recovery_results[symbol] = {"error": str(e)}

        # Add trade recovery results to sync data
        sync_result.data["emergency_trade_recovery"] = trade_recovery_results

        logger.info("Emergency synchronization completed")
        return sync_result
