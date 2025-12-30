"""
Exchange Interface - Abstract base for exchange operations

This module provides a clean abstraction layer for exchange operations,
allowing easy switching between different exchanges (Binance, Coinbase, etc.)
while maintaining consistent data synchronization.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LOSS = "STOP_LOSS"
    TAKE_PROFIT = "TAKE_PROFIT"


class OrderStatus(Enum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


@dataclass
class AccountBalance:
    """Represents account balance information"""

    asset: str
    free: float  # Available balance
    locked: float  # Balance in open orders
    total: float  # Total balance (free + locked)
    last_updated: datetime


@dataclass
class Position:
    """Represents an open position on the exchange"""

    symbol: str
    side: str  # "long" or "short"
    size: float  # Position size
    entry_price: float
    current_price: float
    unrealized_pnl: float
    margin_type: str  # "isolated" or "cross"
    leverage: float
    order_id: str
    open_time: datetime
    last_update_time: datetime


@dataclass
class Order:
    """Represents an order on the exchange"""

    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: float | None  # None for market orders
    status: OrderStatus
    filled_quantity: float
    average_price: float | None
    commission: float
    commission_asset: str
    create_time: datetime
    update_time: datetime
    stop_price: float | None = None
    time_in_force: str = "GTC"


@dataclass
class Trade:
    """Represents a completed trade"""

    trade_id: str
    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    commission: float
    commission_asset: str
    time: datetime


class ExchangeInterface(ABC):
    """
    Abstract base class for exchange operations.

    This interface provides a consistent API for different exchanges,
    allowing the trading bot to work with Binance, Coinbase, or any other
    exchange by implementing this interface.
    """

    def __init__(self, api_key: str, api_secret: str, testnet: bool = False):
        """
        Initialize the exchange interface.

        Args:
            api_key: Exchange API key
            api_secret: Exchange API secret
            testnet: Whether to use testnet/sandbox mode
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self._client = None
        self._initialize_client()

    @abstractmethod
    def _initialize_client(self):
        """Initialize the exchange-specific client"""
        pass

    @abstractmethod
    def test_connection(self) -> bool:
        """Test the connection to the exchange"""
        pass

    @abstractmethod
    def get_account_info(self) -> dict[str, Any]:
        """Get general account information"""
        pass

    @abstractmethod
    def get_balances(self) -> list[AccountBalance]:
        """Get all account balances"""
        pass

    @abstractmethod
    def get_balance(self, asset: str) -> AccountBalance | None:
        """Get balance for a specific asset"""
        pass

    @abstractmethod
    def get_positions(self, symbol: str | None = None) -> list[Position]:
        """Get open positions (for futures/margin trading)"""
        pass

    @abstractmethod
    def get_open_orders(self, symbol: str | None = None) -> list[Order]:
        """Get all open orders"""
        pass

    @abstractmethod
    def get_order(self, order_id: str, symbol: str) -> Order | None:
        """Get specific order by ID"""
        pass

    @abstractmethod
    def get_recent_trades(self, symbol: str, limit: int = 100) -> list[Trade]:
        """Get recent trades for a symbol"""
        pass

    @abstractmethod
    def place_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: float,
        price: float | None = None,
        stop_price: float | None = None,
        time_in_force: str = "GTC",
    ) -> str | None:
        """Place a new order and return order ID"""
        pass

    @abstractmethod
    def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel an existing order"""
        pass

    @abstractmethod
    def cancel_all_orders(self, symbol: str | None = None) -> bool:
        """Cancel all open orders"""
        pass

    @abstractmethod
    def place_stop_loss_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        stop_price: float,
        limit_price: float | None = None,
    ) -> str | None:
        """
        Place a server-side stop-loss order.

        This creates a stop-loss order on the exchange that triggers when
        price reaches stop_price. Using server-side stop-losses ensures
        positions are protected even if the bot goes offline.

        Args:
            symbol: Trading symbol (e.g., BTCUSDT)
            side: Order side (SELL for long positions, BUY for short positions)
            quantity: Amount to sell/buy when stop triggers
            stop_price: Price that triggers the stop order
            limit_price: Optional limit price for STOP_LOSS_LIMIT orders.
                         If None, uses stop_price * 0.99 for sells, * 1.01 for buys.

        Returns:
            Order ID from exchange, or None on failure
        """
        pass

    @abstractmethod
    def get_symbol_info(self, symbol: str) -> dict[str, Any] | None:
        """Get trading symbol information (min qty, price precision, etc.)"""
        pass

    def sync_account_data(self) -> dict[str, Any]:
        """
        Synchronize all account data from the exchange.

        Returns:
            Dictionary containing all synchronized data
        """
        try:
            logger.info("Starting account data synchronization...")

            # Get account information
            account_info = self.get_account_info()

            # Get balances
            balances = self.get_balances()

            # Get positions (if supported)
            positions = self.get_positions()

            # Get open orders
            open_orders = self.get_open_orders()

            sync_data = {
                "timestamp": datetime.now(UTC),
                "account_info": account_info,
                "balances": balances,
                "positions": positions,
                "open_orders": open_orders,
                "sync_successful": True,
            }

            logger.info(
                f"Account sync completed: {len(balances)} balances, "
                f"{len(positions)} positions, {len(open_orders)} open orders"
            )

            return sync_data

        except Exception as e:
            logger.error(f"Account synchronization failed: {e}")
            return {"timestamp": datetime.now(UTC), "sync_successful": False, "error": str(e)}

    def validate_order_parameters(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: float,
        price: float | None = None,
    ) -> tuple[bool, str | None]:
        """
        Validate order parameters before placing.

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Get symbol info
            symbol_info = self.get_symbol_info(symbol)
            if not symbol_info:
                return False, f"Symbol {symbol} not found"

            # Validate quantity
            min_qty = symbol_info.get("min_qty", 0)
            if quantity < min_qty:
                return False, f"Quantity {quantity} below minimum {min_qty}"

            # Validate price for limit orders
            if order_type == OrderType.LIMIT and price is not None:
                min_price = symbol_info.get("min_price", 0)
                if price < min_price:
                    return False, f"Price {price} below minimum {min_price}"

            return True, None

        except Exception as e:
            return False, f"Validation error: {e}"

    def get_total_equity(self, base_currency: str = "USDT") -> float:
        """
        Calculate total equity in base currency.

        Args:
            base_currency: Currency to convert all balances to

        Returns:
            Total equity value
        """
        try:
            balances = self.get_balances()
            total_equity = 0.0

            for balance in balances:
                if balance.asset == base_currency:
                    total_equity += balance.total
                else:
                    # For other assets, we'd need to get current price
                    # This is a simplified version - in practice you'd convert to base currency
                    if balance.total > 0:
                        logger.warning(
                            f"Non-{base_currency} balance detected: {balance.asset} {balance.total}"
                        )

            return total_equity

        except Exception as e:
            logger.error(f"Error calculating total equity: {e}")
            return 0.0
