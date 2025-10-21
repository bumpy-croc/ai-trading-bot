"""
Unified Binance Provider

This module combines both data provider and exchange functionality for Binance,
providing a single interface for all Binance operations including:
- Historical and live data fetching
- Order execution and management
- Account synchronization
- Position management
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Optional

import pandas as pd

from src.config import get_config
from src.utils.symbol_factory import SymbolFactory

from .data_provider import DataProvider
from .exchange_interface import (
    AccountBalance,
    ExchangeInterface,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    Trade,
)

logger = logging.getLogger(__name__)

try:
    from binance.client import Client
    from binance.enums import SIDE_BUY, SIDE_SELL
    from binance.exceptions import BinanceAPIException, BinanceOrderException

    BINANCE_AVAILABLE = True
except ImportError:
    # Fallback for environments without binance library
    logger.warning("Binance library not available - using mock implementation")
    Client = None
    BinanceAPIException = Exception
    BinanceOrderException = Exception
    SIDE_BUY = "BUY"
    SIDE_SELL = "SELL"
    BINANCE_AVAILABLE = False

# Import geo-detection utilities
from src.utils.geo_detection import get_binance_api_endpoint, is_us_location


class BinanceProvider(DataProvider, ExchangeInterface):
    """
    Unified Binance provider that combines data fetching and exchange operations.

    Inherits from both DataProvider and ExchangeInterface to provide complete
    Binance functionality in a single class.
    """

    TIMEFRAME_MAPPING = {
        "1m": Client.KLINE_INTERVAL_1MINUTE if BINANCE_AVAILABLE else "1m",
        "5m": Client.KLINE_INTERVAL_5MINUTE if BINANCE_AVAILABLE else "5m",
        "15m": Client.KLINE_INTERVAL_15MINUTE if BINANCE_AVAILABLE else "15m",
        "1h": Client.KLINE_INTERVAL_1HOUR if BINANCE_AVAILABLE else "1h",
        "4h": Client.KLINE_INTERVAL_4HOUR if BINANCE_AVAILABLE else "4h",
        "1d": Client.KLINE_INTERVAL_1DAY if BINANCE_AVAILABLE else "1d",
    }

    def __init__(
        self, api_key: Optional[str] = None, api_secret: Optional[str] = None, testnet: bool = False
    ):
        """
        Initialize the unified Binance provider.

        Args:
            api_key: Binance API key (optional, will try to get from config)
            api_secret: Binance API secret (optional, will try to get from config)
            testnet: Whether to use testnet/sandbox mode
        """
        # Initialize DataProvider
        DataProvider.__init__(self)

        # Get credentials from config if not provided
        if api_key is None or api_secret is None:
            config = get_config()
            api_key = api_key or config.get("BINANCE_API_KEY")
            api_secret = api_secret or config.get("BINANCE_API_SECRET")

        # SEC-004 Fix: Validate credentials are properly formatted
        api_key, api_secret = self._validate_credentials(api_key, api_secret)

        # Initialize ExchangeInterface
        if api_key and api_secret:
            # ExchangeInterface.__init__ will call self._initialize_client()
            ExchangeInterface.__init__(self, api_key, api_secret, testnet)
        else:
            # Initialize with dummy credentials for data-only operations
            self.api_key = api_key
            self.api_secret = api_secret
            self.testnet = testnet
            self._client = None
            logger.info("Binance provider initialized in read-only mode (no credentials)")
            self._initialize_client()

    @staticmethod
    def _validate_credentials(api_key: Optional[str], api_secret: Optional[str]) -> tuple[str, str]:
        """
        Validate and normalize API credentials.
        
        SEC-004 Fix: Ensure credentials are properly formatted or explicitly missing.
        
        Args:
            api_key: API key to validate
            api_secret: API secret to validate
            
        Returns:
            Tuple of (api_key, api_secret) - empty strings if not provided
            
        Raises:
            ValueError: If credentials are provided but malformed
        """
        # If both are missing/None, return empty strings for read-only mode
        if not api_key and not api_secret:
            return "", ""
        
        # If only one is provided, that's an error
        if bool(api_key) != bool(api_secret):
            raise ValueError(
                "Binance credentials must be provided together. "
                "Either provide both BINANCE_API_KEY and BINANCE_API_SECRET, or neither."
            )
        
        # Validate credential format (reasonable minimum length)
        if api_key and len(str(api_key).strip()) < 20:
            raise ValueError(f"Invalid BINANCE_API_KEY format (too short: {len(str(api_key))} chars)")
        if api_secret and len(str(api_secret).strip()) < 20:
            raise ValueError(f"Invalid BINANCE_API_SECRET format (too short: {len(str(api_secret))} chars)")
        
        return str(api_key).strip() if api_key else "", str(api_secret).strip() if api_secret else ""

    def _initialize_client(self):
        """Initialize Binance client with geo-aware API selection and error handling"""
        logger.debug(f"_initialize_client called - BINANCE_AVAILABLE: {BINANCE_AVAILABLE}")
        
        if not BINANCE_AVAILABLE:
            logger.warning("Binance library not available - using mock client")
            self._client = self._create_offline_client()
            return

        # Determine which Binance API to use based on location
        api_endpoint = get_binance_api_endpoint()
        is_us = is_us_location()
        
        logger.info(f"Geo-detection result: {'US location' if is_us else 'Non-US location'} - using {api_endpoint} API")

        try:
            logger.debug(f"Attempting to create {api_endpoint} client - has_credentials: {bool(self.api_key and self.api_secret)}, testnet: {self.testnet}")
            
            # Create client with appropriate API endpoint
            if self.api_key and self.api_secret:
                logger.debug(f"Creating authenticated {api_endpoint} client...")
                if api_endpoint == "binanceus":
                    # For Binance US, use tld='us' parameter
                    self._client = Client(self.api_key, self.api_secret, testnet=self.testnet, tld='us')
                else:
                    # For global Binance
                    self._client = Client(self.api_key, self.api_secret, testnet=self.testnet)
            else:
                logger.debug(f"Creating public {api_endpoint} client...")
                if api_endpoint == "binanceus":
                    # For Binance US public client
                    self._client = Client(tld='us')
                else:
                    # For global Binance public client
                    self._client = Client()
            
            logger.info(
                f"{api_endpoint.title()} client initialized successfully "
                f"({'with credentials' if self.api_key and self.api_secret else 'public mode'}, "
                f"testnet: {self.testnet})"
            )
                
            # Test the client with a simple operation
            logger.debug("Testing client with server time request...")
            test_response = self._client.get_server_time()
            logger.debug(f"Server time test successful: {test_response}")
            
        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e)

            # Log detailed error information (never log credentials)
            logger.error(
                f"{api_endpoint.title()} Client initialization failed with {error_type}: {error_msg}. "
                f"Credentials available: {bool(self.api_key and self.api_secret)}, "
                f"Testnet mode: {self.testnet}. "
                f"Falling back to offline stub."
            )

            # Log additional context if it's a recursion error
            if "recursion" in error_msg.lower() or "maximum recursion" in error_msg.lower():
                logger.error(
                    "Recursion error detected during Binance client initialization. "
                    "This may indicate a circular dependency or infinite loop in the initialization process. "
                    "Check for circular imports or dependencies in the configuration system."
                )

            self._client = self._create_offline_client()

    def _create_offline_client(self):
        """Create offline client stub for testing"""

        class _OfflineClient:
            """Lightweight stub mimicking the required Binance Client interface for tests."""

            def get_historical_klines(self, *args, **kwargs):
                return []

            def get_klines(self, *args, **kwargs):
                return []

            def get_symbol_ticker(self, *args, **kwargs):
                return {"price": "0"}

            def ping(self):
                return {}

            def get_server_time(self):
                return {"serverTime": 1640995200000}

            def get_account(self):
                return {"balances": [], "canTrade": False}

            def get_open_orders(self, *args, **kwargs):
                return []

            def get_order(self, *args, **kwargs):
                return {}

            def get_my_trades(self, *args, **kwargs):
                return []

            def create_order(self, *args, **kwargs):
                return {"orderId": "12345"}

            def cancel_order(self, *args, **kwargs):
                return {"orderId": "12345"}

            def cancel_all_orders(self, *args, **kwargs):
                return []

            def get_exchange_info(self):
                return {"symbols": []}

        return _OfflineClient()

    # ========================================
    # DataProvider Interface Implementation
    # ========================================

    def _convert_timeframe(self, timeframe: str) -> str:
        """Convert generic timeframe to Binance-specific interval"""
        if timeframe not in self.TIMEFRAME_MAPPING:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        return self.TIMEFRAME_MAPPING[timeframe]

    def _process_klines(self, klines: list) -> pd.DataFrame:
        """Convert raw klines data to standardized DataFrame using base helper"""
        # Binance timestamps are in milliseconds
        # We keep only the first 6 columns which correspond to timestamp and OHLCV
        return self._process_ohlcv([k[:6] for k in klines], timestamp_unit="ms")

    def get_historical_data(
        self, symbol: str, timeframe: str, start: datetime, end: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Fetch historical klines data from Binance"""
        try:
            interval = self._convert_timeframe(timeframe)
            start_ts = int(start.timestamp() * 1000)
            end_ts = int(end.timestamp() * 1000) if end else None

            klines = self._client.get_historical_klines(symbol, interval, start_ts, end_ts)

            df = self._process_klines(klines)
            self.data = df

            if len(df) > 0:
                logger.info(f"Fetched {len(df)} candles from {df.index.min()} to {df.index.max()}")
            else:
                # Check if this is expected (future dates) or an error
                current_time = datetime.now()
                if end is not None and end > current_time:
                    logger.info(f"No data available for future dates: requested {start} to {end}, current time is {current_time}")
                elif end is not None and end > current_time - timedelta(hours=1):
                    logger.info(f"No recent data available yet: requested {start} to {end}, current time is {current_time}")
                else:
                    logger.warning(f"No data returned for {symbol} {timeframe} from {start} to {end}")
            return df

        except Exception as e:
            error_type = type(e).__name__
            logger.error(
                f"Error fetching historical data for {symbol} {timeframe} "
                f"from {start} to {end}: {error_type}: {str(e)}"
            )
            raise

    def get_live_data(self, symbol: str, timeframe: str, limit: int = 100) -> pd.DataFrame:
        """Fetch current market data"""
        try:
            interval = self._convert_timeframe(timeframe)
            klines = self._client.get_klines(symbol=symbol, interval=interval, limit=limit)

            df = self._process_klines(klines)
            self.data = df
            return df

        except Exception as e:
            error_type = type(e).__name__
            logger.error(
                f"Error fetching live data for {symbol} {timeframe} "
                f"(limit: {limit}): {error_type}: {str(e)}"
            )
            raise

    def update_live_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Update the latest market data"""
        try:
            interval = self._convert_timeframe(timeframe)
            latest_kline = self._client.get_klines(symbol=symbol, interval=interval, limit=1)

            if not latest_kline:
                return self.data if self.data is not None else pd.DataFrame()

            latest_df = self._process_klines(latest_kline)

            if self.data is not None:
                # Update or append the latest candle
                self.data = pd.concat(
                    [self.data[~self.data.index.isin(latest_df.index)], latest_df]
                ).sort_index()
            else:
                self.data = latest_df

            return self.data

        except Exception as e:
            error_type = type(e).__name__
            logger.error(
                f"Error updating live data for {symbol} {timeframe}: {error_type}: {str(e)}"
            )
            raise

    def get_current_price(self, symbol: str) -> float:
        """Get latest price for a symbol"""
        try:
            ticker = self._client.get_symbol_ticker(symbol=symbol)
            return float(ticker["price"])
        except Exception as e:
            error_type = type(e).__name__
            logger.error(
                f"Error fetching current price for {symbol}: {error_type}: {str(e)}"
            )
            return 0.0

    # ========================================
    # ExchangeInterface Implementation
    # ========================================

    def test_connection(self) -> bool:
        """Test connection to Binance"""
        if not BINANCE_AVAILABLE or not self._client:
            logger.warning("Binance not available - connection test skipped")
            return False

        try:
            # Test server time
            server_time = self._client.get_server_time()
            logger.info(f"Binance connection test successful - server time: {server_time}")
            return True
        except Exception as e:
            logger.error(f"Binance connection test failed: {e}")
            return False

    def get_account_info(self) -> dict[str, Any]:
        """Get Binance account information"""
        if not BINANCE_AVAILABLE or not self._client:
            logger.warning("Binance not available - returning empty account info")
            return {}

        try:
            account_info = self._client.get_account()
            return {
                "maker_commission": account_info.get("makerCommission"),
                "taker_commission": account_info.get("takerCommission"),
                "buyer_commission": account_info.get("buyerCommission"),
                "seller_commission": account_info.get("sellerCommission"),
                "can_trade": account_info.get("canTrade"),
                "can_withdraw": account_info.get("canWithdraw"),
                "can_deposit": account_info.get("canDeposit"),
                "update_time": account_info.get("updateTime"),
            }
        except Exception as e:
            logger.error(f"Failed to get account info: {e}")
            return {}

    def get_balances(self) -> list[AccountBalance]:
        """Get all account balances"""
        if not BINANCE_AVAILABLE or not self._client:
            logger.warning("Binance not available - returning empty balances")
            return []

        try:
            account_info = self._client.get_account()
            balances = []

            for balance_data in account_info.get("balances", []):
                free = float(balance_data.get("free", 0))
                locked = float(balance_data.get("locked", 0))
                total = free + locked

                if total > 0:  # Only include non-zero balances
                    balance = AccountBalance(
                        asset=balance_data["asset"],
                        free=free,
                        locked=locked,
                        total=total,
                        last_updated=datetime.utcnow(),
                    )
                    balances.append(balance)

            return balances

        except Exception as e:
            logger.error(f"Failed to get balances: {e}")
            return []

    def get_balance(self, asset: str) -> Optional[AccountBalance]:
        """Get balance for a specific asset"""
        if not BINANCE_AVAILABLE or not self._client:
            logger.warning("Binance not available - returning None for balance")
            return None

        try:
            account_info = self._client.get_account()

            for balance_data in account_info.get("balances", []):
                if balance_data["asset"] == asset:
                    free = float(balance_data.get("free", 0))
                    locked = float(balance_data.get("locked", 0))
                    total = free + locked

                    return AccountBalance(
                        asset=asset,
                        free=free,
                        locked=locked,
                        total=total,
                        last_updated=datetime.utcnow(),
                    )

            return None

        except Exception as e:
            logger.error(f"Failed to get balance for {asset}: {e}")
            return None

    def get_positions(self, symbol: Optional[str] = None) -> list[Position]:
        """Get open positions (for spot trading, this returns holdings as positions)"""
        if not BINANCE_AVAILABLE or not self._client:
            logger.warning("Binance not available - returning empty positions")
            return []

        try:
            # For spot trading, we consider holdings as "positions"
            balances = self.get_balances()
            positions = []

            for balance in balances:
                if balance.asset != "USDT" and balance.total > 0:
                    # Get current price for the asset
                    try:
                        ticker = self._client.get_symbol_ticker(
                            symbol=SymbolFactory.to_exchange_symbol(
                                f"{balance.asset}-USD", "binance"
                            )
                        )
                        current_price = float(ticker["price"])

                        position = Position(
                            symbol=f"{balance.asset}USDT",
                            side="long",
                            size=balance.total,
                            entry_price=current_price,  # Simplified - we don't track entry price for holdings
                            current_price=current_price,
                            unrealized_pnl=0.0,  # Simplified
                            margin_type="spot",
                            leverage=1.0,
                            order_id="",  # No order ID for holdings
                            open_time=datetime.utcnow(),  # Simplified
                            last_update_time=datetime.utcnow(),
                        )
                        positions.append(position)

                    except Exception as e:
                        logger.debug(f"Could not get price for {balance.asset}: {e}")

            return positions

        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return []

    def get_open_orders(self, symbol: Optional[str] = None) -> list[Order]:
        """Get all open orders"""
        if not BINANCE_AVAILABLE or not self._client:
            logger.warning("Binance not available - returning empty orders")
            return []

        try:
            if symbol:
                orders_data = self._client.get_open_orders(symbol=symbol)
            else:
                orders_data = self._client.get_open_orders()

            orders = []
            for order_data in orders_data:
                order = Order(
                    order_id=str(order_data["orderId"]),
                    symbol=order_data["symbol"],
                    side=OrderSide.BUY if order_data["side"] == SIDE_BUY else OrderSide.SELL,
                    order_type=self._convert_order_type(order_data["type"]),
                    quantity=float(order_data["origQty"]),
                    price=float(order_data["price"]) if order_data["price"] != "0" else None,
                    status=self._convert_order_status(order_data["status"]),
                    filled_quantity=float(order_data["executedQty"]),
                    average_price=(
                        float(order_data["avgPrice"]) if order_data["avgPrice"] != "0" else None
                    ),
                    commission=0.0,  # Will be updated from trade history
                    commission_asset="",
                    create_time=datetime.fromtimestamp(order_data["time"] / 1000),
                    update_time=datetime.fromtimestamp(order_data["updateTime"] / 1000),
                    stop_price=(
                        float(order_data["stopPrice"]) if order_data.get("stopPrice") else None
                    ),
                    time_in_force=order_data.get("timeInForce", "GTC"),
                )
                orders.append(order)

            return orders

        except Exception as e:
            logger.error(f"Failed to get open orders: {e}")
            return []

    def get_order(self, order_id: str, symbol: str) -> Optional[Order]:
        """Get specific order by ID"""
        if not BINANCE_AVAILABLE or not self._client:
            logger.warning("Binance not available - returning None for order")
            return None

        try:
            order_data = self._client.get_order(symbol=symbol, orderId=order_id)

            order = Order(
                order_id=str(order_data["orderId"]),
                symbol=order_data["symbol"],
                side=OrderSide.BUY if order_data["side"] == SIDE_BUY else OrderSide.SELL,
                order_type=self._convert_order_type(order_data["type"]),
                quantity=float(order_data["origQty"]),
                price=float(order_data["price"]) if order_data["price"] != "0" else None,
                status=self._convert_order_status(order_data["status"]),
                filled_quantity=float(order_data["executedQty"]),
                average_price=(
                    float(order_data["avgPrice"]) if order_data["avgPrice"] != "0" else None
                ),
                commission=0.0,
                commission_asset="",
                create_time=datetime.fromtimestamp(order_data["time"] / 1000),
                update_time=datetime.fromtimestamp(order_data["updateTime"] / 1000),
                stop_price=(
                    float(order_data.get("stopPrice")) if order_data.get("stopPrice") else None
                ),
                time_in_force=order_data.get("timeInForce", "GTC"),
            )

            return order

        except Exception as e:
            logger.error(f"Failed to get order {order_id}: {e}")
            return None

    def get_recent_trades(self, symbol: str, limit: int = 100) -> list[Trade]:
        """Get recent trades for a symbol"""
        if not BINANCE_AVAILABLE or not self._client:
            logger.warning("Binance not available - returning empty trades")
            return []

        try:
            trades_data = self._client.get_my_trades(symbol=symbol, limit=limit)

            trades = []
            for trade_data in trades_data:
                trade = Trade(
                    trade_id=str(trade_data["id"]),
                    order_id=str(trade_data["orderId"]),
                    symbol=trade_data["symbol"],
                    side=OrderSide.BUY if trade_data["isBuyer"] else OrderSide.SELL,
                    quantity=float(trade_data["qty"]),
                    price=float(trade_data["price"]),
                    commission=float(trade_data["commission"]),
                    commission_asset=trade_data["commissionAsset"],
                    time=datetime.fromtimestamp(trade_data["time"] / 1000),
                )
                trades.append(trade)

            return trades

        except Exception as e:
            logger.error(f"Failed to get recent trades for {symbol}: {e}")
            return []

    def place_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: float,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: str = "GTC",
    ) -> Optional[str]:
        """Place a new order and return order ID"""
        if not BINANCE_AVAILABLE or not self._client:
            logger.warning("Binance not available - cannot place order")
            return None

        try:
            # Validate parameters first
            is_valid, error_msg = self.validate_order_parameters(
                symbol, side, order_type, quantity, price
            )
            if not is_valid:
                logger.error(f"Order validation failed: {error_msg}")
                return None

            # Convert to Binance parameters
            binance_side = SIDE_BUY if side == OrderSide.BUY else SIDE_SELL
            binance_type = self._convert_to_binance_order_type(order_type)

            # Prepare order parameters
            order_params = {
                "symbol": symbol,
                "side": binance_side,
                "type": binance_type,
                "quantity": quantity,
            }

            if price is not None:
                order_params["price"] = price

            if stop_price is not None:
                order_params["stopPrice"] = stop_price

            if time_in_force != "GTC":
                order_params["timeInForce"] = time_in_force

            # Place the order
            self._client.create_order(**order_params)

            logger.info(f"Order placed successfully: {symbol} {side.value} {quantity}")

            # In real returns we return order id; in offline mode we may not have it
            return "order"

        except BinanceOrderException as e:
            logger.error(f"Binance order error: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to place order: {e}")
            return None

    def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel an existing order"""
        if not BINANCE_AVAILABLE or not self._client:
            logger.warning("Binance not available - cannot cancel order")
            return False

        try:
            self._client.cancel_order(symbol=symbol, orderId=order_id)
            logger.info(f"Order cancelled successfully: {order_id}")
            return True

        except BinanceOrderException as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False
        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {e}")
            return False

    def cancel_all_orders(self, symbol: Optional[str] = None) -> bool:
        """Cancel all open orders"""
        if not BINANCE_AVAILABLE or not self._client:
            logger.warning("Binance not available - cannot cancel orders")
            return False

        try:
            if symbol:
                self._client.cancel_all_orders(symbol=symbol)
            else:
                # Cancel all orders for all symbols
                open_orders = self.get_open_orders()
                for order in open_orders:
                    self.cancel_order(order.order_id, order.symbol)

            logger.info("All orders cancelled successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to cancel all orders: {e}")
            return False

    def get_symbol_info(self, symbol: str) -> Optional[dict[str, Any]]:
        """Get trading symbol information"""
        if not BINANCE_AVAILABLE or not self._client:
            logger.warning("Binance not available - returning None for symbol info")
            return None

        try:
            exchange_info = self._client.get_exchange_info()

            for symbol_info in exchange_info["symbols"]:
                if symbol_info["symbol"] == SymbolFactory.to_exchange_symbol(symbol, "binance"):
                    # Extract relevant information
                    filters = {f["filterType"]: f for f in symbol_info["filters"]}

                    return {
                        "symbol": symbol,
                        "base_asset": symbol_info["baseAsset"],
                        "quote_asset": symbol_info["quoteAsset"],
                        "status": symbol_info["status"],
                        "min_qty": float(filters.get("LOT_SIZE", {}).get("minQty", 0)),
                        "max_qty": float(filters.get("LOT_SIZE", {}).get("maxQty", float("inf"))),
                        "step_size": float(filters.get("LOT_SIZE", {}).get("stepSize", 0)),
                        "min_price": float(filters.get("PRICE_FILTER", {}).get("minPrice", 0)),
                        "max_price": float(
                            filters.get("PRICE_FILTER", {}).get("maxPrice", float("inf"))
                        ),
                        "tick_size": float(filters.get("PRICE_FILTER", {}).get("tickSize", 0)),
                        "min_notional": float(
                            filters.get("MIN_NOTIONAL", {}).get("minNotional", 0)
                        ),
                    }

            return None

        except Exception as e:
            logger.error(f"Failed to get symbol info for {symbol}: {e}")
            return None

    def _convert_order_type(self, binance_type: str) -> OrderType:
        """Convert Binance order type to our enum"""
        mapping = {
            "MARKET": OrderType.MARKET,
            "LIMIT": OrderType.LIMIT,
            "STOP_LOSS": OrderType.STOP_LOSS,
            "TAKE_PROFIT": OrderType.TAKE_PROFIT,
        }
        return mapping.get(binance_type, OrderType.MARKET)

    def _convert_to_binance_order_type(self, order_type: OrderType) -> str:
        """Convert our order type enum to Binance format"""
        mapping = {
            OrderType.MARKET: "MARKET",
            OrderType.LIMIT: "LIMIT",
            OrderType.STOP_LOSS: "STOP_LOSS",
            OrderType.TAKE_PROFIT: "TAKE_PROFIT",
        }
        return mapping.get(order_type, "MARKET")

    def _convert_order_status(self, binance_status: str) -> OrderStatus:
        """Convert Binance order status to our enum"""
        mapping = {
            "NEW": OrderStatus.PENDING,
            "PARTIALLY_FILLED": OrderStatus.PARTIALLY_FILLED,
            "FILLED": OrderStatus.FILLED,
            "CANCELED": OrderStatus.CANCELLED,
            "REJECTED": OrderStatus.REJECTED,
            "EXPIRED": OrderStatus.EXPIRED,
        }
        return mapping.get(binance_status, OrderStatus.PENDING)


# Aliases for backward compatibility
BinanceDataProvider = BinanceProvider
BinanceExchange = BinanceProvider
