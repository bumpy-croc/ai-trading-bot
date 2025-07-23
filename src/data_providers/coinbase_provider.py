from typing import Optional, Dict, Any, List
from datetime import datetime
import time
import hmac
import hashlib
import base64
import json
import logging
import requests
import pandas as pd

from .data_provider import DataProvider
from .exchange_interface import (
    ExchangeInterface,
    AccountBalance,
    Position,
    Order,
    Trade,
    OrderSide,
    OrderType,
    OrderStatus,
)
from config import get_config

logger = logging.getLogger(__name__)

class CoinbaseProvider(DataProvider, ExchangeInterface):
    """Coinbase data and exchange provider (spot). Implements DataProvider & ExchangeInterface."""

    BASE_URL = "https://api.exchange.coinbase.com"

    # Mapping of generic timeframe to Coinbase granularity (seconds)
    TIMEFRAME_MAPPING = {
        "1m": 60,
        "5m": 300,
        "15m": 900,
        "1h": 3600,
        "6h": 21600,
        "1d": 86400,
    }

    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None, passphrase: Optional[str] = None, testnet: bool = False):
        # Initialize DataProvider
        DataProvider.__init__(self)

        # Credentials from config if not provided
        if api_key is None or api_secret is None:
            config = get_config()
            api_key = api_key or config.get("COINBASE_API_KEY")
            api_secret = api_secret or config.get("COINBASE_API_SECRET")
            passphrase = passphrase or config.get("COINBASE_API_PASSPHRASE")

        # Initialize ExchangeInterface (will handle _initialize_client)
        ExchangeInterface.__init__(self, api_key or "", api_secret or "", testnet)
        self.passphrase = passphrase

    # ---------------------- ExchangeInterface ---------------------
    def _initialize_client(self):
        """Initialise Coinbase client & prepare auth parameters."""
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "CoinbaseProvider/1.0"})
        # Prepare decoded secret for HMAC
        try:
            self._decoded_secret = base64.b64decode(self.api_secret) if self.api_secret else None
        except Exception:
            self._decoded_secret = None

    # ------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------

    def _sign_request(self, timestamp: str, method: str, request_path: str, body: str = "") -> str:
        """Create CB-ACCESS-SIGN header value"""
        if not self._decoded_secret:
            raise ValueError("API secret not configured – cannot sign requests")
        message = f"{timestamp}{method.upper()}{request_path}{body}".encode()
        signature = hmac.new(self._decoded_secret, message, hashlib.sha256).digest()
        return base64.b64encode(signature).decode()

    def _request(self, method: str, path: str, params: Dict[str, Any] = None, body: Dict[str, Any] = None, auth: bool = False):
        """Helper to perform HTTP request with optional Coinbase authentication."""
        url = f"{self.BASE_URL}{path}"
        body_str = json.dumps(body) if body else ""
        headers = {}
        if auth:
            if not (self.api_key and self.api_secret and self.passphrase):
                raise ValueError("Authenticated request requested but API credentials not set")
            timestamp = str(int(time.time()))
            signature = self._sign_request(timestamp, method, path, body_str)
            headers.update({
                "CB-ACCESS-KEY": self.api_key,
                "CB-ACCESS-SIGN": signature,
                "CB-ACCESS-TIMESTAMP": timestamp,
                "CB-ACCESS-PASSPHRASE": self.passphrase,
                "Content-Type": "application/json",
            })
        try:
            response = self._session.request(method, url, params=params, data=body_str if body else None, headers=headers, timeout=15)
            response.raise_for_status()
            if response.text:
                return response.json()
            return {}
        except Exception as e:
            logger.error(f"Coinbase API request error {method} {path}: {e}")
            raise

    def test_connection(self) -> bool:
        try:
            r = self._session.get(f"{self.BASE_URL}/time", timeout=10)
            return r.status_code == 200
        except Exception as e:
            logger.error(f"Coinbase connection test failed: {e}")
            return False

    # The following authenticated endpoints are NOT implemented (Coinbase Advanced Trade API differs).
    # We create stubs to satisfy interface but mark as unimplemented.

    def get_account_info(self) -> Dict[str, Any]:
        try:
            data = self._request("GET", "/accounts", auth=True)
            account_info = {
                "total_accounts": len(data),
                "timestamp": datetime.utcnow(),
            }
            return account_info
        except Exception:
            return {}

    def get_balances(self) -> List[AccountBalance]:
        try:
            accounts = self._request("GET", "/accounts", auth=True)
            balances: List[AccountBalance] = []
            for acct in accounts:
                balance = float(acct.get("balance", "0"))
                available = float(acct.get("available", "0"))
                if balance == 0 and available == 0:
                    continue
                balances.append(
                    AccountBalance(
                        asset=acct.get("currency"),
                        free=available,
                        locked=balance - available,
                        total=balance,
                        last_updated=datetime.utcnow(),
                    )
                )
            return balances
        except Exception as e:
            logger.error(f"Failed to fetch balances: {e}")
            return []

    def get_balance(self, asset: str) -> Optional[AccountBalance]:
        balances = self.get_balances()
        for bal in balances:
            if bal.asset.upper() == asset.upper():
                return bal
        return None

    def get_positions(self, symbol: Optional[str] = None) -> List[Position]:
        logger.info("get_positions not implemented for Coinbase spot API (only holdings)")
        return []

    def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        try:
            params = {"status": "open"}
            if symbol:
                params["product_id"] = self._coinbase_symbol(symbol)
            data = self._request("GET", "/orders", params=params, auth=True)
            orders: List[Order] = []
            for od in data:
                orders.append(
                    Order(
                        order_id=od.get("id"),
                        symbol=od.get("product_id"),
                        side=OrderSide.BUY if od.get("side") == "buy" else OrderSide.SELL,
                        order_type=self._convert_order_type(od.get("type")),
                        quantity=float(od.get("size", 0)),
                        price=float(od.get("price")) if od.get("price") else None,
                        status=self._convert_order_status(od.get("status")),
                        filled_quantity=float(od.get("filled_size", 0)),
                        average_price=float(od.get("executed_value", 0)) / float(od.get("filled_size", 1)) if float(od.get("filled_size", 0)) > 0 else None,
                        commission=0.0,
                        commission_asset="",
                        create_time=datetime.fromisoformat(od.get("created_at")),
                        update_time=datetime.fromisoformat(od.get("done_at")) if od.get("done_at") else datetime.utcnow(),
                        stop_price=float(od.get("stop_price")) if od.get("stop_price") else None,
                        time_in_force=od.get("time_in_force", "GTC"),
                    )
                )
            return orders
        except Exception as e:
            logger.error(f"Failed to get open orders: {e}")
            return []

    def get_order(self, order_id: str, symbol: str) -> Optional[Order]:
        try:
            od = self._request("GET", f"/orders/{order_id}", auth=True)
            return Order(
                order_id=od.get("id"),
                symbol=od.get("product_id"),
                side=OrderSide.BUY if od.get("side") == "buy" else OrderSide.SELL,
                order_type=self._convert_order_type(od.get("type")),
                quantity=float(od.get("size", 0)),
                price=float(od.get("price")) if od.get("price") else None,
                status=self._convert_order_status(od.get("status")),
                filled_quantity=float(od.get("filled_size", 0)),
                average_price=float(od.get("executed_value", 0)) / float(od.get("filled_size", 1)) if float(od.get("filled_size", 0)) > 0 else None,
                commission=0.0,
                commission_asset="",
                create_time=datetime.fromisoformat(od.get("created_at")),
                update_time=datetime.fromisoformat(od.get("done_at")) if od.get("done_at") else datetime.utcnow(),
                stop_price=float(od.get("stop_price")) if od.get("stop_price") else None,
                time_in_force=od.get("time_in_force", "GTC"),
            )
        except Exception as e:
            logger.error(f"Failed to get order {order_id}: {e}")
            return None

    def get_recent_trades(self, symbol: str, limit: int = 100) -> List[Trade]:
        try:
            params = {"product_id": self._coinbase_symbol(symbol), "limit": limit}
            fills = self._request("GET", "/fills", params=params, auth=True)
            trades: List[Trade] = []
            for fl in fills:
                trades.append(
                    Trade(
                        trade_id=fl.get("trade_id"),
                        order_id=fl.get("order_id"),
                        symbol=fl.get("product_id"),
                        side=OrderSide.BUY if fl.get("side") == "buy" else OrderSide.SELL,
                        quantity=float(fl.get("size")),
                        price=float(fl.get("price")),
                        commission=float(fl.get("fee")),
                        commission_asset=fl.get("product_id").split("-")[1],
                        time=datetime.fromisoformat(fl.get("created_at")),
                    )
                )
            return trades
        except Exception as e:
            logger.error(f"Failed to fetch recent trades: {e}")
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
        try:
            cb_type = self._convert_to_cb_type(order_type)
            body: Dict[str, Any] = {
                "product_id": self._coinbase_symbol(symbol),
                "side": side.value.lower(),
                "type": cb_type,
            }
            if cb_type == "market":
                body["size"] = str(quantity)
            else:
                body.update({
                    "size": str(quantity),
                    "price": str(price) if price else None,
                    "time_in_force": time_in_force,
                })
            if cb_type == "stop":
                body["stop_price"] = str(stop_price) if stop_price else None
                body["stop"] = "loss"  # default stop loss

            order = self._request("POST", "/orders", body=body, auth=True)
            return order.get("id")
        except Exception as e:
            logger.error(f"Failed to place order: {e}")
            return None

    def cancel_order(self, order_id: str, symbol: str) -> bool:
        try:
            self._request("DELETE", f"/orders/{order_id}", auth=True)
            return True
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False

    def cancel_all_orders(self, symbol: Optional[str] = None) -> bool:
        try:
            params = {"product_id": self._coinbase_symbol(symbol)} if symbol else None
            self._request("DELETE", "/orders", params=params, auth=True)
            return True
        except Exception as e:
            logger.error(f"Failed to cancel all orders: {e}")
            return False

    def get_symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        try:
            product = self._request("GET", f"/products/{self._coinbase_symbol(symbol)}")
            return {
                "symbol": product.get("id"),
                "base_asset": product.get("base_currency"),
                "quote_asset": product.get("quote_currency"),
                "status": product.get("status"),
                "min_qty": float(product.get("base_min_size", 0)),
                "max_qty": float(product.get("base_max_size", 0)),
                "min_price": float(product.get("min_market_funds", 0)),
                "max_price": float(product.get("max_market_funds", 0)),
                "tick_size": float(product.get("quote_increment", 0)),
            }
        except Exception as e:
            logger.error(f"Failed to fetch symbol info: {e}")
            return None

    # --------------------------- DataProvider --------------------------

    def _convert_timeframe(self, timeframe: str) -> int:
        if timeframe not in self.TIMEFRAME_MAPPING:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        return self.TIMEFRAME_MAPPING[timeframe]

    def _fetch_candles(self, product_id: str, granularity: int, start: datetime = None, end: datetime = None, limit: int = None) -> List[List[Any]]:
        """Fetch candle data from Coinbase public API."""
        params = {"granularity": granularity}
        if start is not None:
            params["start"] = start.isoformat()
        if end is not None:
            params["end"] = end.isoformat()
        url = f"{self.BASE_URL}/products/{product_id}/candles"
        r = self._session.get(url, params=params, timeout=15)
        r.raise_for_status()
        candles = r.json()
        # Coinbase returns [ time, low, high, open, close, volume ] in descending order.
        candles.sort(key=lambda x: x[0])  # ascending
        # Reorder to [timestamp, open, high, low, close, volume]
        formatted = [[c[0], c[3], c[2], c[1], c[4], c[5]] for c in candles]
        if limit is not None:
            formatted = formatted[-limit:]
        return formatted

    def get_historical_data(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: Optional[datetime] = None,
    ) -> pd.DataFrame:
        try:
            granularity = self._convert_timeframe(timeframe)
            product_id = symbol.upper()
            candles = self._fetch_candles(product_id, granularity, start=start, end=end)
            df = self._process_ohlcv(candles, timestamp_unit="s")
            self.data = df
            if not df.empty:
                logger.info(f"Fetched {len(df)} candles for {symbol} from {df.index.min()} to {df.index.max()}")
            return df
        except Exception as e:
            logger.error(f"Error fetching historical data from Coinbase: {e}")
            raise

    def get_live_data(self, symbol: str, timeframe: str, limit: int = 100) -> pd.DataFrame:
        try:
            granularity = self._convert_timeframe(timeframe)
            product_id = symbol.upper()
            candles = self._fetch_candles(product_id, granularity, limit=limit)
            df = self._process_ohlcv(candles, timestamp_unit="s")
            self.data = df
            return df
        except Exception as e:
            logger.error(f"Error fetching live data from Coinbase: {e}")
            raise

    def update_live_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        try:
            if self.data is None or self.data.empty:
                return self.get_live_data(symbol, timeframe, limit=200)
            granularity = self._convert_timeframe(timeframe)
            product_id = symbol.upper()
            candles = self._fetch_candles(product_id, granularity, limit=1)
            if not candles:
                return self.data
            latest_df = self._process_ohlcv(candles, timestamp_unit="s")
            # Merge with existing
            self.data = pd.concat([
                self.data[~self.data.index.isin(latest_df.index)], latest_df
            ]).sort_index()
            return self.data
        except Exception as e:
            logger.error(f"Error updating live data for Coinbase: {e}")
            raise

    def get_current_price(self, symbol: str) -> float:
        try:
            product_id = symbol.upper()
            url = f"{self.BASE_URL}/products/{product_id}/ticker"
            r = self._session.get(url, timeout=10)
            r.raise_for_status()
            data = r.json()
            return float(data["price"])
        except Exception as e:
            logger.error(f"Error fetching current price for {symbol} from Coinbase: {e}")
            return 0.0

# Aliases for convenience
CoinbaseDataProvider = CoinbaseProvider
CoinbaseExchange = CoinbaseProvider