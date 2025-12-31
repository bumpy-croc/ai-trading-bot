import base64
import hashlib
import hmac
import json
import logging
import os
import time
from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd
import requests

from src.config import get_config
from src.infrastructure.network_retry import with_network_retry
from src.trading.symbols.factory import SymbolFactory

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


class CoinbaseProvider(DataProvider, ExchangeInterface):
    """Coinbase data and exchange provider (spot). Implements DataProvider & ExchangeInterface."""

    # Determine environment (sandbox vs production) via env variable
    COINBASE_API_ENV = os.getenv("COINBASE_API_ENV", "sandbox").lower()

    if COINBASE_API_ENV == "sandbox":
        BASE_URL = "https://api-public.sandbox.exchange.coinbase.com"
    else:
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

    def __init__(
        self,
        api_key: str | None = None,
        api_secret: str | None = None,
        passphrase: str | None = None,
        testnet: bool = False,
    ):
        # Initialize DataProvider
        DataProvider.__init__(self)

        config = get_config()

        # Credentials from config if not provided
        if api_key is None:
            api_key = config.get("COINBASE_API_KEY")
        if api_secret is None:
            api_secret = config.get("COINBASE_API_SECRET")
        if passphrase is None:
            passphrase = config.get("COINBASE_API_PASSPHRASE")

        env_name = str(config.get("ENV", "")).lower()
        allow_test_credentials = testnet or env_name in {"test", "testing", "ci"}

        # SEC-004 Fix: Validate credentials are properly formatted
        api_key, api_secret, passphrase = self._validate_credentials(
            api_key,
            api_secret,
            passphrase,
            allow_test_credentials=allow_test_credentials,
        )

        # Initialize ExchangeInterface (will handle _initialize_client)
        ExchangeInterface.__init__(self, api_key or "", api_secret or "", testnet)
        self.passphrase = passphrase

    @staticmethod
    def _validate_credentials(
        api_key: str | None,
        api_secret: str | None,
        passphrase: str | None,
        *,
        allow_test_credentials: bool = False,
    ) -> tuple[str, str, str]:
        """
        Validate and normalize Coinbase API credentials.

        SEC-004 Fix: Ensure credentials are properly formatted or explicitly missing.

        Args:
            api_key: API key to validate
            api_secret: API secret to validate
            passphrase: API passphrase to validate

        Returns:
            Tuple of (api_key, api_secret, passphrase) - empty strings if not provided

        Raises:
            ValueError: If credentials are provided but malformed
        """
        # If all are missing/None, return empty strings for public mode
        if not api_key and not api_secret and not passphrase:
            return "", "", ""

        # Require key and secret to be provided together
        if bool(api_key) != bool(api_secret):
            raise ValueError(
                "Coinbase API key and secret must be provided together. "
                "Either supply both COINBASE_API_KEY and COINBASE_API_SECRET, or neither."
            )

        # Passphrase must align with key/secret usage
        if passphrase and not (api_key and api_secret):
            raise ValueError(
                "Coinbase API passphrase provided without key/secret. "
                "Provide COINBASE_API_KEY, COINBASE_API_SECRET, and COINBASE_API_PASSPHRASE together."
            )
        if api_key and api_secret and not passphrase:
            if allow_test_credentials:
                logger.debug("Coinbase provider allowing missing passphrase for test environment")
                passphrase = ""
            else:
                raise ValueError(
                    "Coinbase credentials must include COINBASE_API_PASSPHRASE when providing API key and secret"
                )

        # Validate credential format (reasonable minimum length)
        if api_key and len(str(api_key).strip()) < 20:
            if allow_test_credentials:
                logger.debug("Coinbase provider allowing short API key for test environment")
            else:
                raise ValueError(
                    f"Invalid COINBASE_API_KEY format (too short: {len(str(api_key))} chars)"
                )
        if api_secret and len(str(api_secret).strip()) < 20:
            if allow_test_credentials:
                logger.debug("Coinbase provider allowing short API secret for test environment")
            else:
                raise ValueError(
                    f"Invalid COINBASE_API_SECRET format (too short: {len(str(api_secret))} chars)"
                )
        if passphrase and len(str(passphrase).strip()) < 5:
            if allow_test_credentials:
                logger.debug("Coinbase provider allowing short API passphrase for test environment")
            else:
                raise ValueError(
                    f"Invalid COINBASE_API_PASSPHRASE format (too short: {len(str(passphrase))} chars)"
                )

        return (
            str(api_key).strip() if api_key else "",
            str(api_secret).strip() if api_secret else "",
            str(passphrase).strip() if passphrase else "",
        )

    # ---------------------- ExchangeInterface ---------------------
    def _initialize_client(self):
        """Initialize Coinbase client & prepare auth parameters."""
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "CoinbaseProvider/1.0"})
        # Prepare decoded secret for HMAC
        try:
            self._decoded_secret = base64.b64decode(self.api_secret) if self.api_secret else None
        except Exception:
            self._decoded_secret = None

    def close(self) -> None:
        """Close the HTTP session and release resources."""
        if hasattr(self, "_session") and self._session is not None:
            try:
                self._session.close()
            except Exception:
                pass  # Ignore errors during cleanup
            self._session = None

    def __del__(self) -> None:
        """Destructor to ensure session is closed when provider is garbage collected."""
        self.close()

    @staticmethod
    def _safe_calculate_average_price(executed_value: Any, filled_size: Any) -> float | None:
        """Safely calculate average price with validation to prevent NaN/inf propagation."""
        import math

        try:
            exec_val = float(executed_value)
            fill_sz = float(filled_size)

            # Validate both values are finite before division
            if not math.isfinite(exec_val) or not math.isfinite(fill_sz):
                return None

            if fill_sz > 0:
                avg_price = exec_val / fill_sz
                # Validate result is finite
                return avg_price if math.isfinite(avg_price) else None
            return None
        except (TypeError, ValueError):
            return None

    # ------------------------------------------------------------
    # The following methods provide implementations for authenticated endpoints of the Coinbase Exchange API.
    # These methods interact with the API to fetch account information and balances.
    # ------------------------------------------------------------

    def _sign_request(self, timestamp: str, method: str, request_path: str, body: str = "") -> str:
        """Create CB-ACCESS-SIGN header value"""
        if not self._decoded_secret:
            raise ValueError("API secret not configured – cannot sign requests")
        message = f"{timestamp}{method.upper()}{request_path}{body}".encode()
        signature = hmac.new(self._decoded_secret, message, hashlib.sha256).digest()
        return base64.b64encode(signature).decode()

    def _request(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        body: dict[str, Any] | None = None,
        auth: bool = False,
    ):
        """Helper to perform HTTP request with optional Coinbase authentication."""
        url = f"{self.BASE_URL}{path}"
        body_str = json.dumps(body) if body else ""
        headers = {}
        if auth:
            if not (self.api_key and self.api_secret and self.passphrase):
                raise ValueError("Authenticated request requested but API credentials not set")
            timestamp = str(int(time.time()))
            signature = self._sign_request(timestamp, method, path, body_str)
            headers.update(
                {
                    "CB-ACCESS-KEY": self.api_key,
                    "CB-ACCESS-SIGN": signature,
                    "CB-ACCESS-TIMESTAMP": timestamp,
                    "CB-ACCESS-PASSPHRASE": self.passphrase,
                    "Content-Type": "application/json",
                }
            )
        # Use retry decorator for network resilience
        @with_network_retry(max_retries=3, base_delay=1.0, max_delay=30.0)
        def _make_request() -> dict[str, Any]:
            response = self._session.request(
                method,
                url,
                params=params,
                data=body_str if body else None,
                headers=headers,
                timeout=15,
            )
            response.raise_for_status()
            if response.text:
                return response.json()
            return {}

        try:
            return _make_request()
        except Exception as e:
            logger.error(f"Coinbase API request error {method} {path} after retries: {e}")
            raise

    def test_connection(self) -> bool:
        try:
            r = self._session.get(f"{self.BASE_URL}/time", timeout=10)
            return r.status_code == 200
        except Exception as e:
            logger.error(f"Coinbase connection test failed: {e}")
            return False

    def get_account_info(self) -> dict[str, Any]:
        try:
            data = self._request("GET", "/accounts", auth=True)
            account_info = {
                "total_accounts": len(data),
                "timestamp": datetime.now(UTC),
            }
            return account_info
        except Exception:
            return {}

    def get_balances(self) -> list[AccountBalance]:
        try:
            accounts = self._request("GET", "/accounts", auth=True)
            balances: list[AccountBalance] = []
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
                        last_updated=datetime.now(UTC),
                    )
                )
            return balances
        except Exception as e:
            logger.error(f"Failed to fetch balances: {e}")
            return []

    def get_balance(self, asset: str) -> AccountBalance | None:
        balances = self.get_balances()
        for bal in balances:
            if bal.asset.upper() == asset.upper():
                return bal
        return None

    def get_positions(self, symbol: str | None = None) -> list[Position]:
        logger.info("get_positions not implemented for Coinbase spot API (only holdings)")
        return []

    def get_open_orders(self, symbol: str | None = None) -> list[Order]:
        try:
            params = {"status": "open"}
            if symbol:
                params["product_id"] = SymbolFactory.to_exchange_symbol(symbol, "coinbase")
            data = self._request("GET", "/orders", params=params, auth=True)
            orders: list[Order] = []
            for od in data:
                orders.append(
                    Order(
                        order_id=od.get("id"),
                        symbol=od.get("product_id"),
                        side=OrderSide.BUY if od.get("side") == "buy" else OrderSide.SELL,
                        order_type=self._convert_order_type(od.get("type")),
                        quantity=float(od.get("size", 0)),
                        price=float(od.get("price")) if od.get("price") else None,
                        status=self._convert_order_status(od.get("status"), od.get("done_reason")),
                        filled_quantity=float(od.get("filled_size", 0)),
                        average_price=(
                            self._safe_calculate_average_price(
                                od.get("executed_value", 0), od.get("filled_size", 0)
                            )
                        ),
                        commission=0.0,
                        commission_asset="",
                        create_time=datetime.fromisoformat(od.get("created_at")),
                        update_time=(
                            datetime.fromisoformat(od.get("done_at"))
                            if od.get("done_at")
                            else datetime.now(UTC)
                        ),
                        stop_price=float(od.get("stop_price")) if od.get("stop_price") else None,
                        time_in_force=od.get("time_in_force", "GTC"),
                    )
                )
            return orders
        except Exception as e:
            logger.error(f"Failed to get open orders: {e}")
            return []

    def get_order(self, order_id: str, symbol: str) -> Order | None:
        try:
            od = self._request("GET", f"/orders/{order_id}", auth=True)
            return Order(
                order_id=od.get("id"),
                symbol=od.get("product_id"),
                side=OrderSide.BUY if od.get("side") == "buy" else OrderSide.SELL,
                order_type=self._convert_order_type(od.get("type")),
                quantity=float(od.get("size", 0)),
                price=float(od.get("price")) if od.get("price") else None,
                status=self._convert_order_status(od.get("status"), od.get("done_reason")),
                filled_quantity=float(od.get("filled_size", 0)),
                average_price=(
                    (float(od.get("executed_value", 0)) / float(od.get("filled_size", 0)))
                    if float(od.get("filled_size", 0)) > 0
                    else None
                ),
                commission=0.0,
                commission_asset="",
                create_time=datetime.fromisoformat(od.get("created_at")),
                update_time=(
                    datetime.fromisoformat(od.get("done_at"))
                    if od.get("done_at")
                    else datetime.now(UTC)
                ),
                stop_price=float(od.get("stop_price")) if od.get("stop_price") else None,
                time_in_force=od.get("time_in_force", "GTC"),
            )
        except Exception as e:
            logger.error(f"Failed to get order {order_id}: {e}")
            return None

    def get_recent_trades(self, symbol: str, limit: int = 100) -> list[Trade]:
        try:
            params = {
                "product_id": SymbolFactory.to_exchange_symbol(symbol, "coinbase"),
                "limit": limit,
            }
            fills = self._request("GET", "/fills", params=params, auth=True)
            trades: list[Trade] = []
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
        price: float | None = None,
        stop_price: float | None = None,
        time_in_force: str = "GTC",
        client_order_id: str | None = None,
    ) -> str | None:
        """
        Place an order on Coinbase Advanced Trade API.

        Note: Coinbase Advanced Trade API supports client_order_id for idempotency.
        """
        try:
            cb_type = self._convert_to_cb_type(order_type)
            body: dict[str, Any] = {
                "product_id": SymbolFactory.to_exchange_symbol(symbol, "coinbase"),
                "side": side.value.lower(),
                "type": cb_type,
            }
            if cb_type == "market":
                body["size"] = str(quantity)
            else:
                body.update(
                    {
                        "size": str(quantity),
                        "price": str(price) if price else None,
                        "time_in_force": time_in_force,
                    }
                )
            if cb_type == "stop":
                body["stop_price"] = str(stop_price) if stop_price else None
                body["stop"] = "loss"  # default stop loss

            # Add client_order_id for idempotency if provided
            if client_order_id:
                body["client_order_id"] = client_order_id

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

    def cancel_all_orders(self, symbol: str | None = None) -> bool:
        try:
            params = (
                {"product_id": SymbolFactory.to_exchange_symbol(symbol, "coinbase")}
                if symbol
                else None
            )
            self._request("DELETE", "/orders", params=params, auth=True)
            return True
        except Exception as e:
            logger.error(f"Failed to cancel all orders: {e}")
            return False

    def place_stop_loss_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        stop_price: float,
        limit_price: float | None = None,
    ) -> str | None:
        """
        Place a server-side stop-loss order on Coinbase.

        Note: Coinbase Advanced Trade API supports stop orders.
        This is a placeholder implementation - full Coinbase stop order
        support would require Advanced Trade API integration.
        """
        logger.warning(
            "Coinbase stop-loss orders not fully implemented - "
            "positions may not be protected if bot goes offline"
        )
        # TODO: Implement Coinbase Advanced Trade API stop orders
        return None

    def get_symbol_info(self, symbol: str) -> dict[str, Any] | None:
        try:
            product = self._request(
                "GET", f"/products/{SymbolFactory.to_exchange_symbol(symbol, 'coinbase')}"
            )
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

    def _fetch_candles(
        self,
        product_id: str,
        granularity: int,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int | None = None,
    ) -> list[list[Any]]:
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

    def _fetch_candles_range(
        self, product_id: str, granularity: int, start: datetime, end: datetime
    ) -> list[list[Any]]:
        """Fetch candles over a range, respecting Coinbase 300-candle limit per request."""
        max_points = 300
        delta_sec = granularity * max_points
        results: list[list[Any]] = []
        chunk_start = start
        while chunk_start < end:
            chunk_end = min(end, chunk_start + timedelta(seconds=delta_sec))
            results.extend(self._fetch_candles(product_id, granularity, chunk_start, chunk_end))
            # API returns inclusive candles → move start forward by granularity to avoid duplicates
            chunk_start = chunk_end + timedelta(seconds=granularity)
        return results

    def get_historical_data(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        try:
            granularity = self._convert_timeframe(timeframe)
            product_id = SymbolFactory.to_exchange_symbol(symbol, "coinbase")
            candles = self._fetch_candles_range(
                product_id, granularity, start, end or datetime.now(UTC)
            )
            df = self._process_ohlcv(candles, timestamp_unit="s")
            self.data = df
            if not df.empty:
                logger.info(
                    f"Fetched {len(df)} candles for {symbol} from {df.index.min()} to {df.index.max()}"
                )
            return df
        except Exception as e:
            logger.error(f"Error fetching historical data from Coinbase: {e}")
            raise

    def get_live_data(self, symbol: str, timeframe: str, limit: int = 100) -> pd.DataFrame:
        try:
            granularity = self._convert_timeframe(timeframe)
            product_id = SymbolFactory.to_exchange_symbol(symbol, "coinbase")
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
            product_id = SymbolFactory.to_exchange_symbol(symbol, "coinbase")
            candles = self._fetch_candles(product_id, granularity, limit=1)
            if not candles:
                return self.data
            latest_df = self._process_ohlcv(candles, timestamp_unit="s")
            # Merge with existing
            self.data = pd.concat(
                [self.data[~self.data.index.isin(latest_df.index)], latest_df]
            ).sort_index()
            return self.data
        except Exception as e:
            logger.error(f"Error updating live data for Coinbase: {e}")
            raise

    def get_current_price(self, symbol: str) -> float:
        """Get latest price for a symbol.

        Raises:
            RuntimeError: If price cannot be fetched from exchange.
                         Caller must handle this to prevent trading with invalid prices.
        """
        try:
            product_id = SymbolFactory.to_exchange_symbol(symbol, "coinbase")
            url = f"{self.BASE_URL}/products/{product_id}/ticker"
            r = self._session.get(url, timeout=10)
            r.raise_for_status()
            data = r.json()
            price = float(data["price"])
            # Validate price is positive to prevent downstream calculation errors
            if price <= 0:
                raise ValueError(f"Invalid price {price} <= 0 for {symbol}")
            return price
        except Exception as e:
            logger.error(f"Error fetching current price for {symbol} from Coinbase: {e}")
            # Don't return 0.0 - that could cause division by zero or infinite position sizes
            # Force caller to handle price fetch failures explicitly
            raise RuntimeError(
                f"Failed to fetch current price for {symbol} from Coinbase: {e}"
            ) from e

    # --------------------------- DataProvider --------------------------

    def _convert_order_type(self, order_type: str) -> OrderType:
        """Convert Coinbase order type to internal OrderType enum."""
        mapping = {
            "market": OrderType.MARKET,
            "limit": OrderType.LIMIT,
            "stop": OrderType.STOP_LOSS,
        }
        return mapping.get(order_type, OrderType.MARKET)

    def _convert_order_status(self, status: str, done_reason: str | None = None) -> OrderStatus:
        """Convert Coinbase order status to internal OrderStatus enum.

        Args:
            status: Coinbase order status
            done_reason: For 'done' orders, the reason (filled, cancelled, etc.)
        """
        if status == "open":
            return OrderStatus.PENDING  # Order is live but not filled
        elif status == "pending":
            return OrderStatus.PENDING  # Order submission pending
        elif status == "active":
            return OrderStatus.PARTIALLY_FILLED  # Order partially executed
        elif status == "done":
            # For done orders, check the reason to determine if filled or cancelled
            if done_reason == "filled":
                return OrderStatus.FILLED
            elif done_reason in ["cancelled", "canceled"]:
                return OrderStatus.CANCELLED
            else:
                # Default to filled for unknown done reasons
                return OrderStatus.FILLED
        else:
            return OrderStatus.PENDING

    def _convert_to_cb_type(self, order_type: str) -> str:
        """Convert internal order type to Coinbase order type."""
        mapping = {
            "market": "market",
            "limit": "limit",
            "stop": "stop",
        }
        if isinstance(order_type, OrderType):
            order_type = order_type.value
        return mapping.get(order_type, "market")


# Aliases for convenience
CoinbaseDataProvider = CoinbaseProvider
CoinbaseExchange = CoinbaseProvider
