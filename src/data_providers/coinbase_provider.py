from typing import Optional, Dict, Any, List
from datetime import datetime
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
        """Initialise Coinbase client. We primarily use public REST calls so keep simple."""
        # Placeholder for real client (e.g., cbpro.PublicClient / AuthenticatedClient).
        # We will simply store session for HTTP requests.
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "CoinbaseProvider/1.0"})

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
        """Not implemented for Coinbase public provider."""
        logger.info("get_account_info not implemented for Coinbase public API")
        return {}

    def get_balances(self) -> List[AccountBalance]:
        logger.info("get_balances not implemented for Coinbase public API")
        return []

    def get_balance(self, asset: str) -> Optional[AccountBalance]:
        logger.info("get_balance not implemented for Coinbase public API")
        return None

    def get_positions(self, symbol: Optional[str] = None) -> List[Position]:
        logger.info("get_positions not implemented for Coinbase spot API (only holdings)")
        return []

    def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        logger.info("get_open_orders not implemented for Coinbase public API")
        return []

    def get_order(self, order_id: str, symbol: str) -> Optional[Order]:
        logger.info("get_order not implemented for Coinbase public API")
        return None

    def get_recent_trades(self, symbol: str, limit: int = 100) -> List[Trade]:
        logger.info("get_recent_trades not implemented for Coinbase public API")
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
        logger.info("place_order not implemented for Coinbase public API")
        return None

    def cancel_order(self, order_id: str, symbol: str) -> bool:
        logger.info("cancel_order not implemented for Coinbase public API")
        return False

    def cancel_all_orders(self, symbol: Optional[str] = None) -> bool:
        logger.info("cancel_all_orders not implemented for Coinbase public API")
        return False

    def get_symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        # Coinbase does not provide rich symbol metadata via a single endpoint like Binance.
        # We'll return basic placeholder info.
        return {
            "symbol": symbol,
            "base_asset": symbol[:-3],
            "quote_asset": symbol[-3:],
        }

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