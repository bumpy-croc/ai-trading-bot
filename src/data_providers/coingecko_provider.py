"""
CoinGecko Data Provider

Provides historical and live cryptocurrency price data from CoinGecko API.
Works reliably from Claude Code web servers where Binance may be blocked.

API Documentation: https://docs.coingecko.com/reference/introduction
"""

import logging
import time
from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd
import requests

from src.infrastructure.network_retry import with_network_retry

from .data_provider import DataProvider

logger = logging.getLogger(__name__)


class CoinGeckoProvider(DataProvider):
    """
    CoinGecko data provider for cryptocurrency price data.

    Free tier limits: 30 calls/minute
    Rate limit strategy: Automatic retry with exponential backoff
    """

    BASE_URL = "https://api.coingecko.com/api/v3"

    # Mapping of standard symbols to CoinGecko coin IDs
    SYMBOL_MAPPING = {
        "BTC-USD": "bitcoin",
        "BTCUSDT": "bitcoin",
        "BTC": "bitcoin",
        "ETH-USD": "ethereum",
        "ETHUSDT": "ethereum",
        "ETH": "ethereum",
        "BNB-USD": "binancecoin",
        "BNBUSDT": "binancecoin",
        "BNB": "binancecoin",
        "XRP-USD": "ripple",
        "XRPUSDT": "ripple",
        "XRP": "ripple",
        "ADA-USD": "cardano",
        "ADAUSDT": "cardano",
        "ADA": "cardano",
        "SOL-USD": "solana",
        "SOLUSDT": "solana",
        "SOL": "solana",
        "DOGE-USD": "dogecoin",
        "DOGEUSDT": "dogecoin",
        "DOGE": "dogecoin",
    }

    # CoinGecko OHLC granularity mapping (days parameter)
    # 1 day = 30-minute candles
    # 7-30 days = 4-hour candles
    # 31-90 days = 4-hour candles
    # 90+ days = 4-day candles
    TIMEFRAME_MAPPING = {
        "1m": None,  # Not supported by CoinGecko OHLC
        "5m": None,  # Not supported
        "15m": None,  # Not supported
        "30m": 1,  # Use 1 day (gives 30m candles)
        "1h": 7,  # Use 7 days (gives 4h candles, we'll need to aggregate)
        "4h": 7,  # Use 7 days (gives 4h candles)
        "1d": 90,  # Use 90 days (gives daily candles)
    }

    def __init__(self, api_key: str | None = None):
        """
        Initialize CoinGecko provider.

        Args:
            api_key: Optional CoinGecko API key for higher rate limits (Pro tier)
        """
        super().__init__()
        self.api_key = api_key
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "ai-trading-bot/1.0"})
        self._last_request_time = 0.0  # Track last request for rate limiting

        if self.api_key:
            self._session.headers.update({"x-cg-pro-api-key": self.api_key})
            logger.info("CoinGecko provider initialized with API key (Pro tier)")
        else:
            logger.info("CoinGecko provider initialized (Free tier: 30 calls/minute)")

    def close(self) -> None:
        """Close the HTTP session and release resources."""
        if hasattr(self, "_session") and self._session is not None:
            try:
                self._session.close()
            except Exception:
                pass
            self._session = None

    def __del__(self) -> None:
        """Destructor to ensure session is closed."""
        self.close()

    def _convert_symbol(self, symbol: str) -> str:
        """Convert trading symbol to CoinGecko coin ID."""
        # Normalize symbol
        symbol = symbol.upper().replace("USDT", "-USD")

        if symbol in self.SYMBOL_MAPPING:
            return self.SYMBOL_MAPPING[symbol]

        # Try to extract base currency
        for separator in ["-", "/"]:
            if separator in symbol:
                base = symbol.split(separator)[0]
                if base in self.SYMBOL_MAPPING:
                    return self.SYMBOL_MAPPING[base]

        # Fallback: assume symbol is the coin name
        logger.warning(
            f"Symbol {symbol} not in mapping, using as-is. "
            f"This may fail if not a valid CoinGecko ID."
        )
        return symbol.lower()

    @with_network_retry(max_retries=3, base_delay=1.0, max_delay=30.0)
    def _request(self, endpoint: str, params: dict[str, Any] | None = None) -> Any:
        """
        Make a request to CoinGecko API with retry logic.

        Args:
            endpoint: API endpoint (e.g., '/coins/bitcoin')
            params: Query parameters

        Returns:
            JSON response data

        Raises:
            requests.HTTPError: If request fails after retries
        """
        # Rate limiting: free tier allows 30 calls/minute = 1 call per 2 seconds
        # Add a 2.5 second delay to be safe
        if not self.api_key:
            current_time = time.time()
            time_since_last = current_time - self._last_request_time
            if time_since_last < 2.5:
                sleep_time = 2.5 - time_since_last
                logger.debug(f"Rate limiting: sleeping {sleep_time:.1f}s")
                time.sleep(sleep_time)
            self._last_request_time = time.time()

        url = f"{self.BASE_URL}{endpoint}"
        response = self._session.get(url, params=params or {}, timeout=15)

        # Handle rate limiting
        if response.status_code == 429:
            retry_after = int(response.headers.get("Retry-After", 60))
            logger.warning(f"Rate limited by CoinGecko. Waiting {retry_after}s before retry...")
            time.sleep(retry_after)
            # Now retry the request
            response = self._session.get(url, params=params or {}, timeout=15)
            response.raise_for_status()

        response.raise_for_status()
        return response.json()

    def get_historical_data(
        self, symbol: str, timeframe: str, start: datetime, end: datetime | None = None
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data from CoinGecko.

        Note: CoinGecko OHLC endpoint returns [timestamp, open, high, low, close]
        without volume. We use market_chart/range for volume data.

        Args:
            symbol: Trading symbol (e.g., 'BTC-USD', 'BTCUSDT')
            timeframe: Timeframe (4h, 1d supported; others limited by CoinGecko)
            start: Start datetime
            end: End datetime (defaults to now)

        Returns:
            DataFrame with OHLCV data
        """
        try:
            coin_id = self._convert_symbol(symbol)
            end = end or datetime.now(UTC)

            # Calculate days parameter for CoinGecko
            days_diff = (end - start).days

            # Fetch OHLC data
            logger.info(f"Fetching {coin_id} OHLC data for {days_diff} days")
            ohlc_data = self._request(
                f"/coins/{coin_id}/ohlc", params={"vs_currency": "usd", "days": days_diff}
            )

            # Fetch volume data from market_chart/range
            from_ts = int(start.timestamp())
            to_ts = int(end.timestamp())
            market_data = self._request(
                f"/coins/{coin_id}/market_chart/range",
                params={"vs_currency": "usd", "from": from_ts, "to": to_ts},
            )

            # Process OHLC data
            if not ohlc_data:
                logger.warning(f"No OHLC data returned for {coin_id}")
                return pd.DataFrame()

            # Convert OHLC format: [timestamp_ms, open, high, low, close]
            df = self._process_ohlcv(
                [[row[0], row[1], row[2], row[3], row[4], 0] for row in ohlc_data],
                timestamp_unit="ms",
            )

            # Merge volume data from market_chart
            if market_data and "total_volumes" in market_data:
                volume_df = pd.DataFrame(
                    market_data["total_volumes"], columns=["timestamp", "volume"]
                )
                volume_df["timestamp"] = pd.to_datetime(volume_df["timestamp"], unit="ms", utc=True)
                volume_df.set_index("timestamp", inplace=True)

                # Align volume with OHLC timestamps (use nearest match)
                if len(volume_df) > 0:
                    # Use merge_asof for efficient nearest-match joining
                    # Both must be timezone-aware for compatibility
                    df_for_merge = df.reset_index()
                    volume_for_merge = volume_df.reset_index()

                    # Ensure timestamp columns have same dtype (both UTC-aware)
                    if df_for_merge["timestamp"].dt.tz is None:
                        df_for_merge["timestamp"] = df_for_merge["timestamp"].dt.tz_localize(UTC)

                    df_merged = pd.merge_asof(
                        df_for_merge.sort_values("timestamp"),
                        volume_for_merge.sort_values("timestamp"),
                        on="timestamp",
                        direction="nearest",
                        suffixes=("", "_vol"),
                    )
                    df_merged.set_index("timestamp", inplace=True)

                    # Update volume column
                    if "volume_vol" in df_merged.columns:
                        df_merged["volume"] = df_merged["volume_vol"]
                        df_merged.drop(columns=["volume_vol"], inplace=True)

                    df = df_merged

            self.data = df

            if len(df) > 0:
                logger.info(
                    f"Fetched {len(df)} candles for {coin_id} from {df.index.min()} to {df.index.max()}"
                )
            else:
                logger.warning(f"No data returned for {coin_id} from {start} to {end}")

            return df

        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            raise

    def get_live_data(self, symbol: str, timeframe: str, limit: int = 100) -> pd.DataFrame:
        """
        Fetch recent OHLCV data.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            limit: Number of candles to fetch

        Returns:
            DataFrame with recent OHLCV data
        """
        try:
            # Calculate appropriate days parameter based on limit and timeframe
            timeframe_hours = {
                "1h": 1,
                "4h": 4,
                "1d": 24,
            }
            hours = timeframe_hours.get(timeframe, 4)
            days = max(1, (limit * hours) // 24)

            end = datetime.now(UTC)
            start = end - timedelta(days=days)

            df = self.get_historical_data(symbol, timeframe, start, end)

            # Limit to requested number of candles
            if len(df) > limit:
                df = df.tail(limit)

            self.data = df
            return df

        except Exception as e:
            logger.error(f"Error fetching live data for {symbol}: {e}")
            raise

    def update_live_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """
        Update the latest candle data.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe

        Returns:
            Updated DataFrame
        """
        try:
            # Fetch latest candle
            latest = self.get_live_data(symbol, timeframe, limit=1)

            if latest.empty:
                return self.data if self.data is not None else pd.DataFrame()

            # Update or append to existing data
            if self.data is not None and not self.data.empty:
                self.data = pd.concat(
                    [self.data[~self.data.index.isin(latest.index)], latest]
                ).sort_index()
            else:
                self.data = latest

            return self.data

        except Exception as e:
            logger.error(f"Error updating live data for {symbol}: {e}")
            raise

    def get_current_price(self, symbol: str) -> float:
        """
        Get latest price for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Current price in USD

        Raises:
            RuntimeError: If price cannot be fetched
        """
        try:
            coin_id = self._convert_symbol(symbol)
            data = self._request(
                "/simple/price", params={"ids": coin_id, "vs_currencies": "usd"}
            )

            if coin_id not in data or "usd" not in data[coin_id]:
                raise ValueError(f"No price data for {coin_id}")

            price = float(data[coin_id]["usd"])

            # Validate price is positive
            if price <= 0:
                raise ValueError(f"Invalid price {price} <= 0 for {symbol}")

            return price

        except Exception as e:
            logger.error(f"Error fetching current price for {symbol}: {e}")
            raise RuntimeError(f"Failed to fetch current price for {symbol}: {e}") from e


# Alias for convenience
CoinGeckoDataProvider = CoinGeckoProvider
