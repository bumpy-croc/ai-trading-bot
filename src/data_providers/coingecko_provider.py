"""
CoinGecko Data Provider

Provides historical and live cryptocurrency price data from CoinGecko API.
Works reliably from Claude Code web servers where Binance may be blocked.

For historical data beyond the CoinGecko free tier limit (365 days),
falls back to Binance public data archives at data.binance.vision.

API Documentation: https://docs.coingecko.com/reference/introduction
"""

import io
import logging
import math
import time
import zipfile
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
import requests

from src.config import get_config
from src.config.constants import DEFAULT_DATA_FETCH_TIMEOUT
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

    # Rate limiting: free tier allows 30 calls/minute = 1 call per 2 seconds
    # Use 2.5 seconds to be safe and prevent 429 errors
    RATE_LIMIT_DELAY_SECONDS = 2.5

    # CoinGecko OHLC endpoint only accepts these specific day values
    VALID_OHLC_DAYS = [1, 7, 14, 30, 90, 180, 365]

    # Chunk size for market_chart/range: <90 days gives hourly granularity
    MARKET_CHART_CHUNK_DAYS = 85

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

        # Configurable timeout via DATA_FETCH_TIMEOUT_SECONDS env var
        config = get_config()
        self._timeout = float(
            config.get("DATA_FETCH_TIMEOUT_SECONDS", DEFAULT_DATA_FETCH_TIMEOUT)
        )

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
            except Exception as e:
                logger.debug("Error closing session: %s", e)
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
            "Symbol %s not in mapping, using as-is. This may fail if not a valid CoinGecko ID.",
            symbol
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
        # Rate limiting: free tier requires delay between requests
        if not self.api_key:
            current_time = time.time()
            time_since_last = current_time - self._last_request_time
            if time_since_last < self.RATE_LIMIT_DELAY_SECONDS:
                sleep_time = self.RATE_LIMIT_DELAY_SECONDS - time_since_last
                logger.debug("Rate limiting: sleeping %.1fs", sleep_time)
                time.sleep(sleep_time)
            self._last_request_time = time.time()

        url = f"{self.BASE_URL}{endpoint}"
        response = self._session.get(url, params=params or {}, timeout=self._timeout)

        # Raise for status - network_retry decorator will handle 429 and other errors
        response.raise_for_status()
        return response.json()

    # Maximum days supported by CoinGecko free tier market_chart endpoint
    COINGECKO_FREE_MAX_DAYS = 365

    # Binance public data archives URL
    BINANCE_DATA_URL = "https://data.binance.vision/data/spot/monthly/klines"

    # Mapping from our symbol format to Binance symbol format
    BINANCE_SYMBOL_MAPPING = {
        "bitcoin": "BTCUSDT",
        "ethereum": "ETHUSDT",
        "binancecoin": "BNBUSDT",
        "ripple": "XRPUSDT",
        "cardano": "ADAUSDT",
        "solana": "SOLUSDT",
        "dogecoin": "DOGEUSDT",
    }

    # Binance timeframe mapping
    BINANCE_TIMEFRAME_MAPPING = {
        "1h": "1h",
        "4h": "4h",
        "1d": "1d",
    }

    def get_historical_data(
        self, symbol: str, timeframe: str, start: datetime, end: datetime | None = None
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data.

        For recent data (within 365 days of now), uses CoinGecko market_chart API.
        For older historical data, falls back to Binance public data archives
        (data.binance.vision) which provides accurate OHLCV data without API keys.

        Args:
            symbol: Trading symbol (e.g., 'BTC-USD', 'BTCUSDT')
            timeframe: Timeframe (1h, 4h, 1d supported)
            start: Start datetime
            end: End datetime (defaults to now)

        Returns:
            DataFrame with OHLCV data
        """
        try:
            coin_id = self._convert_symbol(symbol)
            end = end or datetime.now(UTC)
            days_diff = max(1, (end - start).days)

            now = datetime.now(UTC)
            days_from_now = max(1, (now - start).days)

            # Determine data source based on how far back we need to go
            if days_from_now <= self.COINGECKO_FREE_MAX_DAYS:
                # Recent data: use CoinGecko market_chart
                logger.info(
                    "Fetching %s data for %d days (%s to %s) via CoinGecko market_chart",
                    coin_id, days_diff, start.date(), end.date(),
                )
                df = self._fetch_via_coingecko_market_chart(coin_id, start, end, timeframe)
            else:
                # Historical data: use Binance public archives
                binance_symbol = self.BINANCE_SYMBOL_MAPPING.get(coin_id)
                if binance_symbol:
                    logger.info(
                        "Fetching %s data for %d days (%s to %s) via Binance public archives",
                        binance_symbol, days_diff, start.date(), end.date(),
                    )
                    df = self._fetch_via_binance_archive(
                        binance_symbol, start, end, timeframe
                    )
                else:
                    # Fallback to CoinGecko for unsupported symbols
                    logger.info(
                        "Fetching %s data for %d days (%s to %s) via CoinGecko market_chart",
                        coin_id, days_diff, start.date(), end.date(),
                    )
                    df = self._fetch_via_coingecko_market_chart(
                        coin_id, start, end, timeframe
                    )

            if df is not None and len(df) > 0:
                self.data = df
                logger.info(
                    "Fetched %d candles for %s from %s to %s",
                    len(df), coin_id, df.index.min(), df.index.max(),
                )
            else:
                logger.warning(
                    "No data returned for %s from %s to %s", coin_id, start, end
                )
                return pd.DataFrame()

            return df

        except Exception as e:
            logger.error("Error fetching historical data for %s: %s", symbol, e)
            raise

    def _fetch_via_coingecko_market_chart(
        self, coin_id: str, start: datetime, end: datetime, timeframe: str
    ) -> pd.DataFrame:
        """
        Fetch data via CoinGecko /market_chart endpoint (relative days from now).

        For <=90 days, returns hourly data which is resampled to target timeframe.
        For >90 days, returns daily data points.

        Args:
            coin_id: CoinGecko coin ID (e.g., 'bitcoin')
            start: Start datetime
            end: End datetime
            timeframe: Target timeframe (1h, 4h, 1d)

        Returns:
            DataFrame with OHLCV data
        """
        now = datetime.now(UTC)
        days_from_now = max(1, min((now - start).days + 1, self.COINGECKO_FREE_MAX_DAYS))

        data = self._request(
            f"/coins/{coin_id}/market_chart",
            params={"vs_currency": "usd", "days": days_from_now},
        )

        if not data or "prices" not in data or not data["prices"]:
            logger.warning("No price data from CoinGecko market_chart for %s", coin_id)
            return pd.DataFrame()

        logger.info(
            "CoinGecko market_chart returned %d price points", len(data["prices"])
        )

        # Build price DataFrame
        price_df = pd.DataFrame(data["prices"], columns=["timestamp", "price"])
        price_df["timestamp"] = pd.to_datetime(
            price_df["timestamp"], unit="ms", utc=True
        )
        price_df.set_index("timestamp", inplace=True)
        price_df = price_df[~price_df.index.duplicated(keep="last")]
        price_df.sort_index(inplace=True)

        # Validate prices
        price_df["price"] = pd.to_numeric(price_df["price"], errors="coerce")
        price_df = price_df[price_df["price"].notna() & (price_df["price"] > 0)]

        # Add volume data
        if "total_volumes" in data and data["total_volumes"]:
            vol_df = pd.DataFrame(
                data["total_volumes"], columns=["timestamp", "volume"]
            )
            vol_df["timestamp"] = pd.to_datetime(
                vol_df["timestamp"], unit="ms", utc=True
            )
            vol_df.set_index("timestamp", inplace=True)
            vol_df = vol_df[~vol_df.index.duplicated(keep="last")]
            vol_df["volume"] = pd.to_numeric(
                vol_df["volume"], errors="coerce"
            ).fillna(0)
            vol_df.loc[vol_df["volume"] < 0, "volume"] = 0
            price_df = price_df.join(vol_df, how="left")
            price_df["volume"] = price_df["volume"].fillna(0)
        else:
            price_df["volume"] = 0.0

        if price_df.empty:
            return pd.DataFrame()

        # Filter to requested date range
        price_df = price_df[(price_df.index >= start) & (price_df.index <= end)]

        # Resample to target timeframe
        return self._resample_to_ohlcv(price_df, timeframe)

    def _fetch_via_binance_archive(
        self, binance_symbol: str, start: datetime, end: datetime, timeframe: str
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data from Binance public data archives.

        Downloads monthly zip files from data.binance.vision which contain
        accurate OHLCV data. No API key is required.

        Args:
            binance_symbol: Binance symbol (e.g., 'BTCUSDT')
            start: Start datetime
            end: End datetime
            timeframe: Target timeframe (1h, 4h, 1d)

        Returns:
            DataFrame with OHLCV data
        """
        binance_tf = self.BINANCE_TIMEFRAME_MAPPING.get(timeframe, "1d")
        all_dfs: list[pd.DataFrame] = []

        # Generate list of year-month pairs to fetch
        current = start.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        while current <= end:
            year_month = current.strftime("%Y-%m")
            url = (
                f"{self.BINANCE_DATA_URL}/{binance_symbol}/{binance_tf}/"
                f"{binance_symbol}-{binance_tf}-{year_month}.zip"
            )

            try:
                response = self._session.get(url, timeout=self._timeout)
                if response.status_code == 200:
                    df_month = self._parse_binance_zip(response.content)
                    if df_month is not None and not df_month.empty:
                        all_dfs.append(df_month)
                        logger.debug(
                            "Downloaded %d candles for %s %s",
                            len(df_month), binance_symbol, year_month,
                        )
                elif response.status_code == 404:
                    logger.debug(
                        "No data available for %s %s (404)", binance_symbol, year_month
                    )
                else:
                    logger.warning(
                        "Failed to fetch %s %s: HTTP %d",
                        binance_symbol, year_month, response.status_code,
                    )
            except Exception as e:
                logger.warning(
                    "Error fetching %s %s: %s", binance_symbol, year_month, e
                )

            # Advance to next month
            if current.month == 12:
                current = current.replace(year=current.year + 1, month=1)
            else:
                current = current.replace(month=current.month + 1)

        if not all_dfs:
            logger.warning("No data from Binance archives for %s", binance_symbol)
            return pd.DataFrame()

        # Combine all monthly DataFrames
        combined = pd.concat(all_dfs, ignore_index=False)
        combined = combined[~combined.index.duplicated(keep="last")]
        combined.sort_index(inplace=True)

        # Filter to requested date range
        combined = combined[
            (combined.index >= start) & (combined.index <= end)
        ]

        logger.info(
            "Binance archives: %d candles for %s (%s to %s)",
            len(combined), binance_symbol,
            combined.index.min() if len(combined) > 0 else "N/A",
            combined.index.max() if len(combined) > 0 else "N/A",
        )

        self.data = combined
        return combined

    @staticmethod
    def _parse_binance_zip(content: bytes) -> pd.DataFrame | None:
        """
        Parse a Binance kline zip archive into a DataFrame.

        Binance CSV format:
        open_time, open, high, low, close, volume, close_time,
        quote_volume, trades, taker_buy_base, taker_buy_quote, ignore

        Args:
            content: Raw zip file bytes

        Returns:
            DataFrame with OHLCV data indexed by timestamp, or None on error
        """
        try:
            z = zipfile.ZipFile(io.BytesIO(content))
            csv_names = [n for n in z.namelist() if n.endswith(".csv")]
            if not csv_names:
                return None

            with z.open(csv_names[0]) as f:
                df = pd.read_csv(
                    f,
                    header=None,
                    usecols=[0, 1, 2, 3, 4, 5],
                    names=["timestamp", "open", "high", "low", "close", "volume"],
                )

            # Convert timestamp from milliseconds to UTC datetime
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df.set_index("timestamp", inplace=True)

            # Convert to numeric
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            # Drop invalid rows
            df = df.dropna(subset=["open", "high", "low", "close"])
            for col in ["open", "high", "low", "close"]:
                df = df[df[col] > 0]

            df["volume"] = df["volume"].fillna(0)

            return df

        except Exception as e:
            logger.warning("Error parsing Binance zip archive: %s", e)
            return None

    def _resample_to_ohlcv(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        Resample hourly price/volume data into OHLCV candles at the target timeframe.

        Args:
            df: DataFrame with 'price' and 'volume' columns, indexed by UTC datetime
            timeframe: Target timeframe (1h, 4h, 1d)

        Returns:
            DataFrame with open, high, low, close, volume columns
        """
        resample_freq_map = {
            "1h": "1h",
            "4h": "4h",
            "1d": "1D",
            "1w": "1W",
        }
        freq = resample_freq_map.get(timeframe, "1D")

        resampled = df.resample(freq).agg(
            {"price": ["first", "max", "min", "last"], "volume": "sum"}
        )
        resampled.columns = ["open", "high", "low", "close", "volume"]

        # Drop rows where we have no price data (gaps)
        resampled = resampled.dropna(subset=["open", "close"])

        # Validate all price columns are positive
        for col in ["open", "high", "low", "close"]:
            resampled = resampled[resampled[col] > 0]

        resampled.index.name = "timestamp"

        logger.info(
            "Resampled %d raw points to %d %s candles",
            len(df), len(resampled), timeframe,
        )

        return resampled

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
            logger.error("Error fetching live data for %s: %s", symbol, e)
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
            logger.error("Error updating live data for %s: %s", symbol, e)
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

            # Validate price is positive and finite (reject NaN/Infinity)
            if not math.isfinite(price) or price <= 0:
                raise ValueError(f"Invalid price {price} for {symbol} (must be positive and finite)")

            return price

        except Exception as e:
            logger.error("Error fetching current price for %s: %s", symbol, e)
            raise RuntimeError(f"Failed to fetch current price for {symbol}: {e}") from e


# Alias for convenience
CoinGeckoDataProvider = CoinGeckoProvider
