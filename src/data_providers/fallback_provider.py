"""
Fallback Data Provider

Provides automatic failover between Binance and CoinGecko data providers.
Tries Binance first (preferred for granular data), falls back to CoinGecko
if Binance is unavailable or blocked.
"""

import logging
import math
import threading
from datetime import datetime

import pandas as pd

from .binance_provider import BinanceProvider
from .coingecko_provider import CoinGeckoProvider
from .data_provider import DataProvider

logger = logging.getLogger(__name__)


class FallbackProvider(DataProvider):
    """
    Data provider with automatic failover from Binance to CoinGecko.

    Use this provider in environments where Binance may be blocked (e.g., Claude Code web).
    It automatically tries CoinGecko if Binance fails, providing resilience without
    code changes.

    Usage:
        provider = FallbackProvider()
        df = provider.get_historical_data("BTC-USD", "4h", start, end)
        # Tries Binance first, falls back to CoinGecko if needed
    """

    def __init__(
        self,
        binance_api_key: str | None = None,
        binance_api_secret: str | None = None,
        coingecko_api_key: str | None = None,
        testnet: bool = False,
    ):
        """
        Initialize fallback provider with both Binance and CoinGecko.

        Args:
            binance_api_key: Optional Binance API key
            binance_api_secret: Optional Binance API secret
            coingecko_api_key: Optional CoinGecko API key (for higher rate limits)
            testnet: Whether to use testnet for Binance
        """
        super().__init__()

        # Initialize both providers
        self.primary_provider = BinanceProvider(binance_api_key, binance_api_secret, testnet)
        self.fallback_provider = CoinGeckoProvider(coingecko_api_key)

        self.current_provider = "binance"  # Track which provider is active
        self._binance_failed = False  # Flag to skip Binance if it's consistently failing
        self._lock = threading.Lock()  # Protect mutable shared state

        logger.info("FallbackProvider initialized (Binance → CoinGecko failover)")

    def _is_access_denied_error(self, exception: Exception) -> bool:
        """
        Check if an exception indicates access denied (403 Forbidden).

        Checks HTTP status codes first, falls back to string matching.

        Args:
            exception: The exception to check

        Returns:
            True if exception indicates 403/access denied
        """
        # Maximum depth to traverse exception chain (prevent infinite loops)
        MAX_EXCEPTION_DEPTH = 10

        # Check for HTTP status code in exception chain
        current_exception = exception
        depth = 0
        while current_exception is not None and depth < MAX_EXCEPTION_DEPTH:
            # Check for requests.HTTPError with status code
            if hasattr(current_exception, "response") and hasattr(
                current_exception.response, "status_code"
            ):
                if current_exception.response.status_code == 403:
                    return True

            # Check for BinanceAPIException with status code
            if hasattr(current_exception, "status_code"):
                if current_exception.status_code == 403:
                    return True

            # Move to next exception in chain
            current_exception = (
                current_exception.__cause__ if current_exception.__cause__ else current_exception.__context__
            )
            depth += 1

        # Fall back to string matching as last resort
        error_msg = str(exception).lower()
        return "403" in error_msg or "forbidden" in error_msg or "access denied" in error_msg

    def _normalize_symbol(self, symbol: str, target_provider: str) -> str:
        """
        Convert symbol format for the target provider.

        Args:
            symbol: Input symbol (e.g., 'BTC-USD', 'BTCUSDT')
            target_provider: 'binance' or 'coingecko'

        Returns:
            Symbol in correct format for target provider
        """
        if target_provider == "binance":
            # Binance uses BTCUSDT format
            return symbol.replace("-USD", "USDT").replace("-", "")
        else:
            # CoinGecko uses BTC-USD format (handled by provider's _convert_symbol)
            return symbol

    def get_historical_data(
        self, symbol: str, timeframe: str, start: datetime, end: datetime | None = None
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data with automatic failover.

        Tries Binance first, falls back to CoinGecko if Binance fails.

        Args:
            symbol: Trading symbol (e.g., 'BTC-USD', 'BTCUSDT')
            timeframe: Timeframe (1m, 5m, 15m, 1h, 4h, 1d)
            start: Start datetime
            end: End datetime (defaults to now)

        Returns:
            DataFrame with OHLCV data from either provider
        """
        # Try Binance first (unless it's already failed)
        with self._lock:
            binance_failed = self._binance_failed

        if not binance_failed:
            try:
                binance_symbol = self._normalize_symbol(symbol, "binance")
                logger.debug("Trying Binance provider for %s %s", binance_symbol, timeframe)

                df = self.primary_provider.get_historical_data(
                    binance_symbol, timeframe, start, end
                )

                if df is not None and not df.empty:
                    with self._lock:
                        self.current_provider = "binance"
                        self.data = df
                    logger.info(
                        "✓ Binance: Fetched %d candles for %s %s",
                        len(df), binance_symbol, timeframe
                    )
                    return df
                else:
                    logger.warning("Binance returned empty data for %s", binance_symbol)
                    raise ValueError("Empty data from Binance")

            except Exception as e:
                # Check if it's a 403/blocking error using HTTP status codes
                if self._is_access_denied_error(e):
                    logger.warning(
                        "Binance blocked (403 Forbidden) - will use CoinGecko for future requests"
                    )
                    with self._lock:
                        self._binance_failed = True
                else:
                    logger.warning("Binance failed: %s", e)

        # Fall back to CoinGecko
        try:
            coingecko_symbol = self._normalize_symbol(symbol, "coingecko")
            logger.info("Falling back to CoinGecko for %s %s", coingecko_symbol, timeframe)

            df = self.fallback_provider.get_historical_data(coingecko_symbol, timeframe, start, end)

            if df is not None and not df.empty:
                with self._lock:
                    self.current_provider = "coingecko"
                    self.data = df
                logger.info(
                    "✓ CoinGecko: Fetched %d candles for %s %s",
                    len(df), coingecko_symbol, timeframe
                )
                return df
            else:
                raise ValueError("Empty data from CoinGecko")

        except Exception as e:
            logger.error("Both Binance and CoinGecko failed for %s: %s", symbol, e)
            raise RuntimeError(
                f"Failed to fetch data from both Binance and CoinGecko: {e}"
            ) from e

    def get_live_data(self, symbol: str, timeframe: str, limit: int = 100) -> pd.DataFrame:
        """
        Fetch recent OHLCV data with automatic failover.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            limit: Number of candles to fetch

        Returns:
            DataFrame with recent OHLCV data
        """
        # Try Binance first (unless it's already failed)
        with self._lock:
            binance_failed = self._binance_failed

        if not binance_failed:
            try:
                binance_symbol = self._normalize_symbol(symbol, "binance")
                logger.debug("Trying Binance for live data: %s %s", binance_symbol, timeframe)

                df = self.primary_provider.get_live_data(binance_symbol, timeframe, limit)

                if df is not None and not df.empty:
                    with self._lock:
                        self.current_provider = "binance"
                        self.data = df
                    return df
                else:
                    raise ValueError("Empty data from Binance")

            except Exception as e:
                # Check if it's a 403/blocking error using HTTP status codes
                if self._is_access_denied_error(e):
                    logger.warning("Binance blocked - using CoinGecko")
                    with self._lock:
                        self._binance_failed = True
                else:
                    logger.warning("Binance failed: %s", e)

        # Fall back to CoinGecko
        try:
            coingecko_symbol = self._normalize_symbol(symbol, "coingecko")
            logger.info("Using CoinGecko for live data: %s %s", coingecko_symbol, timeframe)

            df = self.fallback_provider.get_live_data(coingecko_symbol, timeframe, limit)

            if df is not None and not df.empty:
                with self._lock:
                    self.current_provider = "coingecko"
                    self.data = df
                return df
            else:
                raise ValueError("Empty data from CoinGecko")

        except Exception as e:
            logger.error("Both providers failed for %s: %s", symbol, e)
            raise RuntimeError(f"Failed to fetch live data from both providers: {e}") from e

    def update_live_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """
        Update the latest candle data with automatic failover.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe

        Returns:
            Updated DataFrame
        """
        # Use whichever provider is currently active
        with self._lock:
            current_provider = self.current_provider
            binance_failed = self._binance_failed

        if current_provider == "binance" and not binance_failed:
            try:
                binance_symbol = self._normalize_symbol(symbol, "binance")
                df = self.primary_provider.update_live_data(binance_symbol, timeframe)

                if df is not None and not df.empty:
                    with self._lock:
                        self.data = df
                    return df
                else:
                    raise ValueError("Empty data from Binance")

            except Exception as e:
                logger.warning("Binance update failed, switching to CoinGecko: %s", e)
                with self._lock:
                    self._binance_failed = True

        # Use CoinGecko
        try:
            coingecko_symbol = self._normalize_symbol(symbol, "coingecko")
            df = self.fallback_provider.update_live_data(coingecko_symbol, timeframe)

            if df is not None and not df.empty:
                with self._lock:
                    self.current_provider = "coingecko"
                    self.data = df
                return df
            else:
                raise ValueError("Empty data from CoinGecko")

        except Exception as e:
            logger.error("Both providers failed for update: %s", e)
            raise RuntimeError(f"Failed to update data from both providers: {e}") from e

    def get_current_price(self, symbol: str) -> float:
        """
        Get latest price for a symbol with automatic failover.

        Args:
            symbol: Trading symbol

        Returns:
            Current price

        Raises:
            RuntimeError: If both providers fail
        """
        # Try Binance first (unless it's already failed)
        with self._lock:
            binance_failed = self._binance_failed

        if not binance_failed:
            try:
                binance_symbol = self._normalize_symbol(symbol, "binance")
                price = self.primary_provider.get_current_price(binance_symbol)

                # Validate price is positive and finite (reject NaN/Infinity)
                if math.isfinite(price) and price > 0:
                    return price
                else:
                    raise ValueError(f"Invalid price from Binance: {price}")

            except Exception as e:
                # Check if it's a 403/blocking error using HTTP status codes
                if self._is_access_denied_error(e):
                    logger.warning("Binance blocked - using CoinGecko for prices")
                    with self._lock:
                        self._binance_failed = True
                else:
                    logger.warning("Binance price fetch failed: %s", e)

        # Fall back to CoinGecko
        try:
            coingecko_symbol = self._normalize_symbol(symbol, "coingecko")
            price = self.fallback_provider.get_current_price(coingecko_symbol)

            # Validate price is positive and finite (reject NaN/Infinity)
            if math.isfinite(price) and price > 0:
                return price
            else:
                raise ValueError(f"Invalid price from CoinGecko: {price}")

        except Exception as e:
            logger.error("Both providers failed to get price for %s: %s", symbol, e)
            raise RuntimeError(f"Failed to fetch price from both providers: {e}") from e

    def close(self) -> None:
        """Close both provider connections."""
        try:
            self.fallback_provider.close()
        except Exception as e:
            logger.warning("Error closing CoinGecko provider: %s", e)

        # BinanceProvider doesn't have close() method, but we can clean up if needed
        try:
            if hasattr(self.primary_provider, "close"):
                self.primary_provider.close()
        except Exception as e:
            logger.warning("Error closing Binance provider: %s", e)

    def __del__(self) -> None:
        """Destructor to ensure connections are closed."""
        self.close()
