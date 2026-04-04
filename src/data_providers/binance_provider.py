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
import math
import re
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from enum import Enum
from functools import wraps
from typing import Any, TypeVar

import pandas as pd

from src.config import get_config
from src.config.constants import (
    DEFAULT_DATA_FETCH_TIMEOUT,
    DEFAULT_STARTUP_BAN_MAX_RETRIES,
    DEFAULT_STARTUP_BAN_MAX_WAIT,
    DEFAULT_WS_KLINE_STALENESS_THRESHOLD,
    DEFAULT_WS_RECONNECT_MAX_RETRIES,
)
from src.infrastructure.timeout import TimeoutError as InfraTimeoutError
from src.infrastructure.timeout import run_with_timeout
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
    SideEffectType,
    Trade,
)

logger = logging.getLogger(__name__)

try:
    import nest_asyncio

    nest_asyncio.apply()  # Allow nested run_until_complete for TWM event loop

    from binance import ThreadedWebsocketManager
    from binance.client import Client
    from binance.enums import SIDE_BUY, SIDE_SELL
    from binance.exceptions import BinanceAPIException, BinanceOrderException

    # Ensure ws.protocol.State is accessible (not auto-loaded in websockets 13+)
    import websockets.protocol  # noqa: F401

    BINANCE_AVAILABLE = True
except ImportError:
    # Fallback for environments without binance library
    logger.warning("Binance library not available - using mock implementation")
    Client = None
    BinanceAPIException = Exception
    BinanceOrderException = Exception
    ThreadedWebsocketManager = None
    SIDE_BUY = "BUY"
    SIDE_SELL = "SELL"
    BINANCE_AVAILABLE = False

# Import geo-detection utilities
from src.infrastructure.runtime.geo import get_binance_api_endpoint, is_us_location

# Type variable for generic return type
T = TypeVar("T")

# Rate limit error codes from Binance
RATE_LIMIT_ERROR_CODES = {-1003, -1015}  # -1003: Too many requests, -1015: Too many orders

# Definitive reject codes — exchange explicitly refused the order.
# BinanceOrderException and BinanceAPIException both carry these.
# Treating them as ambiguous (return None) creates phantom positions.
DEFINITIVE_REJECT_CODES = {
    -1013,  # LOT_SIZE / MIN_NOTIONAL filter failure
    -1021,  # TIMESTAMP out of recv window
    -1100,  # Illegal characters in parameter
    -1101,  # Too many parameters
    -1102,  # Mandatory parameter missing
    -1106,  # Parameter not required
    -1111,  # Precision over maximum for asset
    -1116,  # Order type not supported
    -2010,  # NEW_ORDER_REJECTED (insufficient balance, etc.)
    -2013,  # Order does not exist
    -2015,  # Invalid API key / permissions
    # Margin-specific rejects
    -3027,  # Margin account not allowed to trade
    -3028,  # Margin account not allowed to borrow
    -3041,  # Balance not sufficient for margin order
    -3067,  # Cross margin account doesn't exist
}

# Stop-loss limit price slippage to ensure fills
# For sells: limit below stop price, for buys: limit above stop price
STOP_LOSS_LIMIT_SLIPPAGE_FACTOR = 0.005  # 0.5% slippage


_BAN_EXPIRY_PATTERN = re.compile(r"banned until (\d{13})")


def _parse_ban_expiry(error_message: str, now_ms: int | None = None) -> float | None:
    """Extract ban expiry timestamp from Binance -1003 error message.

    Args:
        error_message: The error message string from Binance.
        now_ms: Current time in milliseconds (injectable for testing).

    Returns seconds until ban expires, or None if not parseable.
    """
    match = _BAN_EXPIRY_PATTERN.search(str(error_message))
    if match:
        ban_epoch_ms = int(match.group(1))
        if now_ms is None:
            now_ms = int(time.time() * 1000)
        remaining_s = (ban_epoch_ms - now_ms) / 1000
        return max(remaining_s, 0)
    return None


def with_rate_limit_retry(
    max_retries: int = 6, base_delay: float = 1.0, ban_safe: bool = False
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator to handle rate limits with exponential backoff.

    If the error contains a ban expiry timestamp, waits until the ban
    lifts instead of using blind exponential backoff. With 6 retries
    capped at 60s each, handles bans up to ~6 minutes.

    Args:
        max_retries: Maximum number of retry attempts (default 6 to cover
                     common 5-minute Binance bans at 60s per retry)
        base_delay: Base delay in seconds (doubles each retry)
        ban_safe: If True, only retry transient -1015 errors (too many
                  orders). Raises immediately on -1003 (IP ban) to avoid
                  blocking safety-critical paths like stop-loss placement.

    Returns:
        Decorated function that retries on rate limit errors
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except BinanceAPIException as e:
                    error_code = getattr(e, "code", 0)
                    if error_code in RATE_LIMIT_ERROR_CODES:
                        # In ban_safe mode, only retry -1015 (transient).
                        # Raise -1003 (IP ban) immediately — caller handles it.
                        if ban_safe and error_code == -1003:
                            raise
                        if attempt < max_retries:
                            # Use ban expiry if available, otherwise exponential backoff
                            ban_wait = _parse_ban_expiry(str(e))
                            if ban_wait and ban_wait > 0:
                                # Cap at 60s per retry to stay responsive. If ban is
                                # longer, the next retry will re-parse and wait again.
                                delay = min(ban_wait + 2, 60)
                                logger.warning(
                                    "IP banned for %.0fs, waiting %.0fs before retry "
                                    "(attempt %d/%d)",
                                    ban_wait, delay, attempt + 1, max_retries,
                                )
                            else:
                                delay = base_delay * (2**attempt)
                                logger.warning(
                                    "Rate limited (code %d), retrying in %.1fs "
                                    "(attempt %d/%d)",
                                    error_code, delay, attempt + 1, max_retries,
                                )
                            time.sleep(delay)
                            last_exception = e
                            continue
                    raise
                except Exception:
                    raise
            # All retries exhausted
            if last_exception:
                raise last_exception
            raise RuntimeError(f"Rate limit exceeded after {max_retries} retries")

        return wrapper

    return decorator


class WebSocketState(Enum):
    """Connection state for WebSocket streams."""

    DISCONNECTED = "disconnected"
    PRIMARY = "primary"  # WS active, normal operation
    RESYNCING = "resyncing"  # Gap detected, running REST reconciliation
    REST_DEGRADED = "degraded"  # WS failed, using REST polling
    SUSPENDED = "suspended"  # API ban active, waiting for ban expiry


class BinanceProvider(DataProvider, ExchangeInterface):
    """
    Unified Binance provider that combines data fetching and exchange operations.

    Inherits from both DataProvider and ExchangeInterface to provide complete
    Binance functionality in a single class.
    """

    @property
    def is_margin_mode(self) -> bool:
        """Whether provider is operating in cross-margin mode."""
        return self._use_margin

    TIMEFRAME_MAPPING = {
        "1m": Client.KLINE_INTERVAL_1MINUTE if BINANCE_AVAILABLE else "1m",
        "5m": Client.KLINE_INTERVAL_5MINUTE if BINANCE_AVAILABLE else "5m",
        "15m": Client.KLINE_INTERVAL_15MINUTE if BINANCE_AVAILABLE else "15m",
        "1h": Client.KLINE_INTERVAL_1HOUR if BINANCE_AVAILABLE else "1h",
        "4h": Client.KLINE_INTERVAL_4HOUR if BINANCE_AVAILABLE else "4h",
        "1d": Client.KLINE_INTERVAL_1DAY if BINANCE_AVAILABLE else "1d",
    }

    def __init__(
        self, api_key: str | None = None, api_secret: str | None = None, testnet: bool = False
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

        config = get_config()

        # Margin vs spot account routing (A1)
        account_type = config.get("BINANCE_ACCOUNT_TYPE", "spot") or "spot"
        self._use_margin: bool = account_type.lower() == "margin"
        trading_mode = config.get("TRADING_MODE", "paper") or "paper"
        self._is_live: bool = trading_mode.lower() == "live"
        self._margin_symbol_verified: set[str] = set()

        # Get credentials from config if not provided
        if api_key is None:
            api_key = config.get("BINANCE_API_KEY")
        if api_secret is None:
            api_secret = config.get("BINANCE_API_SECRET")

        env_name = str(config.get("ENV", "")).lower()
        allow_test_credentials = testnet or env_name in {"test", "testing", "ci"}

        # SEC-004 Fix: Validate credentials are properly formatted
        api_key, api_secret = self._validate_credentials(
            api_key, api_secret, allow_test_credentials=allow_test_credentials
        )

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

        # WebSocket stream state (initialized regardless of credentials)
        self._twm: ThreadedWebsocketManager | None = None
        self._twm_lock = threading.Lock()  # Guards _ensure_twm() check-then-set
        self._kline_ws_state = WebSocketState.DISCONNECTED
        self._user_ws_state = WebSocketState.DISCONNECTED
        self._kline_socket_key: str | None = None
        self._user_socket_key: str | None = None
        self._on_kline_cb: Callable | None = None
        self._on_user_event_cb: Callable | None = None
        self._active_symbol: str | None = None
        self._active_timeframe: str | None = None
        self._last_kline_event_time = datetime.now(UTC)
        self._last_user_event_time = datetime.now(UTC)
        self._kline_event_received = False  # True after first kline WS event
        self._user_event_received = False  # True after first user WS event

    @staticmethod
    def _validate_credentials(
        api_key: str | None,
        api_secret: str | None,
        *,
        allow_test_credentials: bool = False,
    ) -> tuple[str, str]:
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
            if allow_test_credentials:
                logger.debug("Binance provider allowing short API key for test environment")
            else:
                raise ValueError(
                    f"Invalid BINANCE_API_KEY format (too short: {len(str(api_key))} chars)"
                )
        if api_secret and len(str(api_secret).strip()) < 20:
            if allow_test_credentials:
                logger.debug("Binance provider allowing short API secret for test environment")
            else:
                raise ValueError(
                    f"Invalid BINANCE_API_SECRET format (too short: {len(str(api_secret))} chars)"
                )

        return (
            str(api_key).strip() if api_key else "",
            str(api_secret).strip() if api_secret else "",
        )

    def _initialize_client(self):
        """Initialize Binance client with geo-aware API selection and error handling"""
        logger.debug("_initialize_client called - BINANCE_AVAILABLE: %s", BINANCE_AVAILABLE)

        if not BINANCE_AVAILABLE:
            if self._use_margin and self._is_live:
                raise RuntimeError(
                    "FATAL: Binance library not available but live margin mode requested. "
                    "Cannot fall back to offline stub — placing dummy margin orders "
                    "risks fund loss. Install python-binance or set BINANCE_ACCOUNT_TYPE=spot."
                )
            logger.warning("Binance library not available - using mock client")
            self._client = self._create_offline_client()
            return

        # Determine which Binance API to use based on location
        api_endpoint = get_binance_api_endpoint()
        is_us = is_us_location()

        logger.info(
            f"Geo-detection result: {'US location' if is_us else 'Non-US location'} - using {api_endpoint} API"
        )

        last_error = None
        deadline = time.monotonic() + DEFAULT_STARTUP_BAN_MAX_WAIT
        for attempt in range(DEFAULT_STARTUP_BAN_MAX_RETRIES + 1):
            try:
                self._attempt_client_init(api_endpoint)
                return  # Success
            except RuntimeError:
                raise  # Margin verification failures propagate immediately
            except Exception as e:
                last_error = e
                remaining_budget = deadline - time.monotonic()
                ban_wait = self._handle_startup_ban(
                    e, attempt, DEFAULT_STARTUP_BAN_MAX_RETRIES, remaining_budget
                )
                if ban_wait is None:
                    break  # Non-ban error or exceeded limits — stop retrying
                logger.warning(
                    "Startup attempt %d/%d: IP banned, waiting %.0fs for ban to lift...",
                    attempt + 1,
                    DEFAULT_STARTUP_BAN_MAX_RETRIES + 1,
                    ban_wait,
                )
                time.sleep(ban_wait)

        # All retries exhausted or non-retryable error
        self._handle_init_failure(last_error, api_endpoint)

    def _attempt_client_init(self, api_endpoint: str):
        """Single attempt to create and verify the Binance client."""
        logger.debug(
            "Attempting to create %s client - has_credentials: %s, testnet: %s",
            api_endpoint,
            bool(self.api_key and self.api_secret),
            self.testnet,
        )

        if self.api_key and self.api_secret:
            logger.debug("Creating authenticated %s client...", api_endpoint)
            if api_endpoint == "binanceus":
                client = Client(
                    self.api_key, self.api_secret, testnet=self.testnet, tld="us"
                )
            else:
                client = Client(self.api_key, self.api_secret, testnet=self.testnet)
        else:
            logger.debug("Creating public %s client...", api_endpoint)
            if api_endpoint == "binanceus":
                client = Client(tld="us")
            else:
                client = Client()

        auth_mode = "with credentials" if self.api_key and self.api_secret else "public mode"
        logger.info(
            "%s client initialized successfully (%s, testnet: %s)",
            api_endpoint.title(),
            auth_mode,
            self.testnet,
        )

        logger.debug("Testing client with server time request...")
        test_response = client.get_server_time()
        logger.debug("Server time test successful: %s", test_response)

        # Only promote to self._client after all verification passes
        self._client = client
        if self._use_margin:
            self._verify_margin_account()

    @staticmethod
    def _handle_startup_ban(
        error: Exception,
        attempt: int,
        max_retries: int,
        max_wait: float,
    ) -> float | None:
        """Check if a startup error is a retryable IP ban.

        Returns seconds to wait, or None if not retryable.
        """
        error_code = getattr(error, "code", None)
        if error_code not in RATE_LIMIT_ERROR_CODES:
            return None

        if attempt >= max_retries:
            return None

        ban_wait = _parse_ban_expiry(str(error))
        if ban_wait is None:
            # No parseable expiry — use a short default
            ban_wait = 30.0
        elif ban_wait <= 0:
            # Ban already expired — retry with minimal buffer
            ban_wait = 1.0

        # Add small buffer so we don't land exactly on the expiry edge
        total_wait = ban_wait + 5.0

        if total_wait > max_wait:
            logger.error(
                "IP ban wait (%.0fs) exceeds startup max wait of %ss. Not retrying.",
                total_wait,
                max_wait,
            )
            return None

        return total_wait

    def _handle_init_failure(self, error: Exception | None, api_endpoint: str):
        """Handle final init failure after all retries exhausted."""
        if error is None:
            return

        error_type = type(error).__name__
        error_msg = str(error)

        if self._use_margin and self._is_live:
            logger.error(
                "%s Client initialization failed with %s: %s. "
                "Credentials available: %s, Testnet mode: %s.",
                api_endpoint.title(),
                error_type,
                error_msg,
                bool(self.api_key and self.api_secret),
                self.testnet,
                exc_info=True,
            )
            raise RuntimeError(
                f"FATAL: Cannot initialize Binance client in live margin mode. "
                f"Refusing to fall back to offline stub — placing dummy margin orders "
                f"risks fund loss. Error: {error_type}: {error_msg}"
            ) from error

        logger.error(
            "%s Client initialization failed with %s: %s. "
            "Credentials available: %s, Testnet mode: %s. Falling back to offline stub.",
            api_endpoint.title(),
            error_type,
            error_msg,
            bool(self.api_key and self.api_secret),
            self.testnet,
            exc_info=True,
        )

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

            def get_all_orders(self, *args, **kwargs):
                return []

            # Margin API stubs
            def get_margin_account(self):
                return {
                    "userAssets": [],
                    "tradeEnabled": False,
                    "borrowEnabled": False,
                    "marginLevel": "999",
                }

            def create_margin_order(self, *args, **kwargs):
                return {"orderId": "12345"}

            def cancel_margin_order(self, *args, **kwargs):
                return {"orderId": "12345"}

            def get_margin_order(self, *args, **kwargs):
                return {}

            def get_open_margin_orders(self, *args, **kwargs):
                return []

            def get_all_margin_orders(self, *args, **kwargs):
                return []

            def get_margin_trades(self, *args, **kwargs):
                return []

            def get_margin_interest_history(self, *args, **kwargs):
                return []

            def get_margin_symbol(self, *args, **kwargs):
                return {"isMarginTrade": True, "isBuyAllowed": True, "isSellAllowed": True}

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
        self, symbol: str, timeframe: str, start: datetime, end: datetime | None = None
    ) -> pd.DataFrame:
        """Fetch historical klines data from Binance"""
        try:
            interval = self._convert_timeframe(timeframe)
            start_ts = int(start.timestamp() * 1000)
            end_ts = int(end.timestamp() * 1000) if end else None

            # Wrap API call with timeout to prevent indefinite hangs
            # Configurable via DATA_FETCH_TIMEOUT_SECONDS env var
            data_timeout = get_config().get_float(
                "DATA_FETCH_TIMEOUT_SECONDS", DEFAULT_DATA_FETCH_TIMEOUT
            )
            try:
                klines = run_with_timeout(
                    self._client.get_historical_klines,
                    args=(symbol, interval, start_ts, end_ts),
                    timeout_seconds=data_timeout,
                    operation_name="Binance get_historical_klines",
                )
            except InfraTimeoutError as timeout_err:
                raise TimeoutError(
                    f"Binance API timeout after {data_timeout}s fetching {symbol} {timeframe}"
                ) from timeout_err

            df = self._process_klines(klines)
            self.data = df

            if len(df) > 0:
                logger.info(f"Fetched {len(df)} candles from {df.index.min()} to {df.index.max()}")
            else:
                # Check if this is expected (future dates) or an error
                current_time = datetime.now(UTC)
                if end is not None and end > current_time:
                    logger.info(
                        f"No data available for future dates: requested {start} to {end}, current time is {current_time}"
                    )
                elif end is not None and end > current_time - timedelta(hours=1):
                    logger.info(
                        f"No recent data available yet: requested {start} to {end}, current time is {current_time}"
                    )
                else:
                    logger.warning(
                        f"No data returned for {symbol} {timeframe} from {start} to {end}"
                    )
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
        """Get latest price for a symbol.

        Raises:
            RuntimeError: If price cannot be fetched from exchange.
                         Caller must handle this to prevent trading with invalid prices.
        """
        try:
            ticker = self._client.get_symbol_ticker(symbol=symbol)
            price = float(ticker["price"])
            # Validate price is positive to prevent downstream calculation errors
            if price <= 0:
                raise ValueError(f"Invalid price {price} <= 0 for {symbol}")
            return price
        except Exception as e:
            error_type = type(e).__name__
            logger.error(f"Error fetching current price for {symbol}: {error_type}: {str(e)}")
            # Don't return 0.0 - that could cause division by zero or infinite position sizes
            # Force caller to handle price fetch failures explicitly
            raise RuntimeError(
                f"Failed to fetch current price for {symbol}: {error_type}: {str(e)}"
            ) from e

    # ========================================
    # Margin / Spot Dispatch Layer
    # ========================================

    def _verify_margin_account(self):
        """Verify margin account capabilities at startup."""
        try:
            margin_info = self._client.get_margin_account()
            trade_enabled = margin_info.get("tradeEnabled", False)
            borrow_enabled = margin_info.get("borrowEnabled", False)
            margin_level = margin_info.get("marginLevel", "N/A")

            if not trade_enabled:
                raise RuntimeError("Margin account tradeEnabled=False — cannot trade")
            if not borrow_enabled:
                raise RuntimeError("Margin account borrowEnabled=False — cannot borrow for shorts")

            logger.info(
                "Margin account verified: tradeEnabled=%s, borrowEnabled=%s, marginLevel=%s",
                trade_enabled, borrow_enabled, margin_level,
            )

            # Verify no non-USDT base assets with significant holdings.
            # The bot assumes USDT-only collateral — holding base assets (ETH, BTC)
            # causes MARGIN_BUY to sell existing inventory instead of borrowing,
            # which breaks short position detection in reconciliation.
            dust_threshold = 1.0  # $1 worth considered dust
            quote_assets = {"USDT", "BUSD", "USD", "USDC"}
            for asset_data in margin_info.get("userAssets", []):
                asset_name = asset_data.get("asset", "")
                if asset_name in quote_assets:
                    continue
                net_asset = float(asset_data.get("netAsset", "0"))
                free = float(asset_data.get("free", "0"))
                borrowed = float(asset_data.get("borrowed", "0"))
                if free > 0 and net_asset > 0:
                    # Estimate USD value (rough — just flag if non-trivial)
                    try:
                        ticker = self._client.get_symbol_ticker(symbol=f"{asset_name}USDT")
                        price = float(ticker.get("price", 0))
                        value_usd = free * price
                    except Exception:
                        value_usd = free  # Conservative: treat as $1 per unit
                    if value_usd > dust_threshold:
                        # Warn but don't block — this could be a recovering
                        # long (free > 0, borrowed == 0) or a recovering short
                        # (borrowed > 0). Provider init runs before startup
                        # reconciliation, so blocking here prevents position
                        # recovery. Reconciliation will verify shortly after.
                        if borrowed > 0:
                            logger.warning(
                                "Margin wallet holds %s %s (~$%.2f, borrowed=%.8f) "
                                "— may be a recovering short, reconciliation will verify",
                                free, asset_name, value_usd, borrowed,
                            )
                        else:
                            logger.warning(
                                "Margin wallet holds %s %s (~$%.2f, borrowed=0) "
                                "— may be a recovering long or manual deposit. "
                                "If manual, transfer out before next short entry "
                                "(MARGIN_BUY sells existing inventory before borrowing)",
                                free, asset_name, value_usd,
                            )
        except RuntimeError:
            raise
        except Exception as e:
            # Re-raise rate-limit errors directly so the startup retry loop
            # in _initialize_client can detect and retry them.
            if getattr(e, "code", None) in RATE_LIMIT_ERROR_CODES:
                raise
            if self._is_live:
                raise RuntimeError(f"Failed to verify margin account capabilities: {e}") from e
            logger.warning("Could not verify margin account (non-live mode): %s", e)

    def _verify_margin_symbol(self, symbol: str, side: str | None = None):
        """Verify symbol supports margin trading. Called lazily on first order.

        Args:
            symbol: Trading pair to verify.
            side: Order side (BUY/SELL). If provided, only validates the
                  submitted side so exits/SLs aren't blocked when the
                  exchange disables only one direction.
        """
        try:
            info = self._client.get_margin_symbol(symbol=symbol)
            if not info.get("isMarginTrade", False):
                raise ValueError(f"Symbol {symbol} does not support margin trading")
            # Only check the side being submitted, not both
            if side == "BUY" and not info.get("isBuyAllowed", False):
                raise ValueError(f"Symbol {symbol}: buy not allowed for margin")
            if side == "SELL" and not info.get("isSellAllowed", False):
                raise ValueError(f"Symbol {symbol}: sell not allowed for margin")
            # If no side specified, check both (startup validation)
            if side is None:
                if not info.get("isBuyAllowed", False):
                    raise ValueError(f"Symbol {symbol}: buy not allowed for margin")
                if not info.get("isSellAllowed", False):
                    raise ValueError(f"Symbol {symbol}: sell not allowed for margin")
            logger.info("Margin symbol %s verified: margin trading enabled", symbol)
        except ValueError:
            raise
        except Exception as e:
            raise ValueError(f"Failed to verify margin symbol {symbol}: {e}") from e

    def _call_get_account(self) -> dict:
        """Fetch account data, normalizing margin response to spot format."""
        if self._use_margin:
            raw = self._client.get_margin_account()
            # Normalize userAssets -> balances for downstream compatibility
            raw["balances"] = [
                {
                    "asset": a["asset"],
                    "free": a["free"],
                    "locked": a["locked"],
                    "borrowed": a.get("borrowed", "0"),
                    "interest": a.get("interest", "0"),
                    "netAsset": a.get("netAsset", a["free"]),
                }
                for a in raw.get("userAssets", [])
            ]
            raw["canTrade"] = raw.get("tradeEnabled", False)
            return raw
        return self._client.get_account()

    def _call_create_order(self, **params) -> dict:
        """Place order via margin or spot API."""
        if self._use_margin:
            # Lazy symbol validation on first margin order per side
            symbol = params.get("symbol", "")
            order_side = params.get("side", "")
            # Always use symbol:side key for consistent cache lookup
            cache_key = f"{symbol}:{order_side or 'ANY'}"
            if symbol and cache_key not in self._margin_symbol_verified:
                self._verify_margin_symbol(symbol, side=order_side or None)
                self._margin_symbol_verified.add(cache_key)

            params["isIsolated"] = "FALSE"
            # sideEffectType is already in params if caller set it
            return self._client.create_margin_order(**params)
        # Remove margin-specific params for spot
        params.pop("sideEffectType", None)
        return self._client.create_order(**params)

    def _call_get_order(self, **params) -> dict:
        """Get order via margin or spot API."""
        if self._use_margin:
            params["isIsolated"] = "FALSE"
            return self._client.get_margin_order(**params)
        return self._client.get_order(**params)

    def _call_get_open_orders(self, **params) -> list:
        """Get open orders via margin or spot API."""
        if self._use_margin:
            params["isIsolated"] = "FALSE"
            return self._client.get_open_margin_orders(**params)
        return self._client.get_open_orders(**params)

    def _call_get_my_trades(self, **params) -> list:
        """Get trades via margin or spot API."""
        if self._use_margin:
            params["isIsolated"] = "FALSE"
            return self._client.get_margin_trades(**params)
        return self._client.get_my_trades(**params)

    def _call_cancel_order(self, **params) -> dict:
        """Cancel order via margin or spot API."""
        if self._use_margin:
            params["isIsolated"] = "FALSE"
            return self._client.cancel_margin_order(**params)
        return self._client.cancel_order(**params)

    def _call_get_all_orders(self, **params) -> list:
        """Get all orders via margin or spot API."""
        if self._use_margin:
            params["isIsolated"] = "FALSE"
            return self._client.get_all_margin_orders(**params)
        return self._client.get_all_orders(**params)

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
            account_info = self._call_get_account()
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
            account_info = self._call_get_account()
            balances = []

            for balance_data in account_info.get("balances", []):
                free = float(balance_data.get("free", 0))
                locked = float(balance_data.get("locked", 0))
                if self._use_margin:
                    # In margin mode, netAsset = free + locked - borrowed - interest
                    total = float(balance_data.get("netAsset", free + locked))
                else:
                    total = free + locked

                if total > 0:  # Only include non-zero balances
                    balance = AccountBalance(
                        asset=balance_data["asset"],
                        free=free,
                        locked=locked,
                        total=total,
                        last_updated=datetime.now(UTC),
                    )
                    balances.append(balance)

            return balances

        except Exception as e:
            logger.error(f"Failed to get balances: {e}")
            return []

    def get_margin_borrowed(self, asset: str) -> float | None:
        """Get borrowed amount for a specific asset in cross-margin account.

        Returns the raw borrowed quantity (not netAsset). Used by reconciliation
        to verify short positions still have outstanding debt.
        Returns 0.0 if asset found with no debt, None on error or unavailable.
        Callers must treat None as "unknown" and skip position removal.
        """
        if not self._use_margin or not BINANCE_AVAILABLE or not self._client:
            return None
        try:
            account = self._call_get_account()
            for bal in account.get("balances", []):
                if bal["asset"] == asset:
                    return float(bal.get("borrowed", "0"))
            return 0.0  # Asset not in account — no debt
        except Exception as e:
            logger.warning("Failed to get borrowed amount for %s: %s", asset, e)
            return None  # Unknown — caller must not assume position is closed

    def get_margin_interest_history(
        self,
        asset: str,
        start_time: int | None = None,
        end_time: int | None = None,
        page: int = 1,
    ) -> list[dict]:
        """Get margin interest accrual history for an asset.

        Queries Binance /sapi/v1/margin/interestHistory endpoint.
        Returns list of dicts with keys: txId, interestAccuredTime, asset,
        interest, interestRate, principal, type.
        Returns empty list if not in margin mode or no client available.
        Raises on API/network errors so callers can retry.
        Uses page-based pagination (current=page, size=100).
        """
        if not self._use_margin or not BINANCE_AVAILABLE or not self._client:
            return []
        params: dict[str, Any] = {"asset": asset, "size": 100, "current": page}
        if start_time is not None:
            params["startTime"] = start_time
        if end_time is not None:
            params["endTime"] = end_time
        response = self._client.get_margin_interest_history(**params)
        # Binance returns {rows: [...], total: N} envelope
        if isinstance(response, dict):
            return response.get("rows", [])
        return response if isinstance(response, list) else []

    def get_balance(self, asset: str) -> AccountBalance | None:
        """Get balance for a specific asset"""
        if not BINANCE_AVAILABLE or not self._client:
            logger.warning("Binance not available - returning None for balance")
            return None

        try:
            account_info = self._call_get_account()

            for balance_data in account_info.get("balances", []):
                if balance_data["asset"] == asset:
                    free = float(balance_data.get("free", 0))
                    locked = float(balance_data.get("locked", 0))
                    if self._use_margin:
                        # In margin mode, netAsset = free + locked - borrowed - interest
                        total = float(balance_data.get("netAsset", free + locked))
                    else:
                        total = free + locked

                    return AccountBalance(
                        asset=asset,
                        free=free,
                        locked=locked,
                        total=total,
                        last_updated=datetime.now(UTC),
                    )

            return None

        except Exception as e:
            logger.error(f"Failed to get balance for {asset}: {e}")
            return None

    def get_positions(self, symbol: str | None = None) -> list[Position]:
        """Get open positions (for spot trading, this returns holdings as positions).

        Args:
            symbol: If provided, only fetch position for this symbol (e.g. "ETHUSDT").
                    Saves API weight by skipping ticker calls for unrelated assets.
        """
        if not BINANCE_AVAILABLE or not self._client:
            logger.warning("Binance not available - returning empty positions")
            return []

        try:
            # Filter balances to only the trading symbol's base asset to skip
            # unnecessary ticker API calls for dust/unrelated holdings
            target_base_asset: str | None = None
            if symbol:
                for quote in ("USDT", "BUSD", "USD"):
                    if symbol.endswith(quote) and len(symbol) > len(quote):
                        target_base_asset = symbol[: -len(quote)]
                        break

            # For spot trading, we consider holdings as "positions"
            balances = self.get_balances()
            positions = []

            for balance in balances:
                if balance.asset == "USDT" or balance.total <= 0:
                    continue
                # Skip assets not matching the target symbol to conserve API weight
                if target_base_asset and balance.asset != target_base_asset:
                    continue
                # Get current price for the asset
                try:
                    ticker = self._client.get_symbol_ticker(
                        symbol=SymbolFactory.to_exchange_symbol(
                            f"{balance.asset}-USD", "binance"
                        )
                    )

                    # Validate ticker response before accessing price
                    if not isinstance(ticker, dict) or "price" not in ticker:
                        logger.warning(
                            "Invalid ticker response for %s: %s. Position not loaded.",
                            balance.asset,
                            ticker,
                        )
                        continue

                    current_price = float(ticker["price"])

                    # Validate price is finite to prevent inf/nan in positions
                    if not math.isfinite(current_price) or current_price <= 0:
                        logger.warning(
                            "Invalid price %.8f for %s. Position not loaded.",
                            current_price,
                            balance.asset,
                        )
                        continue

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
                        open_time=datetime.now(UTC),  # Simplified
                        last_update_time=datetime.now(UTC),
                    )
                    positions.append(position)

                except Exception as e:
                    logger.warning(
                        "Failed to load position for %s (balance %.8f): %s",
                        balance.asset,
                        balance.total,
                        e,
                    )

            return positions

        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return []

    def _parse_order_data(self, order_data: dict) -> Order | None:
        """Safely parse order data from Binance API response.

        Args:
            order_data: Raw order data dictionary from Binance API

        Returns:
            Order object or None if data is invalid
        """
        try:
            # Validate required fields exist
            required_fields = [
                "orderId",
                "symbol",
                "side",
                "type",
                "origQty",
                "status",
                "executedQty",
                "time",
                "updateTime",
            ]
            for field in required_fields:
                if field not in order_data:
                    logger.error(
                        "Invalid order data from Binance: missing required field '%s'. Data: %s",
                        field,
                        order_data,
                    )
                    return None

            # Safe extraction with type validation
            order_id_raw = order_data["orderId"]
            if not isinstance(order_id_raw, int | str):
                logger.error("Invalid orderId type: %s", type(order_id_raw))
                return None

            return Order(
                order_id=str(order_id_raw),
                symbol=str(order_data["symbol"]),
                side=OrderSide.BUY if order_data["side"] == SIDE_BUY else OrderSide.SELL,
                order_type=self._convert_order_type(order_data["type"]),
                quantity=float(order_data["origQty"]),
                price=float(order_data.get("price", 0)) if order_data.get("price") != "0" else None,
                status=self._convert_order_status(order_data["status"]),
                filled_quantity=float(order_data.get("executedQty", 0)),
                average_price=self._extract_average_price(order_data),
                commission=0.0,  # Will be updated from trade history
                commission_asset="",
                create_time=datetime.fromtimestamp(int(order_data["time"]) / 1000, tz=UTC),
                update_time=datetime.fromtimestamp(int(order_data["updateTime"]) / 1000, tz=UTC),
                stop_price=(
                    float(order_data["stopPrice"])
                    if order_data.get("stopPrice") and order_data["stopPrice"] != "0"
                    else None
                ),
                time_in_force=order_data.get("timeInForce", "GTC"),
                client_order_id=order_data.get("clientOrderId"),
            )
        except (KeyError, ValueError, TypeError) as e:
            logger.error("Failed to parse order data: %s. Data: %s", e, order_data)
            return None

    @staticmethod
    def _extract_average_price(order_data: dict) -> float | None:
        """Extract average fill price from Binance order data.

        Binance spot endpoints return executedQty and cummulativeQuoteQty
        but not always avgPrice. Falls back to computing the average from
        cummulativeQuoteQty / executedQty when avgPrice is missing or zero.
        """
        avg_price_raw = order_data.get("avgPrice")
        if avg_price_raw and str(avg_price_raw) != "0":
            return float(avg_price_raw)

        # Fallback: derive from cummulativeQuoteQty / executedQty
        exec_qty = float(order_data.get("executedQty", 0))
        if exec_qty > 0:
            cum_quote = float(order_data.get("cummulativeQuoteQty", 0))
            if cum_quote > 0:
                return cum_quote / exec_qty

        return None

    def get_open_orders(self, symbol: str | None = None) -> list[Order]:
        """Get all open orders"""
        if not BINANCE_AVAILABLE or not self._client:
            logger.warning("Binance not available - returning empty orders")
            return []

        try:
            if symbol:
                orders_data = self._call_get_open_orders(symbol=symbol)
            else:
                orders_data = self._call_get_open_orders()

            orders = []
            for order_data in orders_data:
                order = self._parse_order_data(order_data)
                if order is not None:
                    orders.append(order)

            return orders

        except Exception as e:
            logger.error(f"Failed to get open orders: {e}")
            return []

    def get_order(self, order_id: str, symbol: str) -> Order | None:
        """Get specific order by ID.

        Automatically detects whether order_id is a numeric exchange order ID
        or an alphanumeric client order ID (e.g. 'atb_19d360981ab_3a4b0d5a')
        and routes to the appropriate Binance API parameter. Binance's orderId
        parameter only accepts numeric strings matching '^[0-9]{1,20}$'.
        """
        if not BINANCE_AVAILABLE or not self._client:
            logger.warning("Binance not available - returning None for order")
            return None

        # Non-numeric IDs are client order IDs — delegate to existing method
        if not order_id.isdigit():
            logger.debug(
                "Order ID %s is non-numeric, routing to get_order_by_client_id",
                order_id,
            )
            return self.get_order_by_client_id(order_id, symbol)

        try:
            order_data = self._call_get_order(symbol=symbol, orderId=order_id)
            return self._parse_order_data(order_data)

        except Exception as e:
            logger.error("Failed to get order %s: %s", order_id, e)
            return None

    def get_recent_trades(self, symbol: str, limit: int = 100) -> list[Trade]:
        """Get recent trades for a symbol"""
        if not BINANCE_AVAILABLE or not self._client:
            logger.warning("Binance not available - returning empty trades")
            return []

        try:
            trades_data = self._call_get_my_trades(symbol=symbol, limit=limit)

            trades = []
            for trade_data in trades_data:
                # Validate critical fields to prevent KeyError/TypeError from malformed API response
                try:
                    trade_time_ms = trade_data.get("time")
                    if trade_time_ms is None or not isinstance(trade_time_ms, int | float):
                        logger.warning("Skipping trade with invalid timestamp: %s", trade_data)
                        continue

                    trade = Trade(
                        trade_id=str(trade_data["id"]),
                        order_id=str(trade_data["orderId"]),
                        symbol=trade_data["symbol"],
                        side=OrderSide.BUY if trade_data["isBuyer"] else OrderSide.SELL,
                        quantity=float(trade_data["qty"]),
                        price=float(trade_data["price"]),
                        commission=float(trade_data["commission"]),
                        commission_asset=trade_data["commissionAsset"],
                        time=datetime.fromtimestamp(trade_time_ms / 1000),
                    )
                    trades.append(trade)
                except (KeyError, ValueError, TypeError) as e:
                    logger.warning("Skipping malformed trade record: %s. Error: %s", trade_data, e)
                    continue

            return trades

        except Exception as e:
            logger.error(f"Failed to get recent trades for {symbol}: {e}")
            return []

    @with_rate_limit_retry(max_retries=3, base_delay=1.0)
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
        side_effect_type: str | None = None,
    ) -> Order | None:
        """
        Place a new order and return full Order object with fill data.

        Uses client_order_id for idempotency - if provided and an order with the same
        client ID already exists, Binance will reject the duplicate order.
        For market orders, requests FULL response type to capture fill data at placement.
        """
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

            # Request full response for market orders to capture fill data at placement
            if order_type == OrderType.MARKET:
                order_params["newOrderRespType"] = "FULL"

            # Add client order ID for idempotency if provided
            if client_order_id:
                order_params["newClientOrderId"] = client_order_id
                logger.debug(f"Placing order with client ID: {client_order_id}")

            # Pass margin intent to dispatch method (injected into margin orders only)
            if side_effect_type:
                order_params["sideEffectType"] = side_effect_type

            # Place the order
            response = self._call_create_order(**order_params)

            order_id = str(response.get("orderId", ""))
            if not order_id:
                logger.error(f"Order placed but no orderId in response: {response}")
                return None

            # Parse full response into Order object
            order_obj = self._parse_placement_response(response, symbol, side, order_type)

            logger.info(
                f"Order placed successfully: {symbol} {side.value} {quantity} order_id={order_id}"
            )
            return order_obj

        except BinanceOrderException as e:
            error_msg = str(e)
            error_code = getattr(e, "code", None)

            # Check if this is a duplicate client order ID error (idempotency).
            # Require BOTH conditions: -2010 alone covers other rejections
            # (insufficient balance, etc.) that are NOT duplicates.
            if client_order_id and (
                "Duplicate order sent" in error_msg and error_code == -2010
            ):
                logger.warning(
                    f"Duplicate client order ID detected: {client_order_id}. "
                    "This order may have already been placed. Check order status manually."
                )
                return None

            # Definitive rejections: the exchange explicitly refused the order,
            # so it was NOT placed. Raise ValueError so the caller can distinguish
            # this from ambiguous network/timeout errors (which return None).
            if error_code in DEFINITIVE_REJECT_CODES:
                logger.error(
                    "Order definitively rejected by Binance (code=%s): %s",
                    error_code,
                    error_msg,
                )
                raise ValueError(
                    f"Order rejected by exchange (code={error_code}): {error_msg}"
                ) from e

            # Ambiguous error (unknown code, network-adjacent): return None so the
            # caller treats it as "order may or may not have been placed".
            logger.error(f"Binance order error (ambiguous, code={error_code}): {e}")
            return None
        except BinanceAPIException as e:
            # REST API errors surface as BinanceAPIException (not BinanceOrderException).
            # Apply same definitive-reject logic to prevent phantom positions.
            error_code = getattr(e, "code", 0)
            error_msg = getattr(e, "message", str(e))
            if error_code in DEFINITIVE_REJECT_CODES:
                logger.error(
                    "Margin order definitively rejected (code=%s): %s",
                    error_code,
                    error_msg,
                )
                raise ValueError(
                    f"Order rejected by exchange (code={error_code}): {error_msg}"
                ) from e
            logger.error(f"Binance API error placing order (code={error_code}): {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to place order: {e}")
            return None

    def _parse_placement_response(
        self,
        response: dict,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
    ) -> Order:
        """Parse a Binance order placement response into an Order object.

        For FULL responses (market orders), aggregates fill data from the fills array.
        """
        now = datetime.now(UTC)
        order_id = str(response.get("orderId", ""))
        client_oid = response.get("clientOrderId")
        status_str = response.get("status", "NEW")

        # Map Binance status to our OrderStatus
        status_map = {
            "NEW": OrderStatus.PENDING,
            "PARTIALLY_FILLED": OrderStatus.PARTIALLY_FILLED,
            "FILLED": OrderStatus.FILLED,
            "CANCELED": OrderStatus.CANCELLED,
            "REJECTED": OrderStatus.REJECTED,
            "EXPIRED": OrderStatus.EXPIRED,
        }
        status = status_map.get(status_str, OrderStatus.PENDING)

        # Aggregate fill data from fills array (FULL response type)
        fills = response.get("fills", [])
        total_qty = 0.0
        total_cost = 0.0
        total_commission = 0.0
        commission_asset = ""
        for fill in fills:
            qty = float(fill.get("qty", 0))
            px = float(fill.get("price", 0))
            total_qty += qty
            total_cost += qty * px
            total_commission += float(fill.get("commission", 0))
            commission_asset = fill.get("commissionAsset", commission_asset)

        avg_price = total_cost / total_qty if total_qty > 0 else None

        # Fallback to top-level fields if no fills array
        if total_qty == 0:
            total_qty = float(response.get("executedQty", 0))
            cum_quote = float(response.get("cummulativeQuoteQty", 0))
            avg_price = cum_quote / total_qty if total_qty > 0 else None

        return Order(
            order_id=order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=float(response.get("origQty", 0)),
            price=float(response.get("price", 0)) or None,
            status=status,
            filled_quantity=total_qty,
            average_price=avg_price,
            commission=total_commission,
            commission_asset=commission_asset,
            create_time=now,
            update_time=now,
            client_order_id=client_oid,
        )

    # Only retry transient -1015 (too many orders), NOT -1003 (IP ban).
    # IP bans last minutes — sleeping blocks SL placement and leaves the
    # position unprotected. The caller handles -1003 by entering close-only mode.
    @with_rate_limit_retry(max_retries=2, base_delay=1.0, ban_safe=True)
    def place_stop_loss_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        stop_price: float,
        limit_price: float | None = None,
        client_order_id: str | None = None,
        side_effect_type: str | None = None,
    ) -> str | None:
        """
        Place a server-side stop-loss order on Binance.

        Uses STOP_LOSS_LIMIT order type which requires both a stop price
        (trigger) and a limit price (execution price).
        """
        if not BINANCE_AVAILABLE or not self._client:
            logger.warning("Binance not available - cannot place stop-loss order")
            return None

        # Validate stop_price is positive and finite
        if not (stop_price > 0 and math.isfinite(stop_price)):
            logger.error("Invalid stop_price: %s", stop_price)
            return None

        try:
            binance_side = SIDE_BUY if side == OrderSide.BUY else SIDE_SELL

            # Calculate limit price if not provided
            # For sells: limit slightly below stop to ensure fill
            # For buys: limit slightly above stop to ensure fill
            if limit_price is None:
                if side == OrderSide.SELL:
                    limit_price = stop_price * (1 - STOP_LOSS_LIMIT_SLIPPAGE_FACTOR)
                else:
                    limit_price = stop_price * (1 + STOP_LOSS_LIMIT_SLIPPAGE_FACTOR)

            # Round prices to valid tick size
            symbol_info = self.get_symbol_info(symbol)
            if symbol_info:
                # Validate tick_size is numeric before division to prevent TypeError
                tick_size_raw = symbol_info.get("tick_size", 0.01)
                tick_size = float(tick_size_raw) if isinstance(tick_size_raw, int | float) else 0.01
                if tick_size > 0:
                    stop_price = round(stop_price / tick_size) * tick_size
                    limit_price = round(limit_price / tick_size) * tick_size

                # Validate step_size is numeric before division to prevent TypeError
                step_size_raw = symbol_info.get("step_size", 0.00001)
                step_size = (
                    float(step_size_raw) if isinstance(step_size_raw, int | float) else 0.00001
                )
                if step_size > 0:
                    quantity = round(quantity / step_size) * step_size

            sl_params = {
                "symbol": symbol,
                "side": binance_side,
                "type": "STOP_LOSS_LIMIT",
                "quantity": quantity,
                "stopPrice": str(stop_price),
                "price": str(limit_price),
                "timeInForce": "GTC",
            }
            if client_order_id:
                sl_params["newClientOrderId"] = client_order_id
            if side_effect_type:
                sl_params["sideEffectType"] = side_effect_type
            response = self._call_create_order(**sl_params)

            order_id = str(response.get("orderId", ""))
            if not order_id:
                logger.error(f"Stop-loss order placed but no orderId in response: {response}")
                return None

            logger.info(
                f"Stop-loss order placed: {symbol} {side.value} qty={quantity} "
                f"stop={stop_price} limit={limit_price} order_id={order_id}"
            )
            return order_id

        except BinanceOrderException as e:
            logger.error(f"Binance stop-loss order error: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to place stop-loss order: {e}")
            return None

    def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel an existing order"""
        if not BINANCE_AVAILABLE or not self._client:
            logger.warning("Binance not available - cannot cancel order")
            return False

        try:
            self._call_cancel_order(symbol=symbol, orderId=order_id)
            logger.info(f"Order cancelled successfully: {order_id}")
            return True

        except BinanceOrderException as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False
        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {e}")
            return False

    def cancel_all_orders(self, symbol: str | None = None) -> bool:
        """Cancel all open orders"""
        if not BINANCE_AVAILABLE or not self._client:
            logger.warning("Binance not available - cannot cancel orders")
            return False

        try:
            open_orders = self.get_open_orders(symbol=symbol)
            for order in open_orders:
                self.cancel_order(order.order_id, order.symbol)

            logger.info("All orders cancelled successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to cancel all orders: {e}")
            return False

    @with_rate_limit_retry(max_retries=3, base_delay=1.0)
    def get_order_by_client_id(self, client_order_id: str, symbol: str) -> Order | None:
        """Query Binance for an order by our client_order_id (origClientOrderId)."""
        if not BINANCE_AVAILABLE or not self._client:
            return None
        try:
            response = self._call_get_order(
                symbol=symbol, origClientOrderId=client_order_id
            )
            return self._parse_order_data(response)
        except BinanceOrderException as e:
            if "-2013" in str(e):  # Order does not exist
                return None
            logger.error("Failed to get order by client_id %s: %s", client_order_id, e)
            return None
        except Exception as e:
            logger.error("Failed to get order by client_id %s: %s", client_order_id, e)
            return None

    @with_rate_limit_retry(max_retries=3, base_delay=1.0)
    def get_all_orders(
        self, symbol: str, start_time: datetime | None = None, limit: int = 100
    ) -> list[Order]:
        """Get all orders (open + closed) for a symbol within a time window."""
        if not BINANCE_AVAILABLE or not self._client:
            return []
        try:
            params: dict[str, Any] = {"symbol": symbol, "limit": limit}
            if start_time is not None:
                params["startTime"] = int(start_time.timestamp() * 1000)
            raw_orders = self._call_get_all_orders(**params)
            parsed = [self._parse_order_data(o) for o in raw_orders if o]
            return [o for o in parsed if o is not None]
        except Exception as e:
            logger.error("Failed to get all orders for %s: %s", symbol, e)
            return []

    @with_rate_limit_retry(max_retries=3, base_delay=1.0)
    def get_my_trades(
        self, symbol: str, order_id: str | None = None, start_time: datetime | None = None
    ) -> list[Trade]:
        """Get account trades, optionally filtered by order_id or start_time."""
        if not BINANCE_AVAILABLE or not self._client:
            return []
        try:
            params: dict[str, Any] = {"symbol": symbol, "limit": 500}
            if order_id is not None:
                params["orderId"] = int(order_id)
            if start_time is not None:
                params["startTime"] = int(start_time.timestamp() * 1000)
            raw_trades = self._call_get_my_trades(**params)
            result = []
            for t in raw_trades:
                try:
                    trade_time = t.get("time", 0)
                    result.append(
                        Trade(
                            trade_id=str(t["id"]),
                            order_id=str(t["orderId"]),
                            symbol=t["symbol"],
                            side=OrderSide.BUY if t["isBuyer"] else OrderSide.SELL,
                            quantity=float(t["qty"]),
                            price=float(t["price"]),
                            commission=float(t["commission"]),
                            commission_asset=t["commissionAsset"],
                            time=datetime.fromtimestamp(trade_time / 1000, tz=UTC),
                        )
                    )
                except (KeyError, ValueError, TypeError) as e:
                    logger.warning("Skipping malformed trade: %s", e)
            return result
        except Exception as e:
            logger.error("Failed to get my trades for %s: %s", symbol, e)
            return []

    def get_symbol_info(self, symbol: str) -> dict[str, Any] | None:
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


    # ---------- WebSocket Stream Management ----------

    def _ensure_twm(self) -> None:
        """Lazily create the ThreadedWebsocketManager from existing config."""
        with self._twm_lock:
            if self._twm is not None:
                return
            twm_kwargs: dict[str, Any] = {
                "api_key": self.api_key,
                "api_secret": self.api_secret,
            }
            if self.testnet:
                twm_kwargs["testnet"] = True
            api_endpoint = get_binance_api_endpoint()
            if api_endpoint == "binanceus":
                twm_kwargs["tld"] = "us"
            self._twm = ThreadedWebsocketManager(**twm_kwargs)
            self._twm.start()

    def start_kline_stream(
        self, symbol: str, timeframe: str, on_kline: Callable[[dict], None]
    ) -> bool:
        """Start kline stream. Safe for paper mode (no credentials needed).

        Args:
            symbol: Trading pair symbol (e.g. 'BTCUSDT')
            timeframe: Kline interval (e.g. '1h')
            on_kline: Callback receiving raw kline event dicts

        Returns:
            True if stream started successfully, False on failure.
        """
        try:
            self._ensure_twm()
            self._active_symbol = symbol
            self._active_timeframe = timeframe
            self._on_kline_cb = on_kline

            def _kline_callback(msg: dict) -> None:
                """Route kline events, handling errors before user callback."""
                if msg.get("e") == "error":
                    logger.error("Kline WS error: %s", msg.get("m", "unknown"))
                    self._on_kline_disconnect()
                    return
                self._last_kline_event_time = datetime.now(UTC)
                self._kline_event_received = True
                on_kline(msg)

            self._kline_event_received = False  # Reset until first event confirms
            self._kline_socket_key = self._twm.start_kline_socket(
                callback=_kline_callback, symbol=symbol, interval=timeframe
            )
            self._kline_ws_state = WebSocketState.PRIMARY
            self._last_kline_event_time = datetime.now(UTC)
            return True
        except Exception as e:
            logger.error("Failed to start kline stream: %s", e)
            return False

    def start_user_stream(self, on_user_event: Callable[[dict], None]) -> bool:
        """Start user data stream. Requires credentials. Live mode only.

        Args:
            on_user_event: Callback receiving raw user data event dicts

        Returns:
            True if stream started successfully, False on failure.
        """
        try:
            self._ensure_twm()
            self._on_user_event_cb = on_user_event

            def _user_callback(msg: dict) -> None:
                """Route user data events, handling errors before user callback."""
                if msg.get("e") == "error":
                    logger.error("User data WS error: %s", msg.get("m", "unknown"))
                    self._on_user_disconnect()
                    return
                self._last_user_event_time = datetime.now(UTC)
                self._user_event_received = True
                on_user_event(msg)

            if self._use_margin:
                self._user_socket_key = self._twm.start_margin_socket(
                    callback=_user_callback
                )
            else:
                self._user_socket_key = self._twm.start_user_socket(
                    callback=_user_callback
                )
            self._last_user_event_time = datetime.now(UTC)
            self._user_ws_state = WebSocketState.PRIMARY
            return True
        except Exception as e:
            logger.error("Failed to start user data stream: %s", e)
            return False

    def stop_user_stream(self) -> None:
        """Stop only the user data WebSocket stream."""
        if self._user_socket_key and self._twm:
            try:
                self._twm.stop_socket(self._user_socket_key)
            except Exception as e:
                logger.error("Failed to stop user socket: %s", e)
            self._user_socket_key = None
            self._user_ws_state = WebSocketState.DISCONNECTED
            self._user_event_received = False

    def stop_streams(self) -> None:
        """Stop all WebSocket streams. Recreates TWM on reconnect."""
        if self._twm:
            self._twm.stop()
            self._twm = None
            self._kline_socket_key = None
            self._user_socket_key = None
            self._kline_ws_state = WebSocketState.DISCONNECTED
            self._user_ws_state = WebSocketState.DISCONNECTED
            self._kline_event_received = False
            self._user_event_received = False

    @property
    def ws_state(self) -> WebSocketState:
        """Public read access to WebSocket connection state (worst of both streams).

        Severity order: SUSPENDED > REST_DEGRADED > RESYNCING > PRIMARY > DISCONNECTED.
        """
        severity = {
            WebSocketState.DISCONNECTED: 0,
            WebSocketState.PRIMARY: 1,
            WebSocketState.RESYNCING: 2,
            WebSocketState.REST_DEGRADED: 3,
            WebSocketState.SUSPENDED: 4,
        }
        if severity[self._kline_ws_state] >= severity[self._user_ws_state]:
            return self._kline_ws_state
        return self._user_ws_state

    @property
    def ws_healthy(self) -> bool:
        """Kline stream must be alive and have received at least one event."""
        if self._kline_ws_state != WebSocketState.PRIMARY:
            return False
        if not self._kline_event_received:
            return False  # Not yet confirmed — don't prefer WS cache
        kline_age = (datetime.now(UTC) - self._last_kline_event_time).total_seconds()
        return kline_age < DEFAULT_WS_KLINE_STALENESS_THRESHOLD

    def mark_kline_degraded(self) -> None:
        """Transition kline stream to REST_DEGRADED state (thread-safe)."""
        self._kline_ws_state = WebSocketState.REST_DEGRADED
        logger.warning("Kline stream marked REST_DEGRADED")

    def mark_user_degraded(self) -> None:
        """Transition user stream to REST_DEGRADED state (thread-safe)."""
        self._user_ws_state = WebSocketState.REST_DEGRADED
        logger.warning("User stream marked REST_DEGRADED")

    def _on_kline_disconnect(self) -> None:
        """Handle kline WebSocket disconnection. Sets kline state; engine handles resync."""
        self._kline_ws_state = WebSocketState.RESYNCING
        logger.warning("Kline WebSocket disconnected — entering RESYNCING state")

    def _on_user_disconnect(self) -> None:
        """Handle user data WebSocket disconnection. Sets user state; engine handles resync."""
        self._user_ws_state = WebSocketState.RESYNCING
        logger.warning("User data WebSocket disconnected — entering RESYNCING state")

    def reconnect_kline(self) -> bool:
        """Reconnect kline stream with exponential backoff.

        Retries up to DEFAULT_WS_RECONNECT_MAX_RETRIES times before giving up.

        Returns:
            True if reconnect succeeded, False otherwise.
        """
        for attempt in range(1, DEFAULT_WS_RECONNECT_MAX_RETRIES + 1):
            try:
                if self._kline_socket_key and self._twm:
                    self._twm.stop_socket(self._kline_socket_key)
                    self._kline_socket_key = None
                if self.start_kline_stream(
                    self._active_symbol, self._active_timeframe, self._on_kline_cb
                ):
                    return True
            except Exception as e:
                logger.error(
                    "Kline reconnect attempt %d/%d failed: %s",
                    attempt, DEFAULT_WS_RECONNECT_MAX_RETRIES, e,
                )
            if attempt < DEFAULT_WS_RECONNECT_MAX_RETRIES:
                backoff = 2 ** (attempt - 1)
                logger.info("Retrying kline reconnect in %ds...", backoff)
                time.sleep(backoff)
        return False

    def reconnect_user(self, on_user_event: Callable[[dict], None] | None = None) -> bool:
        """Reconnect user data stream with exponential backoff.

        Retries up to DEFAULT_WS_RECONNECT_MAX_RETRIES times before giving up.

        Args:
            on_user_event: Fresh callback for the new stream. If None, reuses
                the previously stored callback (for backward compatibility).

        Returns:
            True if reconnect succeeded, False otherwise.
        """
        callback = on_user_event or self._on_user_event_cb
        if not callback:
            return False
        for attempt in range(1, DEFAULT_WS_RECONNECT_MAX_RETRIES + 1):
            try:
                if self._user_socket_key and self._twm:
                    self._twm.stop_socket(self._user_socket_key)
                if self.start_user_stream(callback):
                    return True
            except Exception as e:
                logger.error(
                    "User stream reconnect attempt %d/%d failed: %s",
                    attempt, DEFAULT_WS_RECONNECT_MAX_RETRIES, e,
                )
            if attempt < DEFAULT_WS_RECONNECT_MAX_RETRIES:
                backoff = 2 ** (attempt - 1)
                logger.info("Retrying user stream reconnect in %ds...", backoff)
                time.sleep(backoff)
        return False


# Aliases for backward compatibility
BinanceDataProvider = BinanceProvider
BinanceExchange = BinanceProvider
