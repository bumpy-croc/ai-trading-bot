"""Thread-safe rolling kline history maintained by WebSocket events.

Sits between the WebSocket kline stream and the trading engine, providing
a DataFrame that matches the format returned by get_live_data().
"""

import logging
import threading
from datetime import UTC, datetime

import pandas as pd

from src.config.constants import DEFAULT_WS_KLINE_STALENESS_THRESHOLD

logger = logging.getLogger(__name__)


class KlineBuffer:
    """Rolling window of OHLCV candles seeded from REST and maintained by WebSocket events.

    The buffer keeps a fixed-size DataFrame (default 500 candles) that mirrors
    what provider.get_live_data() returns. The trading engine reads from this
    buffer instead of making REST calls on every heartbeat.
    """

    def __init__(self, symbol: str, timeframe: str, provider) -> None:
        """Seed the buffer with historical candles from a REST provider.

        Args:
            symbol: Trading pair symbol (e.g. 'BTCUSDT').
            timeframe: Candle timeframe (e.g. '1h', '4h').
            provider: Data provider with a get_live_data() method.
        """
        self._symbol = symbol
        self._timeframe = timeframe
        self._lock = threading.Lock()

        # Seed from REST
        self._df: pd.DataFrame = provider.get_live_data(symbol, timeframe, limit=500)
        self._last_update: datetime = datetime.now(UTC)

        logger.info(
            "KlineBuffer seeded for %s %s with %d candles",
            symbol,
            timeframe,
            len(self._df),
        )

    # --- Public API ---

    def on_kline(self, event: dict) -> None:
        """Process a kline WebSocket event, updating the rolling window.

        Thread-safe. Handles open-candle updates and candle-close transitions.
        Stale events (older than the current tail) are ignored.

        Args:
            event: Raw WebSocket kline event dict with a 'k' key.
        """
        kline = event.get("k", {})
        if not kline or "t" not in kline:
            return

        with self._lock:
            if self._df.empty:
                self._df = self._parse_kline(kline)
                self._last_update = datetime.now(UTC)
                return

            event_ts = pd.Timestamp(kline["t"], unit="ms")
            tail_ts = self._df.index[-1]

            if event_ts < tail_ts:
                return  # Stale event — don't bump freshness timer

            if event_ts == tail_ts:
                # Update current candle (open or closed — same OHLCV write)
                self._update_current_candle(kline)
            else:
                # event_ts > tail_ts — new candle, roll window
                new_row = self._parse_kline(kline)
                self._df = pd.concat([self._df.iloc[1:], new_row])

            self._last_update = datetime.now(UTC)

    def get_dataframe(self) -> pd.DataFrame:
        """Return a thread-safe copy of the current OHLCV DataFrame.

        Returns:
            DataFrame with columns [open, high, low, close, volume] indexed by timestamp.
        """
        with self._lock:
            return self._df.copy()

    @property
    def is_fresh(self) -> bool:
        """Check whether the buffer has been updated recently.

        Returns:
            True if the last update was within the freshness timeout.
        """
        elapsed = (datetime.now(UTC) - self._last_update).total_seconds()
        return elapsed < DEFAULT_WS_KLINE_STALENESS_THRESHOLD

    def resync_from_rest(self, provider, symbol: str, timeframe: str) -> None:
        """Replace the entire buffer with fresh REST data after reconnection.

        Args:
            provider: Data provider with a get_live_data() method.
            symbol: Trading pair symbol.
            timeframe: Candle timeframe.
        """
        new_df = provider.get_live_data(symbol, timeframe, limit=500)

        with self._lock:
            self._df = new_df
            self._last_update = datetime.now(UTC)

        logger.info(
            "KlineBuffer resynced for %s %s with %d candles",
            symbol,
            timeframe,
            len(new_df),
        )

    # --- Private helpers ---

    def _parse_kline(self, kline: dict) -> pd.DataFrame:
        """Parse a kline dict into a single-row DataFrame matching provider format.

        Args:
            kline: WebSocket kline dict with keys t, o, h, l, c, v.

        Returns:
            Single-row DataFrame indexed by UTC timestamp with OHLCV columns.
        """
        ts = pd.Timestamp(kline["t"], unit="ms")
        return pd.DataFrame(
            {
                "open": [float(kline["o"])],
                "high": [float(kline["h"])],
                "low": [float(kline["l"])],
                "close": [float(kline["c"])],
                "volume": [float(kline["v"])],
            },
            index=pd.Index([ts], name="timestamp"),
        )

    def _update_current_candle(self, kline: dict) -> None:
        """Update the tail row's OHLCV in-place from a partial kline event.

        Must be called while holding self._lock.

        Args:
            kline: WebSocket kline dict with keys o, h, l, c, v.
        """
        idx = self._df.index[-1]
        self._df.at[idx, "open"] = float(kline["o"])
        self._df.at[idx, "high"] = float(kline["h"])
        self._df.at[idx, "low"] = float(kline["l"])
        self._df.at[idx, "close"] = float(kline["c"])
        self._df.at[idx, "volume"] = float(kline["v"])
