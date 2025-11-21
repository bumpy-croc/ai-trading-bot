from abc import ABC, abstractmethod
from datetime import datetime

import pandas as pd


class DataProvider(ABC):
    """
    Abstract base class for data providers.
    Implementations should handle different data sources (Binance, CSV, etc.)
    """

    def __init__(self):
        self.data = None

    def _process_ohlcv(self, raw_data, timestamp_unit: str = "s") -> pd.DataFrame:
        """Convert raw OHLCV list data to a standardized DataFrame.

        Args:
            raw_data: Iterable of OHLCV records where the first element is a timestamp followed by
                       open, high, low, close, volume. Additional fields are ignored.
            timestamp_unit: 's' if timestamp is in seconds, 'ms' for milliseconds.
        Returns
            pd.DataFrame indexed by timestamp with columns ['open','high','low','close','volume'].
        """
        if not raw_data:
            return pd.DataFrame()

        # Determine columns – we only keep first six values (timestamp + 5 OHLCV)
        df = pd.DataFrame(raw_data)
        if df.shape[1] < 6:
            raise ValueError(
                "Expected at least 6 columns in OHLCV data (timestamp, open, high, low, close, volume)"
            )

        df = df.iloc[:, 0:6]
        df.columns = ["timestamp", "open", "high", "low", "close", "volume"]

        # Timestamp conversion
        if timestamp_unit == "ms":
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        else:
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")

        # Numeric conversion
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        return df.set_index("timestamp")

    @abstractmethod
    def get_historical_data(
        self, symbol: str, timeframe: str, start: datetime, end: datetime | None = None
    ) -> pd.DataFrame:
        """
        Fetch historical market data.

        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            timeframe: Data timeframe (e.g., '1h', '4h', '1d')
            start: Start datetime
            end: End datetime (optional, defaults to current time)

        Returns:
            DataFrame with OHLCV data
        """
        pass

    @abstractmethod
    def get_live_data(self, symbol: str, timeframe: str, limit: int = 100) -> pd.DataFrame:
        """
        Fetch current market data.

        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            timeframe: Data timeframe (e.g., '1h', '4h', '1d')
            limit: Number of candles to fetch

        Returns:
            DataFrame with OHLCV data
        """
        pass

    @abstractmethod
    def update_live_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """
        Update the latest market data.

        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            timeframe: Data timeframe (e.g., '1h', '4h', '1d')

        Returns:
            DataFrame with updated OHLCV data
        """
        pass

    @abstractmethod
    def get_current_price(self, symbol: str) -> float:
        """Get the latest trade/quote price for a symbol"""
        pass
