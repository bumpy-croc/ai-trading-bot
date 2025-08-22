from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from src.data_providers.data_provider import DataProvider

_TIMEFRAME_TO_FREQ = {
    "1m": "T",
    "5m": "5T",
    "15m": "15T",
    "30m": "30T",
    "1h": "H",
    "4h": "4H",
    "1d": "D",
}


class MockDataProvider(DataProvider):
    """Synthetic data provider used by tests.

    Generates a simple random-walk price series with reasonable OHLCV structure.
    """

    def __init__(
        self,
        interval_seconds: int = 3600,
        num_candles: int = 1000,
        seed: int | None = 42,
        base_price: float = 30000.0,
    ):
        super().__init__()
        self.interval_seconds = interval_seconds
        self.num_candles = num_candles
        self.seed = seed
        self.base_price = base_price
        self.data: pd.DataFrame | None = None

    def _ensure_data(self, start: datetime, end: datetime | None, timeframe: str) -> pd.DataFrame:
        """Create synthetic data covering [start, end] if not already available."""
        freq = _TIMEFRAME_TO_FREQ.get(timeframe, "h")
        if end is None:
            end = datetime.now()
        index = pd.date_range(start=start, end=end, freq=freq)
        if len(index) == 0:
            # Provide at least some data
            index = pd.date_range(end=end, periods=max(self.num_candles, 100), freq=freq)

        if self.seed is not None:
            np.random.seed(self.seed)

        # Random walk for close
        steps = np.random.normal(loc=0.0, scale=0.002, size=len(index))
        close = self.base_price * np.exp(np.cumsum(steps))

        # Construct OHLC around close with small intrabar ranges
        spread = np.maximum(0.0005 * close, 1.0)
        open_ = np.concatenate(([close[0]], close[:-1]))
        high = np.maximum.reduce([open_, close, close + spread])
        low = np.minimum.reduce([open_, close, close - spread])
        volume = np.random.uniform(500, 5000, size=len(index))

        df = pd.DataFrame(
            {
                "open": open_.astype(float),
                "high": high.astype(float),
                "low": low.astype(float),
                "close": close.astype(float),
                "volume": volume.astype(float),
            },
            index=index,
        )
        self.data = df
        return df

    def get_historical_data(
        self, symbol: str, timeframe: str, start: datetime, end: datetime | None = None
    ) -> pd.DataFrame:
        if self.data is None:
            self._ensure_data(start=start, end=end, timeframe=timeframe)
        # Ensure coverage for the requested range
        if self.data is not None:
            data_start, data_end = self.data.index.min(), self.data.index.max()
            if start < data_start or (end and end > data_end):
                self._ensure_data(start=start, end=end, timeframe=timeframe)
            return self.data.loc[start:end] if end is not None else self.data.loc[start:]
        return pd.DataFrame()

    def get_live_data(self, symbol: str, timeframe: str, limit: int = 100) -> pd.DataFrame:
        # Provide the last `limit` candles
        if self.data is None:
            end = datetime.now()
            start = end - timedelta(seconds=self.interval_seconds * max(self.num_candles, limit))
            self._ensure_data(start=start, end=end, timeframe=timeframe)
        return self.data.tail(limit) if self.data is not None else pd.DataFrame()

    def update_live_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        # Append one new candle by rolling forward the random walk
        freq = _TIMEFRAME_TO_FREQ.get(timeframe, "h")
        if self.data is None or len(self.data) == 0:
            return self.get_live_data(symbol, timeframe, limit=1)
        last_idx = self.data.index[-1]
        if freq.endswith("T"):
            minutes = int(freq.replace("T", "")) if freq != "T" else 1
            next_idx = last_idx + timedelta(minutes=minutes)
        elif freq.endswith("h"):
            hours = int(freq.replace("h", "")) if freq != "h" else 1
            next_idx = last_idx + timedelta(hours=hours)
        elif freq == "D":
            next_idx = last_idx + timedelta(days=1)
        else:
            next_idx = last_idx + timedelta(seconds=self.interval_seconds)

        # Simple next step
        if self.seed is not None:
            np.random.seed(int(pd.Timestamp(next_idx).value % (2**32 - 1)))
        step = np.random.normal(loc=0.0, scale=0.002)
        last_close = float(self.data["close"].iloc[-1])
        new_close = max(1.0, last_close * np.exp(step))
        spread = max(1.0, 0.0005 * new_close)
        new_open = last_close
        new_high = max(new_open, new_close, new_close + spread)
        new_low = min(new_open, new_close, new_close - spread)
        new_volume = float(np.random.uniform(500, 5000))

        new_row = pd.DataFrame(
            {
                "open": [new_open],
                "high": [new_high],
                "low": [new_low],
                "close": [new_close],
                "volume": [new_volume],
            },
            index=[next_idx],
        )
        self.data = pd.concat([self.data, new_row])
        return self.data.tail(1)

    def get_current_price(self, symbol: str) -> float:
        if self.data is None or len(self.data) == 0:
            return float(self.base_price)
        return float(self.data["close"].iloc[-1])
