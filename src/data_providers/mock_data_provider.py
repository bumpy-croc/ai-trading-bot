from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from .data_provider import DataProvider


class MockDataProvider(DataProvider):
    """Simple in-memory mock provider for tests and demos."""

    def __init__(self, interval_seconds: int = 60, num_candles: int = 200):
        super().__init__()
        self.interval_seconds = interval_seconds
        self.num_candles = num_candles
        self._live_df = self._generate_series(self.num_candles, self.interval_seconds)

    def _generate_series(self, n: int, step_seconds: int) -> pd.DataFrame:
        now = datetime.now()
        times = [now - timedelta(seconds=step_seconds * (n - i)) for i in range(n)]
        base = 30000.0
        rng = np.random.default_rng(42)
        prices = base + rng.normal(0, 50, size=n).cumsum()
        highs = prices + rng.uniform(0, 25, size=n)
        lows = prices - rng.uniform(0, 25, size=n)
        opens = np.concatenate([[prices[0]], prices[:-1]])
        volumes = rng.uniform(1, 10, size=n)
        df = pd.DataFrame({
            'open': opens,
            'high': highs,
            'low': lows,
            'close': prices,
            'volume': volumes,
        }, index=pd.to_datetime(times))
        df.index.name = 'timestamp'
        return df

    def get_historical_data(self, symbol: str, timeframe: str, start: datetime, end: Optional[datetime] = None) -> pd.DataFrame:
        df = self._live_df.copy()
        mask = (df.index >= start) & ((df.index <= end) if end else True)
        return df.loc[mask]

    def get_live_data(self, symbol: str, timeframe: str, limit: int = 100) -> pd.DataFrame:
        return self._live_df.tail(limit).copy()

    def update_live_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        # Append one more candle to simulate live update
        last_time = self._live_df.index[-1]
        new_time = last_time + timedelta(seconds=self.interval_seconds)
        last_close = float(self._live_df['close'].iloc[-1])
        rng = np.random.default_rng()
        close = last_close + float(rng.normal(0, 10))
        high = max(close, last_close) + float(rng.uniform(0, 5))
        low = min(close, last_close) - float(rng.uniform(0, 5))
        open_price = last_close
        volume = float(rng.uniform(1, 10))
        new_row = pd.DataFrame([[open_price, high, low, close, volume]], columns=['open','high','low','close','volume'], index=[new_time])
        self._live_df = pd.concat([self._live_df, new_row])
        return self._live_df

    def get_current_price(self, symbol: str) -> float:
        return float(self._live_df['close'].iloc[-1])


