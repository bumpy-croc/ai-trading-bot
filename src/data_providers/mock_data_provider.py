import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional
from .data_provider import DataProvider
import random

class MockDataProvider(DataProvider):
    """
    Generates fake OHLCV candles at a rapid interval for stress testing.
    """
    def __init__(self, interval_seconds: int = 5, num_candles: int = 200, start_price: float = 30000.0, volatility: float = 0.001):
        super().__init__()
        self.interval_seconds = interval_seconds
        self.num_candles = num_candles
        self.start_price = start_price
        self.volatility = volatility
        self.data = self._generate_candles()

    def _generate_candles(self) -> pd.DataFrame:
        now = datetime.now()
        times = [now - timedelta(seconds=self.interval_seconds * (self.num_candles - i - 1)) for i in range(self.num_candles)]
        prices = [self.start_price]
        for _ in range(1, self.num_candles):
            change = random.gauss(0, self.volatility) * prices[-1]
            prices.append(max(1, prices[-1] + change))
        df = pd.DataFrame({
            'timestamp': times,
            'open': prices,
            'high': [p * (1 + abs(random.gauss(0, self.volatility/2))) for p in prices],
            'low': [p * (1 - abs(random.gauss(0, self.volatility/2))) for p in prices],
            'close': prices,
            'volume': [random.uniform(0.5, 2.0) for _ in prices],
        })
        df = df.set_index('timestamp')
        return df

    def get_historical_data(self, symbol: str, timeframe: str, start: datetime, end: Optional[datetime] = None) -> pd.DataFrame:
        # Ignore symbol/timeframe for mock, just return generated data in range
        df = self.data
        if end is None:
            end = datetime.now()
        mask = (df.index >= start) & (df.index <= end)
        return df.loc[mask].copy()

    def get_live_data(self, symbol: str, timeframe: str, limit: int = 100) -> pd.DataFrame:
        # Return the last `limit` candles
        return self.data.iloc[-limit:].copy()

    def update_live_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        # Append a new fake candle
        last = self.data.iloc[-1]
        new_time = self.data.index[-1] + timedelta(seconds=self.interval_seconds)
        new_open = last['close']
        change = random.gauss(0, self.volatility) * new_open
        new_close = max(1, new_open + change)
        new_high = max(new_open, new_close) * (1 + abs(random.gauss(0, self.volatility/2)))
        new_low = min(new_open, new_close) * (1 - abs(random.gauss(0, self.volatility/2)))
        new_volume = random.uniform(0.5, 2.0)
        new_row = pd.DataFrame({
            'open': [new_open],
            'high': [new_high],
            'low': [new_low],
            'close': [new_close],
            'volume': [new_volume],
        }, index=[new_time])
        self.data = pd.concat([self.data, new_row])
        # Keep only the last num_candles
        self.data = self.data.iloc[-self.num_candles:]
        return self.data.copy() 