from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from data_providers.data_provider import DataProvider


class MockDataProvider(DataProvider):
	"""In-memory data provider generating synthetic OHLCV for fast tests.

	- Deterministic series using a seeded random walk
	- Supports get_historical_data and simple live updates
	"""

	def __init__(self, interval_seconds: int = 60, num_candles: int = 200, seed: int = 42):
		super().__init__()
		self.interval_seconds = interval_seconds
		self.num_candles = num_candles
		self.seed = seed
		self._init_data()

	def _init_data(self):
		rng = np.random.default_rng(self.seed)
		start = datetime.now() - timedelta(seconds=self.interval_seconds * self.num_candles)
		times = [start + timedelta(seconds=self.interval_seconds * i) for i in range(self.num_candles)]
		base = 30000.0
		# Slight downward drift to mimic bear conditions
		steps = rng.normal(loc=-0.5, scale=10.0, size=self.num_candles)
		prices = base + steps.cumsum()
		opens = np.concatenate([[prices[0]], prices[:-1]])
		highs = np.maximum(opens, prices) + rng.uniform(0.0, 25.0, size=self.num_candles)
		lows = np.minimum(opens, prices) - rng.uniform(0.0, 25.0, size=self.num_candles)
		volumes = rng.lognormal(mean=8.0, sigma=0.25, size=self.num_candles)
		df = pd.DataFrame(
			{
				"open": opens,
				"high": highs,
				"low": lows,
				"close": prices,
				"volume": volumes,
			},
			index=pd.to_datetime(times),
		)
		df.index.name = "timestamp"
		self._live_df = df

	def get_historical_data(self, symbol: str, timeframe: str, start: datetime, end: Optional[datetime] = None) -> pd.DataFrame:
		df = self._live_df.copy()
		mask = (df.index >= pd.to_datetime(start)) & ((df.index <= pd.to_datetime(end)) if end else True)
		return df.loc[mask]

	def get_live_data(self, symbol: str, timeframe: str, limit: int = 100) -> pd.DataFrame:
		return self._live_df.tail(limit).copy()

	def update_live_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
		last_time = self._live_df.index[-1]
		new_time = last_time + timedelta(seconds=self.interval_seconds)
		last_close = float(self._live_df["close"].iloc[-1])
		rng = np.random.default_rng()
		close = last_close + float(rng.normal(loc=-0.5, scale=10.0))
		high = max(close, last_close) + float(rng.uniform(0, 5))
		low = min(close, last_close) - float(rng.uniform(0, 5))
		open_price = last_close
		volume = float(rng.lognormal(mean=8.0, sigma=0.25))
		new_row = pd.DataFrame(
			[[open_price, high, low, close, volume]],
			columns=["open", "high", "low", "close", "volume"],
			index=[new_time],
		)
		self._live_df = pd.concat([self._live_df, new_row])
		return self._live_df

	def get_current_price(self, symbol: str) -> float:
		return float(self._live_df["close"].iloc[-1])

