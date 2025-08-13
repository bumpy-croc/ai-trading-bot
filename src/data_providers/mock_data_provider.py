from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from data_providers.data_provider import DataProvider


@dataclass
class _StreamConfig:
	interval_seconds: int = 5
	num_candles: int = 500
	seed: int = 42


class MockDataProvider(DataProvider):
	"""In-memory data provider generating synthetic OHLCV for fast tests.

	- Deterministic series using a seeded random walk
	- Supports get_historical_data and simple live updates
	"""

	def __init__(self, interval_seconds: int = 5, num_candles: int = 500, seed: int = 42):
		super().__init__()
		self.cfg = _StreamConfig(interval_seconds=interval_seconds, num_candles=num_candles, seed=seed)
		self._init_data()

	def _init_data(self):
		np.random.seed(self.cfg.seed)
		start = datetime.now() - timedelta(seconds=self.cfg.interval_seconds * self.cfg.num_candles)
		timestamps = [start + timedelta(seconds=i * self.cfg.interval_seconds) for i in range(self.cfg.num_candles)]
		price = 30000.0
		prices = []
		volumes = []
		for _ in range(self.cfg.num_candles):
			# Random walk with slight downward drift to resemble bear periods
			ret = np.random.normal(loc=-0.0001, scale=0.002)
			price = max(1.0, price * (1.0 + ret))
			prices.append(price)
			volumes.append(max(1.0, np.random.lognormal(mean=8.0, sigma=0.25)))
		opens = [prices[i - 1] if i > 0 else prices[0] for i in range(len(prices))]
		highs = [max(o, c) * (1 + np.random.uniform(0.0, 0.001)) for o, c in zip(opens, prices)]
		lows = [min(o, c) * (1 - np.random.uniform(0.0, 0.001)) for o, c in zip(opens, prices)]
		df = pd.DataFrame(
			{
				"open": opens,
				"high": highs,
				"low": lows,
				"close": prices,
				"volume": volumes,
			},
			index=pd.to_datetime(timestamps),
		)
		df.index.name = "timestamp"
		self.data = df

	def get_historical_data(
		self, symbol: str, timeframe: str, start: datetime, end: Optional[datetime] = None
	) -> pd.DataFrame:
		# Return slice within requested time window
		end = end or datetime.now()
		mask = (self.data.index >= pd.to_datetime(start)) & (self.data.index <= pd.to_datetime(end))
		return self.data.loc[mask].copy()

	def get_live_data(self, symbol: str, timeframe: str, limit: int = 100) -> pd.DataFrame:
		return self.data.tail(limit).copy()

	def update_live_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
		# Append one new synthetic candle
		last_time = self.data.index[-1]
		last_close = float(self.data["close"].iloc[-1])
		ret = np.random.normal(loc=-0.0001, scale=0.002)
		new_close = max(1.0, last_close * (1.0 + ret))
		new_open = last_close
		new_high = max(new_open, new_close) * (1 + np.random.uniform(0.0, 0.001))
		new_low = min(new_open, new_close) * (1 - np.random.uniform(0.0, 0.001))
		new_vol = max(1.0, np.random.lognormal(mean=8.0, sigma=0.25))
		new_time = last_time + timedelta(seconds=self.cfg.interval_seconds)
		new_row = pd.DataFrame(
			{"open": [new_open], "high": [new_high], "low": [new_low], "close": [new_close], "volume": [new_vol]},
			index=pd.to_datetime([new_time]),
		)
		new_row.index.name = "timestamp"
		self.data = pd.concat([self.data, new_row])
		return self.data.tail(100).copy()

	def get_current_price(self, symbol: str) -> float:
		return float(self.data["close"].iloc[-1])