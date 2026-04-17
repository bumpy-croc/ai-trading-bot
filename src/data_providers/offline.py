"""Offline data providers for deterministic backtests and experiments.

These providers do not hit the network:

* :class:`FixtureProvider` reads pre-captured OHLCV data from a feather file.
* :class:`RandomWalkProvider` generates synthetic series from a seeded RNG.

Both are used by the experimentation framework (``src.experiments``) to keep
tests reproducible without depending on live market data.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_providers.data_provider import DataProvider

_FREQ_MAP = {
    "1m": "1min",
    "5m": "5min",
    "15m": "15min",
    "1h": "1h",
    "4h": "4h",
    "1d": "1d",
}


def _to_utc_timestamp(value: datetime) -> pd.Timestamp:
    """Return a tz-aware UTC ``pd.Timestamp`` regardless of input tz state."""
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


class FixtureProvider(DataProvider):
    """Serve OHLCV data from a pre-captured feather file."""

    def __init__(self, path: Path):
        super().__init__()
        self.path = path
        self.df = self._load()

    def _load(self) -> pd.DataFrame:
        try:
            df = pd.read_feather(self.path)
        except (FileNotFoundError, OSError):
            return pd.DataFrame()
        df.set_index("timestamp", inplace=True)
        # Normalize the index to UTC-aware so comparisons against UTC-aware
        # datetimes from callers (experiments, live pipelines) don't raise
        # ``TypeError: Invalid comparison between dtype=datetime64[ns] and
        # Timestamp``. Fixture files may be stored tz-naive; treat them as
        # UTC by convention.
        if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        return df

    def get_historical_data(
        self, symbol: str, timeframe: str, start: datetime, end: datetime | None = None
    ) -> pd.DataFrame:  # type: ignore[override]
        if self.df.empty:
            return self.df
        end_ts = _to_utc_timestamp(end) if end is not None else pd.Timestamp.now(tz="UTC")
        start_ts = _to_utc_timestamp(start)
        return self.df.loc[(self.df.index >= start_ts) & (self.df.index <= end_ts)].copy()

    def get_live_data(
        self, symbol: str, timeframe: str, limit: int = 100
    ) -> pd.DataFrame:  # type: ignore[override]
        if self.df.empty:
            return self.df
        return self.df.tail(limit).copy()

    def update_live_data(
        self, symbol: str, timeframe: str
    ) -> pd.DataFrame:  # type: ignore[override]
        return self.get_live_data(symbol, timeframe, limit=1)

    def get_current_price(self, symbol: str) -> float:  # type: ignore[override]
        if self.df.empty:
            return 0.0
        return float(self.df["close"].iloc[-1])


class RandomWalkProvider(DataProvider):
    """Generate synthetic OHLCV series using a seeded random walk."""

    def __init__(
        self,
        start: datetime,
        end: datetime,
        timeframe: str = "1h",
        start_price: float = 30000.0,
        vol: float = 0.01,
        seed: int | None = None,
    ):
        super().__init__()
        self.timeframe = timeframe
        self.seed = seed
        self.df = self._generate(start, end, timeframe, start_price, vol)

    @staticmethod
    def _freq(timeframe: str) -> str:
        return _FREQ_MAP.get(timeframe, "1h")

    def _generate(
        self,
        start: datetime,
        end: datetime,
        timeframe: str,
        start_price: float,
        vol: float,
    ) -> pd.DataFrame:
        rng = np.random.default_rng(self.seed)
        # Generate a UTC-aware index so comparisons against UTC-aware caller
        # bounds (experiments pipeline) don't raise TypeError.
        idx = pd.date_range(
            start=_to_utc_timestamp(start),
            end=_to_utc_timestamp(end),
            freq=self._freq(timeframe),
        )
        if len(idx) < 2:
            return pd.DataFrame(
                index=idx, columns=["open", "high", "low", "close", "volume"]
            ).fillna(0.0)

        prices = [start_price]
        for _ in range(1, len(idx)):
            shock = rng.normal(0, vol)
            prices.append(max(1.0, prices[-1] * (1.0 + shock)))
        prices_arr = np.array(prices)
        highs = prices_arr * (1.0 + np.abs(rng.normal(0, vol / 2, size=len(prices_arr))))
        lows = prices_arr * (1.0 - np.abs(rng.normal(0, vol / 2, size=len(prices_arr))))
        opens = np.r_[prices_arr[0], prices_arr[:-1]]
        volume = rng.uniform(1000.0, 10000.0, size=len(prices_arr))
        return pd.DataFrame(
            {
                "open": opens,
                "high": highs,
                "low": lows,
                "close": prices_arr,
                "volume": volume,
            },
            index=idx,
        )

    def get_historical_data(
        self, symbol: str, timeframe: str, start: datetime, end: datetime | None = None
    ) -> pd.DataFrame:  # type: ignore[override]
        end_ts = _to_utc_timestamp(end) if end is not None else pd.Timestamp.now(tz="UTC")
        start_ts = _to_utc_timestamp(start)
        return self.df.loc[(self.df.index >= start_ts) & (self.df.index <= end_ts)].copy()

    def get_live_data(
        self, symbol: str, timeframe: str, limit: int = 100
    ) -> pd.DataFrame:  # type: ignore[override]
        return self.df.tail(limit).copy()

    def update_live_data(
        self, symbol: str, timeframe: str
    ) -> pd.DataFrame:  # type: ignore[override]
        return self.get_live_data(symbol, timeframe, limit=1)

    def get_current_price(self, symbol: str) -> float:  # type: ignore[override]
        if self.df.empty:
            return 0.0
        return float(self.df["close"].iloc[-1])


__all__ = ["FixtureProvider", "RandomWalkProvider"]
