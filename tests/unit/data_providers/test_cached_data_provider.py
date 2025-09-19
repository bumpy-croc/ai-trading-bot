from datetime import datetime

import pandas as pd

from src.data_providers.cached_data_provider import CachedDataProvider
from src.data_providers.data_provider import DataProvider


class DummyDataProvider(DataProvider):
    def __init__(self):
        super().__init__()
        self.historical_calls = 0

    def get_historical_data(self, symbol, timeframe, start, end=None):
        self.historical_calls += 1
        end = end or start
        index = pd.date_range(start=start, end=end, freq="D")
        data = pd.DataFrame(
            {
                "open": [1.0 + i for i in range(len(index))],
                "high": [1.5 + i for i in range(len(index))],
                "low": [0.5 + i for i in range(len(index))],
                "close": [1.2 + i for i in range(len(index))],
                "volume": [100 + i for i in range(len(index))],
            },
            index=index,
        )
        return data

    def get_live_data(self, symbol, timeframe, limit=100):
        return pd.DataFrame({"close": [1.0]})

    def update_live_data(self, symbol, timeframe):
        return pd.DataFrame({"close": [1.1]})

    def get_current_price(self, symbol):
        return 42.0


def test_cached_provider_persists_year_cache(tmp_path):
    provider = DummyDataProvider()
    cached = CachedDataProvider(provider, cache_dir=str(tmp_path), cache_ttl_hours=1)

    start = datetime(2022, 1, 1)
    end = datetime(2022, 1, 2)

    first = cached.get_historical_data("BTCUSDT", "1d", start, end)
    assert provider.historical_calls == 1
    assert not first.empty

    second = cached.get_historical_data("BTCUSDT", "1d", start, end)
    assert provider.historical_calls == 1, "expected year cache to satisfy second request"
    pd.testing.assert_frame_equal(first, second)

    key = cached._generate_year_cache_key("BTCUSDT", "1d", 2022)
    assert len(key) == 64


def test_cached_provider_handles_single_day_range(tmp_path):
    provider = DummyDataProvider()
    cached = CachedDataProvider(provider, cache_dir=str(tmp_path), cache_ttl_hours=1)

    day = datetime(2022, 6, 1)

    result = cached.get_historical_data("ETHUSDT", "1d", day, day)

    assert not result.empty, "Expected data for a zero-length date range"
    assert provider.historical_calls == 1
