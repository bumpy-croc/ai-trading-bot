import shutil
import tempfile
from datetime import datetime

import pandas as pd
import pytest

pytestmark = pytest.mark.unit

try:
    from data_providers.cached_data_provider import CachedDataProvider

    CACHE_AVAILABLE = True
except ImportError:
    CACHE_AVAILABLE = False

    class CachedDataProvider:
        def __init__(self, provider, cache_dir=None):
            self.provider = provider

        def get_historical_data(self, *args, **kwargs):
            return self.provider.get_historical_data(*args, **kwargs)


class TestDataProviderPerformance:
    @pytest.mark.data_provider
    def test_caching_performance(self, mock_data_provider):
        if not CACHE_AVAILABLE:
            pytest.skip("Cache provider not available")

        temp_cache_dir = tempfile.mkdtemp()
        try:
            # Create data that covers the full requested range (Jan 1 to Jan 2 with 1h timeframe)
            dates = pd.date_range(start=datetime(2022, 1, 1), end=datetime(2022, 1, 2), freq="1h")
            mock_data = pd.DataFrame(
                {
                    "open": [50000 + i for i in range(len(dates))],
                    "high": [50100 + i for i in range(len(dates))],
                    "low": [49900 + i for i in range(len(dates))],
                    "close": [50050 + i for i in range(len(dates))],
                    "volume": [100] * len(dates),
                },
                index=dates,
            )
            mock_data_provider.get_historical_data.return_value = mock_data
            cached_provider = CachedDataProvider(mock_data_provider, cache_dir=temp_cache_dir)
            start_date = datetime(2022, 1, 1)
            end_date = datetime(2022, 1, 2)
            _ = cached_provider.get_historical_data("BTCUSDT", "1h", start_date, end_date)
            initial_call_count = mock_data_provider.get_historical_data.call_count
            _ = cached_provider.get_historical_data("BTCUSDT", "1h", start_date, end_date)
            final_call_count = mock_data_provider.get_historical_data.call_count
            assert final_call_count == initial_call_count
        finally:
            shutil.rmtree(temp_cache_dir, ignore_errors=True)
