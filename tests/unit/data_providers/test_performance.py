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
        mock_data = pd.DataFrame(
            {"open": [50000], "high": [50100], "low": [49900], "close": [50050], "volume": [100]},
            index=[datetime(2022, 1, 1)],
        )
        mock_data_provider.get_historical_data.return_value = mock_data
        cached_provider = CachedDataProvider(mock_data_provider)
        start_date = datetime(2022, 1, 1)
        end_date = datetime(2022, 1, 2)
        _ = cached_provider.get_historical_data("BTCUSDT", "1h", start_date, end_date)
        initial_call_count = mock_data_provider.get_historical_data.call_count
        _ = cached_provider.get_historical_data("BTCUSDT", "1h", start_date, end_date)
        final_call_count = mock_data_provider.get_historical_data.call_count
        assert final_call_count == initial_call_count
