import shutil
import tempfile
from datetime import datetime
from unittest.mock import Mock

import pandas as pd
import pytest

pytestmark = pytest.mark.unit

try:
    from data_providers.cached_data_provider import CachedDataProvider

    CACHE_AVAILABLE = True
except ImportError:
    CACHE_AVAILABLE = False
    CachedDataProvider = Mock


@pytest.mark.skipif(not CACHE_AVAILABLE, reason="Cache provider not available")
class TestCachedDataProvider:
    @pytest.mark.data_provider
    def test_cached_provider_initialization(self, mock_data_provider):
        cached_provider = CachedDataProvider(mock_data_provider)
        assert cached_provider.provider == mock_data_provider
        assert hasattr(cached_provider, "cache")

    @pytest.mark.data_provider
    def test_cached_provider_first_call(self, mock_data_provider):
        mock_data = pd.DataFrame(
            {"open": [50000], "high": [50100], "low": [49900], "close": [50050], "volume": [100]},
            index=[datetime(2022, 1, 1)],
        )
        mock_data_provider.get_historical_data.return_value = mock_data
        temp_cache_dir = tempfile.mkdtemp()
        try:
            cached_provider = CachedDataProvider(mock_data_provider, cache_dir=temp_cache_dir)
            result = cached_provider.get_historical_data(
                "BTCUSDT", "1h", datetime(2022, 1, 1), datetime(2022, 1, 2)
            )
            assert mock_data_provider.get_historical_data.call_count >= 1
            assert result is not None
        finally:
            shutil.rmtree(temp_cache_dir, ignore_errors=True)

    @pytest.mark.data_provider
    def test_cached_provider_subsequent_calls(self, mock_data_provider):
        mock_data = pd.DataFrame(
            {"open": [50000], "high": [50100], "low": [49900], "close": [50050], "volume": [100]},
            index=[datetime(2022, 1, 1)],
        )
        mock_data_provider.get_historical_data.return_value = mock_data
        cached_provider = CachedDataProvider(mock_data_provider)
        start_date = datetime(2022, 1, 1)
        end_date = datetime(2022, 1, 2)
        result1 = cached_provider.get_historical_data("BTCUSDT", "1h", start_date, end_date)
        result2 = cached_provider.get_historical_data("BTCUSDT", "1h", start_date, end_date)
        initial_call_count = mock_data_provider.get_historical_data.call_count
        result3 = cached_provider.get_historical_data("BTCUSDT", "1h", start_date, end_date)
        final_call_count = mock_data_provider.get_historical_data.call_count
        assert final_call_count == initial_call_count
        assert len(result1) == len(result2) == len(result3)

    @pytest.mark.data_provider
    def test_cached_provider_error_handling(self, mock_data_provider):
        temp_cache_dir = tempfile.mkdtemp()
        try:
            cached_provider = CachedDataProvider(mock_data_provider, cache_dir=temp_cache_dir)
            mock_data_provider.get_historical_data.side_effect = Exception("Provider error")
            result = cached_provider.get_historical_data(
                "BTCUSDT", "1h", datetime(2022, 1, 1), datetime(2022, 1, 2)
            )
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 0
        finally:
            shutil.rmtree(temp_cache_dir, ignore_errors=True)
