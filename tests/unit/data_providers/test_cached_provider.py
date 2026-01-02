import shutil
import tempfile
from datetime import datetime
from unittest.mock import Mock

import pandas as pd
import pytest

from src.data_providers.data_provider import DataProvider

pytestmark = pytest.mark.unit

try:
    from src.data_providers.cached_data_provider import CachedDataProvider

    CACHE_AVAILABLE = True
except ImportError:
    CACHE_AVAILABLE = False
    CachedDataProvider = Mock

# Check if parquet support is available (required for cache persistence tests)
try:
    import pyarrow  # noqa: F401

    PARQUET_AVAILABLE = True
except ImportError:
    PARQUET_AVAILABLE = False


class DummyDataProvider(DataProvider):
    """Simple data provider for testing cache behavior without mocks."""

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
    @pytest.mark.skipif(not PARQUET_AVAILABLE, reason="pyarrow not available for parquet support")
    def test_cached_provider_subsequent_calls(self, mock_data_provider):
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
            result1 = cached_provider.get_historical_data("BTCUSDT", "1h", start_date, end_date)
            result2 = cached_provider.get_historical_data("BTCUSDT", "1h", start_date, end_date)
            initial_call_count = mock_data_provider.get_historical_data.call_count
            result3 = cached_provider.get_historical_data("BTCUSDT", "1h", start_date, end_date)
            final_call_count = mock_data_provider.get_historical_data.call_count
            assert final_call_count == initial_call_count
            assert len(result1) == len(result2) == len(result3)
        finally:
            shutil.rmtree(temp_cache_dir, ignore_errors=True)

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


@pytest.mark.skipif(not CACHE_AVAILABLE, reason="Cache provider not available")
@pytest.mark.skipif(not PARQUET_AVAILABLE, reason="pyarrow not available for parquet support")
class TestCachedDataProviderWithDummyProvider:
    """Tests using a real (non-mock) data provider to verify cache behavior."""

    @pytest.mark.data_provider
    def test_cached_provider_persists_year_cache(self, tmp_path):
        """Test that year-based caching works correctly."""
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

        # Verify cache key is a SHA256 hash (64 hex chars)
        key = cached._generate_year_cache_key("BTCUSDT", "1d", 2022)
        assert len(key) == 64

    @pytest.mark.data_provider
    def test_cached_provider_handles_single_day_range(self, tmp_path):
        """Test that single-day (zero-length) date ranges are handled correctly."""
        provider = DummyDataProvider()
        cached = CachedDataProvider(provider, cache_dir=str(tmp_path), cache_ttl_hours=1)

        day = datetime(2022, 6, 1)

        result = cached.get_historical_data("ETHUSDT", "1d", day, day)

        assert not result.empty, "Expected data for a zero-length date range"
        assert provider.historical_calls == 1
