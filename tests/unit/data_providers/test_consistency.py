import pytest
import pandas as pd
from datetime import datetime

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


class TestDataConsistency:
    @pytest.mark.data_provider
    def test_data_format_consistency(self, mock_data_provider):
        mock_data = pd.DataFrame({'open': [50000], 'high': [50100], 'low': [49900], 'close': [50050], 'volume': [100]}, index=[datetime(2022, 1, 1)])
        mock_data_provider.get_historical_data.return_value = mock_data
        providers = [mock_data_provider]
        if CACHE_AVAILABLE:
            providers.append(CachedDataProvider(mock_data_provider))
        results = []
        for provider in providers:
            df = provider.get_historical_data("BTCUSDT", "1h", datetime(2022, 1, 1), datetime(2022, 1, 2))
            if not df.empty:
                results.append(df)
        if len(results) > 1:
            required_columns = {'open', 'high', 'low', 'close', 'volume'}
            for df in results:
                actual_columns = set(df.columns)
                assert required_columns.issubset(actual_columns)
                for col in required_columns:
                    assert pd.api.types.is_numeric_dtype(df[col])

    @pytest.mark.data_provider
    def test_data_type_consistency(self, sample_ohlcv_data):
        assert pd.api.types.is_numeric_dtype(sample_ohlcv_data['open'])
        assert pd.api.types.is_numeric_dtype(sample_ohlcv_data['high'])
        assert pd.api.types.is_numeric_dtype(sample_ohlcv_data['low'])
        assert pd.api.types.is_numeric_dtype(sample_ohlcv_data['close'])
        assert pd.api.types.is_numeric_dtype(sample_ohlcv_data['volume'])
        assert (sample_ohlcv_data['high'] >= sample_ohlcv_data['low']).all()
        assert (sample_ohlcv_data['volume'] >= 0).all()

    @pytest.mark.data_provider
    def test_timestamp_consistency(self, sample_ohlcv_data):
        assert isinstance(sample_ohlcv_data.index, pd.DatetimeIndex)
        assert sample_ohlcv_data.index.is_monotonic_increasing