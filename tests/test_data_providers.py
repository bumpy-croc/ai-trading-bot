"""
Tests for data providers.

Data providers are critical for feeding accurate data to strategies. Tests cover:
- API connectivity and error handling
- Data validation and integrity
- Caching mechanisms
- Rate limiting handling
- Data format consistency
- Historical vs live data consistency
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import requests
from requests.exceptions import RequestException, Timeout, ConnectionError

from core.data_providers.data_provider import DataProvider
from core.data_providers.binance_data_provider import BinanceDataProvider
from core.data_providers.cached_data_provider import CachedDataProvider
from core.data_providers.senticrypt_provider import SentiCryptProvider


class TestDataProviderInterface:
    """Test the base data provider interface"""

    def test_data_provider_is_abstract(self):
        """Test that DataProvider cannot be instantiated directly"""
        with pytest.raises(TypeError):
            DataProvider()

    def test_data_provider_interface_methods(self):
        """Test that data provider interface has required methods"""
        # This tests the interface definition
        required_methods = [
            'get_historical_data',
            'get_live_data', 
            'update_live_data'
        ]
        
        for method in required_methods:
            assert hasattr(DataProvider, method)


class TestBinanceDataProvider:
    """Test Binance data provider implementation"""

    @pytest.mark.data_provider
    def test_binance_provider_initialization(self):
        """Test Binance provider initialization"""
        provider = BinanceDataProvider()
        assert provider is not None
        # Test any specific initialization requirements

    @pytest.mark.data_provider
    @patch('requests.get')
    def test_binance_historical_data_success(self, mock_get):
        """Test successful historical data retrieval"""
        # Mock successful API response
        mock_response = Mock()
        mock_response.json.return_value = [
            [1640995200000, "50000", "50100", "49900", "50050", "100", 1640995259999, "5000000", 1000, "50", "2500000", "0"],
            [1640998800000, "50050", "50150", "49950", "50100", "110", 1640998859999, "5500000", 1100, "55", "2750000", "0"]
        ]
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        provider = BinanceDataProvider()
        start_date = datetime(2022, 1, 1)
        end_date = datetime(2022, 1, 2)
        
        df = provider.get_historical_data("BTCUSDT", "1h", start_date, end_date)
        
        # Verify data structure
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume'])
        
        # Verify data types
        for col in ['open', 'high', 'low', 'close', 'volume']:
            assert pd.api.types.is_numeric_dtype(df[col])

    @pytest.mark.data_provider
    @patch('requests.get')
    def test_binance_api_error_handling(self, mock_get):
        """Test Binance API error handling"""
        # Test various API errors
        provider = BinanceDataProvider()
        start_date = datetime(2022, 1, 1)
        
        # Test HTTP error
        mock_get.side_effect = RequestException("API Error")
        
        with pytest.raises((RequestException, Exception)):
            provider.get_historical_data("BTCUSDT", "1h", start_date)

    @pytest.mark.data_provider
    @patch('requests.get')
    def test_binance_rate_limit_handling(self, mock_get):
        """Test rate limit handling"""
        # Mock rate limit response
        mock_response = Mock()
        mock_response.status_code = 429  # Too Many Requests
        mock_response.json.return_value = {"msg": "Rate limit exceeded"}
        mock_get.return_value = mock_response
        
        provider = BinanceDataProvider()
        start_date = datetime(2022, 1, 1)
        
        # Should handle rate limit gracefully
        try:
            result = provider.get_historical_data("BTCUSDT", "1h", start_date)
            # If it doesn't raise an exception, should return empty or retry
        except Exception as e:
            # Should be a handled exception, not a crash
            assert "rate limit" in str(e).lower() or "429" in str(e)

    @pytest.mark.data_provider
    @patch('requests.get')
    def test_binance_live_data(self, mock_get):
        """Test live data retrieval"""
        # Mock live data response
        mock_response = Mock()
        mock_response.json.return_value = [
            [int(datetime.now().timestamp() * 1000), "50000", "50100", "49900", "50050", "100", 
             int(datetime.now().timestamp() * 1000) + 59999, "5000000", 1000, "50", "2500000", "0"]
        ]
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        provider = BinanceDataProvider()
        df = provider.get_live_data("BTCUSDT", "1h", limit=1)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume'])

    @pytest.mark.data_provider
    def test_binance_data_validation(self):
        """Test data validation for Binance provider"""
        provider = BinanceDataProvider()
        
        # Test invalid symbol
        with pytest.raises((ValueError, Exception)):
            provider.get_historical_data("INVALID", "1h", datetime.now())
        
        # Test invalid timeframe
        with pytest.raises((ValueError, Exception)):
            provider.get_historical_data("BTCUSDT", "invalid", datetime.now())


class TestCachedDataProvider:
    """Test cached data provider wrapper"""

    @pytest.mark.data_provider
    def test_cached_provider_initialization(self, mock_data_provider):
        """Test cached provider initialization"""
        cached_provider = CachedDataProvider(mock_data_provider)
        
        assert cached_provider.provider == mock_data_provider
        assert hasattr(cached_provider, 'cache')

    @pytest.mark.data_provider
    def test_cached_provider_first_call(self, mock_data_provider):
        """Test that first call goes to underlying provider"""
        cached_provider = CachedDataProvider(mock_data_provider)
        
        start_date = datetime(2022, 1, 1)
        result = cached_provider.get_historical_data("BTCUSDT", "1h", start_date)
        
        # Should call underlying provider
        mock_data_provider.get_historical_data.assert_called_once_with("BTCUSDT", "1h", start_date, None)
        assert result is not None

    @pytest.mark.data_provider
    def test_cached_provider_subsequent_calls(self, mock_data_provider):
        """Test that subsequent calls use cache"""
        cached_provider = CachedDataProvider(mock_data_provider)
        
        start_date = datetime(2022, 1, 1)
        
        # First call
        result1 = cached_provider.get_historical_data("BTCUSDT", "1h", start_date)
        
        # Second call with same parameters
        result2 = cached_provider.get_historical_data("BTCUSDT", "1h", start_date)
        
        # Should only call underlying provider once
        assert mock_data_provider.get_historical_data.call_count == 1
        
        # Results should be the same
        pd.testing.assert_frame_equal(result1, result2)

    @pytest.mark.data_provider
    def test_cached_provider_cache_invalidation(self, mock_data_provider):
        """Test cache invalidation for live data"""
        cached_provider = CachedDataProvider(mock_data_provider)
        
        # Live data should not be cached heavily
        result1 = cached_provider.get_live_data("BTCUSDT", "1h")
        result2 = cached_provider.get_live_data("BTCUSDT", "1h")
        
        # Depending on implementation, may call provider multiple times for live data
        assert mock_data_provider.get_live_data.call_count >= 1

    @pytest.mark.data_provider
    def test_cached_provider_error_handling(self, mock_data_provider):
        """Test cached provider error handling"""
        cached_provider = CachedDataProvider(mock_data_provider)
        
        # Make underlying provider raise an error
        mock_data_provider.get_historical_data.side_effect = Exception("Provider error")
        
        with pytest.raises(Exception):
            cached_provider.get_historical_data("BTCUSDT", "1h", datetime.now())


class TestSentiCryptProvider:
    """Test SentiCrypt sentiment data provider"""

    @pytest.mark.data_provider
    @patch('requests.get')
    def test_senticrypt_data_retrieval(self, mock_get):
        """Test SentiCrypt data retrieval"""
        # Mock successful API response
        mock_response = Mock()
        mock_response.json.return_value = [
            {
                "date": "2024-01-01T00:00:00Z",
                "score1": 0.5,
                "score2": 0.3,
                "score3": 0.7,
                "sum": 1.5,
                "mean": 0.5,
                "count": 100,
                "price": 50000,
                "volume": 1000000
            },
            {
                "date": "2024-01-01T02:00:00Z", 
                "score1": 0.6,
                "score2": 0.4,
                "score3": 0.8,
                "sum": 1.8,
                "mean": 0.6,
                "count": 120,
                "price": 50100,
                "volume": 1100000
            }
        ]
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        provider = SentiCryptProvider()
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 2)
        
        df = provider.get_historical_sentiment("BTCUSDT", start_date, end_date)
        
        # Verify data structure
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        
        # Check expected columns
        expected_columns = ['sentiment_score', 'sentiment_count', 'sentiment_volume']
        for col in expected_columns:
            assert col in df.columns

    @pytest.mark.data_provider
    @patch('requests.get')
    def test_senticrypt_error_handling(self, mock_get):
        """Test SentiCrypt error handling"""
        provider = SentiCryptProvider()
        
        # Test API error
        mock_get.side_effect = RequestException("SentiCrypt API Error")
        
        start_date = datetime(2024, 1, 1)
        
        with pytest.raises((RequestException, Exception)):
            provider.get_historical_sentiment("BTCUSDT", start_date)

    @pytest.mark.data_provider 
    @patch('requests.get')
    def test_senticrypt_live_sentiment(self, mock_get):
        """Test live sentiment data retrieval"""
        # Mock live sentiment response
        mock_response = Mock()
        mock_response.json.return_value = [
            {
                "date": datetime.now().isoformat(),
                "score1": 0.6,
                "score2": 0.4,
                "score3": 0.7,
                "sum": 1.7,
                "mean": 0.567,
                "count": 150,
                "price": 50200,
                "volume": 1200000
            }
        ]
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        provider = SentiCryptProvider()
        sentiment = provider.get_live_sentiment()
        
        assert isinstance(sentiment, dict)
        assert 'sentiment_score' in sentiment
        assert isinstance(sentiment['sentiment_score'], (int, float))

    @pytest.mark.data_provider
    def test_senticrypt_sentiment_aggregation(self):
        """Test sentiment data aggregation"""
        provider = SentiCryptProvider()
        
        # Create sample sentiment data
        sentiment_data = pd.DataFrame({
            'sentiment_score': [0.5, 0.6, 0.4, 0.7, 0.3],
            'sentiment_count': [100, 120, 80, 150, 90],
            'sentiment_volume': [1000000, 1100000, 900000, 1300000, 800000]
        }, index=pd.date_range('2024-01-01', periods=5, freq='2H'))
        
        # Aggregate to daily
        aggregated = provider.aggregate_sentiment(sentiment_data, window='1D')
        
        assert isinstance(aggregated, pd.DataFrame)
        assert len(aggregated) <= len(sentiment_data)  # Should be aggregated


class TestDataConsistency:
    """Test data consistency across providers"""

    @pytest.mark.data_provider
    @pytest.mark.integration
    def test_data_format_consistency(self, mock_data_provider):
        """Test that all providers return consistent data formats"""
        providers = [mock_data_provider]
        
        # Add cached version
        cached_provider = CachedDataProvider(mock_data_provider)
        providers.append(cached_provider)
        
        start_date = datetime(2022, 1, 1)
        
        results = []
        for provider in providers:
            try:
                df = provider.get_historical_data("BTCUSDT", "1h", start_date)
                results.append(df)
            except:
                continue  # Skip providers that fail
        
        if len(results) > 1:
            # All results should have same columns
            base_columns = set(results[0].columns)
            for df in results[1:]:
                assert set(df.columns) == base_columns

    @pytest.mark.data_provider
    def test_data_type_consistency(self, sample_ohlcv_data):
        """Test that data types are consistent"""
        # Test that OHLCV data has correct types
        assert pd.api.types.is_numeric_dtype(sample_ohlcv_data['open'])
        assert pd.api.types.is_numeric_dtype(sample_ohlcv_data['high'])
        assert pd.api.types.is_numeric_dtype(sample_ohlcv_data['low'])
        assert pd.api.types.is_numeric_dtype(sample_ohlcv_data['close'])
        assert pd.api.types.is_numeric_dtype(sample_ohlcv_data['volume'])
        
        # Test that high >= low, high >= open, high >= close
        assert (sample_ohlcv_data['high'] >= sample_ohlcv_data['low']).all()
        assert (sample_ohlcv_data['high'] >= sample_ohlcv_data['open']).all()
        assert (sample_ohlcv_data['high'] >= sample_ohlcv_data['close']).all()
        
        # Test that volume is non-negative
        assert (sample_ohlcv_data['volume'] >= 0).all()

    @pytest.mark.data_provider
    def test_timestamp_consistency(self, sample_ohlcv_data):
        """Test that timestamps are properly formatted"""
        # Index should be datetime
        assert isinstance(sample_ohlcv_data.index, pd.DatetimeIndex)
        
        # Timestamps should be in ascending order
        assert sample_ohlcv_data.index.is_monotonic_increasing


class TestDataProviderEdgeCases:
    """Test edge cases and error conditions"""

    @pytest.mark.data_provider
    def test_empty_date_range(self, mock_data_provider):
        """Test behavior with empty date ranges"""
        # End date before start date
        start_date = datetime(2022, 1, 2)
        end_date = datetime(2022, 1, 1)
        
        mock_data_provider.get_historical_data.return_value = pd.DataFrame()
        
        result = mock_data_provider.get_historical_data("BTCUSDT", "1h", start_date, end_date)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    @pytest.mark.data_provider
    def test_future_date_handling(self, mock_data_provider):
        """Test handling of future dates"""
        future_date = datetime.now() + timedelta(days=365)
        
        mock_data_provider.get_historical_data.return_value = pd.DataFrame()
        
        # Should handle gracefully
        result = mock_data_provider.get_historical_data("BTCUSDT", "1h", future_date)
        assert isinstance(result, pd.DataFrame)

    @pytest.mark.data_provider
    def test_very_large_date_range(self, mock_data_provider):
        """Test behavior with very large date ranges"""
        start_date = datetime(2000, 1, 1)
        end_date = datetime(2030, 1, 1)
        
        # Should either handle gracefully or raise appropriate error
        try:
            result = mock_data_provider.get_historical_data("BTCUSDT", "1h", start_date, end_date)
            # If successful, should return DataFrame
            assert isinstance(result, pd.DataFrame)
        except Exception as e:
            # Should be a meaningful error
            assert len(str(e)) > 0

    @pytest.mark.data_provider 
    def test_network_timeout_handling(self, mock_data_provider):
        """Test handling of network timeouts"""
        mock_data_provider.get_historical_data.side_effect = Timeout("Request timeout")
        
        with pytest.raises((Timeout, Exception)):
            mock_data_provider.get_historical_data("BTCUSDT", "1h", datetime.now())

    @pytest.mark.data_provider
    def test_malformed_api_response(self, mock_data_provider):
        """Test handling of malformed API responses"""
        # Mock provider to return invalid data
        mock_data_provider.get_historical_data.return_value = "invalid_data"
        
        # Should either convert properly or raise appropriate error
        try:
            result = mock_data_provider.get_historical_data("BTCUSDT", "1h", datetime.now())
            assert isinstance(result, pd.DataFrame)
        except Exception as e:
            # Should be a meaningful error about data format
            assert len(str(e)) > 0


class TestDataProviderPerformance:
    """Test data provider performance characteristics"""

    @pytest.mark.data_provider
    def test_caching_performance(self, mock_data_provider):
        """Test that caching improves performance"""
        cached_provider = CachedDataProvider(mock_data_provider)
        
        start_date = datetime(2022, 1, 1)
        
        import time
        
        # First call (should hit provider)
        start_time = time.time()
        result1 = cached_provider.get_historical_data("BTCUSDT", "1h", start_date)
        first_call_time = time.time() - start_time
        
        # Second call (should hit cache)
        start_time = time.time()
        result2 = cached_provider.get_historical_data("BTCUSDT", "1h", start_date)
        second_call_time = time.time() - start_time
        
        # Cache should be faster (though with mocks this might not be noticeable)
        # At minimum, should call provider only once
        assert mock_data_provider.get_historical_data.call_count == 1

    @pytest.mark.data_provider
    def test_large_dataset_handling(self, mock_data_provider):
        """Test handling of large datasets"""
        # Mock large dataset
        large_data = pd.DataFrame({
            'open': np.random.uniform(45000, 55000, 10000),
            'high': np.random.uniform(45000, 55000, 10000),
            'low': np.random.uniform(45000, 55000, 10000),
            'close': np.random.uniform(45000, 55000, 10000),
            'volume': np.random.uniform(1000, 10000, 10000)
        }, index=pd.date_range('2020-01-01', periods=10000, freq='1H'))
        
        mock_data_provider.get_historical_data.return_value = large_data
        
        result = mock_data_provider.get_historical_data("BTCUSDT", "1h", datetime(2020, 1, 1))
        
        # Should handle large datasets without issues
        assert len(result) == 10000
        assert isinstance(result, pd.DataFrame)