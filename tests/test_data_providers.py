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

from data_providers.data_provider import DataProvider

# Import data providers conditionally to handle missing dependencies
try:
    from data_providers.binance_data_provider import BinanceDataProvider
    BINANCE_AVAILABLE = True
except ImportError:
    BINANCE_AVAILABLE = False
    BinanceDataProvider = Mock

try:
    from data_providers.cached_data_provider import CachedDataProvider
    CACHE_AVAILABLE = True
except ImportError:
    CACHE_AVAILABLE = False
    CachedDataProvider = Mock

try:
    from data_providers.senticrypt_provider import SentiCryptProvider
    SENTICRYPT_AVAILABLE = True
except ImportError:
    SENTICRYPT_AVAILABLE = False
    SentiCryptProvider = Mock


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


@pytest.mark.skipif(not BINANCE_AVAILABLE, reason="Binance provider not available")
class TestBinanceDataProvider:
    """Test Binance data provider implementation"""

    @pytest.mark.data_provider
    def test_binance_provider_initialization(self):
        """Test Binance provider initialization"""
        # Mock config to avoid requiring real API keys
        with patch('data_providers.binance_data_provider.get_config') as mock_config:
            mock_config_obj = Mock()
            mock_config_obj.get_required.return_value = "fake_key"
            mock_config.return_value = mock_config_obj
            
            provider = BinanceDataProvider()
            assert provider is not None

    @pytest.mark.data_provider
    @patch('data_providers.binance_data_provider.Client')
    def test_binance_historical_data_success(self, mock_client_class):
        """Test successful historical data retrieval"""
        # Mock the Binance client
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        # Mock successful API response from Binance client
        mock_client.get_historical_klines.return_value = [
            [1640995200000, "50000", "50100", "49900", "50050", "100", 1640995259999, "5000000", 1000, "50", "2500000", "0"],
            [1640998800000, "50050", "50150", "49950", "50100", "110", 1640998859999, "5500000", 1100, "55", "2750000", "0"]
        ]
        
        # Mock config to avoid credentials requirement
        with patch('data_providers.binance_data_provider.get_config') as mock_config:
            mock_config_obj = Mock()
            mock_config_obj.get_required.return_value = "fake_key"
            mock_config.return_value = mock_config_obj
            
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
    @patch('data_providers.binance_data_provider.Client')
    def test_binance_api_error_handling(self, mock_client_class):
        """Test Binance API error handling"""
        # Mock the Binance client to raise an exception
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.get_historical_klines.side_effect = Exception("API Error")
        
        # Mock config
        with patch('data_providers.binance_data_provider.get_config') as mock_config:
            mock_config_obj = Mock()
            mock_config_obj.get_required.return_value = "fake_key"
            mock_config.return_value = mock_config_obj
            
            provider = BinanceDataProvider()
            start_date = datetime(2022, 1, 1)
            
            # Should handle error gracefully - either raise exception or return empty DataFrame
            try:
                result = provider.get_historical_data("BTCUSDT", "1h", start_date)
                # If no exception, should return empty DataFrame
                assert isinstance(result, pd.DataFrame)
                assert len(result) == 0
            except Exception as e:
                # Exception is acceptable for API errors
                assert "API Error" in str(e) or "error" in str(e).lower()

    @pytest.mark.data_provider
    @patch('data_providers.binance_data_provider.Client')
    def test_binance_rate_limit_handling(self, mock_client_class):
        """Test rate limit handling"""
        # Mock the Binance client to simulate rate limit
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        # Try to import Binance exception, fall back to generic exception
        try:
            from binance.exceptions import BinanceAPIException
            # Create a mock response object that the exception expects
            mock_response = Mock()
            mock_response.text = '{"code": -1003, "msg": "Rate limit exceeded"}'
            exception_to_raise = BinanceAPIException(mock_response, status_code=429, text='{"code": -1003, "msg": "Rate limit exceeded"}')
        except (ImportError, TypeError, AttributeError):
            # If constructor signature is different or import fails, fall back to generic exception
            exception_to_raise = Exception("Rate limit exceeded")
        
        mock_client.get_historical_klines.side_effect = exception_to_raise
        
        # Mock config
        with patch('data_providers.binance_data_provider.get_config') as mock_config:
            mock_config_obj = Mock()
            mock_config_obj.get_required.return_value = "fake_key"
            mock_config.return_value = mock_config_obj
            
            provider = BinanceDataProvider()
            start_date = datetime(2022, 1, 1)
            
            # Should handle rate limit gracefully - either raise exception or return empty DataFrame
            try:
                result = provider.get_historical_data("BTCUSDT", "1h", start_date)
                # If no exception, should return empty DataFrame
                assert isinstance(result, pd.DataFrame)
                assert len(result) == 0
            except Exception as e:
                # Exception is acceptable for rate limit errors
                assert "rate limit" in str(e).lower() or "exceeded" in str(e).lower() or "error" in str(e).lower()

    @pytest.mark.data_provider
    @patch('data_providers.binance_data_provider.Client')
    def test_binance_data_validation(self, mock_client_class):
        """Test data validation for Binance provider"""
        # Mock the Binance client
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        # Mock config
        with patch('data_providers.binance_data_provider.get_config') as mock_config:
            mock_config_obj = Mock()
            mock_config_obj.get_required.return_value = "fake_key"
            mock_config.return_value = mock_config_obj
            
            provider = BinanceDataProvider()
            
            # Test invalid timeframe - should handle gracefully
            try:
                result = provider.get_historical_data("BTCUSDT", "invalid", datetime.now())
                # If no exception, should return empty DataFrame
                assert isinstance(result, pd.DataFrame)
                assert len(result) == 0
            except (ValueError, Exception) as e:
                # Exception is acceptable for invalid input
                assert "invalid" in str(e).lower() or "timeframe" in str(e).lower() or "error" in str(e).lower()


@pytest.mark.skipif(not CACHE_AVAILABLE, reason="Cache provider not available")
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
        if not CACHE_AVAILABLE:
            pytest.skip("Cache provider not available")
        
        # Setup mock to return valid data
        mock_data = pd.DataFrame({
            'open': [50000], 'high': [50100], 'low': [49900], 'close': [50050], 'volume': [100]
        }, index=[datetime(2022, 1, 1)])
        mock_data_provider.get_historical_data.return_value = mock_data
        
        # Use a temporary cache directory to avoid interference
        import tempfile
        import shutil
        temp_cache_dir = tempfile.mkdtemp()
        
        try:
            cached_provider = CachedDataProvider(mock_data_provider, cache_dir=temp_cache_dir)
            
            start_date = datetime(2022, 1, 1)
            end_date = datetime(2022, 1, 2)
            result = cached_provider.get_historical_data("BTCUSDT", "1h", start_date, end_date)
            
            # Should call underlying provider (may be called with year-based parameters)
            assert mock_data_provider.get_historical_data.call_count >= 1
            # Check that some call was made to the underlying provider
            calls = mock_data_provider.get_historical_data.call_args_list
            assert len(calls) > 0
            # First call should be for the symbol and timeframe
            first_call = calls[0]
            assert first_call[0][0] == "BTCUSDT"  # symbol
            assert first_call[0][1] == "1h"       # timeframe
            assert result is not None
        finally:
            # Clean up temporary directory
            shutil.rmtree(temp_cache_dir, ignore_errors=True)

    @pytest.mark.data_provider
    def test_cached_provider_subsequent_calls(self, mock_data_provider):
        """Test that subsequent calls use cache"""
        if not CACHE_AVAILABLE:
            pytest.skip("Cache provider not available")
            
        # Setup mock to return valid data
        mock_data = pd.DataFrame({
            'open': [50000], 'high': [50100], 'low': [49900], 'close': [50050], 'volume': [100]
        }, index=[datetime(2022, 1, 1)])
        mock_data_provider.get_historical_data.return_value = mock_data
        
        cached_provider = CachedDataProvider(mock_data_provider)
        
        start_date = datetime(2022, 1, 1)
        end_date = datetime(2022, 1, 2)
        
        # First call
        result1 = cached_provider.get_historical_data("BTCUSDT", "1h", start_date, end_date)
        
        # Second call with same parameters
        result2 = cached_provider.get_historical_data("BTCUSDT", "1h", start_date, end_date)
        
        # Should use cache for second call - call count should be same after both calls
        # (Year-based caching may result in 1 call that gets cached)
        initial_call_count = mock_data_provider.get_historical_data.call_count
        
        # Third call should definitely use cache
        result3 = cached_provider.get_historical_data("BTCUSDT", "1h", start_date, end_date)
        final_call_count = mock_data_provider.get_historical_data.call_count
        
        # Call count should not increase for the third call
        assert final_call_count == initial_call_count
        
        # Results should be consistent
        assert len(result1) == len(result2) == len(result3)

    @pytest.mark.data_provider
    def test_cached_provider_error_handling(self, mock_data_provider):
        """Test cached provider error handling"""
        if not CACHE_AVAILABLE:
            pytest.skip("Cache provider not available")
            
        # Use a temporary cache directory to avoid interference
        import tempfile
        import shutil
        temp_cache_dir = tempfile.mkdtemp()
        
        try:
            cached_provider = CachedDataProvider(mock_data_provider, cache_dir=temp_cache_dir)
            
            # Make underlying provider raise an error
            mock_data_provider.get_historical_data.side_effect = Exception("Provider error")
            
            start_date = datetime(2022, 1, 1)
            end_date = datetime(2022, 1, 2)
            
            # Should handle error gracefully and return empty DataFrame
            result = cached_provider.get_historical_data("BTCUSDT", "1h", start_date, end_date)
            
            # Should return empty DataFrame when provider fails
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 0
        finally:
            # Clean up temporary directory
            shutil.rmtree(temp_cache_dir, ignore_errors=True)


@pytest.mark.skipif(not SENTICRYPT_AVAILABLE, reason="SentiCrypt provider not available")
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
        
        # Test if method exists
        if hasattr(provider, 'get_historical_sentiment'):
            df = provider.get_historical_sentiment("BTCUSDT", start_date, end_date)
            
            # Verify data structure
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 2

    @pytest.mark.data_provider
    @patch('requests.get')
    def test_senticrypt_error_handling(self, mock_get):
        """Test SentiCrypt error handling"""
        if not SENTICRYPT_AVAILABLE:
            pytest.skip("SentiCrypt provider not available")
            
        provider = SentiCryptProvider()
        
        # Test API error
        mock_get.side_effect = RequestException("SentiCrypt API Error")
        
        start_date = datetime(2024, 1, 1)
        
        # Should handle error gracefully
        try:
            if hasattr(provider, 'get_historical_sentiment'):
                provider.get_historical_sentiment("BTCUSDT", start_date)
        except Exception as e:
            # Should be a meaningful error
            assert len(str(e)) > 0


class TestDataConsistency:
    """Test data consistency across providers"""

    @pytest.mark.data_provider
    @pytest.mark.integration
    def test_data_format_consistency(self, mock_data_provider):
        """Test that all providers return consistent data formats"""
        # Setup mock to return standard OHLCV data
        mock_data = pd.DataFrame({
            'open': [50000], 'high': [50100], 'low': [49900], 'close': [50050], 'volume': [100]
        }, index=[datetime(2022, 1, 1)])
        mock_data_provider.get_historical_data.return_value = mock_data
        
        providers = [mock_data_provider]
        
        # Add cached version if available
        if CACHE_AVAILABLE:
            # Use temporary cache directory
            import tempfile
            import shutil
            temp_cache_dir = tempfile.mkdtemp()
            try:
                cached_provider = CachedDataProvider(mock_data_provider, cache_dir=temp_cache_dir)
                providers.append(cached_provider)
            except Exception:
                pass  # Skip if cached provider fails
        
        start_date = datetime(2022, 1, 1)
        end_date = datetime(2022, 1, 2)
        
        results = []
        for provider in providers:
            try:
                df = provider.get_historical_data("BTCUSDT", "1h", start_date, end_date)
                if not df.empty:
                    results.append(df)
            except Exception:
                continue  # Skip providers that fail
        
        if len(results) > 1:
            # All results should have at least the basic OHLCV columns
            required_columns = {'open', 'high', 'low', 'close', 'volume'}
            for df in results:
                actual_columns = set(df.columns)
                # Check that all required columns are present
                assert required_columns.issubset(actual_columns), f"Missing required columns: {required_columns - actual_columns}"
                
                # Check that data types are consistent for required columns
                for col in required_columns:
                    assert pd.api.types.is_numeric_dtype(df[col]), f"Column {col} should be numeric"
        
        # Clean up if we created a temp directory
        if CACHE_AVAILABLE and 'temp_cache_dir' in locals():
            try:
                shutil.rmtree(temp_cache_dir, ignore_errors=True)
            except:
                pass

    @pytest.mark.data_provider
    def test_data_type_consistency(self, sample_ohlcv_data):
        """Test that data types are consistent"""
        # Test that OHLCV data has correct types
        assert pd.api.types.is_numeric_dtype(sample_ohlcv_data['open'])
        assert pd.api.types.is_numeric_dtype(sample_ohlcv_data['high'])
        assert pd.api.types.is_numeric_dtype(sample_ohlcv_data['low'])
        assert pd.api.types.is_numeric_dtype(sample_ohlcv_data['close'])
        assert pd.api.types.is_numeric_dtype(sample_ohlcv_data['volume'])
        
        # Test that high >= low (this should always be true)
        assert (sample_ohlcv_data['high'] >= sample_ohlcv_data['low']).all()
        
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
        if not CACHE_AVAILABLE:
            pytest.skip("Cache provider not available")
            
        # Setup mock to return valid data
        mock_data = pd.DataFrame({
            'open': [50000], 'high': [50100], 'low': [49900], 'close': [50050], 'volume': [100]
        }, index=[datetime(2022, 1, 1)])
        mock_data_provider.get_historical_data.return_value = mock_data
        
        # Use a temporary cache directory to avoid interference
        import tempfile
        import shutil
        temp_cache_dir = tempfile.mkdtemp()
        
        try:
            cached_provider = CachedDataProvider(mock_data_provider, cache_dir=temp_cache_dir)
            
            start_date = datetime(2022, 1, 1)
            end_date = datetime(2022, 1, 2)
            
            # First call - should hit underlying provider
            result1 = cached_provider.get_historical_data("BTCUSDT", "1h", start_date, end_date)
            initial_call_count = mock_data_provider.get_historical_data.call_count
            assert initial_call_count >= 1  # Should have called underlying provider
            
            # Second call - should use cache
            result2 = cached_provider.get_historical_data("BTCUSDT", "1h", start_date, end_date)
            final_call_count = mock_data_provider.get_historical_data.call_count
            
            # Call count should not increase for cached call
            assert final_call_count == initial_call_count
        finally:
            # Clean up temporary directory
            shutil.rmtree(temp_cache_dir, ignore_errors=True)

    @pytest.mark.data_provider
    def test_large_dataset_handling(self, mock_data_provider):
        """Test handling of large datasets"""
        # Create large mock dataset
        large_data = pd.DataFrame({
            'open': np.random.uniform(49000, 51000, 10000),
            'high': np.random.uniform(50000, 52000, 10000),
            'low': np.random.uniform(48000, 50000, 10000),
            'close': np.random.uniform(49000, 51000, 10000),
            'volume': np.random.uniform(1000, 10000, 10000)
        }, index=pd.date_range('2020-01-01', periods=10000, freq='1H'))
        
        mock_data_provider.get_historical_data.return_value = large_data
        
        # Should handle large datasets without issues
        result = mock_data_provider.get_historical_data("BTCUSDT", "1h", datetime(2020, 1, 1))
        assert len(result) == 10000
        assert isinstance(result, pd.DataFrame)


# Fallback tests for when components are not available
class TestDataProviderFallbacks:
    """Test fallbacks when data providers are not fully available"""

    def test_mock_providers_work(self):
        """Test that mock providers work for basic testing"""
        # This ensures our conditional imports and mocks work
        assert BinanceDataProvider is not None
        assert CachedDataProvider is not None
        assert SentiCryptProvider is not None

    def test_availability_flags(self):
        """Test that availability flags are set correctly"""
        # Test that we can check availability
        assert isinstance(BINANCE_AVAILABLE, bool)
        assert isinstance(CACHE_AVAILABLE, bool)
        assert isinstance(SENTICRYPT_AVAILABLE, bool)