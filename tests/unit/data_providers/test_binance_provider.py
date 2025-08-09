import pytest
from unittest.mock import Mock, patch
from datetime import datetime
import pandas as pd

pytestmark = pytest.mark.unit

try:
    from data_providers.binance_provider import BinanceProvider
    BINANCE_AVAILABLE = True
except ImportError:
    BINANCE_AVAILABLE = False
    BinanceProvider = Mock


@pytest.mark.skipif(not BINANCE_AVAILABLE, reason="Binance provider not available")
class TestBinanceDataProvider:
    @pytest.mark.data_provider
    def test_binance_provider_initialization(self):
        with patch('data_providers.binance_provider.get_config') as mock_config:
            mock_config_obj = Mock()
            mock_config_obj.get_required.return_value = "fake_key"
            mock_config.return_value = mock_config_obj
            provider = BinanceProvider()
            assert provider is not None

    @pytest.mark.data_provider
    @patch('data_providers.binance_provider.Client')
    def test_binance_historical_data_success(self, mock_client_class):
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.get_historical_klines.return_value = [
            [1640995200000, "50000", "50100", "49900", "50050", "100", 1640995259999, "5000000", 1000, "50", "2500000", "0"],
            [1640998800000, "50050", "50150", "49950", "50100", "110", 1640998859999, "5500000", 1100, "55", "2750000", "0"],
        ]
        with patch('data_providers.binance_provider.get_config') as mock_config:
            mock_config_obj = Mock()
            mock_config_obj.get_required.return_value = "fake_key"
            mock_config.return_value = mock_config_obj
            provider = BinanceProvider()
            df = provider.get_historical_data("BTCUSDT", "1h", datetime(2022, 1, 1), datetime(2022, 1, 2))
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume'])

    @pytest.mark.data_provider
    @patch('data_providers.binance_provider.Client')
    def test_binance_api_error_handling(self, mock_client_class):
        mock_client = Mock(); mock_client_class.return_value = mock_client
        mock_client.get_historical_klines.side_effect = Exception("API Error")
        with patch('data_providers.binance_provider.get_config') as mock_config:
            mock_config_obj = Mock(); mock_config_obj.get_required.return_value = "fake_key"; mock_config.return_value = mock_config_obj
            provider = BinanceProvider()
            try:
                result = provider.get_historical_data("BTCUSDT", "1h", datetime(2022, 1, 1))
                assert isinstance(result, pd.DataFrame)
                assert len(result) == 0
            except Exception as e:
                assert "api" in str(e).lower() or "error" in str(e).lower()

    @pytest.mark.data_provider
    @patch('data_providers.binance_provider.Client')
    def test_binance_rate_limit_handling(self, mock_client_class):
        mock_client = Mock(); mock_client_class.return_value = mock_client
        try:
            from binance.exceptions import BinanceAPIException
            mock_response = Mock(); mock_response.text = '{"code": -1003, "msg": "Rate limit exceeded"}'
            exception_to_raise = BinanceAPIException(mock_response, status_code=429, text=mock_response.text)
        except (ImportError, TypeError, AttributeError):
            exception_to_raise = Exception("Rate limit exceeded")
        mock_client.get_historical_klines.side_effect = exception_to_raise
        with patch('data_providers.binance_provider.get_config') as mock_config:
            mock_config_obj = Mock(); mock_config_obj.get_required.return_value = "fake_key"; mock_config.return_value = mock_config_obj
            provider = BinanceProvider()
            try:
                result = provider.get_historical_data("BTCUSDT", "1h", datetime(2022, 1, 1))
                assert isinstance(result, pd.DataFrame)
                assert len(result) == 0
            except Exception as e:
                assert any(s in str(e).lower() for s in ["rate limit", "exceeded", "error"])

    @pytest.mark.data_provider
    @patch('data_providers.binance_provider.Client')
    def test_binance_data_validation(self, mock_client_class):
        mock_client = Mock(); mock_client_class.return_value = mock_client
        with patch('data_providers.binance_provider.get_config') as mock_config:
            mock_config_obj = Mock(); mock_config_obj.get_required.return_value = "fake_key"; mock_config.return_value = mock_config_obj
            provider = BinanceProvider()
            try:
                result = provider.get_historical_data("BTCUSDT", "invalid", datetime.now())
                assert isinstance(result, pd.DataFrame)
                assert len(result) == 0
            except (ValueError, Exception) as e:
                assert any(s in str(e).lower() for s in ["invalid", "timeframe", "error"])