import pytest
import pandas as pd
from datetime import datetime
from unittest.mock import Mock, patch

from data_providers.coinbase_provider import CoinbaseProvider

pytestmark = pytest.mark.unit


class TestCoinbaseProvider:
    @pytest.mark.data_provider
    def test_initialization_without_keys(self):
        provider = CoinbaseProvider()
        assert provider is not None

    @pytest.mark.data_provider
    @patch("data_providers.coinbase_provider.requests.Session.get")
    def test_historical_data_fetch(self, mock_get):
        sample_candles = [
            [1640998800, 49000, 51000, 49500, 50000, 12.3],
            [1640995200, 48000, 50000, 48500, 49500, 10.0],
        ]
        mock_response = Mock(); mock_response.status_code = 200; mock_response.json.return_value = sample_candles
        mock_get.return_value = mock_response
        provider = CoinbaseProvider()
        start_date = datetime.utcfromtimestamp(1640995200)
        end_date = datetime.utcfromtimestamp(1640998800)
        df = provider.get_historical_data("BTC-USD", "1h", start_date, end_date)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        for col in ["open", "high", "low", "close", "volume"]:
            assert col in df.columns
            assert pd.api.types.is_numeric_dtype(df[col])
        assert df.index[0] < df.index[1]

    @pytest.mark.data_provider
    @patch("data_providers.coinbase_provider.requests.Session.get")
    def test_current_price(self, mock_get):
        mock_response = Mock(); mock_response.status_code = 200; mock_response.json.return_value = {"price": "12345.67"}
        mock_get.return_value = mock_response
        provider = CoinbaseProvider()
        price = provider.get_current_price("BTC-USD")
        assert price == 12345.67