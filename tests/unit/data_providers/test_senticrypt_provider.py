from datetime import datetime
from unittest.mock import Mock, patch

import pandas as pd
import pytest

pytestmark = pytest.mark.unit

try:
    from data_providers.senticrypt_provider import SentiCryptProvider

    SENTICRYPT_AVAILABLE = True
except ImportError:
    SENTICRYPT_AVAILABLE = False
    SentiCryptProvider = Mock


@pytest.mark.skipif(not SENTICRYPT_AVAILABLE, reason="SentiCrypt provider not available")
class TestSentiCryptProvider:
    @pytest.mark.data_provider
    @patch("requests.get")
    def test_senticrypt_data_retrieval(self, mock_get):
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
                "volume": 1000000,
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
                "volume": 1100000,
            },
        ]
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        provider = SentiCryptProvider()
        df = provider.get_historical_sentiment(
            "BTCUSDT", datetime(2024, 1, 1), datetime(2024, 1, 2)
        )
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2

    @pytest.mark.data_provider
    @patch("requests.get")
    def test_senticrypt_error_handling(self, mock_get):
        provider = SentiCryptProvider()
        from requests.exceptions import RequestException

        mock_get.side_effect = RequestException("SentiCrypt API Error")
        try:
            provider.get_historical_sentiment("BTCUSDT", datetime(2024, 1, 1))
        except Exception as e:
            assert len(str(e)) > 0
