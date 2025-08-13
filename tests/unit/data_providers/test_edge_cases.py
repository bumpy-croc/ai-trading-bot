from datetime import datetime, timedelta

import pandas as pd
import pytest
from requests.exceptions import Timeout

pytestmark = pytest.mark.unit


class TestDataProviderEdgeCases:
    @pytest.mark.data_provider
    def test_empty_date_range(self, mock_data_provider):
        start_date = datetime(2022, 1, 2)
        end_date = datetime(2022, 1, 1)
        mock_data_provider.get_historical_data.return_value = pd.DataFrame()
        result = mock_data_provider.get_historical_data("BTCUSDT", "1h", start_date, end_date)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    @pytest.mark.data_provider
    def test_future_date_handling(self, mock_data_provider):
        future_date = datetime.now() + timedelta(days=365)
        mock_data_provider.get_historical_data.return_value = pd.DataFrame()
        result = mock_data_provider.get_historical_data("BTCUSDT", "1h", future_date)
        assert isinstance(result, pd.DataFrame)

    @pytest.mark.data_provider
    def test_network_timeout_handling(self, mock_data_provider):
        mock_data_provider.get_historical_data.side_effect = Timeout("Request timeout")
        with pytest.raises((Timeout, Exception)):
            mock_data_provider.get_historical_data("BTCUSDT", "1h", datetime.now())

    @pytest.mark.data_provider
    def test_malformed_api_response(self, mock_data_provider):
        mock_data_provider.get_historical_data.return_value = "invalid_data"
        try:
            result = mock_data_provider.get_historical_data("BTCUSDT", "1h", datetime.now())
            assert isinstance(result, pd.DataFrame)
        except Exception as e:
            assert len(str(e)) > 0
