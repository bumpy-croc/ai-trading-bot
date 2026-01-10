import pandas as pd
import pytest

pytestmark = pytest.mark.integration

from src.engines.live.trading_engine import LiveTradingEngine


class TestDataValidation:
    @pytest.mark.live_trading
    def test_empty_data_handling(self, mock_strategy, mock_data_provider):
        engine = LiveTradingEngine(strategy=mock_strategy, data_provider=mock_data_provider)
        mock_data_provider.get_live_data.return_value = pd.DataFrame()
        result = engine._get_latest_data("BTCUSDT", "1h")
        assert result is None or isinstance(result, pd.DataFrame)

    @pytest.mark.live_trading
    def test_malformed_data_handling(self, mock_strategy, mock_data_provider):
        engine = LiveTradingEngine(strategy=mock_strategy, data_provider=mock_data_provider)
        mock_data_provider.get_live_data.return_value = pd.DataFrame(
            {"price": [50000], "vol": [1000]}
        )
        try:
            _ = engine._get_latest_data("BTCUSDT", "1h")
        except Exception as e:
            assert any(k in str(e).lower() for k in ["column", "key"])  # graceful error

    @pytest.mark.live_trading
    def test_api_rate_limit_handling(self, mock_strategy, mock_data_provider):
        engine = LiveTradingEngine(
            strategy=mock_strategy, data_provider=mock_data_provider, max_consecutive_errors=5
        )
        from requests.exceptions import RequestException

        mock_data_provider.get_live_data.side_effect = RequestException("Rate limit exceeded")
        result = engine._get_latest_data("BTCUSDT", "1h")
        assert result is None
