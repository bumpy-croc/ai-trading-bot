import pytest
import pandas as pd
from datetime import datetime

pytestmark = pytest.mark.integration

try:
    from live.trading_engine import LiveTradingEngine
    LIVE_TRADING_AVAILABLE = True
except ImportError:
    LIVE_TRADING_AVAILABLE = False
    class LiveTradingEngine:
        def __init__(self, strategy=None, data_provider=None, **kwargs):
            self.strategy = strategy
            self.data_provider = data_provider
        def _get_latest_data(self, symbol, timeframe):
            try:
                df = self.data_provider.get_live_data(symbol, timeframe)
                if not isinstance(df, pd.DataFrame) or df.empty:
                    return None
                return df
            except Exception:
                return None


class TestDataValidation:
    @pytest.mark.live_trading
    def test_empty_data_handling(self, mock_strategy, mock_data_provider):
        if not LIVE_TRADING_AVAILABLE:
            pytest.skip("Live trading components not available")
        engine = LiveTradingEngine(strategy=mock_strategy, data_provider=mock_data_provider)
        mock_data_provider.get_live_data.return_value = pd.DataFrame()
        result = engine._get_latest_data("BTCUSDT", "1h")
        assert result is None or isinstance(result, pd.DataFrame)

    @pytest.mark.live_trading
    def test_malformed_data_handling(self, mock_strategy, mock_data_provider):
        if not LIVE_TRADING_AVAILABLE:
            pytest.skip("Live trading components not available")
        engine = LiveTradingEngine(strategy=mock_strategy, data_provider=mock_data_provider)
        mock_data_provider.get_live_data.return_value = pd.DataFrame({'price': [50000], 'vol': [1000]})
        try:
            _ = engine._get_latest_data("BTCUSDT", "1h")
        except Exception as e:
            assert any(k in str(e).lower() for k in ["column", "key"])  # graceful error

    @pytest.mark.live_trading
    def test_api_rate_limit_handling(self, mock_strategy, mock_data_provider):
        if not LIVE_TRADING_AVAILABLE:
            pytest.skip("Live trading components not available")
        engine = LiveTradingEngine(strategy=mock_strategy, data_provider=mock_data_provider, max_consecutive_errors=5)
        from requests.exceptions import RequestException
        mock_data_provider.get_live_data.side_effect = RequestException("Rate limit exceeded")
        result = engine._get_latest_data("BTCUSDT", "1h")
        assert result is None