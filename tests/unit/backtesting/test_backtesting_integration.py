"""Backtesting integration checks with auxiliary services."""

from datetime import datetime

from src.backtesting.engine import Backtester
from src.strategies.ml_basic import MlBasic


class TestBacktestingIntegration:
    """Ensure integrations such as database and sentiment work under test."""

    def test_strategy_integration(self, mock_data_provider, sample_ohlcv_data):
        """Strategies should plug into the backtester cleanly."""

        adaptive_strategy = MlBasic()
        mock_data_provider.get_historical_data.return_value = sample_ohlcv_data

        backtester = Backtester(
            strategy=adaptive_strategy, data_provider=mock_data_provider, initial_balance=10000
        )

        results = backtester.run("BTCUSDT", "1h", datetime(2024, 1, 1))

        assert isinstance(results, dict)
        assert "total_trades" in results

    def test_database_logging_integration(self, mock_data_provider, sample_ohlcv_data):
        """Database logging flag should attach a DB manager."""

        strategy = MlBasic()
        mock_data_provider.get_historical_data.return_value = sample_ohlcv_data

        backtester = Backtester(
            strategy=strategy,
            data_provider=mock_data_provider,
            initial_balance=10000,
            log_to_database=True,
        )

        results = backtester.run("BTCUSDT", "1h", datetime(2024, 1, 1))

        assert isinstance(results, dict)
        assert backtester.db_manager is not None

    def test_sentiment_integration(
        self, mock_data_provider, mock_sentiment_provider, sample_ohlcv_data
    ):
        """Sentiment providers should integrate without issue."""

        strategy = MlBasic()
        mock_data_provider.get_historical_data.return_value = sample_ohlcv_data

        backtester = Backtester(
            strategy=strategy,
            data_provider=mock_data_provider,
            sentiment_provider=mock_sentiment_provider,
            initial_balance=10000,
        )

        results = backtester.run("BTCUSDT", "1h", datetime(2024, 1, 1))

        assert isinstance(results, dict)
        assert backtester.sentiment_provider == mock_sentiment_provider
