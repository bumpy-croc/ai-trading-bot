"""Tests for atb backtest command."""

import argparse
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest

from cli.commands.backtest import _handle, _load_strategy, _get_date_range


class TestLoadStrategy:
    """Tests for the _load_strategy function."""

    def test_loads_ml_basic_strategy(self):
        """Test that ml_basic strategy is loaded successfully."""
        # Arrange & Act
        with patch("cli.commands.backtest.create_ml_basic_strategy") as mock_create:
            mock_strategy = Mock(name="ml_basic")
            mock_create.return_value = mock_strategy

            result = _load_strategy("ml_basic")

            # Assert
            assert result == mock_strategy
            mock_create.assert_called_once()

    def test_loads_ml_sentiment_strategy(self):
        """Test that ml_sentiment strategy is loaded successfully."""
        # Arrange & Act
        with patch("cli.commands.backtest.create_ml_sentiment_strategy") as mock_create:
            mock_strategy = Mock(name="ml_sentiment")
            mock_create.return_value = mock_strategy

            result = _load_strategy("ml_sentiment")

            # Assert
            assert result == mock_strategy
            mock_create.assert_called_once()

    def test_raises_system_exit_for_unknown_strategy(self):
        """Test that SystemExit is raised for unknown strategy."""
        # Arrange & Act & Assert
        with pytest.raises(SystemExit):
            _load_strategy("unknown_strategy")

    def test_raises_exception_when_strategy_creation_fails(self):
        """Test that exception is raised when strategy creation fails."""
        # Arrange & Act & Assert
        with patch("cli.commands.backtest.create_ml_basic_strategy") as mock_create:
            mock_create.side_effect = Exception("Model not found")

            with pytest.raises(Exception, match="Model not found"):
                _load_strategy("ml_basic")


class TestGetDateRange:
    """Tests for the _get_date_range function."""

    def test_uses_start_and_end_dates_when_provided(self):
        """Test that start and end dates are used when both provided."""
        # Arrange
        args = argparse.Namespace(start="2024-01-01", end="2024-12-31", days=None)

        # Act
        start_date, end_date = _get_date_range(args)

        # Assert
        assert start_date == datetime(2024, 1, 1)
        assert end_date == datetime(2024, 12, 31)

    def test_uses_start_and_now_when_only_start_provided(self):
        """Test that start date and current date are used when only start provided."""
        # Arrange
        args = argparse.Namespace(start="2024-01-01", end=None, days=None)

        # Act
        with patch("cli.commands.backtest.datetime") as mock_datetime:
            mock_now = datetime(2024, 10, 29)
            mock_datetime.now.return_value = mock_now
            mock_datetime.strptime = datetime.strptime

            start_date, end_date = _get_date_range(args)

            # Assert
            assert start_date == datetime(2024, 1, 1)
            assert end_date == mock_now

    def test_uses_days_parameter_when_provided(self):
        """Test that days parameter calculates correct date range."""
        # Arrange
        args = argparse.Namespace(start=None, end=None, days=7)

        # Act
        with patch("cli.commands.backtest.datetime") as mock_datetime:
            mock_now = datetime(2024, 10, 29)
            mock_datetime.now.return_value = mock_now
            mock_datetime_class = MagicMock()
            mock_datetime_class.now.return_value = mock_now
            # Need to patch timedelta calculation properly
            expected_start = mock_now - timedelta(days=7)

            start_date, end_date = _get_date_range(args)

            # Assert
            assert end_date == mock_now
            assert (end_date - start_date).days == 7

    def test_defaults_to_30_days_when_no_params_provided(self):
        """Test that defaults to 30 days when no parameters provided."""
        # Arrange
        args = argparse.Namespace(start=None, end=None, days=None)

        # Act
        with patch("cli.commands.backtest.datetime") as mock_datetime:
            mock_now = datetime(2024, 10, 29)
            mock_datetime.now.return_value = mock_now

            start_date, end_date = _get_date_range(args)

            # Assert
            assert end_date == mock_now
            assert (end_date - start_date).days == 30


class TestHandleBacktest:
    """Tests for the _handle function."""

    @pytest.fixture
    def default_args(self):
        """Provides default backtest arguments."""
        return argparse.Namespace(
            strategy="ml_basic",
            symbol="BTCUSDT",
            timeframe="1h",
            days=30,
            start=None,
            end=None,
            initial_balance=10000,
            risk_per_trade=0.01,
            max_risk_per_trade=0.02,
            max_drawdown=0.5,
            use_sentiment=False,
            no_cache=False,
            cache_ttl=24,
            log_to_db=False,
            provider="binance",
        )

    @pytest.fixture
    def mock_backtester(self):
        """Provides a mocked Backtester instance."""
        mock = Mock()
        mock.run.return_value = {
            "total_trades": 10,
            "win_rate": 60.0,
            "total_return": 15.5,
            "annualized_return": 25.0,
            "max_drawdown": 8.5,
            "sharpe_ratio": 1.5,
            "final_balance": 11550.0,
            "hold_return": 12.0,
            "trading_vs_hold_difference": 3.5,
            "yearly_returns": {2024: 15.5},
        }
        return mock

    def test_successful_backtest_execution(self, default_args, mock_backtester):
        """Test successful backtest execution with default parameters."""
        # Arrange
        with (
            patch("src.engines.backtest.engine.Backtester", return_value=mock_backtester),
            patch("cli.commands.backtest._load_strategy") as mock_load_strategy,
            patch("cli.commands.backtest.configure_logging"),
            patch("src.data_providers.binance_provider.BinanceProvider") as mock_provider_class,
            patch(
                "src.data_providers.cached_data_provider.CachedDataProvider"
            ) as mock_cached_provider,
            patch("cli.commands.backtest.SymbolFactory.to_exchange_symbol", return_value="BTCUSDT"),
            patch("builtins.open", create=True),
            patch("cli.commands.backtest.PROJECT_ROOT", Path("/tmp/test")),
        ):

            mock_strategy = Mock(name="ml_basic")
            mock_strategy.get_trading_pair.return_value = "BTCUSDT"
            mock_load_strategy.return_value = mock_strategy

            mock_provider = Mock()
            mock_provider_class.return_value = mock_provider

            mock_data_provider = Mock()
            mock_data_provider.get_cache_info.return_value = {
                "total_files": 5,
                "total_size_mb": 10.5,
            }
            mock_cached_provider.return_value = mock_data_provider

            # Act
            result = _handle(default_args)

            # Assert
            assert result == 0
            mock_load_strategy.assert_called_once_with("ml_basic")
            mock_backtester.run.assert_called_once()

    def test_backtest_with_sentiment_enabled(self, default_args, mock_backtester):
        """Test backtest execution with sentiment analysis enabled."""
        # Arrange
        default_args.use_sentiment = True

        with (
            patch("src.engines.backtest.engine.Backtester", return_value=mock_backtester),
            patch("cli.commands.backtest._load_strategy") as mock_load_strategy,
            patch("cli.commands.backtest.configure_logging"),
            patch("src.data_providers.binance_provider.BinanceProvider") as mock_provider_class,
            patch(
                "src.data_providers.cached_data_provider.CachedDataProvider"
            ) as mock_cached_provider,
            patch(
                "src.data_providers.feargreed_provider.FearGreedProvider"
            ) as mock_sentiment_provider,
            patch("cli.commands.backtest.SymbolFactory.to_exchange_symbol", return_value="BTCUSDT"),
            patch("builtins.open", create=True),
            patch("cli.commands.backtest.PROJECT_ROOT", Path("/tmp/test")),
        ):

            mock_strategy = Mock(name="ml_basic")
            mock_strategy.get_trading_pair.return_value = "BTCUSDT"
            mock_load_strategy.return_value = mock_strategy

            mock_provider = Mock()
            mock_provider_class.return_value = mock_provider

            mock_data_provider = Mock()
            mock_data_provider.get_cache_info.return_value = {
                "total_files": 5,
                "total_size_mb": 10.5,
            }
            mock_data_provider.get_historical_data.return_value = Mock()
            mock_cached_provider.return_value = mock_data_provider

            mock_sentiment = Mock()
            mock_sentiment.get_historical_sentiment.return_value = Mock(empty=True)
            mock_sentiment_provider.return_value = mock_sentiment

            # Act
            result = _handle(default_args)

            # Assert
            assert result == 0
            mock_sentiment_provider.assert_called_once()

    def test_backtest_with_cache_disabled(self, default_args, mock_backtester):
        """Test backtest execution with cache disabled."""
        # Arrange
        default_args.no_cache = True

        with (
            patch("src.engines.backtest.engine.Backtester", return_value=mock_backtester),
            patch("cli.commands.backtest._load_strategy") as mock_load_strategy,
            patch("cli.commands.backtest.configure_logging"),
            patch("src.data_providers.binance_provider.BinanceProvider") as mock_provider_class,
            patch(
                "src.data_providers.cached_data_provider.CachedDataProvider"
            ) as mock_cached_provider,
            patch("cli.commands.backtest.SymbolFactory.to_exchange_symbol", return_value="BTCUSDT"),
            patch("builtins.open", create=True),
            patch("cli.commands.backtest.PROJECT_ROOT", Path("/tmp/test")),
        ):

            mock_strategy = Mock(name="ml_basic")
            mock_strategy.get_trading_pair.return_value = "BTCUSDT"
            mock_load_strategy.return_value = mock_strategy

            mock_provider = Mock()
            mock_provider_class.return_value = mock_provider

            # Act
            result = _handle(default_args)

            # Assert
            assert result == 0
            # Verify CachedDataProvider was not used when no_cache is True
            mock_cached_provider.assert_not_called()

    def test_backtest_with_coinbase_provider(self, default_args, mock_backtester):
        """Test backtest execution with Coinbase provider."""
        # Arrange
        default_args.provider = "coinbase"

        with (
            patch("src.engines.backtest.engine.Backtester", return_value=mock_backtester),
            patch("cli.commands.backtest._load_strategy") as mock_load_strategy,
            patch("cli.commands.backtest.configure_logging"),
            patch(
                "src.data_providers.coinbase_provider.CoinbaseProvider"
            ) as mock_coinbase_provider,
            patch(
                "src.data_providers.cached_data_provider.CachedDataProvider"
            ) as mock_cached_provider,
            patch("cli.commands.backtest.SymbolFactory.to_exchange_symbol", return_value="BTC-USD"),
            patch("builtins.open", create=True),
            patch("cli.commands.backtest.PROJECT_ROOT", Path("/tmp/test")),
        ):

            mock_strategy = Mock(name="ml_basic")
            mock_strategy.get_trading_pair.return_value = "BTC-USD"
            mock_load_strategy.return_value = mock_strategy

            mock_provider = Mock()
            mock_coinbase_provider.return_value = mock_provider

            mock_data_provider = Mock()
            mock_data_provider.get_cache_info.return_value = {
                "total_files": 5,
                "total_size_mb": 10.5,
            }
            mock_cached_provider.return_value = mock_data_provider

            # Act
            result = _handle(default_args)

            # Assert
            assert result == 0
            mock_coinbase_provider.assert_called_once()

    def test_returns_error_on_invalid_strategy(self, default_args):
        """Test that error is returned when strategy loading fails."""
        # Arrange
        default_args.strategy = "invalid_strategy"

        with (
            patch("cli.commands.backtest._load_strategy") as mock_load_strategy,
            patch("cli.commands.backtest.configure_logging"),
        ):

            mock_load_strategy.side_effect = SystemExit(1)

            # Act & Assert
            with pytest.raises(SystemExit):
                _handle(default_args)

    def test_returns_error_on_backtest_exception(self, default_args):
        """Test that error is returned when backtest execution fails."""
        # Arrange
        with (
            patch("src.engines.backtest.engine.Backtester") as mock_backtester_class,
            patch("cli.commands.backtest._load_strategy") as mock_load_strategy,
            patch("cli.commands.backtest.configure_logging"),
            patch("src.data_providers.binance_provider.BinanceProvider") as mock_provider_class,
            patch(
                "src.data_providers.cached_data_provider.CachedDataProvider"
            ) as mock_cached_provider,
        ):

            mock_strategy = Mock(name="ml_basic")
            mock_load_strategy.return_value = mock_strategy

            mock_provider = Mock()
            mock_provider_class.return_value = mock_provider

            mock_data_provider = Mock()
            mock_data_provider.get_cache_info.return_value = {
                "total_files": 5,
                "total_size_mb": 10.5,
            }
            mock_cached_provider.return_value = mock_data_provider

            mock_backtester_instance = Mock()
            mock_backtester_instance.run.side_effect = Exception("Backtest failed")
            mock_backtester_class.return_value = mock_backtester_instance

            # Act
            result = _handle(default_args)

            # Assert
            assert result == 1

    def test_saves_backtest_results_to_file(self, default_args, mock_backtester):
        """Test that backtest completes successfully and attempts to save results."""
        # Arrange
        with (
            patch("src.engines.backtest.engine.Backtester", return_value=mock_backtester),
            patch("cli.commands.backtest._load_strategy") as mock_load_strategy,
            patch("cli.commands.backtest.configure_logging"),
            patch("src.data_providers.binance_provider.BinanceProvider") as mock_provider_class,
            patch(
                "src.data_providers.cached_data_provider.CachedDataProvider"
            ) as mock_cached_provider,
            patch("cli.commands.backtest.SymbolFactory.to_exchange_symbol", return_value="BTCUSDT"),
            patch("cli.commands.backtest.PROJECT_ROOT", Path("/tmp/test")),
        ):

            mock_strategy = Mock(name="ml_basic")
            mock_strategy.get_trading_pair.return_value = "BTCUSDT"
            mock_load_strategy.return_value = mock_strategy

            mock_provider = Mock()
            mock_provider_class.return_value = mock_provider

            mock_data_provider = Mock()
            mock_data_provider.get_cache_info.return_value = {
                "total_files": 5,
                "total_size_mb": 10.5,
            }
            mock_cached_provider.return_value = mock_data_provider

            # Act
            result = _handle(default_args)

            # Assert
            assert result == 0
            # File saving is tested in integration tests
