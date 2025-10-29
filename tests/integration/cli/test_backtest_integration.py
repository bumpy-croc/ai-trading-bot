"""Integration tests for atb backtest command."""

import argparse
import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, Mock

import pandas as pd
import pytest

from cli.commands.backtest import _handle


@pytest.mark.integration
class TestBacktestIntegration:
    """Integration tests for backtest command with real components."""

    @pytest.fixture
    def temp_project_root(self):
        """Provides a temporary project root directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            logs_dir = tmppath / "logs" / "backtest"
            logs_dir.mkdir(parents=True)
            yield tmppath

    @pytest.fixture
    def mock_historical_data(self):
        """Provides mock historical OHLCV data."""
        # Create 7 days of hourly data
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=7), periods=168, freq="1h"  # 7 days * 24 hours
        )
        data = {
            "timestamp": dates,
            "open": [40000 + i * 10 for i in range(168)],
            "high": [40100 + i * 10 for i in range(168)],
            "low": [39900 + i * 10 for i in range(168)],
            "close": [40000 + i * 10 for i in range(168)],
            "volume": [100 + i for i in range(168)],
        }
        df = pd.DataFrame(data)
        df.set_index("timestamp", inplace=True)
        return df

    def test_end_to_end_backtest_with_small_dataset(self, temp_project_root, mock_historical_data):
        """Test complete backtest workflow with small dataset."""
        # Arrange
        args = argparse.Namespace(
            strategy="ml_basic",
            symbol="BTCUSDT",
            timeframe="1h",
            days=7,
            start=None,
            end=None,
            initial_balance=10000,
            risk_per_trade=0.01,
            max_risk_per_trade=0.02,
            max_drawdown=0.5,
            use_sentiment=False,
            no_cache=True,  # Disable cache for integration test
            cache_ttl=24,
            log_to_db=False,  # Disable DB logging for integration test
            provider="binance",
        )

        # Act
        with (
            patch("cli.commands.backtest.PROJECT_ROOT", temp_project_root),
            patch("src.data_providers.binance_provider.BinanceProvider") as mock_provider_class,
            patch("cli.commands.backtest.configure_logging"),
        ):

            mock_provider = Mock()
            mock_provider.get_historical_data.return_value = mock_historical_data
            mock_provider_class.return_value = mock_provider

            result = _handle(args)

            # Assert
            assert result == 0

            # Verify log file was created
            logs_dir = temp_project_root / "logs" / "backtest"
            log_files = list(logs_dir.glob("*.json"))
            assert len(log_files) == 1

            # Verify log file contains expected data
            with open(log_files[0], "r") as f:
                log_data = json.load(f)
                # Strategy is logged as class name (MlBasic) not identifier (ml_basic)
                assert log_data["strategy"] == "MlBasic"
                assert log_data["symbol"] == "BTCUSDT"
                assert log_data["timeframe"] == "1h"
                assert "results" in log_data
                assert "total_trades" in log_data["results"]

    def test_backtest_creates_output_files(self, temp_project_root, mock_historical_data):
        """Test that backtest creates expected output files."""
        # Arrange
        args = argparse.Namespace(
            strategy="ml_basic",
            symbol="BTCUSDT",
            timeframe="1h",
            days=7,
            start=None,
            end=None,
            initial_balance=10000,
            risk_per_trade=0.01,
            max_risk_per_trade=0.02,
            max_drawdown=0.5,
            use_sentiment=False,
            no_cache=True,
            cache_ttl=24,
            log_to_db=False,
            provider="binance",
        )

        # Act
        with (
            patch("cli.commands.backtest.PROJECT_ROOT", temp_project_root),
            patch("src.data_providers.binance_provider.BinanceProvider") as mock_provider_class,
            patch("cli.commands.backtest.configure_logging"),
        ):

            mock_provider = Mock()
            mock_provider.get_historical_data.return_value = mock_historical_data
            mock_provider_class.return_value = mock_provider

            result = _handle(args)

            # Assert
            assert result == 0

            # Verify logs directory was created
            logs_dir = temp_project_root / "logs" / "backtest"
            assert logs_dir.exists()
            assert logs_dir.is_dir()

    def test_backtest_with_date_range(self, temp_project_root, mock_historical_data):
        """Test backtest with explicit start and end dates."""
        # Arrange
        start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        end_date = datetime.now().strftime("%Y-%m-%d")

        args = argparse.Namespace(
            strategy="ml_basic",
            symbol="BTCUSDT",
            timeframe="1h",
            days=None,
            start=start_date,
            end=end_date,
            initial_balance=10000,
            risk_per_trade=0.01,
            max_risk_per_trade=0.02,
            max_drawdown=0.5,
            use_sentiment=False,
            no_cache=True,
            cache_ttl=24,
            log_to_db=False,
            provider="binance",
        )

        # Act
        with (
            patch("cli.commands.backtest.PROJECT_ROOT", temp_project_root),
            patch("src.data_providers.binance_provider.BinanceProvider") as mock_provider_class,
            patch("cli.commands.backtest.configure_logging"),
        ):

            mock_provider = Mock()
            mock_provider.get_historical_data.return_value = mock_historical_data
            mock_provider_class.return_value = mock_provider

            result = _handle(args)

            # Assert
            assert result == 0

            # Verify log file contains correct date range
            logs_dir = temp_project_root / "logs" / "backtest"
            log_files = list(logs_dir.glob("*.json"))
            assert len(log_files) == 1

            with open(log_files[0], "r") as f:
                log_data = json.load(f)
                assert start_date in log_data["start_date"]
                assert end_date in log_data["end_date"]

    def test_backtest_performance_metrics_calculated(self, temp_project_root, mock_historical_data):
        """Test that backtest calculates all expected performance metrics."""
        # Arrange
        args = argparse.Namespace(
            strategy="ml_basic",
            symbol="BTCUSDT",
            timeframe="1h",
            days=7,
            start=None,
            end=None,
            initial_balance=10000,
            risk_per_trade=0.01,
            max_risk_per_trade=0.02,
            max_drawdown=0.5,
            use_sentiment=False,
            no_cache=True,
            cache_ttl=24,
            log_to_db=False,
            provider="binance",
        )

        # Act
        with (
            patch("cli.commands.backtest.PROJECT_ROOT", temp_project_root),
            patch("src.data_providers.binance_provider.BinanceProvider") as mock_provider_class,
            patch("cli.commands.backtest.configure_logging"),
        ):

            mock_provider = Mock()
            mock_provider.get_historical_data.return_value = mock_historical_data
            mock_provider_class.return_value = mock_provider

            result = _handle(args)

            # Assert
            assert result == 0

            # Verify all required metrics are present
            logs_dir = temp_project_root / "logs" / "backtest"
            log_files = list(logs_dir.glob("*.json"))

            with open(log_files[0], "r") as f:
                log_data = json.load(f)
                results = log_data["results"]

                required_metrics = [
                    "total_trades",
                    "win_rate",
                    "total_return",
                    "annualized_return",
                    "max_drawdown",
                    "sharpe_ratio",
                    "final_balance",
                    "hold_return",
                    "trading_vs_hold_difference",
                ]

                for metric in required_metrics:
                    assert metric in results, f"Missing metric: {metric}"
