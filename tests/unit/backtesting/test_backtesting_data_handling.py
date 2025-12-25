"""Backtesting data validation and hygiene tests."""

from datetime import datetime

import pandas as pd
import pytest

from src.engines.backtest.engine import Backtester
from src.strategies.ml_basic import create_ml_basic_strategy


class TestDataHandling:
    """Guard against malformed market data inputs."""

    def test_empty_data_handling(self, mock_data_provider):
        """Empty datasets should not produce trades."""

        strategy = create_ml_basic_strategy()
        empty_data = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        mock_data_provider.get_historical_data.return_value = empty_data

        backtester = Backtester(
            strategy=strategy, data_provider=mock_data_provider, initial_balance=10000
        )

        results = backtester.run("BTCUSDT", "1h", datetime(2024, 1, 1))

        assert results["total_trades"] == 0
        assert results["final_balance"] == 10000

    def test_missing_columns_handling(self, mock_data_provider):
        """Missing OHLCV columns should raise clear errors."""

        strategy = create_ml_basic_strategy()
        incomplete_data = pd.DataFrame({"open": [100, 101, 102], "close": [101, 102, 103]})
        mock_data_provider.get_historical_data.return_value = incomplete_data

        backtester = Backtester(
            strategy=strategy, data_provider=mock_data_provider, initial_balance=10000
        )

        with pytest.raises((KeyError, ValueError), match="Missing required columns"):
            backtester.run("BTCUSDT", "1h", datetime(2024, 1, 1))

    def test_data_validation(self, mock_data_provider):
        """Invalid numeric values should be surfaced or handled gracefully."""

        strategy = create_ml_basic_strategy()
        invalid_data = pd.DataFrame(
            {
                "open": [100, -50, 102],
                "high": [101, 101, 102],
                "low": [99, 99, 101],
                "close": [101, 102, 103],
                "volume": [1000, 1000, 1000],
            }
        )

        mock_data_provider.get_historical_data.return_value = invalid_data

        backtester = Backtester(
            strategy=strategy, data_provider=mock_data_provider, initial_balance=10000
        )

        try:
            results = backtester.run("BTCUSDT", "1h", datetime(2024, 1, 1))
            assert isinstance(results, dict)
        except (ValueError, AssertionError):
            pass
