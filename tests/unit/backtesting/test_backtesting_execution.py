"""Tests that execute full backtesting runs and validate outputs."""

from datetime import datetime

import pandas as pd

from src.backtesting.engine import Backtester
from src.risk.risk_manager import RiskParameters
from src.strategies.ml_basic import create_ml_basic_strategy


class TestBacktestingExecution:
    """Exercise the Backtester run loop."""

    def test_basic_backtest_execution(self, mock_data_provider, sample_ohlcv_data):
        """A standard run should return populated metrics."""

        strategy = create_ml_basic_strategy()
        risk_params = RiskParameters()

        backtester = Backtester(
            strategy=strategy,
            data_provider=mock_data_provider,
            risk_parameters=risk_params,
            initial_balance=10000,
        )

        mock_data_provider.get_historical_data.return_value = sample_ohlcv_data

        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 7)

        results = backtester.run("BTCUSDT", "1h", start_date, end_date)

        assert isinstance(results, dict)
        required_keys = ["total_trades", "win_rate", "total_return", "final_balance"]
        for key in required_keys:
            assert key in results

        assert results["total_trades"] >= 0
        assert 0 <= results["win_rate"] <= 100
        assert results["final_balance"] > 0

    def test_backtest_with_no_trades(self, mock_data_provider):
        """No signals should yield zero trades while preserving balance."""

        strategy = create_ml_basic_strategy()
        no_signal_data = pd.DataFrame(
            {
                "open": [100, 100, 100, 100, 100],
                "high": [101, 101, 101, 101, 101],
                "low": [99, 99, 99, 99, 99],
                "close": [100, 100, 100, 100, 100],
                "volume": [1000, 1000, 1000, 1000, 1000],
            },
            index=pd.date_range("2024-01-01", periods=5, freq="1h"),
        )

        mock_data_provider.get_historical_data.return_value = no_signal_data

        backtester = Backtester(
            strategy=strategy, data_provider=mock_data_provider, initial_balance=10000
        )

        results = backtester.run("BTCUSDT", "1h", datetime(2024, 1, 1))

        assert results["total_trades"] == 0
        assert results["final_balance"] == 10000
        assert results["total_return"] == 0.0

    def test_backtest_performance_metrics(self, mock_data_provider, sample_ohlcv_data):
        """Performance statistics should be included and bounded."""

        strategy = create_ml_basic_strategy()
        mock_data_provider.get_historical_data.return_value = sample_ohlcv_data

        backtester = Backtester(
            strategy=strategy, data_provider=mock_data_provider, initial_balance=10000
        )

        results = backtester.run("BTCUSDT", "1h", datetime(2024, 1, 1))

        assert "total_return" in results
        assert "win_rate" in results
        assert results["total_return"] >= -100
        assert 0 <= results["win_rate"] <= 100
