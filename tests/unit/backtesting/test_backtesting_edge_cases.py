"""Edge case and stress tests for the Backtester."""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from src.engines.backtest.engine import Backtester
from src.strategies.ml_basic import create_ml_basic_strategy


class TestBacktestingEdgeCases:
    """Validate behaviour on unusual datasets."""

    def test_single_data_point(self, mock_data_provider):
        """Single row inputs should not break execution."""

        strategy = create_ml_basic_strategy()
        single_data = pd.DataFrame(
            {"open": [100], "high": [101], "low": [99], "close": [100.5], "volume": [1000]},
            index=[datetime(2024, 1, 1, 10, 0)],
        )

        mock_data_provider.get_historical_data.return_value = single_data

        backtester = Backtester(
            strategy=strategy, data_provider=mock_data_provider, initial_balance=10000
        )

        results = backtester.run("BTCUSDT", "1h", datetime(2024, 1, 1))

        assert isinstance(results, dict)
        assert results["total_trades"] == 0

    @pytest.mark.slow
    def test_very_large_dataset(self, mock_data_provider):
        """Large datasets should remain processable."""

        strategy = create_ml_basic_strategy(fast_mode=True)
        n_points = 10000
        large_data = pd.DataFrame(
            {
                "open": np.random.randn(n_points) + 100,
                "high": np.random.randn(n_points) + 101,
                "low": np.random.randn(n_points) + 99,
                "close": np.random.randn(n_points) + 100,
                "volume": np.random.randint(1000, 10000, n_points),
            },
            index=pd.date_range("2024-01-01", periods=n_points, freq="1h"),
        )

        mock_data_provider.get_historical_data.return_value = large_data

        backtester = Backtester(
            strategy=strategy, data_provider=mock_data_provider, initial_balance=10000
        )

        results = backtester.run("BTCUSDT", "1h", datetime(2024, 1, 1))

        assert isinstance(results, dict)
        assert "total_trades" in results

    def test_concurrent_trades_handling(self, mock_data_provider, sample_ohlcv_data):
        """Concurrent trade scenarios should not error."""

        strategy = create_ml_basic_strategy()
        mock_data_provider.get_historical_data.return_value = sample_ohlcv_data

        backtester = Backtester(
            strategy=strategy, data_provider=mock_data_provider, initial_balance=10000
        )

        results = backtester.run("BTCUSDT", "1h", datetime(2024, 1, 1))

        assert isinstance(results, dict)
        assert results["total_trades"] >= 0
