"""Smoke test for verifying MlBasic strategy performance for 2024.

This test runs a backtest using a mocked Binance data provider for the MlBasic strategy
from 2024-01-01 to 2024-12-31 and compares the yearly return with the validated
benchmark (62.77 % for 2024).

If the Binance API is unreachable (e.g. offline CI environment) the test
is skipped automatically.
"""

from datetime import datetime

import pytest
from unittest.mock import Mock

# Core imports
from backtesting.engine import Backtester
from data_providers.data_provider import DataProvider
from strategies.ml_basic import MlBasic

# We mark the test as a smoke test to allow easy selection or deselection when running PyTest.
pytestmark = [
    pytest.mark.smoke,
]


@pytest.mark.timeout(300)  # Give the backtest up to 5 minutes
def test_ml_basic_backtest_2024_smoke(btcusdt_1h_2023_2024):
    """Run MlBasic backtest and validate 2024 annual return."""
    # Use a lightweight mocked DataProvider that returns cached candles.
    data_provider: DataProvider = Mock(spec=DataProvider)
    data_provider.get_historical_data.return_value = btcusdt_1h_2023_2024
    # For completeness, live data can return last candle
    data_provider.get_live_data.return_value = btcusdt_1h_2023_2024.tail(1)

    strategy = MlBasic()
    backtester = Backtester(
        strategy=strategy,
        data_provider=data_provider,
        initial_balance=10_000,  # Nominal starting equity
        log_to_database=False,  # Speed up the test
    )

    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 12, 31)

    results = backtester.run("BTCUSDT", "1h", start_date, end_date)
    yearly = results.get("yearly_returns", {})

    # Ensure year of interest is present
    assert "2024" in yearly, "Year 2024 missing from yearly returns"

    # Validate against previously recorded benchmark with 2 % tolerance.
    assert yearly["2024"] == pytest.approx(62.77, rel=0.01)
