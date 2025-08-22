"""Smoke test for verifying MlBasic strategy performance for 2024.

This test runs a backtest using a mocked Binance data provider for the MlBasic strategy
from 2024-01-01 to 2024-12-31 and compares the yearly return with the validated
benchmark (73.81 % for 2024).

If the Binance API is unreachable (e.g. offline CI environment) the test
is skipped automatically.

TODO: Consider lightening this test for CI environments by:
- Reducing the time period from 1 year to 1-3 months
- Using lower resolution data (4h or 1d instead of 1h)
- This would significantly reduce execution time while maintaining test coverage
"""

import os
from datetime import datetime
from unittest.mock import Mock

import pytest

# Core imports
from src.backtesting.engine import Backtester
from src.data_providers.data_provider import DataProvider
from src.strategies.ml_basic import MlBasic

# We mark the test as a smoke test to allow easy selection or deselection when running PyTest.
pytestmark = [
    pytest.mark.smoke,
    pytest.mark.slow,
    pytest.mark.mock_only,
]


@pytest.mark.timeout(300)  # Give the backtest up to 5 minutes
@pytest.mark.slow
def test_ml_basic_backtest_2024_smoke(btcusdt_1h_2023_2024):
    """Run MlBasic backtest and validate 2024 annual return."""
    # Use a lightweight mocked DataProvider that returns cached candles.
    data_provider: DataProvider = Mock(spec=DataProvider)
    data_provider.get_historical_data.return_value = btcusdt_1h_2023_2024
    # For completeness, live data can return last candle
    data_provider.get_live_data.return_value = btcusdt_1h_2023_2024.tail(1)

    # Stabilize engine path to match baseline
    os.environ["USE_PREDICTION_ENGINE"] = "1"
    os.environ["ENGINE_BATCH_INFERENCE"] = "0"

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

    # Validate performance: require >= 73% with a 2% relative margin (>= 71.54)
    actual = yearly["2024"]
    min_allowed = 73.0 * (1 - 0.02)
    assert (
        actual >= min_allowed
    ), f"2024 return {actual:.2f}% is below minimum allowed {min_allowed:.2f}%"


@pytest.mark.fast
@pytest.mark.mock_only
def test_ml_basic_engine_parity_short_slice(btcusdt_1h_2023_2024):
    """Compare predictions engine-off vs engine-on over a short slice."""
    import os

    df = btcusdt_1h_2023_2024.iloc[:500].copy()

    # Engine OFF
    os.environ["USE_PREDICTION_ENGINE"] = "0"
    s_off = MlBasic()
    df_off = s_off.calculate_indicators(df)

    # Engine ON
    os.environ["USE_PREDICTION_ENGINE"] = "1"
    os.environ["ENGINE_BATCH_INFERENCE"] = "0"
    s_on = MlBasic()
    df_on = s_on.calculate_indicators(df)

    # Align indices with valid predictions
    start = s_off.sequence_length
    preds_off = df_off["onnx_pred"].iloc[start:]
    preds_on = df_on["onnx_pred"].iloc[start:]

    # Basic sanity
    assert len(preds_off) == len(preds_on)
    import numpy as np

    # Relative error metrics (avoid division by zero)
    denom = np.maximum(np.abs(preds_off.values), 1e-6)
    rel_err = np.abs(preds_off.values - preds_on.values) / denom
    # Direction agreement
    dir_off = np.sign(np.diff(preds_off.values))
    dir_on = np.sign(np.diff(preds_on.values))
    direction_agreement = np.mean(dir_off == dir_on)

    # Tolerances: predictions should be very close up to small numerical drift; direction should mostly match
    assert np.nanmedian(rel_err) <= 0.002  # 0.2% median relative error
    assert np.nanmax(rel_err) <= 0.02  # 2% worst-case relative error
    assert direction_agreement >= 0.9
