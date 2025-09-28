"""Smoke tests for verifying MlBasic strategy performance and engine parity.

This module contains tests for validating MlBasic strategy behavior including:
- Engine parity tests comparing engine-on vs engine-off predictions
- Strategy unit tests for core functionality

If the Binance API is unreachable (e.g. offline CI environment) the tests
are skipped automatically.
"""

import os
from datetime import datetime
from unittest.mock import Mock

import pandas as pd
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


