import numpy as np
import pandas as pd

from performance.metrics import (
    brier_score_direction,
    directional_accuracy,
    mean_absolute_error,
    mean_absolute_percentage_error,
)


def test_prediction_metrics_basic():
    idx = pd.date_range("2024-01-01", periods=6, freq="1h")
    # Simulate increasing actuals
    actual = pd.Series([100, 101, 102, 103, 104, 105], index=idx)
    # Predictions slightly lagging
    pred = pd.Series([99, 100.5, 101.5, 103, 104.5, 105.5], index=idx)

    acc = directional_accuracy(pred, actual)
    mae = mean_absolute_error(pred, actual)
    mape = mean_absolute_percentage_error(pred, actual)

    assert 0.0 <= acc <= 100.0
    assert mae >= 0.0
    assert mape >= 0.0

    prob_up = pd.Series(np.clip(np.random.rand(len(idx)), 0, 1), index=idx)
    actual_up = (actual.diff() > 0).astype(float)
    bs = brier_score_direction(prob_up.fillna(0.5), actual_up.fillna(0.0))
    assert 0.0 <= bs <= 1.0
