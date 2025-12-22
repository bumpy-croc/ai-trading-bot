"""Tests for regime calibration and evaluation helpers."""

from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

from src.regime.detector import RegimeConfig, RegimeDetector
from src.regime.enhanced_detector import (
    calibrate_regime_detector,
    evaluate_regime_accuracy,
    plot_regime_accuracy,
)


def _create_price_series(length: int = 200) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=length, freq="1H")
    base_price = 100.0

    rng = np.random.default_rng(42)
    trend = np.concatenate(
        [
            np.linspace(0, 5, length // 2, endpoint=False),
            np.linspace(5, -2, length - length // 2),
        ]
    )
    noise = rng.normal(0, 0.2, length)
    prices = base_price + trend + noise

    df = pd.DataFrame(
        {
            "open": prices * 0.999,
            "high": prices * 1.001,
            "low": prices * 0.999,
            "close": prices,
            "volume": rng.uniform(1_000, 5_000, length),
        },
        index=index,
    )
    return df


def test_calibrate_regime_detector_recovers_best_config() -> None:
    df = _create_price_series()

    ground_truth_config = RegimeConfig(
        slope_window=30,
        atr_window=14,
        trend_threshold=0.0005,
        r2_min=0.2,
        atr_high_percentile=0.7,
        atr_percentile_lookback=60,
        hysteresis_k=2,
        min_dwell=5,
    )
    truth_detector = RegimeDetector(ground_truth_config)
    labelled = truth_detector.annotate(df.copy())
    labelled["target_trend"] = labelled["trend_label"]
    labelled["target_vol"] = labelled["vol_label"]

    slope_windows = (20, 30)
    atr_windows = (10, 14)
    trend_thresholds = (0.0, 0.0005)
    r2_mins = (0.1, 0.2)
    atr_percentiles = (0.6, 0.7)

    result = calibrate_regime_detector(
        labelled,
        target_trend_col="target_trend",
        target_vol_col="target_vol",
        slope_windows=slope_windows,
        atr_windows=atr_windows,
        trend_thresholds=trend_thresholds,
        r2_mins=r2_mins,
        atr_percentiles=atr_percentiles,
        base_config=ground_truth_config,
    )

    assert result.config.slope_window == 30
    assert result.config.atr_window == 14
    assert abs(result.config.trend_threshold - 0.0005) < 1e-9
    assert result.metrics.support > 0
    assert result.metrics.accuracy == 1.0
    assert result.metrics.trend_accuracy == 1.0
    assert result.metrics.volatility_accuracy == 1.0
    expected_combos = (
        len(slope_windows)
        * len(atr_windows)
        * len(trend_thresholds)
        * len(r2_mins)
        * len(atr_percentiles)
    )
    assert result.tried_configs == expected_combos
    assert "regime_correct" in result.evaluation_frame.columns


def test_evaluate_regime_accuracy_handles_missing_targets() -> None:
    df = _create_price_series(50)
    detector = RegimeDetector()
    annotated = detector.annotate(df.copy())
    annotated["target_trend"] = np.nan
    annotated["target_vol"] = np.nan

    metrics, evaluation = evaluate_regime_accuracy(
        annotated,
        target_trend_col="target_trend",
        target_vol_col="target_vol",
    )

    assert np.isnan(metrics.accuracy)
    assert metrics.support == 0
    assert "trend_correct" in evaluation.columns


def test_plot_regime_accuracy_returns_figure() -> None:
    df = _create_price_series(80)
    detector = RegimeDetector()
    annotated = detector.annotate(df.copy())
    annotated["target_trend"] = annotated["trend_label"]
    annotated["target_vol"] = annotated["vol_label"]

    metrics, evaluation = evaluate_regime_accuracy(
        annotated,
        target_trend_col="target_trend",
        target_vol_col="target_vol",
    )
    assert metrics.support > 0

    fig = plot_regime_accuracy(evaluation, window=10)
    assert fig is not None
    ax = fig.axes[0]
    assert len(ax.lines) == 3
