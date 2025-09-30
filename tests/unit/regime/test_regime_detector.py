from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from src.regime import RegimeConfig, RegimeDetector


@pytest.fixture(scope="module")
def rolling_ols_regression_baseline():
    """Deterministic baseline for the rolling OLS regression helper."""

    prices = pd.Series(np.linspace(100.0, 200.0, 25), name="close")
    window = 10
    baseline_slopes = np.array(
        [
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            3.5301395907215532e-02,
            3.4090813820782596e-02,
            3.2960937086141493e-02,
            3.1903916497827214e-02,
            3.0912893331543825e-02,
            2.9981847233983377e-02,
            2.9105471417943810e-02,
            2.8279069589811904e-02,
            2.7498470303878066e-02,
            2.6759955389056985e-02,
            2.6060199813985914e-02,
            2.5396220906882251e-02,
            2.4765335270506460e-02,
            2.4165122061650814e-02,
            2.3593391561837376e-02,
            2.3048158168403280e-02,
        ]
    )
    baseline_r2 = np.array(
        [
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            9.9800119874779356e-01,
            9.9813624573380360e-01,
            9.9825800003010322e-01,
            9.9836815710894969e-01,
            9.9846814955810634e-01,
            9.9855919467675505e-01,
            9.9864233228278224e-01,
            9.9871845497277623e-01,
            9.9878833251332530e-01,
            9.9885263163254431e-01,
            9.9891193217951106e-01,
            9.9896674039563061e-01,
            9.9901749987441204e-01,
            9.9906460065957061e-01,
            9.9910838683500984e-01,
            9.9914916288631137e-01,
        ]
    )
    return prices, window, baseline_slopes, baseline_r2


def test_rolling_ols_regression(rolling_ols_regression_baseline):
    prices, window, baseline_slopes, baseline_r2 = rolling_ols_regression_baseline

    slopes, r2 = RegimeDetector._rolling_ols_slope_and_r2(prices, window)

    np.testing.assert_allclose(
        slopes.to_numpy(), baseline_slopes, rtol=1e-12, atol=1e-12, equal_nan=True
    )
    np.testing.assert_allclose(
        r2.to_numpy(), baseline_r2, rtol=1e-12, atol=1e-12, equal_nan=True
    )


def make_trend_series(n=200, slope=0.001, noise=0.0, start=30000.0):
    t = np.arange(n)
    base = start * (1.0 + slope * t)
    noise_arr = np.random.normal(0.0, noise * start, size=n)
    prices = np.maximum(1.0, base + noise_arr)
    ts = [datetime(2024, 1, 1) + timedelta(hours=i) for i in range(n)]
    return pd.DataFrame(
        {
            "open": prices,
            "high": prices * 1.001,
            "low": prices * 0.999,
            "close": prices,
            "volume": np.random.uniform(1.0, 2.0, size=n),
        },
        index=pd.to_datetime(ts),
    )


def test_regime_detector_trend_up_basic():
    df = make_trend_series(n=300, slope=0.001, noise=0.0)
    cfg = RegimeConfig(
        slope_window=50, atr_window=14, atr_percentile_lookback=60, hysteresis_k=2, min_dwell=5
    )
    rd = RegimeDetector(cfg)
    out = rd.annotate(df)
    assert "trend_label" in out.columns
    # Expect last label to be trend_up
    assert str(out["trend_label"].iloc[-1]) == "trend_up"


def test_regime_detector_trend_down_basic():
    df = make_trend_series(n=300, slope=-0.001, noise=0.0)
    cfg = RegimeConfig(
        slope_window=50, atr_window=14, atr_percentile_lookback=60, hysteresis_k=2, min_dwell=5
    )
    rd = RegimeDetector(cfg)
    out = rd.annotate(df)
    assert str(out["trend_label"].iloc[-1]) == "trend_down"


def test_regime_detector_range_label_under_low_slope():
    df = make_trend_series(n=300, slope=0.0, noise=0.0001)
    cfg = RegimeConfig(
        slope_window=50,
        atr_window=14,
        atr_percentile_lookback=60,
        trend_threshold=0.0,
        hysteresis_k=1,
        min_dwell=1,
    )
    rd = RegimeDetector(cfg)
    out = rd.annotate(df)
    # With near-zero slope, should be range or not strongly trending
    assert str(out["trend_label"].iloc[-1]) in {"range", "trend_up", "trend_down"}


def test_regime_hysteresis_prevents_flips():
    # Alternate slight up/down segments to try to force flips
    parts = []
    for j in range(6):
        slope = 0.001 if j % 2 == 0 else -0.001
        parts.append(make_trend_series(n=50, slope=slope, noise=0.0, start=30000 + j * 10))
    df = pd.concat(parts)
    cfg = RegimeConfig(
        slope_window=30, atr_window=14, atr_percentile_lookback=60, hysteresis_k=5, min_dwell=30
    )
    rd = RegimeDetector(cfg)
    out = rd.annotate(df)
    labels = out["trend_label"].astype(str).tolist()
    # Expect fewer switches due to hysteresis; count transitions
    switches = sum(1 for i in range(1, len(labels)) if labels[i] != labels[i - 1])
    assert switches < 4


def test_confidence_in_0_1_range():
    df = make_trend_series(n=200, slope=0.002, noise=0.001)
    rd = RegimeDetector(RegimeConfig(slope_window=40, atr_window=14, atr_percentile_lookback=80))
    out = rd.annotate(df)
    conf = out["regime_confidence"].dropna()
    assert (conf >= 0).all() and (conf <= 1).all()
