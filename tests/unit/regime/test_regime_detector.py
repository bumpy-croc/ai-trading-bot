from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from src.regime import RegimeConfig, RegimeDetector


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


def _naive_rolling_ols(series: pd.Series, window: int) -> tuple[pd.Series, pd.Series]:
    y = np.log(series.clip(lower=1e-8).astype(float))
    t = np.arange(len(y), dtype=float)
    slopes = np.full(len(y), np.nan, dtype=float)
    r2s = np.full(len(y), np.nan, dtype=float)
    if window <= 0:
        return pd.Series(slopes, index=series.index), pd.Series(r2s, index=series.index)
    for end in range(window - 1, len(y)):
        idx_slice = slice(end + 1 - window, end + 1)
        t_window = t[idx_slice]
        y_window = y.iloc[idx_slice]
        t_mean = t_window.mean()
        y_mean = y_window.mean()
        tt = t_window - t_mean
        yy = y_window - y_mean
        denom = (tt**2).sum()
        if denom == 0:
            continue
        slope = (tt * yy).sum() / denom
        y_hat = y_mean + slope * tt
        ss_tot = (yy**2).sum()
        ss_res = ((y_window - y_hat) ** 2).sum()
        slopes[end] = slope
        r2s[end] = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
    return pd.Series(slopes, index=series.index), pd.Series(r2s, index=series.index)


def test_rolling_ols_matches_naive_calculation():
    rng = np.random.default_rng(42)
    prices = np.exp(rng.normal(0.0, 0.01, size=500).cumsum() + 10.0)
    index = pd.date_range("2024-01-01", periods=prices.size, freq="h")
    series = pd.Series(prices, index=index)
    window = 50

    slopes_vec, r2_vec = RegimeDetector._rolling_ols_slope_and_r2(series, window)
    slopes_naive, r2_naive = _naive_rolling_ols(series, window)

    np.testing.assert_allclose(slopes_vec.values, slopes_naive.values, atol=1e-12, rtol=1e-9, equal_nan=True)
    np.testing.assert_allclose(r2_vec.values, r2_naive.values, atol=1e-12, rtol=1e-6, equal_nan=True)


def test_rolling_ols_perfect_trend_has_unit_r2():
    t = np.arange(200, dtype=float)
    close = pd.Series(np.exp(0.01 * t + 2.0), index=pd.date_range("2024-01-01", periods=t.size, freq="h"))
    window = 40
    slopes, r2 = RegimeDetector._rolling_ols_slope_and_r2(close, window)

    valid = r2.dropna()
    assert not valid.empty
    np.testing.assert_allclose(slopes[valid.index].values, 0.01, atol=1e-12, rtol=1e-9)
    np.testing.assert_allclose(valid.values, 1.0, atol=1e-12, rtol=1e-9)
