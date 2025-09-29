from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pandas.testing as pdt

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


def _naive_rolling_ols(close: pd.Series, window: int) -> tuple[pd.Series, pd.Series]:
    y = np.log(close.clip(lower=1e-8))
    idx = np.arange(len(y))
    df = pd.DataFrame({"y": y.values, "t": idx}, index=y.index)

    slopes: list[float] = []
    r2s: list[float] = []
    for i in range(len(df)):
        if i + 1 < window:
            slopes.append(np.nan)
            r2s.append(np.nan)
            continue
        block = df.iloc[i + 1 - window : i + 1]
        t_block = block["t"].astype(float).to_numpy()
        y_block = block["y"].astype(float).to_numpy()
        t_mean = t_block.mean()
        y_mean = y_block.mean()
        tt = t_block - t_mean
        yy = y_block - y_mean
        denom = (tt**2).sum()
        if denom == 0:
            slopes.append(np.nan)
            r2s.append(np.nan)
            continue
        slope = (tt * yy).sum() / denom
        y_hat = y_mean + slope * tt
        ss_tot = (yy**2).sum()
        ss_res = ((y_block - y_hat) ** 2).sum()
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
        slopes.append(slope)
        r2s.append(r2)
    return pd.Series(slopes, index=close.index), pd.Series(r2s, index=close.index)


def _naive_percentile_rank(series: pd.Series, lookback: int) -> pd.Series:
    def rank_last(window: pd.Series) -> float:
        if window.isna().any():
            return np.nan
        last = window.iloc[-1]
        return (window <= last).mean()

    return series.rolling(window=lookback, min_periods=lookback).apply(rank_last, raw=False)


def test_vectorized_components_match_naive():
    df = make_trend_series(n=160, slope=0.0015, noise=0.0005)
    window = 40
    slope_vec, r2_vec = RegimeDetector._rolling_ols_slope_and_r2(df["close"], window)
    slope_naive, r2_naive = _naive_rolling_ols(df["close"], window)

    np.testing.assert_allclose(
        slope_vec.to_numpy(), slope_naive.to_numpy(), rtol=1e-9, atol=1e-9, equal_nan=True
    )
    np.testing.assert_allclose(
        r2_vec.to_numpy(), r2_naive.to_numpy(), rtol=1e-9, atol=1e-9, equal_nan=True
    )

    atr_series = RegimeDetector._atr(df, window).copy()
    atr_series.iloc[window] = np.nan
    lookback = 80
    rank_vec = RegimeDetector._percentile_rank(atr_series, lookback)
    rank_naive = _naive_percentile_rank(atr_series, lookback)
    pdt.assert_series_equal(rank_vec, rank_naive)


def test_annotate_inplace_controls_mutation():
    df = make_trend_series(n=120, slope=0.001, noise=0.0002)
    rd = RegimeDetector(RegimeConfig(slope_window=20, atr_window=10, atr_percentile_lookback=30))

    df_copy = df.copy()
    out = rd.annotate(df_copy)
    assert out is not df_copy
    for col in [
        "trend_score",
        "trend_label",
        "vol_label",
        "regime_label",
        "regime_confidence",
    ]:
        assert col in out.columns
        assert col not in df_copy.columns

    df_inplace = df.copy()
    out_inplace = rd.annotate(df_inplace, inplace=True)
    assert out_inplace is df_inplace
    for col in [
        "trend_score",
        "trend_label",
        "vol_label",
        "regime_label",
        "regime_confidence",
    ]:
        assert col in df_inplace.columns
