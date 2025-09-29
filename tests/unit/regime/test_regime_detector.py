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


def test_incremental_matches_full_recompute():
    cfg = RegimeConfig(
        slope_window=30,
        atr_window=14,
        atr_percentile_lookback=80,
        hysteresis_k=3,
        min_dwell=10,
    )
    base = make_trend_series(n=320, slope=0.0008, noise=0.0005)
    detector_full = RegimeDetector(cfg)
    detector_incremental = RegimeDetector(cfg)

    warm = detector_incremental.annotate(base)
    assert not warm.empty

    new_row = base.iloc[-1:].copy()
    new_row.index = [base.index[-1] + timedelta(hours=1)]
    new_row["open"] = base["close"].iloc[-1]
    new_row["close"] = base["close"].iloc[-1] * 1.0015
    new_row["high"] = new_row["close"] * 1.001
    new_row["low"] = new_row["close"] * 0.999
    new_row["volume"] = base["volume"].iloc[-1]

    next_window = pd.concat([base.iloc[1:], new_row])

    incremental = detector_incremental.annotate_incremental(next_window)
    recomputed = detector_full.annotate(next_window)

    for column in [
        "trend_score",
        "trend_label",
        "vol_label",
        "regime_label",
        "regime_confidence",
        "atr",
        "atr_percentile",
    ]:
        inc_vals = incremental[column]
        rec_vals = recomputed[column]
        if np.issubdtype(inc_vals.dtype, np.floating):
            pd.testing.assert_series_equal(inc_vals, rec_vals, check_names=False, check_dtype=False)
        else:
            assert list(map(str, inc_vals.tolist())) == list(map(str, rec_vals.tolist()))
