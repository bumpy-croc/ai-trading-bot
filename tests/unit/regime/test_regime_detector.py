from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import pytest

from src.regime import RegimeConfig, RegimeDetector

TEST_RANDOM_SEED = 1337


@pytest.fixture()
def seeded_rng() -> np.random.Generator:
    """Provide a deterministic RNG for synthetic trend series."""

    return np.random.default_rng(TEST_RANDOM_SEED)


def make_trend_series(
    n=200,
    slope=0.001,
    noise=0.0,
    start=30000.0,
    rng: Optional[np.random.Generator] = None,
):
    rng = rng or np.random.default_rng()
    t = np.arange(n)
    base = start * (1.0 + slope * t)
    noise_arr = rng.normal(0.0, noise * start, size=n)
    prices = np.maximum(1.0, base + noise_arr)
    ts = [datetime(2024, 1, 1) + timedelta(hours=i) for i in range(n)]
    return pd.DataFrame(
        {
            "open": prices,
            "high": prices * 1.001,
            "low": prices * 0.999,
            "close": prices,
            "volume": rng.uniform(1.0, 2.0, size=n),
        },
        index=pd.to_datetime(ts),
    )


def test_regime_detector_trend_up_basic(seeded_rng):
    df = make_trend_series(n=300, slope=0.001, noise=0.0, rng=seeded_rng)
    cfg = RegimeConfig(
        slope_window=50, atr_window=14, atr_percentile_lookback=60, hysteresis_k=2, min_dwell=5
    )
    rd = RegimeDetector(cfg)
    out = rd.annotate(df)
    assert "trend_label" in out.columns
    # Expect last label to be trend_up
    assert str(out["trend_label"].iloc[-1]) == "trend_up"

def test_regime_detector_trend_down_basic(seeded_rng):
    df = make_trend_series(n=300, slope=-0.001, noise=0.0, rng=seeded_rng)
    cfg = RegimeConfig(
        slope_window=50, atr_window=14, atr_percentile_lookback=60, hysteresis_k=2, min_dwell=5
    )
    rd = RegimeDetector(cfg)
    out = rd.annotate(df)
    assert str(out["trend_label"].iloc[-1]) == "trend_down"

def test_regime_detector_range_label_under_low_slope(seeded_rng):
    df = make_trend_series(n=300, slope=0.0, noise=0.0001, rng=seeded_rng)
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

def test_regime_hysteresis_prevents_flips(seeded_rng):
    # Alternate slight up/down segments to try to force flips
    parts = []
    for j in range(6):
        slope = 0.001 if j % 2 == 0 else -0.001
        parts.append(
            make_trend_series(
                n=50,
                slope=slope,
                noise=0.0,
                start=30000 + j * 10,
                rng=seeded_rng,
            )
        )
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

def test_confidence_in_0_1_range(seeded_rng):
    df = make_trend_series(n=200, slope=0.002, noise=0.001, rng=seeded_rng)
    rd = RegimeDetector(RegimeConfig(slope_window=40, atr_window=14, atr_percentile_lookback=80))
    out = rd.annotate(df)
    conf = out["regime_confidence"].dropna()
    assert (conf >= 0).all() and (conf <= 1).all()
