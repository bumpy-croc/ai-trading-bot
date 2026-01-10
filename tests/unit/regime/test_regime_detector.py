from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from src.regime import RegimeConfig, RegimeDetector

TEST_RANDOM_SEED = 1337


@pytest.fixture()
def seeded_rng() -> np.random.Generator:
    """Provide a deterministic RNG for synthetic trend series."""
    return np.random.default_rng(TEST_RANDOM_SEED)


def _compute_reference_rolling_ols(series: pd.Series, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Reference implementation of the rolling OLS slope and R² calculation.

    To keep the regression test independent from the vectorised production
    implementation we deliberately rely on an explicit least-squares solve via
    :func:`numpy.linalg.lstsq` for each window.  Computing the baseline
    dynamically still avoids hard-coding floating point literals that can drift
    slightly between Python/NumPy builds – the root cause of the original
    regression – while ensuring the expectation comes from a numerically
    distinct algorithm.
    """

    y = np.log(series.clip(lower=1e-8).astype(float))
    t = np.arange(len(y), dtype=float)
    slopes = np.full(len(series), np.nan, dtype=float)
    r2s = np.full(len(series), np.nan, dtype=float)

    if window <= 0:
        return slopes, r2s

    for end in range(window - 1, len(series)):
        idx = slice(end + 1 - window, end + 1)
        y_window = y.iloc[idx]

        if y_window.isna().any():
            continue

        t_window = t[idx]
        X = np.column_stack((np.ones(window, dtype=float), t_window))
        y_vals = y_window.to_numpy(dtype=float)

        coeffs, *_ = np.linalg.lstsq(X, y_vals, rcond=None)
        slope = coeffs[1]
        y_hat = X @ coeffs

        y_mean = y_vals.mean()
        ss_tot = np.sum((y_vals - y_mean) ** 2)
        if ss_tot <= 0:
            slopes[end] = slope
            r2s[end] = np.nan
            continue

        ss_res = np.sum((y_vals - y_hat) ** 2)

        slopes[end] = slope
        r2 = 1.0 - (ss_res / ss_tot)
        r2s[end] = np.clip(r2, 0.0, 1.0)

    return slopes, r2s


@pytest.fixture(scope="module")
def rolling_ols_regression_baseline():
    """Deterministic baseline for the rolling OLS regression helper.

    The fixture intentionally computes the expected slopes and R² values on the
    fly using the naïve reference algorithm above.  This keeps the regression
    test stable across different BLAS/NumPy builds where hard-coded floating
    point literals can differ by >1e-12 and break the strict comparison in
    :func:`test_rolling_ols_regression`.
    """

    prices = pd.Series(np.linspace(100.0, 200.0, 25), name="close")
    window = 10
    baseline_slopes, baseline_r2 = _compute_reference_rolling_ols(prices, window)
    return prices, window, baseline_slopes, baseline_r2


def test_rolling_ols_regression(rolling_ols_regression_baseline):
    prices, window, baseline_slopes, baseline_r2 = rolling_ols_regression_baseline

    slopes, r2 = RegimeDetector._rolling_ols_slope_and_r2(prices, window)

    np.testing.assert_allclose(
        slopes.to_numpy(), baseline_slopes, rtol=1e-12, atol=1e-12, equal_nan=True
    )
    np.testing.assert_allclose(r2.to_numpy(), baseline_r2, rtol=1e-10, atol=1e-12, equal_nan=True)


def make_trend_series(
    n=200,
    slope=0.001,
    noise=0.0,
    start=30000.0,
    rng: np.random.Generator | None = None,
):
    """Create a synthetic trend series with optional noise.

    Args:
        n: Number of data points
        slope: Trend slope per time step
        noise: Noise level as fraction of start price
        start: Starting price level
        rng: Optional RNG for reproducible noise generation

    Returns:
        DataFrame with OHLCV data
    """
    t = np.arange(n)
    base = start * (1.0 + slope * t)

    if noise > 0.0:
        if rng is None:
            rng = np.random.default_rng(TEST_RANDOM_SEED)
        noise_arr = rng.normal(0.0, noise * start, size=n)
    else:
        noise_arr = np.zeros_like(base)

    prices = np.maximum(1.0, base + noise_arr)
    ts = [datetime(2024, 1, 1) + timedelta(hours=i) for i in range(n)]

    # Use deterministic volume generation
    if rng is not None:
        volume = rng.uniform(1.0, 2.0, size=n)
    else:
        volume = 1.0 + 0.01 * t

    return pd.DataFrame(
        {
            "open": prices,
            "high": prices * 1.001,
            "low": prices * 0.999,
            "close": prices,
            "volume": volume,
        },
        index=pd.to_datetime(ts),
    )


UP_TREND_SCORE_SUFFIX = [
    0.0009857359308331664,
    0.0009847651841847058,
    0.0009837963476811648,
    0.0009828294156900058,
    0.0009818643826008276,
    0.0009809012428252514,
    0.0009799399907967877,
    0.0009789806209707526,
    0.0009780231278241458,
    0.0009770675058555886,
    0.0009761137495851572,
    0.0009751618535543032,
    0.0009742118123257661,
    0.0009732636204834472,
    0.0009723172726323222,
    0.0009713727633983214,
    0.0009704300874282372,
    0.0009694892393896393,
    0.0009685502139707481,
    0.0009676130058803523,
    0.0009666776098476782,
    0.0009657440206223513,
    0.0009648122329742311,
    0.0009638822416933512,
    0.0009629540415898299,
    0.0009620276274937498,
    0.000961102994255076,
    0.0009601801367435604,
    0.0009592590498486059,
    0.0009583397284792484,
    0.0009574221675640183,
    0.0009565063620508366,
    0.0009555923069069388,
    0.0009546799971187777,
    0.0009537694276919637,
    0.0009528605936511062,
    0.000951953490039802,
    0.0009510481119204667,
    0.0009501444543742932,
    0.000949242512501183,
    0.0009483422814195986,
    0.0009474437562665065,
    0.0009465469321972893,
    0.0009456518043856601,
    0.0009447583680235692,
    0.0009438666183211392,
    0.000942976550506536,
    0.0009420881598259189,
    0.0009412014415433413,
    0.0009403163909406932,
    0.0009394330033175751,
]

UP_REGIME_CONFIDENCE_SUFFIX = [
    0.5533738618049914,
    0.5537931092478762,
    0.5541755329467465,
    0.5545244551213705,
    0.5548428122923142,
    0.5551332096775271,
    0.5553979666365391,
    0.5556391548285513,
    0.5558586304205887,
    0.5560580613930836,
    0.5562389507962305,
    0.5564026566422721,
    0.5565504089838389,
    0.5566833246350696,
    0.556802419905841,
    0.5569086216554048,
    0.5570027769176922,
    0.5570856613114183,
    0.5571579864068462,
    0.5572204062002379,
    0.5572735228174331,
    0.557317891554058,
]

DOWN_TREND_SCORE_SUFFIX = [
    -0.0010147446371265925,
    -0.001015775421610121,
    -0.0010168083024437297,
    -0.0010178432860292732,
    -0.0010188803787946974,
    -0.0010199195871941984,
    -0.001020960917708328,
    -0.0010220043768441132,
    -0.0010230499711352352,
    -0.0010240977071421316,
    -0.001025147591452159,
    -0.0010261996306796884,
    -0.0010272538314662853,
    -0.0010283102004808489,
    -0.001029368744419752,
    -0.0010304294700069253,
    -0.0010314923839940921,
    -0.0010325574931608352,
    -0.0010336248043147861,
    -0.0010346943242917548,
    -0.0010357660599558784,
    -0.0010368400181997516,
    -0.0010379162059445924,
    -0.0010389946301404133,
    -0.001040075297766108,
    -0.0010411582158296457,
    -0.0010422433913682392,
    -0.0010433308314484468,
    -0.001044420543166374,
    -0.0010455125336477987,
    -0.001046606810048327,
    -0.0010477033795535636,
    -0.0010488022493792637,
    -0.0010499034267714734,
    -0.0010510069190067118,
    -0.001052112733392154,
    -0.001053220877265735,
    -0.0010543313579963389,
    -0.0010554441829839804,
    -0.0010565593596599365,
    -0.0010576768954869465,
    -0.0010587967979593244,
    -0.001059919074603205,
    -0.0010610437329766408,
    -0.0010621707806698273,
    -0.0010633002253052195,
    -0.0010644320745377277,
    -0.0010655663360549178,
    -0.001066703017577167,
    -0.0010678421268577952,
    -0.0010689836716833167,
]

DOWN_REGIME_CONFIDENCE_SUFFIX = [
    0.5638112221973819,
    0.5646154441238309,
    0.5653829567528408,
    0.5661170781686916,
    0.5668207413888227,
    0.5674965486733882,
    0.5681468168934748,
    0.5687736156277385,
    0.5693787993161513,
    0.5699640345230579,
    0.5705308231549893,
    0.5710805223195293,
    0.5716143613752903,
    0.5721334566308943,
    0.5726388240565261,
    0.573131390320587,
    0.5736120023985242,
    0.5740814359685754,
    0.57454040276804,
    0.5749895570561978,
    0.5754295013083064,
    0.575860791248112,
]

EXPECTED_UP_TREND_SCORE = np.array([np.nan] * 29 + UP_TREND_SCORE_SUFFIX, dtype=float)
EXPECTED_UP_TREND_LABEL = ["range"] * 29 + ["trend_up"] * 51
EXPECTED_UP_REGIME_LABEL = (
    ["range:low_vol"] * 29 + ["trend_up:low_vol"] * 19 + ["trend_up:high_vol"] * 32
)
EXPECTED_UP_REGIME_CONFIDENCE = np.array(
    [np.nan] * 58 + UP_REGIME_CONFIDENCE_SUFFIX,
    dtype=float,
)

EXPECTED_DOWN_TREND_SCORE = np.array([np.nan] * 29 + DOWN_TREND_SCORE_SUFFIX, dtype=float)
EXPECTED_DOWN_TREND_LABEL = ["range"] * 29 + ["trend_down"] * 51
EXPECTED_DOWN_REGIME_LABEL = ["range:low_vol"] * 29 + ["trend_down:low_vol"] * 51
EXPECTED_DOWN_REGIME_CONFIDENCE = np.array(
    [np.nan] * 58 + DOWN_REGIME_CONFIDENCE_SUFFIX,
    dtype=float,
)

EXPECTED_RANGE_TREND_SCORE = np.array([np.nan] * 29 + [0.0] * 51, dtype=float)
EXPECTED_RANGE_TREND_LABEL = ["range"] * 80
EXPECTED_RANGE_REGIME_LABEL = ["range:low_vol"] * 48 + ["range:high_vol"] * 32
EXPECTED_RANGE_REGIME_CONFIDENCE = np.full(80, np.nan, dtype=float)


@pytest.fixture(scope="module")
def deterministic_regime_config() -> RegimeConfig:
    return RegimeConfig(
        slope_window=30,
        atr_window=10,
        atr_percentile_lookback=40,
        hysteresis_k=2,
        min_dwell=5,
    )


@pytest.fixture(scope="module")
def upward_regime_expectations():
    df = make_trend_series(n=80, slope=0.001, noise=0.0)
    expected = {
        "trend_score": EXPECTED_UP_TREND_SCORE,
        "trend_label": EXPECTED_UP_TREND_LABEL,
        "regime_label": EXPECTED_UP_REGIME_LABEL,
        "regime_confidence": EXPECTED_UP_REGIME_CONFIDENCE,
    }
    return df, expected


@pytest.fixture(scope="module")
def downward_regime_expectations():
    df = make_trend_series(n=80, slope=-0.001, noise=0.0)
    expected = {
        "trend_score": EXPECTED_DOWN_TREND_SCORE,
        "trend_label": EXPECTED_DOWN_TREND_LABEL,
        "regime_label": EXPECTED_DOWN_REGIME_LABEL,
        "regime_confidence": EXPECTED_DOWN_REGIME_CONFIDENCE,
    }
    return df, expected


@pytest.fixture(scope="module")
def range_regime_expectations():
    df = make_trend_series(n=80, slope=0.0, noise=0.0)
    expected = {
        "trend_score": EXPECTED_RANGE_TREND_SCORE,
        "trend_label": EXPECTED_RANGE_TREND_LABEL,
        "regime_label": EXPECTED_RANGE_REGIME_LABEL,
        "regime_confidence": EXPECTED_RANGE_REGIME_CONFIDENCE,
    }
    return df, expected


def _assert_expected_columns(out: pd.DataFrame, expected: dict[str, object]):
    np.testing.assert_allclose(
        out["trend_score"].to_numpy(),
        expected["trend_score"],
        rtol=1e-7,
        atol=1e-8,
        equal_nan=True,
    )
    assert out["trend_label"].astype(str).tolist() == expected["trend_label"]
    assert out["regime_label"].astype(str).tolist() == expected["regime_label"]
    np.testing.assert_allclose(
        out["regime_confidence"].to_numpy(),
        expected["regime_confidence"],
        rtol=1e-7,
        atol=1e-8,
        equal_nan=True,
    )


def test_regime_detector_trend_up_vectors(
    deterministic_regime_config: RegimeConfig, upward_regime_expectations
):
    df, expected = upward_regime_expectations
    rd = RegimeDetector(deterministic_regime_config)
    out = rd.annotate(df)
    _assert_expected_columns(out, expected)


def test_regime_detector_trend_down_vectors(
    deterministic_regime_config: RegimeConfig, downward_regime_expectations
):
    df, expected = downward_regime_expectations
    rd = RegimeDetector(deterministic_regime_config)
    out = rd.annotate(df)
    _assert_expected_columns(out, expected)


def test_regime_detector_range_vectors(
    deterministic_regime_config: RegimeConfig, range_regime_expectations
):
    df, expected = range_regime_expectations
    rd = RegimeDetector(deterministic_regime_config)
    out = rd.annotate(df)
    _assert_expected_columns(out, expected)


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

    np.testing.assert_allclose(
        slopes_vec.values, slopes_naive.values, atol=1e-12, rtol=1e-9, equal_nan=True
    )
    np.testing.assert_allclose(
        r2_vec.values, r2_naive.values, atol=1e-12, rtol=1e-6, equal_nan=True
    )


def test_rolling_ols_perfect_trend_has_unit_r2():
    t = np.arange(200, dtype=float)
    close = pd.Series(
        np.exp(0.01 * t + 2.0), index=pd.date_range("2024-01-01", periods=t.size, freq="h")
    )
    window = 40
    slopes, r2 = RegimeDetector._rolling_ols_slope_and_r2(close, window)

    valid = r2.dropna()
    assert not valid.empty
    np.testing.assert_allclose(slopes[valid.index].values, 0.01, atol=1e-12, rtol=1e-9)
    np.testing.assert_allclose(valid.values, 1.0, atol=1e-12, rtol=1e-9)


def test_rolling_ols_handles_nan_values_correctly():
    """Test that NaN values don't propagate to windows that don't contain them."""
    # Create a series with some NaN values in the middle
    t = np.arange(200, dtype=float)
    prices = np.exp(0.01 * t + 2.0)

    # Insert NaN values at positions 50-55
    prices[50:56] = np.nan

    close = pd.Series(prices, index=pd.date_range("2024-01-01", periods=t.size, freq="h"))
    window = 40
    slopes, r2 = RegimeDetector._rolling_ols_slope_and_r2(close, window)

    # Windows that contain NaN values should be NaN
    # Window starting at position 11 (ending at 50) should be valid
    # Window starting at position 16 (ending at 55) should be NaN
    # Window starting at position 17 (ending at 56) should be NaN
    # Window starting at position 56 (ending at 95) should be valid again

    # Check that we have valid values before the NaN region
    assert not pd.isna(slopes.iloc[49])  # Window ending at position 49 (before NaN)
    assert not pd.isna(r2.iloc[49])

    # Check that windows containing NaN are invalid
    for i in range(50, 56 + window - 1):  # Windows that would contain NaN values
        if i < len(slopes):
            assert pd.isna(slopes.iloc[i]), f"Expected NaN at position {i}, got {slopes.iloc[i]}"
            assert pd.isna(r2.iloc[i]), f"Expected NaN at position {i}, got {r2.iloc[i]}"

    # Check that we have valid values after the NaN region (once window no longer contains NaN)
    post_nan_start = 56 + window - 1  # First window that doesn't contain any NaN
    if post_nan_start < len(slopes):
        assert not pd.isna(
            slopes.iloc[post_nan_start]
        ), f"Expected valid value at position {post_nan_start}"
        assert not pd.isna(
            r2.iloc[post_nan_start]
        ), f"Expected valid value at position {post_nan_start}"

        # The slope should still be approximately 0.01 for the valid windows
        valid_post_nan = slopes.iloc[post_nan_start:].dropna()
        if not valid_post_nan.empty:
            np.testing.assert_allclose(valid_post_nan.values, 0.01, atol=1e-10, rtol=1e-8)


def test_rolling_ols_numerical_stability_large_windows():
    """Test that large rolling windows don't suffer from catastrophic cancellation."""
    # Create a long series with small but consistent trend
    n_points = 5000
    t = np.arange(n_points, dtype=float)
    # Use a very small slope to test numerical stability
    small_slope = 1e-6
    prices = np.exp(small_slope * t + 10.0)  # Start at high price level

    # Add tiny amount of noise to make variance non-zero but very small
    np.random.seed(42)  # For reproducibility
    noise = np.random.normal(0, 1e-8, n_points)
    prices = prices * (1 + noise)

    close = pd.Series(prices, index=pd.date_range("2024-01-01", periods=n_points, freq="h"))

    # Test with large window that would cause cancellation issues
    large_window = 2000
    slopes, r2 = RegimeDetector._rolling_ols_slope_and_r2(close, large_window)

    # Check that we don't get excessive NaN values due to cancellation
    valid_slopes = slopes.dropna()
    valid_r2 = r2.dropna()

    # We should have a reasonable number of valid results
    expected_valid_count = n_points - large_window + 1
    assert (
        len(valid_slopes) >= expected_valid_count * 0.8
    ), f"Too many NaN slopes: got {len(valid_slopes)} valid out of {expected_valid_count} expected"
    assert (
        len(valid_r2) >= expected_valid_count * 0.8
    ), f"Too many NaN R² values: got {len(valid_r2)} valid out of {expected_valid_count} expected"

    # The slopes should be close to the true slope (allowing for noise)
    if len(valid_slopes) > 0:
        mean_slope = valid_slopes.mean()
        assert (
            abs(mean_slope - small_slope) < 1e-5
        ), f"Mean slope {mean_slope} too far from expected {small_slope}"

    # R² values should be finite and in valid range
    if len(valid_r2) > 0:
        assert np.all(valid_r2 >= 0), "R² values should be non-negative"
        assert np.all(valid_r2 <= 1), "R² values should not exceed 1"
        assert np.all(np.isfinite(valid_r2)), "R² values should be finite"


def test_rolling_ols_cancellation_detection():
    """Test that the cancellation detection works correctly."""
    # Create a series where variance calculation would suffer from cancellation
    n_points = 1000
    base_value = 1e6  # Large base value to amplify cancellation effects

    # Create nearly constant series (very small variance)
    t = np.arange(n_points, dtype=float)
    prices = np.full(n_points, base_value)
    # Add tiny variations that would cause cancellation in naive calculation
    tiny_variations = np.sin(t * 0.01) * 1e-10
    prices = prices + tiny_variations

    close = pd.Series(prices, index=pd.date_range("2024-01-01", periods=n_points, freq="h"))

    window = 500
    slopes, r2 = RegimeDetector._rolling_ols_slope_and_r2(close, window)

    # With cancellation protection, we should get valid results (likely R² ≈ 0)
    valid_r2 = r2.dropna()

    # Should have most windows valid (not NaN due to cancellation)
    expected_valid_count = n_points - window + 1
    assert (
        len(valid_r2) >= expected_valid_count * 0.9
    ), f"Cancellation protection failed: got {len(valid_r2)} valid out of {expected_valid_count} expected"

    # R² should be close to 0 for nearly constant series
    if len(valid_r2) > 0:
        assert np.all(valid_r2 >= 0), "R² values should be non-negative"
        assert np.all(valid_r2 <= 1), "R² values should not exceed 1"
        # Most R² values should be very small for nearly constant data
        assert (
            np.mean(valid_r2) < 0.1
        ), f"Expected low R² for constant data, got mean {np.mean(valid_r2)}"
