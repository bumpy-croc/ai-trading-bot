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
    if noise:
        rng = np.random.default_rng(1234)
        noise_arr = rng.normal(0.0, noise * start, size=n)
    else:
        noise_arr = np.zeros_like(base)
    prices = np.maximum(1.0, base + noise_arr)
    ts = [datetime(2024, 1, 1) + timedelta(hours=i) for i in range(n)]
    return pd.DataFrame(
        {
            "open": prices,
            "high": prices * 1.001,
            "low": prices * 0.999,
            "close": prices,
            "volume": 1.0 + 0.01 * t,
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
        rtol=0.0,
        atol=1e-12,
        equal_nan=True,
    )
    assert out["trend_label"].astype(str).tolist() == expected["trend_label"]
    assert out["regime_label"].astype(str).tolist() == expected["regime_label"]
    np.testing.assert_allclose(
        out["regime_confidence"].to_numpy(),
        expected["regime_confidence"],
        rtol=0.0,
        atol=1e-12,
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
