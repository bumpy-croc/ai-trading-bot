"""Unit tests for the regime-gated exposure governor (#802)."""

from __future__ import annotations

import pytest

from src.config.constants import DEFAULT_EXPOSURE_CAP_UNKNOWN, DEFAULT_EXPOSURE_CAPS
from src.strategies.components.exposure_governor import ExposureGovernor

pytestmark = pytest.mark.unit


class _Label:
    def __init__(self, value: str):
        self.value = value


class _Regime:
    def __init__(self, trend: str, vol: str):
        self.trend = _Label(trend)
        self.volatility = _Label(vol)


@pytest.mark.parametrize(
    "trend,vol,expected_key",
    [
        ("trend_up", "low_vol", "trend_up_low_vol"),
        ("trend_up", "high_vol", "trend_up_high_vol"),
        ("trend_down", "low_vol", "trend_down_low_vol"),
        ("trend_down", "high_vol", "trend_down_high_vol"),
        ("range", "low_vol", "range_low_vol"),
        ("range", "high_vol", "range_high_vol"),
    ],
)
def test_regime_cap_maps_each_regime(trend, vol, expected_key):
    gov = ExposureGovernor(enabled=True)
    assert gov.regime_cap(_Regime(trend, vol)) == DEFAULT_EXPOSURE_CAPS[expected_key]


def test_regime_cap_none_is_most_conservative():
    gov = ExposureGovernor(enabled=True)
    assert gov.regime_cap(None) == DEFAULT_EXPOSURE_CAP_UNKNOWN


def test_regime_cap_unknown_trend_is_conservative():
    gov = ExposureGovernor(enabled=True)

    class NoTrend:
        trend = None
        volatility = _Label("low_vol")

    assert gov.regime_cap(NoTrend()) == DEFAULT_EXPOSURE_CAP_UNKNOWN


def test_regime_cap_normal_vol_treated_as_low_vol():
    # 'normal' volatility (not high_vol) uses the low_vol cap.
    gov = ExposureGovernor(enabled=True)
    assert (
        gov.regime_cap(_Regime("trend_up", "normal")) == DEFAULT_EXPOSURE_CAPS["trend_up_low_vol"]
    )


def test_cap_fraction_passes_through_when_headroom_ample():
    gov = ExposureGovernor(enabled=True)
    # trend_up_low_vol cap 0.50, no existing exposure, propose 0.10 -> unchanged
    allowed, reason = gov.cap_fraction(
        0.10, regime=_Regime("trend_up", "low_vol"), gross_exposure_fraction=0.0
    )
    assert allowed == 0.10
    assert reason is None


def test_cap_fraction_trims_to_headroom():
    gov = ExposureGovernor(enabled=True)
    # bear high_vol cap 0.15, existing 0.10 -> headroom 0.05; propose 0.10 -> 0.05
    allowed, reason = gov.cap_fraction(
        0.10, regime=_Regime("trend_down", "high_vol"), gross_exposure_fraction=0.10
    )
    assert allowed == pytest.approx(0.05)
    assert reason is not None and "capped" in reason


def test_cap_fraction_blocks_when_cap_reached():
    gov = ExposureGovernor(enabled=True)
    allowed, reason = gov.cap_fraction(
        0.10, regime=_Regime("trend_down", "high_vol"), gross_exposure_fraction=0.20
    )
    assert allowed == 0.0
    assert reason is not None and "reached" in reason


def test_cap_fraction_extra_factor_tightens_cap():
    gov = ExposureGovernor(enabled=True)
    # cap 0.50 * 0.5 = 0.25; existing 0.20 -> headroom 0.05
    allowed, reason = gov.cap_fraction(
        0.10,
        regime=_Regime("trend_up", "low_vol"),
        gross_exposure_fraction=0.20,
        extra_factor=0.5,
    )
    assert allowed == pytest.approx(0.05)
    assert reason is not None


def test_cap_fraction_ignores_nonfinite_current_exposure():
    gov = ExposureGovernor(enabled=True)
    allowed, _ = gov.cap_fraction(
        0.10, regime=_Regime("trend_up", "low_vol"), gross_exposure_fraction=float("nan")
    )
    assert allowed == 0.10  # NaN treated as 0 exposure


def test_disabled_by_default_reads_feature_flag(monkeypatch):
    monkeypatch.delenv("FEATURE_ENABLE_EXPOSURE_GOVERNOR", raising=False)
    gov = ExposureGovernor()  # no override -> reads flag (default OFF)
    assert gov.enabled is False


def test_feature_flag_env_enables(monkeypatch):
    monkeypatch.setenv("FEATURE_ENABLE_EXPOSURE_GOVERNOR", "true")
    assert ExposureGovernor().enabled is True


def test_rejects_out_of_range_caps():
    with pytest.raises(ValueError, match="in \\(0, 1\\]"):
        ExposureGovernor(caps={"trend_up_low_vol": 1.5}, enabled=True)
    with pytest.raises(ValueError, match="in \\(0, 1\\]"):
        ExposureGovernor(unknown_cap=0.0, enabled=True)
