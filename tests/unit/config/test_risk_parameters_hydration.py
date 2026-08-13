"""Parity tripwire: ``RiskParameters()`` must equal the ratified limits.

Design §3.9.4 (docs/architecture/proposals/2026-07-14_single-source-risk-config.md).
Before hydration, ``src/config/risk-limits.json`` was inert — the Board ratified
a file that governed nothing. These tests bind the two together so any future
drift between the ratified file and the constructed defaults fails CI.
"""

from __future__ import annotations

import pytest

from src.config.risk_limits import RiskLimitsError, get_risk_limits
from src.risk.risk_manager import RiskParameters

# field name on RiskParameters -> dotted accessor on RiskLimits
RATIFIED_FIELD_MAP = {
    "base_risk_per_trade": "position.base_risk_per_trade_pct",
    "max_risk_per_trade": "position.max_risk_per_trade_pct",
    "max_position_size": "position.max_position_size_pct",
    "max_daily_risk": "portfolio.max_daily_risk_pct",
    "max_drawdown": "portfolio.max_drawdown_pct",
    "max_correlated_exposure": "portfolio.max_correlated_exposure_pct",
}


def _resolve(limits, dotted: str) -> float:
    section, key = dotted.split(".")
    return getattr(getattr(limits, section), key)


@pytest.mark.fast
class TestRatifiedDefaultParity:
    """A bare ``RiskParameters()`` yields Board-ratified values by construction."""

    @pytest.mark.parametrize(("field", "dotted"), sorted(RATIFIED_FIELD_MAP.items()))
    def test_bare_construction_matches_ratified_value(self, field: str, dotted: str) -> None:
        assert getattr(RiskParameters(), field) == _resolve(get_risk_limits(), dotted)

    def test_every_hydrated_field_is_covered_by_this_test(self) -> None:
        """The map above is the contract; a new hydrated field must join it."""
        assert set(RiskParameters._RATIFIED_FIELD_MAP) == set(RATIFIED_FIELD_MAP)


@pytest.mark.fast
class TestExplicitArgumentsWin:
    """Caller intent overrides ratified defaults — tightening is always allowed."""

    def test_explicit_tighter_value_is_preserved(self) -> None:
        assert RiskParameters(max_position_size=0.05).max_position_size == 0.05

    def test_explicit_value_equal_to_ratified_is_preserved(self) -> None:
        ratified = get_risk_limits().position.max_position_size_pct
        assert RiskParameters(max_position_size=ratified).max_position_size == ratified

    def test_explicit_values_still_pass_validation(self) -> None:
        with pytest.raises(ValueError, match="max_position_size"):
            RiskParameters(max_position_size=0.0)

    def test_round_trip_through_asdict_preserves_values(self) -> None:
        from dataclasses import asdict

        original = RiskParameters(max_position_size=0.07, max_drawdown=0.11)
        clone = RiskParameters(**asdict(original))
        assert clone.max_position_size == 0.07
        assert clone.max_drawdown == 0.11


@pytest.mark.fast
class TestFailsClosed:
    """A missing/invalid ratified file must not silently fall back to a literal."""

    def test_construction_raises_when_limits_unloadable(self, monkeypatch) -> None:
        def _boom() -> None:
            raise RiskLimitsError("simulated invalid risk-limits.json")

        monkeypatch.setattr("src.risk.risk_manager.get_risk_limits", _boom)
        with pytest.raises(RiskLimitsError):
            RiskParameters()

    def test_fully_specified_construction_still_needs_no_file(self, monkeypatch) -> None:
        """Explicit values for every ratified field skip the loader entirely."""

        def _boom() -> None:
            raise RiskLimitsError("simulated invalid risk-limits.json")

        monkeypatch.setattr("src.risk.risk_manager.get_risk_limits", _boom)
        params = RiskParameters(
            base_risk_per_trade=0.01,
            max_risk_per_trade=0.02,
            max_position_size=0.10,
            max_daily_risk=0.05,
            max_drawdown=0.15,
            max_correlated_exposure=0.10,
        )
        assert params.max_position_size == 0.10
