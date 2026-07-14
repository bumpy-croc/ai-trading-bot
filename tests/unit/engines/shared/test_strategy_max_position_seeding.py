"""Tests for the shared strategy ``max_fraction`` seeding helper.

One seam feeds ``RiskParameters.max_position_size`` from a strategy's
``get_risk_overrides()['max_fraction']`` for BOTH the backtest CLI and the
experiment harness, so the two can never drift apart again (GH #1021). The
acceptance rules here mirror the backtest CLI's original inline logic
bit-for-bit: dict overrides only, numeric values only, (0, 1] only.
"""

from __future__ import annotations

import pytest

from src.engines.shared.risk_configuration import resolve_strategy_max_position_size

pytestmark = pytest.mark.fast


class _StubStrategy:
    """Minimal strategy exposing configurable risk overrides."""

    def __init__(self, overrides):
        self._overrides = overrides

    def get_risk_overrides(self):
        return self._overrides


class TestResolveStrategyMaxPositionSize:
    def test_returns_valid_max_fraction(self):
        assert resolve_strategy_max_position_size(_StubStrategy({"max_fraction": 0.25})) == 0.25

    def test_full_allocation_is_accepted(self):
        """Trend-following style strategies request up to 100% allocation."""
        assert resolve_strategy_max_position_size(_StubStrategy({"max_fraction": 0.95})) == 0.95
        assert resolve_strategy_max_position_size(_StubStrategy({"max_fraction": 1})) == 1.0

    def test_returns_float_type(self):
        result = resolve_strategy_max_position_size(_StubStrategy({"max_fraction": 1}))
        assert isinstance(result, float)

    def test_strategy_without_override_hook_returns_none(self):
        assert resolve_strategy_max_position_size(object()) is None

    def test_non_dict_overrides_return_none(self):
        assert resolve_strategy_max_position_size(_StubStrategy(None)) is None
        assert resolve_strategy_max_position_size(_StubStrategy([("max_fraction", 0.25)])) is None

    def test_missing_max_fraction_key_returns_none(self):
        assert resolve_strategy_max_position_size(_StubStrategy({"base_fraction": 0.1})) is None

    @pytest.mark.parametrize("invalid", [0, 0.0, -0.5, 1.5, "0.3", None])
    def test_invalid_max_fraction_values_return_none(self, invalid):
        assert resolve_strategy_max_position_size(_StubStrategy({"max_fraction": invalid})) is None

    def test_override_hook_exceptions_propagate(self):
        """Fail loud, matching the CLI: a broken get_risk_overrides() must not
        be silently swallowed into default sizing."""

        class _Broken:
            def get_risk_overrides(self):
                raise RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            resolve_strategy_max_position_size(_Broken())
