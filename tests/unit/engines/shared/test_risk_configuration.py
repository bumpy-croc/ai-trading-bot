"""Tests for shared risk configuration builders (regression for #760)."""

import pytest

from src.config.constants import DEFAULT_MAX_HOLDING_HOURS
from src.engines.shared.risk_configuration import build_time_exit_policy
from src.position_management.time_exits import TimeExitPolicy

pytestmark = pytest.mark.unit


class _StrategyWithOverrides:
    def __init__(self, overrides):
        self._overrides = overrides

    def get_risk_overrides(self):
        return self._overrides


class _Params:
    def __init__(self, **attrs):
        for key, value in attrs.items():
            setattr(self, key, value)


class _RiskManager:
    def __init__(self, params):
        self.params = params


class TestBuildTimeExitPolicy:
    @pytest.mark.fast
    def test_builds_policy_from_strategy_overrides(self):
        """Regression for #760: a max_holding_hours config must produce a
        policy (the old code passed nonexistent kwargs and always returned
        None)."""
        strategy = _StrategyWithOverrides({"time_exits": {"max_holding_hours": 24}})

        policy = build_time_exit_policy(strategy)

        assert isinstance(policy, TimeExitPolicy)
        assert policy.max_holding_hours == 24

    @pytest.mark.fast
    def test_maps_full_config_shape(self):
        """The helper maps the same config keys as both engines' builders."""
        strategy = _StrategyWithOverrides(
            {
                "time_exits": {
                    "max_holding_hours": 48,
                    "end_of_day_flat": True,
                    "weekend_flat": True,
                    "market_timezone": "America/New_York",
                    "time_restrictions": {
                        "no_overnight": True,
                        "no_weekend": False,
                        "trading_hours_only": True,
                    },
                }
            }
        )

        policy = build_time_exit_policy(strategy)

        assert isinstance(policy, TimeExitPolicy)
        assert policy.max_holding_hours == 48
        assert policy.end_of_day_flat is True
        assert policy.weekend_flat is True
        assert policy.market_timezone == "America/New_York"
        assert policy.time_restrictions.no_overnight is True
        assert policy.time_restrictions.no_weekend is False
        assert policy.time_restrictions.trading_hours_only is True

    @pytest.mark.fast
    def test_defaults_applied_for_missing_keys(self):
        """Keys omitted from the config fall back to their DEFAULT_* constants."""
        strategy = _StrategyWithOverrides({"time_exits": {"weekend_flat": True}})

        policy = build_time_exit_policy(strategy)

        assert isinstance(policy, TimeExitPolicy)
        assert policy.max_holding_hours == DEFAULT_MAX_HOLDING_HOURS
        assert policy.weekend_flat is True

    @pytest.mark.fast
    def test_falls_back_to_risk_manager_time_exits_params(self):
        """Engines read params.time_exits — the shared helper must too."""
        strategy = _StrategyWithOverrides(None)
        risk_manager = _RiskManager(_Params(time_exits={"max_holding_hours": 12}))

        policy = build_time_exit_policy(strategy, risk_manager)

        assert isinstance(policy, TimeExitPolicy)
        assert policy.max_holding_hours == 12

    @pytest.mark.fast
    def test_falls_back_to_legacy_max_holding_hours_param(self):
        """The helper's original params.max_holding_hours fallback still works."""
        strategy = _StrategyWithOverrides(None)
        risk_manager = _RiskManager(_Params(max_holding_hours=6))

        policy = build_time_exit_policy(strategy, risk_manager)

        assert isinstance(policy, TimeExitPolicy)
        assert policy.max_holding_hours == 6

    @pytest.mark.fast
    def test_returns_none_without_config(self):
        """No time_exits config (None override or missing attribute) returns None."""
        strategy = _StrategyWithOverrides(None)

        assert build_time_exit_policy(strategy) is None
        assert build_time_exit_policy(object()) is None

    @pytest.mark.fast
    def test_non_dict_config_returns_none(self):
        """A non-dict time_exits value emits a warning and returns None."""
        strategy = _StrategyWithOverrides({"time_exits": "24h"})

        assert build_time_exit_policy(strategy) is None

    @pytest.mark.fast
    def test_invalid_time_restrictions_type_returns_none(self):
        """A non-dict time_restrictions value triggers the except handler and returns None."""
        strategy = _StrategyWithOverrides({"time_exits": {"time_restrictions": "invalid"}})

        assert build_time_exit_policy(strategy) is None
