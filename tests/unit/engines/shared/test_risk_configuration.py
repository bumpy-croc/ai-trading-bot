"""Tests for shared risk configuration builders (regression for #760, #971)."""

import pytest

from src.config.constants import (
    DEFAULT_BREAKEVEN_THRESHOLD,
    DEFAULT_MAX_HOLDING_HOURS,
    DEFAULT_TRAILING_ACTIVATION_THRESHOLD,
    DEFAULT_TRAILING_DISTANCE_ATR_MULT,
    DEFAULT_TRAILING_DISTANCE_PCT,
)
from src.engines.shared.risk_configuration import (
    build_early_cut_policy,
    build_time_exit_policy,
    build_trailing_stop_policy,
)
from src.position_management.early_cut import EarlyCutPolicy
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


def _default_params() -> "_Params":
    """RiskParameters-shaped defaults for the trailing-stop fields."""
    return _Params(
        trailing_activation_threshold=DEFAULT_TRAILING_ACTIVATION_THRESHOLD,
        trailing_distance_pct=DEFAULT_TRAILING_DISTANCE_PCT,
        trailing_atr_multiplier=DEFAULT_TRAILING_DISTANCE_ATR_MULT,
        breakeven_threshold=DEFAULT_BREAKEVEN_THRESHOLD,
        breakeven_buffer=0.001,
    )


class TestBuildTrailingStopPolicyExistingBehavior:
    """Regression pins: existing config shapes must produce identical policies.

    These cases mirror the strategies that declare a ``trailing_stop``
    override today (hyper_growth, kelly_momentum, leveraged_regime,
    momentum_leverage, chaos_test, ensemble_weighted) plus the params-only
    path, so the #971 expressibility rework provably changes nothing for
    any existing config.
    """

    @pytest.mark.fast
    def test_hyper_growth_shaped_cfg_with_default_params(self):
        """hyper_growth's cfg: params atr multiplier still leaks in (pinned)."""
        strategy = _StrategyWithOverrides(
            {
                "trailing_stop": {
                    "activation_threshold": 0.03,
                    "trailing_distance_pct": 0.015,
                    "breakeven_threshold": 0.05,
                    "breakeven_buffer": 0.008,
                }
            }
        )
        policy = build_trailing_stop_policy(strategy, _RiskManager(_default_params()))

        assert policy is not None
        assert policy.activation_threshold == 0.03
        assert policy.trailing_distance_pct == 0.015
        # Key absent from cfg -> params fallback (pre-existing behavior,
        # kept bit-identical; the TrailingStopManager prefers pct distance
        # so this leaked multiplier is inert at runtime).
        assert policy.atr_multiplier == DEFAULT_TRAILING_DISTANCE_ATR_MULT
        assert policy.breakeven_threshold == 0.05
        assert policy.breakeven_buffer == 0.008

    @pytest.mark.fast
    def test_cfg_without_breakeven_keys_falls_back_to_params(self):
        """chaos_test's cfg shape: missing breakeven keys use params values."""
        strategy = _StrategyWithOverrides(
            {
                "trailing_stop": {
                    "activation_threshold": 0.005,
                    "trailing_distance_pct": 0.003,
                }
            }
        )
        policy = build_trailing_stop_policy(strategy, _RiskManager(_default_params()))

        assert policy is not None
        assert policy.activation_threshold == 0.005
        assert policy.trailing_distance_pct == 0.003
        assert policy.breakeven_threshold == DEFAULT_BREAKEVEN_THRESHOLD
        assert policy.breakeven_buffer == 0.001

    @pytest.mark.fast
    def test_params_only_path(self):
        """No strategy cfg: params-driven policy (unchanged)."""
        strategy = _StrategyWithOverrides(None)
        policy = build_trailing_stop_policy(strategy, _RiskManager(_default_params()))

        assert policy is not None
        assert policy.activation_threshold == DEFAULT_TRAILING_ACTIVATION_THRESHOLD
        assert policy.trailing_distance_pct == DEFAULT_TRAILING_DISTANCE_PCT
        assert policy.atr_multiplier == DEFAULT_TRAILING_DISTANCE_ATR_MULT

    @pytest.mark.fast
    def test_no_cfg_no_params_returns_none(self):
        assert build_trailing_stop_policy(_StrategyWithOverrides(None)) is None

    @pytest.mark.fast
    def test_inert_extra_keys_ignored(self):
        """ensemble_weighted's cfg carries extra keys ('enabled': True,
        'distance', 'step') that must stay inert."""
        strategy = _StrategyWithOverrides(
            {
                "trailing_stop": {
                    "enabled": True,
                    "activation_threshold": 0.04,
                    "distance": 0.02,
                    "step": 0.01,
                    "cooldown_hours": 4,
                }
            }
        )
        policy = build_trailing_stop_policy(strategy, _RiskManager(_default_params()))

        assert policy is not None
        assert policy.activation_threshold == 0.04
        # No trailing_distance_pct key -> params fallback (unchanged).
        assert policy.trailing_distance_pct == DEFAULT_TRAILING_DISTANCE_PCT


class TestBuildTrailingStopPolicyExpressibility:
    """#971: config must be able to genuinely control trailing/breakeven."""

    @pytest.mark.fast
    def test_enabled_false_disables_trailing_entirely(self):
        strategy = _StrategyWithOverrides({"trailing_stop": {"enabled": False}})

        assert build_trailing_stop_policy(strategy, _RiskManager(_default_params())) is None

    @pytest.mark.fast
    def test_explicit_none_atr_mult_stops_params_leak(self):
        strategy = _StrategyWithOverrides(
            {
                "trailing_stop": {
                    "activation_threshold": 0.03,
                    "trailing_distance_pct": 0.02,
                    "trailing_distance_atr_mult": None,
                }
            }
        )
        policy = build_trailing_stop_policy(strategy, _RiskManager(_default_params()))

        assert policy is not None
        assert policy.trailing_distance_pct == 0.02
        assert policy.atr_multiplier is None

    @pytest.mark.fast
    def test_explicit_none_breakeven_disables_breakeven(self):
        strategy = _StrategyWithOverrides(
            {
                "trailing_stop": {
                    "activation_threshold": 0.03,
                    "trailing_distance_pct": 0.02,
                    "breakeven_threshold": None,
                }
            }
        )
        policy = build_trailing_stop_policy(strategy, _RiskManager(_default_params()))

        assert policy is not None
        assert policy.breakeven_threshold is None

    @pytest.mark.fast
    def test_breakeven_only_policy_without_trailing_distance(self):
        """Breakeven-at-+X% without any trailing distance is expressible."""
        strategy = _StrategyWithOverrides(
            {
                "trailing_stop": {
                    "activation_threshold": 0.05,
                    "trailing_distance_pct": None,
                    "trailing_distance_atr_mult": None,
                    "breakeven_threshold": 0.05,
                    "breakeven_buffer": 0.005,
                }
            }
        )
        policy = build_trailing_stop_policy(strategy, _RiskManager(_default_params()))

        assert policy is not None
        assert policy.trailing_distance_pct is None
        assert policy.atr_multiplier is None
        assert policy.breakeven_threshold == 0.05
        assert policy.breakeven_buffer == 0.005


class TestBuildEarlyCutPolicy:
    """#971: MFE-conditioned early-cut config channels."""

    @pytest.mark.fast
    def test_builds_from_strategy_overrides(self):
        strategy = _StrategyWithOverrides(
            {"early_cut": {"mfe_threshold_pct": 0.015, "evaluation_window_hours": 18}}
        )
        policy = build_early_cut_policy(strategy)

        assert isinstance(policy, EarlyCutPolicy)
        assert policy.mfe_threshold_pct == 0.015
        assert policy.evaluation_window_hours == 18

    @pytest.mark.fast
    def test_falls_back_to_risk_manager_params(self):
        """Expressible via RiskParameters(early_cut=...) — the same channel
        the exit-geometry experiment used for time_exits."""
        strategy = _StrategyWithOverrides(None)
        risk_manager = _RiskManager(
            _Params(early_cut={"mfe_threshold_pct": 0.02, "evaluation_window_hours": 12})
        )
        policy = build_early_cut_policy(strategy, risk_manager)

        assert isinstance(policy, EarlyCutPolicy)
        assert policy.mfe_threshold_pct == 0.02
        assert policy.evaluation_window_hours == 12

    @pytest.mark.fast
    def test_strategy_overrides_win_over_params(self):
        strategy = _StrategyWithOverrides(
            {"early_cut": {"mfe_threshold_pct": 0.015, "evaluation_window_hours": 18}}
        )
        risk_manager = _RiskManager(
            _Params(early_cut={"mfe_threshold_pct": 0.02, "evaluation_window_hours": 12})
        )
        policy = build_early_cut_policy(strategy, risk_manager)

        assert policy is not None
        assert policy.mfe_threshold_pct == 0.015

    @pytest.mark.fast
    def test_returns_none_without_config(self):
        """DEFAULT OFF: no config anywhere means no policy."""
        assert build_early_cut_policy(_StrategyWithOverrides(None)) is None
        assert build_early_cut_policy(object()) is None
        assert (
            build_early_cut_policy(
                _StrategyWithOverrides(None), _RiskManager(_Params(early_cut=None))
            )
            is None
        )

    @pytest.mark.fast
    def test_enabled_false_returns_none(self):
        strategy = _StrategyWithOverrides(
            {
                "early_cut": {
                    "enabled": False,
                    "mfe_threshold_pct": 0.015,
                    "evaluation_window_hours": 18,
                }
            }
        )
        assert build_early_cut_policy(strategy) is None

    @pytest.mark.fast
    def test_invalid_config_returns_none(self):
        """Malformed config disables the policy (warn, never crash the engine)."""
        assert build_early_cut_policy(_StrategyWithOverrides({"early_cut": "18h"})) is None
        assert (
            build_early_cut_policy(
                _StrategyWithOverrides({"early_cut": {"mfe_threshold_pct": 0.015}})
            )
            is None
        )
        assert (
            build_early_cut_policy(
                _StrategyWithOverrides(
                    {"early_cut": {"mfe_threshold_pct": -1, "evaluation_window_hours": 18}}
                )
            )
            is None
        )
