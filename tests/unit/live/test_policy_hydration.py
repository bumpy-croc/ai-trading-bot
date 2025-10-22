from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.live.trading_engine import LiveTradingEngine
from src.position_management.partial_manager import PartialExitPolicy
from src.position_management.trailing_stops import TrailingStopPolicy
from src.risk.risk_manager import RiskParameters
from src.strategies.components.policies import (
    PartialExitPolicyDescriptor,
    PolicyBundle,
    TrailingStopPolicyDescriptor,
)
from tests.mocks import MockDatabaseManager

pytestmark = pytest.mark.unit


def _build_engine(
    monkeypatch: pytest.MonkeyPatch,
    *,
    enable_partial_operations: bool,
    risk_parameters: RiskParameters,
    trailing_stop_policy: TrailingStopPolicy | None = None,
) -> LiveTradingEngine:
    """Create a LiveTradingEngine instance with a mocked database manager."""

    monkeypatch.setattr("src.live.trading_engine.DatabaseManager", MockDatabaseManager)

    strategy = Mock()
    strategy.name = "MockStrategy"
    strategy.trading_pair = "BTCUSDT"
    strategy.get_risk_overrides.return_value = {}

    data_provider = Mock()

    engine = LiveTradingEngine(
        strategy=strategy,
        data_provider=data_provider,
        enable_live_trading=False,
        resume_from_last_balance=False,
        enable_partial_operations=enable_partial_operations,
        trailing_stop_policy=trailing_stop_policy,
        enable_dynamic_risk=False,
        risk_parameters=risk_parameters,
    )

    return engine


def test_component_policy_hydration_respects_disabled_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    risk_params = RiskParameters(
        partial_exit_targets=[],
        partial_exit_sizes=[],
        scale_in_thresholds=[],
        scale_in_sizes=[],
        trailing_activation_threshold=None,
        trailing_distance_pct=None,
        trailing_atr_multiplier=None,
    )

    engine = _build_engine(
        monkeypatch,
        enable_partial_operations=False,
        risk_parameters=risk_params,
        trailing_stop_policy=None,
    )

    assert engine.partial_manager is None
    assert engine.trailing_stop_policy is None

    bundle = PolicyBundle(
        partial_exit=PartialExitPolicyDescriptor(exit_targets=[0.05], exit_sizes=[0.5]),
        trailing_stop=TrailingStopPolicyDescriptor(
            activation_threshold=0.01,
            trailing_distance_pct=0.005,
            breakeven_threshold=0.02,
            breakeven_buffer=0.001,
        ),
    )

    engine._apply_policies_from_decision(SimpleNamespace(policies=bundle))

    assert engine.partial_manager is None
    assert engine.trailing_stop_policy is None


def test_component_policy_hydration_updates_when_opted_in(monkeypatch: pytest.MonkeyPatch) -> None:
    risk_params = RiskParameters()

    engine = _build_engine(
        monkeypatch,
        enable_partial_operations=True,
        risk_parameters=risk_params,
        trailing_stop_policy=None,
    )

    assert isinstance(engine.partial_manager, PartialExitPolicy)
    assert isinstance(engine.trailing_stop_policy, TrailingStopPolicy)

    bundle = PolicyBundle(
        partial_exit=PartialExitPolicyDescriptor(
            exit_targets=[0.08],
            exit_sizes=[0.4],
            scale_in_thresholds=[0.03],
            scale_in_sizes=[0.25],
            max_scale_ins=2,
        ),
        trailing_stop=TrailingStopPolicyDescriptor(
            activation_threshold=0.025,
            trailing_distance_pct=0.01,
            breakeven_threshold=0.04,
            breakeven_buffer=0.002,
        ),
    )

    engine._apply_policies_from_decision(SimpleNamespace(policies=bundle))

    assert isinstance(engine.partial_manager, PartialExitPolicy)
    assert engine.partial_manager.exit_targets == pytest.approx([0.08])
    assert engine.partial_manager.exit_sizes == pytest.approx([0.4])
    assert engine.partial_manager.scale_in_thresholds == pytest.approx([0.03])
    assert engine.partial_manager.scale_in_sizes == pytest.approx([0.25])
    assert engine.partial_manager.max_scale_ins == 2

    assert isinstance(engine.trailing_stop_policy, TrailingStopPolicy)
    assert engine.trailing_stop_policy.activation_threshold == pytest.approx(0.025)
    assert engine.trailing_stop_policy.trailing_distance_pct == pytest.approx(0.01)
    assert engine.trailing_stop_policy.breakeven_threshold == pytest.approx(0.04)
    assert engine.trailing_stop_policy.breakeven_buffer == pytest.approx(0.002)
