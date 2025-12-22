from __future__ import annotations

from datetime import datetime

import pytest

from src.backtesting.engine import Backtester
from src.data_providers.mock_data_provider import MockDataProvider
from src.position_management.partial_manager import PartialExitPolicy
from src.position_management.trailing_stops import TrailingStopPolicy
from src.strategies.components import (
    DynamicRiskDescriptor,
    PartialExitPolicyDescriptor,
    PolicyBundle,
    Signal,
    SignalDirection,
    StrategyRuntime,
    TrailingStopPolicyDescriptor,
)
from src.strategies.components.strategy import TradingDecision


class DummyRuntimeStrategy:
    name = "DummyRuntimeStrategy"
    warmup_period = 0

    def get_feature_generators(self):
        return []

    def prepare_runtime(self, dataset):  # pragma: no cover - simple stub
        return None

    def process_candle(self, df, index, balance, current_positions=None):  # pragma: no cover
        raise NotImplementedError

    def finalize_runtime(self):  # pragma: no cover - simple stub
        return None

    def get_risk_overrides(self):
        return {}


@pytest.fixture
def backtester() -> Backtester:
    runtime = StrategyRuntime(DummyRuntimeStrategy())
    return Backtester(
        strategy=runtime,
        data_provider=MockDataProvider(num_candles=10),
        partial_manager=PartialExitPolicy(exit_targets=[0.02], exit_sizes=[0.5]),
        trailing_stop_policy=TrailingStopPolicy(
            activation_threshold=0.01,
            trailing_distance_pct=0.005,
        ),
        enable_dynamic_risk=False,
    )


@pytest.mark.skip(reason="_apply_policies_from_decision method was removed in develop branch refactoring")
def test_apply_policies_from_decision_updates_backtester_state(backtester: Backtester) -> None:
    decision = TradingDecision(
        timestamp=datetime.utcnow(),
        signal=Signal(
            direction=SignalDirection.BUY,
            strength=1.0,
            confidence=1.0,
            metadata={},
        ),
        position_size=1_000.0,
        regime=None,
        risk_metrics={},
        execution_time_ms=0.0,
        metadata={},
        policies=PolicyBundle(
            partial_exit=PartialExitPolicyDescriptor(
                exit_targets=[0.03],
                exit_sizes=[0.25],
            ),
            trailing_stop=TrailingStopPolicyDescriptor(
                activation_threshold=0.05,
                trailing_distance_pct=0.01,
            ),
            dynamic_risk=DynamicRiskDescriptor(
                performance_window_days=15,
                drawdown_thresholds=[0.1],
                risk_reduction_factors=[0.4],
                recovery_thresholds=[0.05],
            ),
        ),
    )

    backtester._apply_policies_from_decision(decision)

    assert backtester.partial_manager.exit_targets == [0.03]
    assert backtester.partial_manager.exit_sizes == [0.25]
    assert backtester.trailing_stop_policy is not None
    assert backtester.trailing_stop_policy.activation_threshold == pytest.approx(0.05)
    assert backtester.trailing_stop_policy.trailing_distance_pct == pytest.approx(0.01)
    assert backtester.enable_dynamic_risk is True
    assert backtester.dynamic_risk_manager is not None
    assert (
        backtester.dynamic_risk_manager.config.performance_window_days == 15
    )
    assert backtester.dynamic_risk_manager.config.drawdown_thresholds == [0.1]
