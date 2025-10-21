"""Tests for the strategy component position management integration."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from src.position_management.dynamic_risk import DynamicRiskConfig
from src.position_management.partial_manager import PositionState
from src.risk.risk_manager import RiskParameters
from src.strategies.components.position_sizer import (
    FixedFractionSizer,
    PolicyDrivenPositionSizer,
    PositionManagementSuite,
    PositionSizer,
)


@dataclass
class DummyDirection:
    value: str


@dataclass
class DummySignal:
    direction: DummyDirection
    confidence: float = 1.0
    strength: float = 1.0

    @classmethod
    def long(cls, *, confidence: float = 1.0, strength: float = 1.0) -> DummySignal:
        return cls(direction=DummyDirection("buy"), confidence=confidence, strength=strength)


class ZeroSizer(PositionSizer):
    def __init__(self) -> None:
        super().__init__("zero_sizer")

    def calculate_size(
        self,
        signal: DummySignal,
        balance: float,
        risk_amount: float,
        regime=None,
    ) -> float:
        return 0.0


def test_policy_driven_sizer_applies_dynamic_risk_reduction():
    base = FixedFractionSizer(fraction=0.1)
    suite = PositionManagementSuite.from_risk_parameters()
    sizer = PolicyDrivenPositionSizer(base, suite, min_fraction=0.0, max_fraction=1.0)

    signal = DummySignal.long()
    balance = 1000.0
    risk_amount = 500.0

    # Establish a previous peak to trigger a drawdown reduction
    sizer.update_performance_state(current_balance=balance, peak_balance=1500.0)

    base_size = base.calculate_size(signal, balance, risk_amount)
    adjusted_size = sizer.calculate_size(signal, balance, risk_amount)

    assert adjusted_size < base_size
    assert sizer.last_adjustments is not None
    assert sizer.last_adjustments.position_size_factor < 1.0


def test_policy_driven_sizer_respects_zero_base_size():
    base = ZeroSizer()
    suite = PositionManagementSuite.from_risk_parameters()
    sizer = PolicyDrivenPositionSizer(base, suite)

    signal = DummySignal.long()
    balance = 1_000.0
    risk_amount = 500.0

    assert sizer.calculate_size(signal, balance, risk_amount) == 0.0


def test_policy_driven_sizer_exposes_partial_and_scale_in_policies():
    params = RiskParameters(
        partial_exit_targets=[0.01],
        partial_exit_sizes=[0.5],
        scale_in_thresholds=[0.005],
        scale_in_sizes=[0.25],
        max_scale_ins=1,
    )
    suite = PositionManagementSuite.from_risk_parameters(
        params,
        dynamic_risk_config=DynamicRiskConfig(enabled=False),
    )
    sizer = PolicyDrivenPositionSizer(FixedFractionSizer(fraction=0.05), suite)

    position = PositionState(
        entry_price=100.0,
        side="long",
        original_size=1.0,
        current_size=1.0,
    )

    scale_in = sizer.evaluate_scale_in(position, current_price=101.0)
    assert scale_in is not None
    assert scale_in["type"] == "scale_in"

    partials = sizer.evaluate_partial_exits(position, current_price=102.0)
    assert partials
    assert partials[0]["type"] == "partial_exit"


def test_policy_driven_sizer_wraps_trailing_time_and_mfe_features():
    params = RiskParameters(
        trailing_activation_threshold=0.01,
        trailing_distance_pct=0.005,
        breakeven_threshold=0.015,
        time_exits={
            "max_holding_hours": 1,
            "time_restrictions": {"no_weekend": True},
        },
    )
    suite = PositionManagementSuite.from_risk_parameters(
        params,
        dynamic_risk_config=DynamicRiskConfig(enabled=False),
    )
    sizer = PolicyDrivenPositionSizer(FixedFractionSizer(fraction=0.05), suite)

    new_stop, activated, breakeven = sizer.update_trailing_stop(
        side="long",
        entry_price=100.0,
        current_price=103.0,
        existing_stop=None,
        position_fraction=1.0,
        atr=1.0,
    )
    assert activated is True
    assert new_stop is not None

    entry_time = datetime(2024, 1, 5, 12, tzinfo=UTC)
    now_time = entry_time + timedelta(hours=2)
    should_exit, reason = sizer.check_time_exit(entry_time, now_time)
    assert should_exit is True
    assert reason is not None
    assert sizer.next_scheduled_exit(entry_time, now_time) is not None

    metrics = sizer.update_mfe_mae(
        position_key="trade-1",
        entry_price=100.0,
        current_price=104.0,
        side="long",
        position_fraction=1.0,
        current_time=now_time,
    )
    assert metrics is not None
    assert metrics.mfe > 0

