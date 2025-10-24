"""Serializable descriptors for engine position-management policies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PartialExitPolicyDescriptor:
    """Descriptor mirroring :class:`PartialExitPolicy` configuration."""

    exit_targets: list[float]
    exit_sizes: list[float]
    scale_in_thresholds: list[float] = field(default_factory=list)
    scale_in_sizes: list[float] = field(default_factory=list)
    max_scale_ins: int = 0

    def to_policy(self):
        from src.position_management.partial_manager import PartialExitPolicy

        return PartialExitPolicy(
            exit_targets=list(self.exit_targets),
            exit_sizes=list(self.exit_sizes),
            scale_in_thresholds=list(self.scale_in_thresholds),
            scale_in_sizes=list(self.scale_in_sizes),
            max_scale_ins=int(self.max_scale_ins),
        )


@dataclass(frozen=True)
class TrailingStopPolicyDescriptor:
    """Descriptor for :class:`TrailingStopPolicy`."""

    activation_threshold: float
    trailing_distance_pct: float | None = None
    atr_multiplier: float | None = None
    breakeven_threshold: float | None = None
    breakeven_buffer: float = 0.0

    def to_policy(self):
        from src.position_management.trailing_stops import TrailingStopPolicy

        return TrailingStopPolicy(
            activation_threshold=float(self.activation_threshold),
            trailing_distance_pct=float(self.trailing_distance_pct)
            if self.trailing_distance_pct is not None
            else None,
            atr_multiplier=float(self.atr_multiplier)
            if self.atr_multiplier is not None
            else None,
            breakeven_threshold=float(self.breakeven_threshold)
            if self.breakeven_threshold is not None
            else None,
            breakeven_buffer=float(self.breakeven_buffer),
        )


@dataclass(frozen=True)
class DynamicRiskDescriptor:
    """Descriptor for :class:`DynamicRiskConfig`."""

    enabled: bool = True
    performance_window_days: int = 30
    drawdown_thresholds: list[float] | None = None
    risk_reduction_factors: list[float] | None = None
    recovery_thresholds: list[float] | None = None
    volatility_adjustment_enabled: bool = True
    volatility_window_days: int = 30
    high_volatility_threshold: float = 0.03
    low_volatility_threshold: float = 0.01
    volatility_risk_multipliers: tuple[float, float] = (0.7, 1.3)

    def to_config(self):
        from src.position_management.dynamic_risk import DynamicRiskConfig

        return DynamicRiskConfig(
            enabled=bool(self.enabled),
            performance_window_days=int(self.performance_window_days),
            drawdown_thresholds=list(self.drawdown_thresholds)
            if self.drawdown_thresholds is not None
            else None,
            risk_reduction_factors=list(self.risk_reduction_factors)
            if self.risk_reduction_factors is not None
            else None,
            recovery_thresholds=list(self.recovery_thresholds)
            if self.recovery_thresholds is not None
            else None,
            volatility_adjustment_enabled=bool(self.volatility_adjustment_enabled),
            volatility_window_days=int(self.volatility_window_days),
            high_volatility_threshold=float(self.high_volatility_threshold),
            low_volatility_threshold=float(self.low_volatility_threshold),
            volatility_risk_multipliers=tuple(self.volatility_risk_multipliers),
        )


@dataclass(frozen=True)
class PolicyBundle:
    """Aggregate container for optional policy descriptors."""

    partial_exit: PartialExitPolicyDescriptor | None = None
    trailing_stop: TrailingStopPolicyDescriptor | None = None
    dynamic_risk: DynamicRiskDescriptor | None = None

    def is_empty(self) -> bool:
        return not any((self.partial_exit, self.trailing_stop, self.dynamic_risk))

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if self.partial_exit is not None:
            payload["partial_exit"] = self.partial_exit.__dict__
        if self.trailing_stop is not None:
            payload["trailing_stop"] = self.trailing_stop.__dict__
        if self.dynamic_risk is not None:
            payload["dynamic_risk"] = self.dynamic_risk.__dict__
        return payload
