"""Unified risk configuration logic for trading engines.

This module provides consistent risk configuration building and merging
for both backtesting and live trading engines.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from src.config.constants import DEFAULT_BREAKEVEN_BUFFER, DEFAULT_BREAKEVEN_THRESHOLD

if TYPE_CHECKING:
    from src.position_management.dynamic_risk import DynamicRiskConfig
    from src.position_management.time_exits import TimeExitPolicy
    from src.position_management.trailing_stops import TrailingStopPolicy
    from src.risk.risk_manager import RiskManager

logger = logging.getLogger(__name__)


def merge_dynamic_risk_config(
    base_config: DynamicRiskConfig,
    strategy: Any,
) -> DynamicRiskConfig:
    """Merge strategy risk overrides with base dynamic risk configuration.

    Args:
        base_config: Base dynamic risk configuration.
        strategy: Strategy with optional get_risk_overrides() method.

    Returns:
        Merged DynamicRiskConfig with strategy overrides applied.
    """
    from src.position_management.dynamic_risk import DynamicRiskConfig

    try:
        strategy_overrides = (
            strategy.get_risk_overrides()
            if strategy and hasattr(strategy, "get_risk_overrides")
            else None
        )
        if not strategy_overrides or "dynamic_risk" not in strategy_overrides:
            return base_config

        dynamic_overrides = strategy_overrides["dynamic_risk"]

        return DynamicRiskConfig(
            enabled=dynamic_overrides.get("enabled", base_config.enabled),
            performance_window_days=dynamic_overrides.get(
                "performance_window_days", base_config.performance_window_days
            ),
            drawdown_thresholds=dynamic_overrides.get(
                "drawdown_thresholds", base_config.drawdown_thresholds
            ),
            risk_reduction_factors=dynamic_overrides.get(
                "risk_reduction_factors", base_config.risk_reduction_factors
            ),
            recovery_thresholds=dynamic_overrides.get(
                "recovery_thresholds", base_config.recovery_thresholds
            ),
            volatility_adjustment_enabled=dynamic_overrides.get(
                "volatility_adjustment_enabled",
                base_config.volatility_adjustment_enabled,
            ),
            volatility_window_days=dynamic_overrides.get(
                "volatility_window_days", base_config.volatility_window_days
            ),
            high_volatility_threshold=dynamic_overrides.get(
                "high_volatility_threshold", base_config.high_volatility_threshold
            ),
            low_volatility_threshold=dynamic_overrides.get(
                "low_volatility_threshold", base_config.low_volatility_threshold
            ),
            volatility_risk_multipliers=dynamic_overrides.get(
                "volatility_risk_multipliers", base_config.volatility_risk_multipliers
            ),
        )
    except Exception as e:
        logger.debug("Failed to merge strategy dynamic risk overrides: %s", e)
        return base_config


def build_trailing_stop_policy(
    strategy: Any,
    risk_manager: RiskManager | None = None,
) -> TrailingStopPolicy | None:
    """Build trailing stop policy from strategy/risk overrides.

    Args:
        strategy: Strategy with optional get_risk_overrides() method.
        risk_manager: Risk manager with optional params attribute.

    Returns:
        TrailingStopPolicy if configuration exists, None otherwise.
    """
    from src.position_management.trailing_stops import TrailingStopPolicy

    # Get overrides from strategy
    try:
        overrides = (
            strategy.get_risk_overrides()
            if hasattr(strategy, "get_risk_overrides")
            else None
        )
    except Exception:
        overrides = None

    cfg = None
    if overrides and isinstance(overrides, dict):
        cfg = overrides.get("trailing_stop")

    # Get params from risk manager
    params = getattr(risk_manager, "params", None) if risk_manager else None

    if cfg or params:
        # Activation threshold
        activation = (cfg.get("activation_threshold") if cfg else None) or (
            params.trailing_activation_threshold if params else None
        )

        # Trailing distance
        dist_pct = cfg.get("trailing_distance_pct") if cfg else None
        atr_mult = cfg.get("trailing_distance_atr_mult") if cfg else None
        if atr_mult is None and params is not None:
            atr_mult = params.trailing_atr_multiplier

        # Breakeven settings
        breakeven_threshold = (cfg.get("breakeven_threshold") if cfg else None) or (
            params.breakeven_threshold if params else None
        )
        breakeven_buffer = (cfg.get("breakeven_buffer") if cfg else None) or (
            params.breakeven_buffer if params else None
        )

        # Check if params have distance settings
        params_has_distance = bool(
            params
            and (
                params.trailing_distance_pct is not None
                or params.trailing_atr_multiplier is not None
            )
        )

        if activation and (dist_pct is not None or atr_mult is not None or params_has_distance):
            return TrailingStopPolicy(
                activation_threshold=float(activation),
                trailing_distance_pct=(
                    float(dist_pct)
                    if dist_pct is not None
                    else (
                        float(params.trailing_distance_pct)
                        if params and params.trailing_distance_pct is not None
                        else None
                    )
                ),
                atr_multiplier=float(atr_mult) if atr_mult is not None else None,
                breakeven_threshold=(
                    float(breakeven_threshold) if breakeven_threshold is not None else DEFAULT_BREAKEVEN_THRESHOLD
                ),
                breakeven_buffer=(
                    float(breakeven_buffer) if breakeven_buffer is not None else DEFAULT_BREAKEVEN_BUFFER
                ),
            )

    return None


def build_time_exit_policy(
    strategy: Any,
    risk_manager: RiskManager | None = None,
) -> TimeExitPolicy | None:
    """Build time exit policy from strategy/risk overrides.

    Args:
        strategy: Strategy with optional get_risk_overrides() method.
        risk_manager: Risk manager with optional params attribute.

    Returns:
        TimeExitPolicy if configuration exists, None otherwise.
    """
    from src.position_management.time_exits import TimeExitPolicy

    # Get overrides from strategy
    try:
        overrides = (
            strategy.get_risk_overrides()
            if hasattr(strategy, "get_risk_overrides")
            else None
        )
    except Exception:
        overrides = None

    time_cfg = None
    if overrides and isinstance(overrides, dict):
        time_cfg = overrides.get("time_exits")

    # Fallback to risk manager params
    if not time_cfg and risk_manager:
        params = getattr(risk_manager, "params", None)
        if params:
            # Check if params has time exit settings
            max_holding = getattr(params, "max_holding_hours", None)
            if max_holding is not None:
                time_cfg = {"max_holding_hours": max_holding}

    if time_cfg:
        max_holding = time_cfg.get("max_holding_hours")
        exit_time_str = time_cfg.get("exit_time")
        exit_days = time_cfg.get("exit_days")

        if max_holding is not None:
            return TimeExitPolicy(
                max_holding_hours=int(max_holding),
                exit_time=exit_time_str,
                exit_days=exit_days,
            )

    return None


def extract_risk_overrides(strategy: Any) -> dict[str, Any]:
    """Extract risk overrides from a strategy.

    Args:
        strategy: Strategy with optional get_risk_overrides() method.

    Returns:
        Dictionary of risk overrides, or empty dict if none.
    """
    try:
        if hasattr(strategy, "get_risk_overrides"):
            overrides = strategy.get_risk_overrides()
            if overrides and isinstance(overrides, dict):
                return overrides
    except Exception as e:
        logger.debug("Failed to extract risk overrides: %s", e)

    return {}


def get_risk_parameters(
    risk_manager: RiskManager | None,
) -> Any | None:
    """Get risk parameters from a risk manager.

    Args:
        risk_manager: Risk manager with optional params attribute.

    Returns:
        RiskParameters object or None.
    """
    if risk_manager is None:
        return None
    return getattr(risk_manager, "params", None)


__all__ = [
    "merge_dynamic_risk_config",
    "build_trailing_stop_policy",
    "build_time_exit_policy",
    "extract_risk_overrides",
    "get_risk_parameters",
]
