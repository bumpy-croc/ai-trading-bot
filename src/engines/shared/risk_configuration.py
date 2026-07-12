"""Unified risk configuration logic for trading engines.

This module provides consistent risk configuration building and merging
for both backtesting and live trading engines.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from src.config.constants import (
    DEFAULT_BREAKEVEN_BUFFER,
    DEFAULT_BREAKEVEN_THRESHOLD,
    DEFAULT_END_OF_DAY_FLAT,
    DEFAULT_MARKET_TIMEZONE,
    DEFAULT_MAX_HOLDING_HOURS,
    DEFAULT_TIME_RESTRICTIONS,
    DEFAULT_WEEKEND_FLAT,
)

if TYPE_CHECKING:
    from src.position_management.dynamic_risk import DynamicRiskConfig
    from src.position_management.early_cut import EarlyCutPolicy
    from src.position_management.partial_manager import PartialExitPolicy
    from src.position_management.time_exits import TimeExitPolicy
    from src.position_management.trailing_stops import TrailingStopPolicy
    from src.risk.risk_manager import RiskManager

logger = logging.getLogger(__name__)

# Sentinel distinguishing "key absent from config" (fall back to risk
# parameters) from an explicit None (honored as-is, e.g. "no ATR distance").
_UNSET = object()


def _resolve_cfg_value(cfg: dict[str, Any] | None, key: str, fallback: Any) -> Any:
    """Key-presence resolution: an explicit config key wins (even ``None``);
    an absent key falls back to the risk-parameter value."""
    if cfg is not None and key in cfg:
        return cfg[key]
    return fallback


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

    Resolution is key-presence based (#971 expressibility): a key present in
    the strategy's ``trailing_stop`` overrides is authoritative — including
    an explicit ``None`` (e.g. ``"trailing_distance_atr_mult": None`` blocks
    the risk-parameter ATR fallback, ``"breakeven_threshold": None`` disables
    the breakeven move). A key that is absent falls back to the risk-manager
    parameters, exactly as before. ``{"enabled": False}`` disables trailing
    entirely (no fallback policy is built).

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
            strategy.get_risk_overrides() if hasattr(strategy, "get_risk_overrides") else None
        )
    except Exception:
        overrides = None

    cfg = None
    if overrides and isinstance(overrides, dict):
        cfg = overrides.get("trailing_stop")

    if cfg is not None and not isinstance(cfg, dict):
        logger.warning("Ignoring non-dict trailing_stop config: %r", cfg)
        cfg = None

    # Explicit opt-out: the strategy/config declared trailing off.
    if cfg is not None and cfg.get("enabled") is False:
        return None

    # Get params from risk manager
    params = getattr(risk_manager, "params", None) if risk_manager else None

    if cfg or params:
        activation = _resolve_cfg_value(
            cfg, "activation_threshold", params.trailing_activation_threshold if params else None
        )
        dist_pct = _resolve_cfg_value(
            cfg, "trailing_distance_pct", params.trailing_distance_pct if params else None
        )
        atr_mult = _resolve_cfg_value(
            cfg, "trailing_distance_atr_mult", params.trailing_atr_multiplier if params else None
        )

        # Breakeven settings. _UNSET (no cfg key AND no params value) maps to
        # the DEFAULT_* constants at construction — matching the previous
        # or-chain, where only a None params value fell through to defaults;
        # a params value of 0.0 is preserved as 0.0 (which TrailingStopPolicy
        # treats as breakeven disabled).
        params_be = params.breakeven_threshold if params else None
        breakeven_threshold = _resolve_cfg_value(
            cfg, "breakeven_threshold", params_be if params_be is not None else _UNSET
        )
        params_bb = params.breakeven_buffer if params else None
        breakeven_buffer = _resolve_cfg_value(
            cfg, "breakeven_buffer", params_bb if params_bb is not None else _UNSET
        )
        if breakeven_buffer is None:
            breakeven_buffer = _UNSET

        # A breakeven threshold declared explicitly by the strategy config
        # supports a breakeven-only policy (no trailing distance). Distances
        # resolved from params alone do NOT trigger this rescue, so the
        # pre-#971 "no distance anywhere -> no policy" outcome is preserved.
        cfg_has_breakeven = (
            cfg is not None
            and "breakeven_threshold" in cfg
            and cfg["breakeven_threshold"] is not None
        )

        if activation and (dist_pct is not None or atr_mult is not None or cfg_has_breakeven):
            try:
                return TrailingStopPolicy(
                    activation_threshold=float(activation),
                    trailing_distance_pct=(float(dist_pct) if dist_pct is not None else None),
                    atr_multiplier=float(atr_mult) if atr_mult is not None else None,
                    breakeven_threshold=(
                        DEFAULT_BREAKEVEN_THRESHOLD
                        if breakeven_threshold is _UNSET
                        else (
                            float(breakeven_threshold) if breakeven_threshold is not None else None
                        )
                    ),
                    breakeven_buffer=(
                        DEFAULT_BREAKEVEN_BUFFER
                        if breakeven_buffer is _UNSET
                        else float(breakeven_buffer)
                    ),
                )
            except (TypeError, ValueError) as e:
                logger.warning(
                    "Failed to build TrailingStopPolicy due to invalid configuration values: %s",
                    e,
                )
                return None

    return None


def build_early_cut_policy(
    strategy: Any,
    risk_manager: RiskManager | None = None,
) -> EarlyCutPolicy | None:
    """Build the MFE-conditioned early-cut policy from strategy/risk overrides.

    DEFAULT OFF: returns None unless the strategy's ``early_cut`` overrides
    or ``RiskParameters.early_cut`` provide both ``mfe_threshold_pct`` and
    ``evaluation_window_hours``. Strategy overrides win over risk parameters
    (same precedence as ``time_exits``). ``{"enabled": False}`` disables the
    policy regardless of other keys.

    Args:
        strategy: Strategy with optional get_risk_overrides() method.
        risk_manager: Risk manager with optional params attribute.

    Returns:
        EarlyCutPolicy if a complete, valid configuration exists, else None.
    """
    from src.position_management.early_cut import EarlyCutPolicy

    cfg = extract_risk_overrides(strategy).get("early_cut")

    if cfg is None and risk_manager is not None:
        params = getattr(risk_manager, "params", None)
        cfg = getattr(params, "early_cut", None) if params else None

    if cfg is None:
        return None
    if not isinstance(cfg, dict):
        logger.warning("Ignoring non-dict early_cut config: %r", cfg)
        return None
    if cfg.get("enabled") is False:
        return None
    if "mfe_threshold_pct" not in cfg or "evaluation_window_hours" not in cfg:
        logger.warning(
            "Ignoring incomplete early_cut config %r: both mfe_threshold_pct "
            "and evaluation_window_hours are required",
            cfg,
        )
        return None

    try:
        return EarlyCutPolicy(
            mfe_threshold_pct=float(cfg["mfe_threshold_pct"]),
            evaluation_window_hours=float(cfg["evaluation_window_hours"]),
        )
    except (TypeError, ValueError) as e:
        logger.warning("Failed to build EarlyCutPolicy from config %r: %s", cfg, e)
        return None


def build_time_exit_policy(
    strategy: Any,
    risk_manager: RiskManager | None = None,
) -> TimeExitPolicy | None:
    """Build time exit policy from strategy/risk overrides.

    Maps the same ``time_exits`` config shape as both engines' builders
    (``Backtester._build_time_exit_policy`` and the live engine's inline
    construction): ``max_holding_hours``, ``end_of_day_flat``, ``weekend_flat``,
    ``market_timezone``, ``time_restrictions``.

    Args:
        strategy: Strategy with optional get_risk_overrides() method.
        risk_manager: Risk manager with optional params attribute. Both the
            engines' ``params.time_exits`` dict and the legacy
            ``params.max_holding_hours`` attribute are honored as fallbacks.

    Returns:
        TimeExitPolicy if configuration exists, None otherwise.
    """
    from src.position_management.time_exits import TimeExitPolicy, TimeRestrictions

    # Get overrides from strategy
    try:
        overrides = (
            strategy.get_risk_overrides() if hasattr(strategy, "get_risk_overrides") else None
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
            time_cfg = getattr(params, "time_exits", None)
            if not time_cfg:
                # Legacy single-setting fallback
                max_holding = getattr(params, "max_holding_hours", None)
                if max_holding is not None:
                    time_cfg = {"max_holding_hours": max_holding}

    if not time_cfg:
        return None
    if not isinstance(time_cfg, dict):
        logger.warning("Ignoring non-dict time_exits config: %r", time_cfg)
        return None

    try:
        restrictions_cfg = time_cfg.get("time_restrictions")
        if restrictions_cfg is None:
            restrictions_cfg = DEFAULT_TIME_RESTRICTIONS
        restrictions = TimeRestrictions(
            no_overnight=bool(restrictions_cfg.get("no_overnight", False)),
            no_weekend=bool(restrictions_cfg.get("no_weekend", False)),
            trading_hours_only=bool(restrictions_cfg.get("trading_hours_only", False)),
        )

        return TimeExitPolicy(
            max_holding_hours=time_cfg.get("max_holding_hours", DEFAULT_MAX_HOLDING_HOURS),
            end_of_day_flat=time_cfg.get("end_of_day_flat", DEFAULT_END_OF_DAY_FLAT),
            weekend_flat=time_cfg.get("weekend_flat", DEFAULT_WEEKEND_FLAT),
            market_timezone=time_cfg.get("market_timezone", DEFAULT_MARKET_TIMEZONE),
            time_restrictions=restrictions,
        )
    except (TypeError, ValueError, AttributeError) as e:
        logger.warning("Failed to build TimeExitPolicy from config %r: %s", time_cfg, e)
        return None


def build_partial_exit_policy(
    strategy: Any,
    risk_parameters: Any | None = None,
) -> PartialExitPolicy:
    """Build the partial exit/scale-in policy for an engine.

    Strategy-declared ``partial_operations`` overrides win; risk-parameter
    defaults (which hydrate ``DEFAULT_PARTIAL_EXIT_TARGETS``) apply only when
    the strategy specifies nothing. Used by both engines so partial-config
    hydration stays in parity.

    Args:
        strategy: Strategy with optional get_risk_overrides() method.
        risk_parameters: RiskParameters fallback; a default instance is
            created when None.

    Returns:
        PartialExitPolicy built from strategy overrides or risk parameters.
    """
    from src.position_management.partial_manager import PartialExitPolicy
    from src.risk.risk_manager import RiskParameters

    overrides = extract_risk_overrides(strategy)
    partial_config = overrides.get("partial_operations")
    if isinstance(partial_config, dict):
        return PartialExitPolicy(
            exit_targets=partial_config.get("exit_targets", []),
            exit_sizes=partial_config.get("exit_sizes", []),
            scale_in_thresholds=partial_config.get("scale_in_thresholds", []),
            scale_in_sizes=partial_config.get("scale_in_sizes", []),
            max_scale_ins=partial_config.get("max_scale_ins", 0),
        )

    rp = risk_parameters if risk_parameters is not None else RiskParameters()
    return PartialExitPolicy(
        exit_targets=rp.partial_exit_targets or [],
        exit_sizes=rp.partial_exit_sizes or [],
        scale_in_thresholds=rp.scale_in_thresholds or [],
        scale_in_sizes=rp.scale_in_sizes or [],
        max_scale_ins=rp.max_scale_ins,
    )


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
    "build_partial_exit_policy",
    "build_early_cut_policy",
    "extract_risk_overrides",
    "get_risk_parameters",
]
