"""
Kelly Momentum Strategy - Component-Based Implementation

An aggressive growth strategy that combines momentum-based signal generation
with Kelly Criterion position sizing to maximize geometric growth rate.

Core approach:
1. Momentum signals identify trending moves worth entering
2. Kelly Criterion sizes positions optimally based on live trade statistics
3. Volatility-adjusted risk management provides adaptive stop losses
4. Regime detection scales down exposure in unfavorable conditions

Half-Kelly is used by default to balance growth against drawdown risk.
"""

from __future__ import annotations

from src.config.constants import (
    DEFAULT_KELLY_FRACTION,
    DEFAULT_KELLY_LOOKBACK_TRADES,
    DEFAULT_KELLY_MIN_TRADES,
    DEFAULT_STRATEGY_MIN_CONFIDENCE_AGGRESSIVE,
)
from src.strategies.components import (
    EnhancedRegimeDetector,
    KellyCriterionSizer,
    MomentumSignalGenerator,
    Strategy,
    VolatilityRiskManager,
)


def create_kelly_momentum_strategy(
    name: str = "KellyMomentum",
    momentum_entry_threshold: float = 0.01,
    strong_momentum_threshold: float = 0.025,
    base_risk: float = 0.08,
    kelly_fraction: float = DEFAULT_KELLY_FRACTION,
    min_trades: int = DEFAULT_KELLY_MIN_TRADES,
    lookback_trades: int = DEFAULT_KELLY_LOOKBACK_TRADES,
    fallback_fraction: float = 0.03,
    take_profit_pct: float = 0.20,
) -> Strategy:
    """
    Create Kelly Momentum strategy using component composition.

    Pairs momentum-based signals with Kelly Criterion position sizing
    for aggressive but mathematically grounded position management.

    Args:
        name: Strategy name
        momentum_entry_threshold: Minimum momentum percentage to enter (default 1%)
        strong_momentum_threshold: Strong momentum threshold (default 2.5%)
        base_risk: Base risk percentage for stop loss (default 8%)
        kelly_fraction: Fraction of full Kelly to use (default 0.5 = half-Kelly)
        min_trades: Minimum trades before Kelly activates
        lookback_trades: Rolling window size for trade statistics
        fallback_fraction: Fixed fraction used during cold start
        take_profit_pct: Target profit percentage for scaling out

    Returns:
        Configured Strategy instance
    """
    signal_generator = MomentumSignalGenerator(
        name=f"{name}_signals",
        momentum_entry_threshold=momentum_entry_threshold,
        strong_momentum_threshold=strong_momentum_threshold,
    )

    risk_manager = VolatilityRiskManager(
        base_risk=base_risk,
        atr_multiplier=2.0,
        min_risk=0.005,
        max_risk=0.12,
    )

    position_sizer = KellyCriterionSizer(
        kelly_fraction=kelly_fraction,
        min_trades=min_trades,
        lookback_trades=lookback_trades,
        fallback_fraction=fallback_fraction,
    )

    regime_detector = EnhancedRegimeDetector()

    strategy = Strategy(
        name=name,
        signal_generator=signal_generator,
        risk_manager=risk_manager,
        position_sizer=position_sizer,
        regime_detector=regime_detector,
    )

    strategy.base_position_size = fallback_fraction
    strategy.take_profit_pct = take_profit_pct

    strategy.set_risk_overrides(
        {
            "position_sizer": "kelly_criterion",
            "kelly_fraction": kelly_fraction,
            "min_fraction": 0.01,
            "max_fraction": 0.20,
            "stop_loss_pct": base_risk,
            "take_profit_pct": take_profit_pct,
            "dynamic_risk": {
                "enabled": True,
                "drawdown_thresholds": [0.10, 0.20, 0.30],
                "risk_reduction_factors": [0.8, 0.6, 0.4],
                "recovery_thresholds": [0.05, 0.10],
            },
            "partial_operations": {
                "exit_targets": [0.05, 0.10, 0.20],
                "exit_sizes": [0.25, 0.25, 0.50],
                "scale_in_thresholds": [0.02, 0.04],
                "scale_in_sizes": [0.30, 0.20],
                "max_scale_ins": 2,
            },
            "trailing_stop": {
                "activation_threshold": 0.04,
                "trailing_distance_pct": 0.02,
                "breakeven_threshold": 0.06,
                "breakeven_buffer": 0.01,
            },
        }
    )

    return strategy
