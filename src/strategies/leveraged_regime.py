"""
Leveraged Regime Strategy - Component-Based Implementation

Combines regime detection with dynamic leverage to amplify returns during
confirmed trends and reduce exposure during uncertain markets.

Key features:
1. Regime-aware leverage: scales position size by regime-derived multiplier
2. Bear regime protection: holds no positions (cash) in bearish markets
3. Bull regime amplification: increases exposure with leverage during uptrends
4. Smooth transitions: exponential decay prevents abrupt leverage jumps
5. Conviction scaling: leverage increases as regime persists longer
"""

from __future__ import annotations

from src.config.constants import (
    DEFAULT_LEVERAGE_DECAY_RATE,
    DEFAULT_MAX_LEVERAGE,
    DEFAULT_MIN_REGIME_BARS,
    DEFAULT_STRATEGY_BASE_FRACTION,
    DEFAULT_STRATEGY_MIN_CONFIDENCE,
)
from src.strategies.components import (
    ConfidenceWeightedSizer,
    EnhancedRegimeDetector,
    MLBasicSignalGenerator,
    MomentumSignalGenerator,
    Strategy,
    VolatilityRiskManager,
)
from src.strategies.components.leverage_manager import LeverageManager


def create_leveraged_regime_strategy(
    name: str = "LeveragedRegime",
    signal_source: str = "momentum",
    base_risk: float = 0.05,
    base_fraction: float = DEFAULT_STRATEGY_BASE_FRACTION,
    min_confidence: float = DEFAULT_STRATEGY_MIN_CONFIDENCE,
    max_leverage: float = DEFAULT_MAX_LEVERAGE,
    leverage_decay_rate: float = DEFAULT_LEVERAGE_DECAY_RATE,
    min_regime_bars: int = DEFAULT_MIN_REGIME_BARS,
    take_profit_pct: float = 0.10,
    stop_loss_pct: float = 0.05,
) -> Strategy:
    """Create a leveraged regime strategy using component composition.

    Combines a signal generator (momentum or ML) with a LeverageManager
    that scales position sizes based on the current market regime.

    Args:
        name: Strategy name.
        signal_source: Signal generator type, either "momentum" or "ml".
        base_risk: Base risk percentage for stop loss.
        base_fraction: Base position size as fraction of balance.
        min_confidence: Minimum signal confidence to take a position.
        max_leverage: Maximum leverage multiplier (safety cap).
        leverage_decay_rate: Exponential decay rate for leverage transitions.
        min_regime_bars: Minimum bars before conviction scaling begins.
        take_profit_pct: Target profit percentage.
        stop_loss_pct: Stop loss percentage.

    Returns:
        Configured Strategy instance with leverage manager attached.
    """
    # Create signal generator based on source preference
    if signal_source == "ml":
        signal_generator = MLBasicSignalGenerator(name=f"{name}_signals")
    else:
        signal_generator = MomentumSignalGenerator(name=f"{name}_signals")

    # Create risk manager with moderate stops
    risk_manager = VolatilityRiskManager(
        base_risk=base_risk,
        atr_multiplier=2.0,
        min_risk=0.005,
        max_risk=0.15,
    )

    # Create position sizer
    position_sizer = ConfidenceWeightedSizer(
        base_fraction=base_fraction,
        min_confidence=min_confidence,
    )

    # Create regime detector
    regime_detector = EnhancedRegimeDetector()

    # Create leverage manager
    leverage_manager = LeverageManager(
        max_leverage=max_leverage,
        decay_rate=leverage_decay_rate,
        min_regime_bars=min_regime_bars,
    )

    # Compose strategy
    strategy = Strategy(
        name=name,
        signal_generator=signal_generator,
        risk_manager=risk_manager,
        position_sizer=position_sizer,
        regime_detector=regime_detector,
    )

    # Attach leverage manager for position size amplification
    strategy.leverage_manager = leverage_manager

    # Expose configuration for validation
    strategy.base_position_size = base_fraction
    strategy.take_profit_pct = take_profit_pct

    # Engine-level risk overrides
    strategy.set_risk_overrides(
        {
            "position_sizer": "confidence_weighted",
            "base_fraction": base_fraction,
            "min_fraction": 0.01,
            "max_fraction": base_fraction * max_leverage,
            "stop_loss_pct": stop_loss_pct,
            "take_profit_pct": take_profit_pct,
            "leverage": {
                "enabled": True,
                "max_leverage": max_leverage,
                "decay_rate": leverage_decay_rate,
                "min_regime_bars": min_regime_bars,
            },
            "dynamic_risk": {
                "enabled": True,
                "drawdown_thresholds": [0.10, 0.20, 0.30],
                "risk_reduction_factors": [0.8, 0.6, 0.4],
                "recovery_thresholds": [0.05, 0.10],
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
