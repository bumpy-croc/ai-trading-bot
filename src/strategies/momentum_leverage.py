"""
Momentum Leverage Strategy - Component-Based Implementation

A simplified but aggressive strategy designed specifically to beat buy-and-hold
by implementing the core techniques that actually work:

1. Pseudo-leverage through concentrated position sizing (up to 50% per trade)
2. Pure momentum following with trend confirmation
3. Volatility-based position scaling
4. Extended profit targets to capture full moves
5. Quick re-entry after exits

Research-backed approach:
- Focus on capturing upward moves with aggressive position sizes
- Use momentum to time entries for maximum effect
- Hold positions longer to capture full trend moves
- Re-enter quickly to minimize time out of market

Key insight: Beat buy-and-hold by being MORE aggressive, not more conservative.
"""

from typing import Any

from src.strategies.components import (
    Strategy,
    MomentumSignalGenerator,
    VolatilityRiskManager,
    ConfidenceWeightedSizer,
    EnhancedRegimeDetector,
)


def create_momentum_leverage_strategy(
    name: str = "MomentumLeverage",
    momentum_entry_threshold: float = 0.01,
    strong_momentum_threshold: float = 0.025,
    base_risk: float = 0.10,
    base_fraction: float = 0.70,
    min_position_size_ratio: float = 0.40,
    max_position_size_ratio: float = 0.95,
    take_profit_pct: float = 0.35,
) -> Strategy:
    """
    Create Momentum Leverage strategy using component composition.
    
    This strategy uses aggressive momentum-based signals with volatility-adjusted
    risk management and confidence-weighted position sizing for maximum returns.
    
    Args:
        name: Strategy name
        momentum_entry_threshold: Minimum momentum percentage to enter (default 1%)
        strong_momentum_threshold: Strong momentum threshold (default 2.5%)
        base_risk: Base risk percentage for stop loss (default 10%)
        base_fraction: Target base position size fraction before sizer cap (default 70%)
        min_position_size_ratio: Minimum position fraction applied during drawdowns
        max_position_size_ratio: Maximum position fraction allowed during strong trends
        take_profit_pct: Target profit percentage for scaling out
    
    Returns:
        Configured Strategy instance
    """
    # Create momentum signal generator with aggressive thresholds
    signal_generator = MomentumSignalGenerator(
        name=f"{name}_signals",
        momentum_entry_threshold=momentum_entry_threshold,
        strong_momentum_threshold=strong_momentum_threshold,
    )
    
    # Create volatility-based risk manager with wide stop loss
    # This allows positions to breathe and capture full trend moves
    risk_manager = VolatilityRiskManager(
        base_risk=base_risk,  # 10% stop loss (very wide)
        atr_multiplier=2.0,   # Volatility adjustment
        min_risk=0.005,       # 0.5% minimum
        max_risk=0.15,        # 15% maximum
    )
    
    # Create aggressive position sizer
    # ConfidenceWeightedSizer caps at 50% base_fraction for safety
    effective_base_fraction = min(base_fraction, 0.5)
    position_sizer = ConfidenceWeightedSizer(
        base_fraction=effective_base_fraction,  # 50% base allocation cap for safety
        min_confidence=0.2,           # Lower threshold for aggressive trading
    )
    
    # Create regime detector
    regime_detector = EnhancedRegimeDetector()
    
    strategy = Strategy(
        name=name,
        signal_generator=signal_generator,
        risk_manager=risk_manager,
        position_sizer=position_sizer,
        regime_detector=regime_detector,
    )

    # Surface key parameters for compatibility with legacy engine hooks
    strategy.base_position_size = base_fraction
    strategy.base_fraction = effective_base_fraction
    strategy.min_position_size_ratio = min_position_size_ratio
    strategy.max_position_size_ratio = max_position_size_ratio
    strategy.stop_loss_pct = base_risk
    strategy.take_profit_pct = take_profit_pct
    strategy.min_confidence = position_sizer.min_confidence
    strategy.trading_pair = "BTCUSDT"

    # Restore aggressive runtime overrides consumed by backtesting/live engines
    strategy._risk_overrides = {
        "position_sizer": "confidence_weighted",
        "base_fraction": base_fraction,
        "min_fraction": min_position_size_ratio,
        "max_fraction": max_position_size_ratio,
        "stop_loss_pct": base_risk,
        "take_profit_pct": take_profit_pct,
        "dynamic_risk": {
            "enabled": True,
            "drawdown_thresholds": [0.25, 0.35, 0.45],
            "risk_reduction_factors": [0.95, 0.85, 0.75],
            "recovery_thresholds": [0.12, 0.25],
        },
        "partial_operations": {
            "exit_targets": [0.08, 0.15, 0.25],
            "exit_sizes": [0.20, 0.30, 0.50],
            "scale_in_thresholds": [0.02, 0.04],
            "scale_in_sizes": [0.40, 0.30],
            "max_scale_ins": 3,
        },
        "trailing_stop": {
            "activation_threshold": 0.06,
            "trailing_distance_pct": 0.03,
            "breakeven_threshold": 0.10,
            "breakeven_buffer": 0.02,
        },
    }

    return strategy
