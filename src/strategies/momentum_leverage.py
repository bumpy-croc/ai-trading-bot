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
    base_fraction: float = 0.5,
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
        base_fraction: Base position size fraction (default 50%, max allowed)
    
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
    position_sizer = ConfidenceWeightedSizer(
        base_fraction=base_fraction,  # 50% base allocation (aggressive)
        min_confidence=0.2,           # Lower threshold for aggressive trading
    )
    
    # Create regime detector
    regime_detector = EnhancedRegimeDetector()
    
    return Strategy(
        name=name,
        signal_generator=signal_generator,
        risk_manager=risk_manager,
        position_sizer=position_sizer,
        regime_detector=regime_detector,
    )


# Backward compatibility wrapper - will be removed after engine migration (Task 2 & 3)
class MomentumLeverage:
    """
    Legacy class wrapper for backward compatibility.
    
    This allows existing code to continue using `MomentumLeverage()` while
    internally using the new component-based factory function.
    
    Deprecated: Use create_momentum_leverage_strategy() instead.
    This wrapper will be removed once the backtesting and live engines
    are updated to use factory functions directly.
    """
    
    def __new__(
        cls,
        name: str = "MomentumLeverage",
        momentum_entry_threshold: float = 0.01,
        strong_momentum_threshold: float = 0.025,
        base_risk: float = 0.10,
        base_fraction: float = 0.5,
        **kwargs: Any
    ) -> Strategy:
        """Create strategy instance using factory function."""
        return create_momentum_leverage_strategy(
            name=name,
            momentum_entry_threshold=momentum_entry_threshold,
            strong_momentum_threshold=strong_momentum_threshold,
            base_risk=base_risk,
            base_fraction=base_fraction,
        )