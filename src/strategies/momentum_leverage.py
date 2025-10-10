"""
Momentum Leverage Strategy - Component-Based Implementation

A simplified but aggressive strategy designed specifically to beat buy-and-hold
by implementing the core techniques that actually work:

1. Pseudo-leverage through concentrated position sizing (up to 80% per trade)
2. Pure momentum following with trend confirmation
3. Volatility-based position scaling
4. Extended profit targets to capture full moves
5. Quick re-entry after exits

Research-backed approach:
- Focus on capturing 80% of upward moves with 80% position sizes
- Use momentum to time entries for maximum effect
- Hold positions longer to capture full trend moves
- Re-enter quickly to minimize time out of market

Key insight: Beat buy-and-hold by being MORE aggressive, not more conservative.
"""

from typing import Any, Optional

from src.strategies.adapters.legacy_adapter import LegacyStrategyAdapter
from src.strategies.components import (
    ConfidenceWeightedSizer,
    EnhancedRegimeDetector,
    MomentumSignalGenerator,
    VolatilityRiskManager,
)


class MomentumLeverage(LegacyStrategyAdapter):
    """
    Pure momentum strategy with pseudo-leverage to beat buy-and-hold
    
    This implementation wraps the component-based strategy with LegacyStrategyAdapter
    to maintain backward compatibility with the existing BaseStrategy interface.
    """
    
    # Ultra-aggressive configuration for beating buy-and-hold (Cycle 3)
    BASE_POSITION_SIZE = 0.70  # 70% base allocation (pseudo 1.75x leverage)
    MIN_POSITION_SIZE_RATIO = 0.40  # 40% minimum (always highly leveraged)
    MAX_POSITION_SIZE_RATIO = 0.95  # 95% maximum (pseudo 2.5x leverage)
    
    # Optimized risk management for maximum trend capture
    STOP_LOSS_PCT = 0.10  # 10% stop loss (very wide to capture full moves)
    TAKE_PROFIT_PCT = 0.35  # 35% take profit (capture massive moves)
    
    # Aggressive momentum thresholds
    MOMENTUM_ENTRY_THRESHOLD = 0.01   # 1% momentum to enter (lower threshold)
    STRONG_MOMENTUM_THRESHOLD = 0.025 # 2.5% for maximum position
    TREND_LOOKBACK = 15  # 15-period for faster response
    
    def __init__(self, name: str = "MomentumLeverage"):
        # Create components
        signal_generator, risk_manager, position_sizer, regime_detector = self._create_components(name)
        
        # Initialize adapter with components
        super().__init__(
            signal_generator=signal_generator,
            risk_manager=risk_manager,
            position_sizer=position_sizer,
            regime_detector=regime_detector,
            name=name
        )
        
        # Preserve legacy attributes for backward compatibility
        self.trading_pair = "BTCUSDT"

    def _create_components(self, name: str):
        """Create the component instances with Momentum Leverage configuration."""
        
        # Create momentum signal generator
        signal_generator = MomentumSignalGenerator(
            name=f"{name}_signals",
            momentum_entry_threshold=self.MOMENTUM_ENTRY_THRESHOLD,
            strong_momentum_threshold=self.STRONG_MOMENTUM_THRESHOLD,
        )
        
        # Create volatility-based risk manager
        # Note: VolatilityRiskManager uses base_risk, not stop_loss_pct
        # Converting 10% stop loss to risk percentage (10% = 0.10)
        risk_manager = VolatilityRiskManager(
            base_risk=self.STOP_LOSS_PCT,
            atr_multiplier=2.0,
            min_risk=0.005,
            max_risk=0.15,
        )
        
        # Create aggressive position sizer
        # Note: ConfidenceWeightedSizer max base_fraction is 0.5, so we cap it
        position_sizer = ConfidenceWeightedSizer(
            base_fraction=min(0.5, self.BASE_POSITION_SIZE),  # Cap at 50%
            min_confidence=0.2,  # Lower threshold for aggressive trading
        )
        
        # Create regime detector
        regime_detector = EnhancedRegimeDetector()
        
        return signal_generator, risk_manager, position_sizer, regime_detector
    
    def get_risk_overrides(self) -> Optional[dict[str, Any]]:
        """Aggressive risk management for beating buy-and-hold"""
        return {
            "position_sizer": "confidence_weighted",
            "base_fraction": self.BASE_POSITION_SIZE,
            "min_fraction": self.MIN_POSITION_SIZE_RATIO,
            "max_fraction": self.MAX_POSITION_SIZE_RATIO,
            "stop_loss_pct": self.STOP_LOSS_PCT,
            "take_profit_pct": self.TAKE_PROFIT_PCT,
            "dynamic_risk": {
                "enabled": True,
                "drawdown_thresholds": [0.25, 0.35, 0.45],  # Accept much higher drawdowns
                "risk_reduction_factors": [0.95, 0.85, 0.75], # Minimal reduction
                "recovery_thresholds": [0.12, 0.25],         # Higher recovery thresholds
            },
            "partial_operations": {
                "exit_targets": [0.08, 0.15, 0.25],  # 8%, 15%, 25%
                "exit_sizes": [0.20, 0.30, 0.50],     # 20%, 30%, 50%
                "scale_in_thresholds": [0.02, 0.04], # 2%, 4%
                "scale_in_sizes": [0.4, 0.3],         # 40%, 30%
                "max_scale_ins": 3,
            },
            "trailing_stop": {
                "activation_threshold": 0.06,   # 6% before trailing
                "trailing_distance_pct": 0.03,  # 3% trailing distance
                "breakeven_threshold": 0.10,    # 10% before breakeven
                "breakeven_buffer": 0.02,       # 2% buffer
            },
        }
    
    def get_parameters(self) -> dict:
        """Return strategy parameters"""
        return {
            "name": self.name,
            "base_position_size": self.BASE_POSITION_SIZE,
            "max_position_size": self.MAX_POSITION_SIZE_RATIO,
            "stop_loss_pct": self.STOP_LOSS_PCT,
            "take_profit_pct": self.TAKE_PROFIT_PCT,
            "momentum_entry_threshold": self.MOMENTUM_ENTRY_THRESHOLD,
            "strong_momentum_threshold": self.STRONG_MOMENTUM_THRESHOLD,
            "trend_lookback": self.TREND_LOOKBACK,
        }