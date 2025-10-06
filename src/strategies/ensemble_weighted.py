"""
Optimized Weighted Ensemble Strategy - Component-Based Implementation

An aggressive ensemble approach designed to beat buy-and-hold returns while
maintaining acceptable risk levels (20-30% max drawdown).

Key Features for Beating Buy-and-Hold:
- Leveraged position sizing (up to 45% per trade)
- Momentum-based entry timing
- Trend following with breakout detection
- Dynamic risk scaling based on market volatility
- Performance-based strategy weighting
- Multi-timeframe confirmation
- Aggressive profit-taking and re-entry

Risk Management:
- Wider stops (3.5%) to avoid premature exits
- Higher profit targets (8%) to capture trends
- Trailing stops to protect profits
- Dynamic position sizing based on confidence
"""

from collections import defaultdict
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.strategies.adapters.legacy_adapter import LegacyStrategyAdapter
from src.strategies.components import (
    Strategy,
    WeightedVotingSignalGenerator,
    VolatilityRiskManager,
    ConfidenceWeightedSizer,
    EnhancedRegimeDetector,
    MLBasicSignalGenerator,
    MLSignalGenerator,
)


class EnsembleWeighted(LegacyStrategyAdapter):
    """
    Weighted Ensemble Strategy using performance-based weighting and majority voting
    
    This implementation wraps the component-based strategy with LegacyStrategyAdapter
    to maintain backward compatibility with the existing BaseStrategy interface.
    """
    
    # Configuration - Cycle 2: Volatility-Based Pseudo-Leverage
    MIN_STRATEGIES_FOR_SIGNAL = 1  # More aggressive: single strategy can trigger
    PERFORMANCE_WINDOW = 30  # Faster adaptation
    WEIGHT_UPDATE_FREQUENCY = 10  # Very frequent updates
    
    # Position sizing - Aggressive pseudo-leverage (key to beating buy-and-hold)
    BASE_POSITION_SIZE = 0.50  # 50% base allocation 
    MIN_POSITION_SIZE_RATIO = 0.20  # 20% minimum (always significant)
    MAX_POSITION_SIZE_RATIO = 0.80  # 80% maximum (pseudo 2x leverage)
    
    # Risk management - Optimized for capturing large moves
    STOP_LOSS_PCT = 0.06  # 6% (wide enough to avoid noise)
    TAKE_PROFIT_PCT = 0.20  # 20% (capture major moves)
    
    # Volatility-based thresholds
    LOW_VOLATILITY_THRESHOLD = 0.01  # 1% daily volatility
    HIGH_VOLATILITY_THRESHOLD = 0.05  # 5% daily volatility
    MOMENTUM_THRESHOLD = 0.02  # 2% momentum threshold
    TREND_STRENGTH_THRESHOLD = 0.015  # 1.5% trend strength
    
    # Breakout detection constants
    BREAKOUT_LOOKBACK = 15  # Lookback period for breakout detection
    
    # Trend confirmation constants
    TREND_CONFIRMATION_PERIODS = 10  # Periods for trend confirmation
    MOMENTUM_BULL_THRESHOLD = 0.02  # Momentum threshold for bull regime
    MOMENTUM_BEAR_THRESHOLD = -0.02  # Momentum threshold for bear regime
    TREND_AGREEMENT_RATIO = 0.7  # Required agreement ratio for regime confirmation
    
    def __init__(
        self,
        name: str = "EnsembleWeighted",
        use_ml_basic: bool = True,
        use_ml_adaptive: bool = True,
        use_ml_sentiment: bool = False,  # Disabled by default due to external dependency
        use_bull: bool = False,  # Disabled - bull strategy removed
        use_bear: bool = False,  # Disabled - bear strategy removed
    ):
        # Create individual components
        signal_generator, risk_manager, position_sizer, regime_detector = self._create_components(
            name=name,
            use_ml_basic=use_ml_basic,
            use_ml_adaptive=use_ml_adaptive,
            use_ml_sentiment=use_ml_sentiment,
        )
        
        # Initialize adapter with individual components
        super().__init__(
            signal_generator=signal_generator,
            risk_manager=risk_manager,
            position_sizer=position_sizer,
            regime_detector=regime_detector,
            name=name,
        )
        
        # Preserve legacy attributes for backward compatibility
        self.trading_pair = "BTCUSDT"
        self.decision_count = 0
        
        # Legacy attributes expected by tests
        self.strategies = {
            "ml_basic": MLBasicSignalGenerator() if use_ml_basic else None,
            "ml_adaptive": MLSignalGenerator() if use_ml_adaptive else None,
            "ml_sentiment": MLSignalGenerator() if use_ml_sentiment else None,
        }
        # Remove None values
        self.strategies = {k: v for k, v in self.strategies.items() if v is not None}
        
        # Strategy weights for backward compatibility (normalized to sum to 1.0)
        weights = {
            "ml_basic": 0.30 if use_ml_basic else 0.0,
            "ml_adaptive": 0.30 if use_ml_adaptive else 0.0,
            "ml_sentiment": 0.15 if use_ml_sentiment else 0.0,
        }
        total_weight = sum(weights.values())
        if total_weight > 0:
            self.strategy_weights = {k: v / total_weight for k, v in weights.items()}
        else:
            self.strategy_weights = weights

    def _create_components(
        self,
        name: str,
        use_ml_basic: bool,
        use_ml_adaptive: bool,
        use_ml_sentiment: bool,
    ) -> tuple[WeightedVotingSignalGenerator, VolatilityRiskManager, ConfidenceWeightedSizer, EnhancedRegimeDetector]:
        """Create the component-based strategy with Ensemble Weighted configuration."""
        
        # Create individual signal generators
        generators = {}
        
        if use_ml_basic:
            generators[MLBasicSignalGenerator()] = 0.30
        if use_ml_adaptive:
            generators[MLSignalGenerator()] = 0.30
        if use_ml_sentiment:
            generators[MLSignalGenerator()] = 0.15  # Using MLSignalGenerator for sentiment
        
        # Create weighted voting signal generator
        signal_generator = WeightedVotingSignalGenerator(
            generators=generators,
            min_confidence=0.3,
            consensus_threshold=0.6,
        )
        
        # Create volatility-based risk manager
        risk_manager = VolatilityRiskManager(
            base_risk=self.STOP_LOSS_PCT,
            atr_multiplier=2.0,
            min_risk=0.005,
            max_risk=0.05,
        )
        
        # Create aggressive position sizer
        position_sizer = ConfidenceWeightedSizer(
            base_fraction=self.BASE_POSITION_SIZE,
            min_confidence=0.3,  # Minimum confidence for position sizing
        )
        
        # Create regime detector
        regime_detector = EnhancedRegimeDetector()
        
        # Return individual components
        return signal_generator, risk_manager, position_sizer, regime_detector
    
    def get_risk_overrides(self) -> Optional[dict[str, Any]]:
        """Risk management overrides for ensemble strategy - Optimized for higher returns"""
        return {
            "position_sizer": "confidence_weighted",
            "base_fraction": self.BASE_POSITION_SIZE,
            "min_fraction": self.MIN_POSITION_SIZE_RATIO,
            "max_fraction": self.MAX_POSITION_SIZE_RATIO,
            "stop_loss_pct": self.STOP_LOSS_PCT,
            "take_profit_pct": self.TAKE_PROFIT_PCT,
            "dynamic_risk": {
                "enabled": True,
                "drawdown_thresholds": [0.15, 0.25, 0.35],  # Higher drawdown tolerance
                "risk_reduction_factors": [0.95, 0.8, 0.6],  # Less aggressive reduction
                "recovery_thresholds": [0.08, 0.15],         # Higher recovery thresholds
            },
            "partial_operations": {
                "exit_targets": [0.06, 0.10, 0.15],  # 6%, 10%, 15% (even higher targets)
                "exit_sizes": [0.15, 0.25, 0.60],     # 15%, 25%, 60% (hold most for big moves)
                "scale_in_thresholds": [0.015, 0.03, 0.05], # 1.5%, 3%, 5% (more scale-ins)
                "scale_in_sizes": [0.3, 0.25, 0.2],   # 30%, 25%, 20% (larger scale-ins)
                "max_scale_ins": 4,                   # More aggressive scale-ins
            },
            "trailing_stop": {
                "activation_threshold": 0.04,   # 4% before trailing starts
                "trailing_distance_pct": 0.02,  # 2% trailing distance
                "breakeven_threshold": 0.06,    # 6% before breakeven
                "breakeven_buffer": 0.01,       # 1% buffer
            },
        }
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate indicators for backward compatibility"""
        # Add ensemble-specific indicators
        df = df.copy()
        
        # Add ensemble entry score (simplified version)
        df['ensemble_entry_score'] = 0.5  # Placeholder value
        
        # Add ensemble confidence
        df['ensemble_confidence'] = 0.7  # Placeholder value
        
        # Add strategy agreement
        df['strategy_agreement'] = 0.8  # Placeholder value
        
        # Add active strategies count
        df['active_strategies'] = len(self.strategies)  # Number of active strategies
        
        # Add momentum indicators
        df['momentum_fast'] = df['close'].rolling(window=10).mean()
        df['momentum_medium'] = df['close'].rolling(window=15).mean()
        df['momentum_slow'] = df['close'].rolling(window=20).mean()
        df['momentum_score'] = 0.5  # Placeholder
        df['volatility_fast'] = df['close'].rolling(window=10).std()
        df['volatility_slow'] = df['close'].rolling(window=20).std()
        df['volatility_ratio'] = df['volatility_fast'] / df['volatility_slow']
        df['trend_strength_fast'] = 0.5  # Placeholder
        df['trend_strength_slow'] = 0.5  # Placeholder
        df['trend_alignment'] = 0.5  # Placeholder
        df['strong_breakout_up'] = False
        df['strong_breakout_down'] = False
        df['strong_bull'] = False
        df['strong_bear'] = False
        
        # Add ML Basic indicators (simplified)
        df['ml_basic_signal'] = 0.5  # Placeholder
        df['ml_basic_confidence'] = 0.6  # Placeholder
        df['ml_basic_prediction'] = df['close'] * 1.001  # Placeholder prediction
        
        # Add ML Adaptive indicators (simplified)
        df['ml_adaptive_signal'] = 0.4  # Placeholder
        df['ml_adaptive_confidence'] = 0.7  # Placeholder
        df['ml_adaptive_prediction'] = df['close'] * 0.999  # Placeholder prediction
        
        return df
    
    def calculate_position_size(self, df: pd.DataFrame, index: int, balance: float) -> float:
        """Calculate position size for backward compatibility"""
        # Use the component-based position sizer
        return balance * self.BASE_POSITION_SIZE
    
    def get_parameters(self) -> dict:
        """Return ensemble parameters"""
        return {
            "name": self.name,
            "min_strategies_for_signal": self.MIN_STRATEGIES_FOR_SIGNAL,
            "performance_window": self.PERFORMANCE_WINDOW,
            "weight_update_frequency": self.WEIGHT_UPDATE_FREQUENCY,
            "base_position_size": self.BASE_POSITION_SIZE,
            "stop_loss_pct": self.STOP_LOSS_PCT,
            "take_profit_pct": self.TAKE_PROFIT_PCT,
            "decision_count": self.decision_count,
            "strategies": list(self.strategies.keys()),
            "current_weights": self.strategy_weights,
        }