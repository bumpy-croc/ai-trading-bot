"""
Regime Adaptive Strategy

A meta-strategy that automatically switches between different underlying strategies
based on detected market regimes for optimal performance across market cycles.
"""

import logging
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
from datetime import datetime

from src.strategies.base import BaseStrategy
from src.strategies.ml_basic import MlBasic
from src.strategies.momentum_leverage import MomentumLeverage
from src.strategies.bear import BearStrategy
from src.strategies.ensemble_weighted import EnsembleWeighted
from src.regime.detector import RegimeDetector, RegimeConfig, TrendLabel, VolLabel

logger = logging.getLogger(__name__)


class RegimeAdaptive(BaseStrategy):
    """
    Regime-adaptive strategy that switches between optimal strategies based on market conditions
    """
    
    def __init__(self, name: str = "RegimeAdaptive"):
        super().__init__(name)
        self.trading_pair = "BTCUSDT"
        
        # Initialize component strategies
        self.strategies = {
            'momentum_leverage': MomentumLeverage(name="MomentumLeverage_Component"),
            'ml_basic': MlBasic(name="MlBasic_Component"),
            'ensemble_weighted': EnsembleWeighted(name="Ensemble_Component"),
        }
        
        # Add bear strategy if available
        try:
            self.strategies['bear'] = BearStrategy(name="Bear_Component")
        except Exception as e:
            logger.warning(f"BearStrategy not available: {e}")
        
        # Enhanced regime detector configuration
        regime_config = RegimeConfig(
            slope_window=40,           # Balanced response time
            hysteresis_k=3,           # Moderate stability
            min_dwell=15,             # Require 15 periods of stability
            trend_threshold=0.002,     # 0.2% threshold for trend detection
            r2_min=0.3,               # Require decent trend quality
            atr_high_percentile=0.7    # 70th percentile for high volatility
        )
        self.regime_detector = RegimeDetector(regime_config)
        
        # Strategy mapping based on regime
        self.strategy_mapping = {
            f"{TrendLabel.TREND_UP.value}:{VolLabel.LOW.value}": 'momentum_leverage',     # Bull + Low Vol = Aggressive
            f"{TrendLabel.TREND_UP.value}:{VolLabel.HIGH.value}": 'ensemble_weighted',   # Bull + High Vol = Diversified
            f"{TrendLabel.TREND_DOWN.value}:{VolLabel.LOW.value}": 'bear',               # Bear + Low Vol = Short
            f"{TrendLabel.TREND_DOWN.value}:{VolLabel.HIGH.value}": 'bear',              # Bear + High Vol = Conservative Short
            f"{TrendLabel.RANGE.value}:{VolLabel.LOW.value}": 'ml_basic',                # Range + Low Vol = ML
            f"{TrendLabel.RANGE.value}:{VolLabel.HIGH.value}": 'ml_basic',               # Range + High Vol = Conservative ML
        }
        
        # Position size multipliers by regime (risk adjustment)
        self.position_multipliers = {
            f"{TrendLabel.TREND_UP.value}:{VolLabel.LOW.value}": 1.0,      # Full size in ideal bull conditions
            f"{TrendLabel.TREND_UP.value}:{VolLabel.HIGH.value}": 0.7,     # Reduced in volatile bull
            f"{TrendLabel.TREND_DOWN.value}:{VolLabel.LOW.value}": 0.6,     # Cautious in bear
            f"{TrendLabel.TREND_DOWN.value}:{VolLabel.HIGH.value}": 0.4,    # Very cautious in volatile bear
            f"{TrendLabel.RANGE.value}:{VolLabel.LOW.value}": 0.6,          # Moderate in stable range
            f"{TrendLabel.RANGE.value}:{VolLabel.HIGH.value}": 0.3,         # Conservative in choppy range
        }
        
        # Current state
        self.current_strategy_name = 'ml_basic'  # Default fallback
        self.current_regime = f"{TrendLabel.RANGE.value}:{VolLabel.LOW.value}"
        self.regime_confidence = 0.0
        self.regime_duration = 0
        self.last_switch_index = 0
        
        # Switching parameters
        self.min_confidence = 0.4      # Minimum confidence to switch
        self.min_regime_duration = 12  # Minimum periods in regime before switching
        self.switch_cooldown = 20      # Minimum periods between switches
        
        # Performance tracking
        self.switch_history = []
        self.regime_history = []
        
        logger.info(f"Initialized {self.name} with regime-adaptive switching")
        logger.info(f"Available strategies: {list(self.strategies.keys())}")
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate indicators for all strategies and regime detection"""
        
        # Apply regime detection first
        df_with_regime = self.regime_detector.annotate(df.copy())
        
        # Calculate indicators for all component strategies
        for strategy_name, strategy in self.strategies.items():
            try:
                strategy_df = strategy.calculate_indicators(df.copy())
                # Add strategy-specific columns with prefixes to avoid conflicts
                for col in strategy_df.columns:
                    if col not in ["open", "high", "low", "close", "volume"]:
                        df_with_regime[f"{strategy_name}_{col}"] = strategy_df[col]
            except Exception as e:
                logger.warning(f"Failed to calculate indicators for {strategy_name}: {e}")
        
        return df_with_regime
    
    def _update_regime_state(self, df: pd.DataFrame, index: int) -> Dict[str, Any]:
        """Update current regime state and determine if switch is needed"""
        
        if index < 50:  # Need sufficient data for regime detection
            return {'should_switch': False, 'reason': 'insufficient_data'}
        
        # Get current regime from detector
        trend_label, vol_label, confidence = self.regime_detector.current_labels(df.iloc[:index+1])
        current_regime = f"{trend_label}:{vol_label}"
        
        # Update regime tracking
        if current_regime == self.current_regime:
            self.regime_duration += 1
        else:
            self.regime_duration = 1
            
        self.current_regime = current_regime
        self.regime_confidence = confidence
        
        # Record regime history for analysis
        self.regime_history.append({
            'index': index,
            'regime': current_regime,
            'confidence': confidence,
            'trend_label': trend_label,
            'vol_label': vol_label
        })
        
        # Determine optimal strategy for current regime
        optimal_strategy = self.strategy_mapping.get(current_regime, 'ml_basic')
        
        # Check if we should switch strategies
        switch_decision = {
            'should_switch': False,
            'reason': '',
            'current_regime': current_regime,
            'optimal_strategy': optimal_strategy,
            'confidence': confidence,
            'regime_duration': self.regime_duration
        }
        
        # Switching criteria
        if confidence < self.min_confidence:
            switch_decision['reason'] = f'low_confidence_{confidence:.3f}'
            return switch_decision
        
        if self.regime_duration < self.min_regime_duration:
            switch_decision['reason'] = f'regime_not_stable_{self.regime_duration}'
            return switch_decision
        
        if index - self.last_switch_index < self.switch_cooldown:
            switch_decision['reason'] = f'cooldown_{index - self.last_switch_index}'
            return switch_decision
        
        if optimal_strategy == self.current_strategy_name:
            switch_decision['reason'] = f'already_optimal_{optimal_strategy}'
            return switch_decision
        
        # All checks passed - recommend switch
        switch_decision['should_switch'] = True
        switch_decision['reason'] = 'regime_stable_high_confidence'
        
        return switch_decision
    
    def _execute_strategy_switch(self, switch_decision: Dict[str, Any], index: int) -> bool:
        """Execute strategy switch and record it"""
        
        if not switch_decision['should_switch']:
            return False
        
        old_strategy = self.current_strategy_name
        new_strategy = switch_decision['optimal_strategy']
        
        # Validate new strategy exists
        if new_strategy not in self.strategies:
            logger.warning(f"Strategy {new_strategy} not available, using ml_basic")
            new_strategy = 'ml_basic'
        
        # Execute switch
        self.current_strategy_name = new_strategy
        self.last_switch_index = index
        
        # Record the switch
        switch_record = {
            'index': index,
            'from_strategy': old_strategy,
            'to_strategy': new_strategy,
            'regime': switch_decision['current_regime'],
            'confidence': switch_decision['confidence'],
            'reason': switch_decision['reason']
        }
        
        self.switch_history.append(switch_record)
        
        logger.info(f"Strategy switch at index {index}: {old_strategy} → {new_strategy} "
                   f"(regime: {switch_decision['current_regime']}, confidence: {switch_decision['confidence']:.3f})")
        
        return True
    
    def check_entry_conditions(self, df: pd.DataFrame, index: int) -> bool:
        """Check entry conditions using regime-adaptive strategy selection"""
        
        # Update regime state and check for strategy switch
        switch_decision = self._update_regime_state(df, index)
        if switch_decision['should_switch']:
            self._execute_strategy_switch(switch_decision, index)
        
        # Use current active strategy for entry decision
        current_strategy = self.strategies[self.current_strategy_name]
        
        # Create strategy-specific dataframe
        strategy_df = self._get_strategy_dataframe(df, self.current_strategy_name)
        
        return current_strategy.check_entry_conditions(strategy_df, index)
    
    def check_exit_conditions(self, df: pd.DataFrame, index: int, entry_price: float) -> bool:
        """Check exit conditions using current active strategy"""
        
        current_strategy = self.strategies[self.current_strategy_name]
        strategy_df = self._get_strategy_dataframe(df, self.current_strategy_name)
        
        return current_strategy.check_exit_conditions(strategy_df, index, entry_price)
    
    def calculate_position_size(self, df: pd.DataFrame, index: int, balance: float) -> float:
        """Calculate position size with regime-based risk adjustment"""
        
        current_strategy = self.strategies[self.current_strategy_name]
        strategy_df = self._get_strategy_dataframe(df, self.current_strategy_name)
        
        # Get base position size from active strategy
        base_size = current_strategy.calculate_position_size(strategy_df, index, balance)
        
        # Apply regime-based position multiplier for risk management
        regime_multiplier = self.position_multipliers.get(self.current_regime, 0.5)
        
        # Additional confidence-based adjustment
        confidence_multiplier = max(0.5, min(1.0, self.regime_confidence))
        
        return base_size * regime_multiplier * confidence_multiplier
    
    def calculate_stop_loss(self, df: pd.DataFrame, index: int, price: float, side: str = "long") -> float:
        """Calculate stop loss using current active strategy"""
        
        current_strategy = self.strategies[self.current_strategy_name]
        strategy_df = self._get_strategy_dataframe(df, self.current_strategy_name)
        
        return current_strategy.calculate_stop_loss(strategy_df, index, price, side)
    
    def _get_strategy_dataframe(self, df: pd.DataFrame, strategy_name: str) -> pd.DataFrame:
        """Create strategy-specific dataframe with renamed columns"""
        
        # Start with base OHLCV columns
        strategy_df = df[["open", "high", "low", "close", "volume"]].copy()
        
        # Add strategy-specific indicator columns (remove prefix)
        strategy_prefix = f"{strategy_name}_"
        for col in df.columns:
            if col.startswith(strategy_prefix):
                new_col_name = col.replace(strategy_prefix, "")
                strategy_df[new_col_name] = df[col]
        
        return strategy_df
    
    def get_risk_overrides(self) -> Optional[Dict[str, Any]]:
        """Get risk overrides from current active strategy"""
        
        current_strategy = self.strategies[self.current_strategy_name]
        
        # Get base risk overrides from active strategy
        if hasattr(current_strategy, 'get_risk_overrides'):
            base_overrides = current_strategy.get_risk_overrides() or {}
        else:
            base_overrides = {}
        
        # Add regime-aware adjustments
        regime_overrides = {
            "regime_adaptive": {
                "enabled": True,
                "current_regime": self.current_regime,
                "regime_confidence": self.regime_confidence,
                "active_strategy": self.current_strategy_name,
                "position_multiplier": self.position_multipliers.get(self.current_regime, 0.5)
            }
        }
        
        # Merge overrides
        base_overrides.update(regime_overrides)
        
        return base_overrides
    
    def get_parameters(self) -> dict:
        """Return strategy parameters including regime-adaptive configuration"""
        
        return {
            "name": self.name,
            "available_strategies": list(self.strategies.keys()),
            "current_strategy": self.current_strategy_name,
            "current_regime": self.current_regime,
            "regime_confidence": self.regime_confidence,
            "regime_duration": self.regime_duration,
            "total_switches": len(self.switch_history),
            "min_confidence": self.min_confidence,
            "min_regime_duration": self.min_regime_duration,
            "switch_cooldown": self.switch_cooldown,
            "strategy_mapping": self.strategy_mapping,
            "position_multipliers": self.position_multipliers
        }
    
    def get_regime_analysis(self) -> Dict[str, Any]:
        """Get detailed regime and switching analysis"""
        
        if not self.regime_history:
            return {}
        
        # Regime distribution
        regimes = [r['regime'] for r in self.regime_history]
        regime_counts = pd.Series(regimes).value_counts()
        
        # Switch analysis
        switch_strategies = [s['to_strategy'] for s in self.switch_history]
        strategy_usage = pd.Series(switch_strategies).value_counts() if switch_strategies else pd.Series()
        
        # Confidence statistics
        confidences = [r['confidence'] for r in self.regime_history]
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        return {
            "regime_distribution": regime_counts.to_dict(),
            "strategy_usage": strategy_usage.to_dict(),
            "total_switches": len(self.switch_history),
            "average_confidence": avg_confidence,
            "current_regime": self.current_regime,
            "current_strategy": self.current_strategy_name,
            "switch_history": self.switch_history[-10:],  # Last 10 switches
            "recent_regimes": self.regime_history[-50:]    # Last 50 regime readings
        }