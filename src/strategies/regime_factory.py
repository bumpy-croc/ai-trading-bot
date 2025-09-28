"""
Regime-Aware Strategy Factory

A clean implementation that separates regime detection from strategy execution.
This factory selects the appropriate strategy based on detected market regimes.
"""

import logging
from typing import Dict, Any, Optional, Type
from dataclasses import dataclass
from enum import Enum

import pandas as pd

from src.strategies.base import BaseStrategy
from src.regime.detector import RegimeDetector, RegimeConfig, TrendLabel, VolLabel

logger = logging.getLogger(__name__)


class RegimeType(str, Enum):
    """Market regime types"""
    BULL_LOW_VOL = "bull_low_vol"
    BULL_HIGH_VOL = "bull_high_vol"
    BEAR_LOW_VOL = "bear_low_vol"
    BEAR_HIGH_VOL = "bear_high_vol"
    RANGE_LOW_VOL = "range_low_vol"
    RANGE_HIGH_VOL = "range_high_vol"


@dataclass
class RegimeStrategyConfig:
    """Configuration for regime-based strategy selection"""
    
    # Strategy mappings
    bull_low_vol_strategy: str = "momentum_leverage"
    bull_high_vol_strategy: str = "ensemble_weighted"
    bear_low_vol_strategy: str = "bear"
    bear_high_vol_strategy: str = "bear"
    range_low_vol_strategy: str = "ml_basic"
    range_high_vol_strategy: str = "ml_basic"
    
    # Position size multipliers by regime
    bull_low_vol_multiplier: float = 1.0
    bull_high_vol_multiplier: float = 0.7
    bear_low_vol_multiplier: float = 0.6
    bear_high_vol_multiplier: float = 0.4
    range_low_vol_multiplier: float = 0.6
    range_high_vol_multiplier: float = 0.3
    
    # Switching parameters
    min_confidence: float = 0.4
    min_regime_duration: int = 12
    switch_cooldown: int = 20
    fallback_strategy: str = "ml_basic"


class RegimeStrategyFactory:
    """
    Factory that creates regime-aware strategies by combining regime detection
    with appropriate underlying strategies.
    """
    
    def __init__(
        self,
        regime_config: Optional[RegimeConfig] = None,
        strategy_config: Optional[RegimeStrategyConfig] = None
    ):
        self.regime_detector = RegimeDetector(regime_config or RegimeConfig())
        self.strategy_config = strategy_config or RegimeStrategyConfig()
        
        # Strategy registry - maps strategy names to classes
        self.strategy_registry: Dict[str, Type[BaseStrategy]] = {}
        self._register_default_strategies()
        
        # Current state
        self.current_regime: Optional[RegimeType] = None
        self.current_strategy_name: Optional[str] = None
        self.regime_confidence: float = 0.0
        self.regime_duration: int = 0
        self.last_switch_index: int = 0
        
        # History tracking
        self.regime_history: list = []
        self.switch_history: list = []
    
    def _register_default_strategies(self):
        """Register default strategy classes"""
        try:
            from src.strategies.ml_basic import MlBasic
            from src.strategies.momentum_leverage import MomentumLeverage
            from src.strategies.ensemble_weighted import EnsembleWeighted
            from src.strategies.bear import BearStrategy
            
            self.strategy_registry.update({
                "ml_basic": MlBasic,
                "momentum_leverage": MomentumLeverage,
                "ensemble_weighted": EnsembleWeighted,
                "bear": BearStrategy,
            })
        except ImportError as e:
            logger.warning(f"Some strategies not available: {e}")
    
    def register_strategy(self, name: str, strategy_class: Type[BaseStrategy]):
        """Register a new strategy class"""
        self.strategy_registry[name] = strategy_class
        logger.info(f"Registered strategy: {name}")
    
    def _detect_regime_type(self, trend_label: str, vol_label: str) -> RegimeType:
        """Convert regime labels to regime type"""
        if trend_label == TrendLabel.TREND_UP.value:
            return RegimeType.BULL_HIGH_VOL if vol_label == VolLabel.HIGH.value else RegimeType.BULL_LOW_VOL
        elif trend_label == TrendLabel.TREND_DOWN.value:
            return RegimeType.BEAR_HIGH_VOL if vol_label == VolLabel.HIGH.value else RegimeType.BEAR_LOW_VOL
        else:  # Range
            return RegimeType.RANGE_HIGH_VOL if vol_label == VolLabel.HIGH.value else RegimeType.RANGE_LOW_VOL
    
    def _get_strategy_name_for_regime(self, regime_type: RegimeType) -> str:
        """Get strategy name for regime type"""
        mapping = {
            RegimeType.BULL_LOW_VOL: self.strategy_config.bull_low_vol_strategy,
            RegimeType.BULL_HIGH_VOL: self.strategy_config.bull_high_vol_strategy,
            RegimeType.BEAR_LOW_VOL: self.strategy_config.bear_low_vol_strategy,
            RegimeType.BEAR_HIGH_VOL: self.strategy_config.bear_high_vol_strategy,
            RegimeType.RANGE_LOW_VOL: self.strategy_config.range_low_vol_strategy,
            RegimeType.RANGE_HIGH_VOL: self.strategy_config.range_high_vol_strategy,
        }
        return mapping.get(regime_type, self.strategy_config.fallback_strategy)
    
    def _get_position_multiplier(self, regime_type: RegimeType) -> float:
        """Get position size multiplier for regime"""
        mapping = {
            RegimeType.BULL_LOW_VOL: self.strategy_config.bull_low_vol_multiplier,
            RegimeType.BULL_HIGH_VOL: self.strategy_config.bull_high_vol_multiplier,
            RegimeType.BEAR_LOW_VOL: self.strategy_config.bear_low_vol_multiplier,
            RegimeType.BEAR_HIGH_VOL: self.strategy_config.bear_high_vol_multiplier,
            RegimeType.RANGE_LOW_VOL: self.strategy_config.range_low_vol_multiplier,
            RegimeType.RANGE_HIGH_VOL: self.strategy_config.range_high_vol_multiplier,
        }
        return mapping.get(regime_type, 0.5)
    
    def create_regime_aware_strategy(
        self,
        name: str = "RegimeAware",
        initial_strategy: Optional[str] = None
    ) -> "RegimeAwareStrategy":
        """
        Create a regime-aware strategy that automatically selects
        the appropriate underlying strategy based on market regime.
        """
        return RegimeAwareStrategy(
            name=name,
            factory=self,
            initial_strategy=initial_strategy
        )


class RegimeAwareStrategy(BaseStrategy):
    """
    A strategy that automatically adapts to market regimes by delegating
    to the appropriate underlying strategy.
    """
    
    def __init__(
        self,
        name: str,
        factory: RegimeStrategyFactory,
        initial_strategy: Optional[str] = None
    ):
        super().__init__(name)
        self.factory = factory
        self.current_strategy: Optional[BaseStrategy] = None
        self.current_strategy_name = initial_strategy or factory.strategy_config.fallback_strategy
        
        # Initialize with default strategy
        self._load_strategy(self.current_strategy_name)
    
    def _load_strategy(self, strategy_name: str) -> bool:
        """Load a strategy by name"""
        try:
            if strategy_name not in self.factory.strategy_registry:
                logger.warning(f"Strategy {strategy_name} not found, using fallback")
                strategy_name = self.factory.strategy_config.fallback_strategy
            
            strategy_class = self.factory.strategy_registry[strategy_name]
            self.current_strategy = strategy_class(name=f"{strategy_name}_Component")
            self.current_strategy_name = strategy_name
            
            logger.info(f"Loaded strategy: {strategy_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to load strategy {strategy_name}: {e}")
            return False
    
    def _update_regime_and_switch(self, df: pd.DataFrame, index: int) -> bool:
        """Update regime detection and switch strategy if needed"""
        
        if index < 50:  # Need sufficient data
            return False
        
        # Detect current regime
        trend_label, vol_label, confidence = self.factory.regime_detector.current_labels(df.iloc[:index+1])
        regime_type = self.factory._detect_regime_type(trend_label, vol_label)
        
        # Update regime tracking
        if regime_type == self.factory.current_regime:
            self.factory.regime_duration += 1
        else:
            self.factory.regime_duration = 1
            self.factory.current_regime = regime_type
        
        self.factory.regime_confidence = confidence
        
        # Record regime history
        self.factory.regime_history.append({
            'index': index,
            'regime_type': regime_type.value,
            'confidence': confidence,
            'trend_label': trend_label,
            'vol_label': vol_label
        })
        
        # Check if we should switch strategies
        optimal_strategy = self.factory._get_strategy_name_for_regime(regime_type)
        
        # Switching criteria
        if confidence < self.factory.strategy_config.min_confidence:
            return False
        
        if self.factory.regime_duration < self.factory.strategy_config.min_regime_duration:
            return False
        
        if index - self.factory.last_switch_index < self.factory.strategy_config.switch_cooldown:
            return False
        
        if optimal_strategy == self.current_strategy_name:
            return False
        
        # Execute switch
        old_strategy = self.current_strategy_name
        success = self._load_strategy(optimal_strategy)
        
        if success:
            self.factory.last_switch_index = index
            
            # Record the switch
            switch_record = {
                'index': index,
                'from_strategy': old_strategy,
                'to_strategy': optimal_strategy,
                'regime_type': regime_type.value,
                'confidence': confidence
            }
            self.factory.switch_history.append(switch_record)
            
            logger.info(f"Strategy switch: {old_strategy} → {optimal_strategy} "
                       f"(regime: {regime_type.value}, confidence: {confidence:.3f})")
        
        return success
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate indicators using regime detection and current strategy"""
        
        # Apply regime detection
        df_with_regime = self.factory.regime_detector.annotate(df.copy())
        
        # Calculate indicators for current strategy only
        if self.current_strategy:
            try:
                strategy_df = self.current_strategy.calculate_indicators(df.copy())
                # Add strategy-specific columns with prefix
                for col in strategy_df.columns:
                    if col not in ["open", "high", "low", "close", "volume"]:
                        df_with_regime[f"{self.current_strategy_name}_{col}"] = strategy_df[col]
            except Exception as e:
                logger.warning(f"Failed to calculate indicators for {self.current_strategy_name}: {e}")
        
        return df_with_regime
    
    def check_entry_conditions(self, df: pd.DataFrame, index: int) -> bool:
        """Check entry conditions using current strategy"""
        
        # Update regime and potentially switch strategies
        self._update_regime_and_switch(df, index)
        
        if not self.current_strategy:
            return False
        
        # Create strategy-specific dataframe
        strategy_df = self._get_strategy_dataframe(df)
        
        return self.current_strategy.check_entry_conditions(strategy_df, index)
    
    def check_exit_conditions(self, df: pd.DataFrame, index: int, entry_price: float) -> bool:
        """Check exit conditions using current strategy"""
        
        if not self.current_strategy:
            return False
        
        strategy_df = self._get_strategy_dataframe(df)
        return self.current_strategy.check_exit_conditions(strategy_df, index, entry_price)
    
    def calculate_position_size(self, df: pd.DataFrame, index: int, balance: float) -> float:
        """Calculate position size with regime-based adjustment"""
        
        if not self.current_strategy:
            return 0.0
        
        strategy_df = self._get_strategy_dataframe(df)
        
        # Get base position size from current strategy
        base_size = self.current_strategy.calculate_position_size(strategy_df, index, balance)
        
        # Apply regime-based multiplier
        if self.factory.current_regime:
            regime_multiplier = self.factory._get_position_multiplier(self.factory.current_regime)
            confidence_multiplier = max(0.5, min(1.0, self.factory.regime_confidence))
            return base_size * regime_multiplier * confidence_multiplier
        
        return base_size
    
    def calculate_stop_loss(self, df: pd.DataFrame, index: int, price: float, side: str = "long") -> float:
        """Calculate stop loss using current strategy"""
        
        if not self.current_strategy:
            return price * 0.95  # Default 5% stop loss
        
        strategy_df = self._get_strategy_dataframe(df)
        return self.current_strategy.calculate_stop_loss(strategy_df, index, price, side)
    
    def _get_strategy_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create strategy-specific dataframe with renamed columns"""
        
        # Start with base OHLCV columns
        strategy_df = df[["open", "high", "low", "close", "volume"]].copy()
        
        # Add strategy-specific indicator columns (remove prefix)
        if self.current_strategy_name:
            strategy_prefix = f"{self.current_strategy_name}_"
            for col in df.columns:
                if col.startswith(strategy_prefix):
                    new_col_name = col.replace(strategy_prefix, "")
                    strategy_df[new_col_name] = df[col]
        
        return strategy_df
    
    def get_risk_overrides(self) -> Optional[Dict[str, Any]]:
        """Get risk overrides from current strategy with regime adjustments"""
        
        base_overrides = {}
        if self.current_strategy and hasattr(self.current_strategy, 'get_risk_overrides'):
            base_overrides = self.current_strategy.get_risk_overrides() or {}
        
        # Add regime-aware adjustments
        regime_overrides = {
            "regime_aware": {
                "enabled": True,
                "current_regime": self.factory.current_regime.value if self.factory.current_regime else "unknown",
                "regime_confidence": self.factory.regime_confidence,
                "active_strategy": self.current_strategy_name,
                "position_multiplier": self.factory._get_position_multiplier(self.factory.current_regime) if self.factory.current_regime else 0.5
            }
        }
        
        base_overrides.update(regime_overrides)
        return base_overrides
    
    def get_parameters(self) -> dict:
        """Return strategy parameters"""
        
        return {
            "name": self.name,
            "type": "regime_aware",
            "current_strategy": self.current_strategy_name,
            "current_regime": self.factory.current_regime.value if self.factory.current_regime else "unknown",
            "regime_confidence": self.factory.regime_confidence,
            "regime_duration": self.factory.regime_duration,
            "total_switches": len(self.factory.switch_history),
            "available_strategies": list(self.factory.strategy_registry.keys()),
            "strategy_config": {
                "min_confidence": self.factory.strategy_config.min_confidence,
                "min_regime_duration": self.factory.strategy_config.min_regime_duration,
                "switch_cooldown": self.factory.strategy_config.switch_cooldown
            }
        }
    
    def get_regime_analysis(self) -> Dict[str, Any]:
        """Get detailed regime and switching analysis"""
        
        if not self.factory.regime_history:
            return {}
        
        # Regime distribution
        regimes = [r['regime_type'] for r in self.factory.regime_history]
        regime_counts = pd.Series(regimes).value_counts()
        
        # Switch analysis
        switch_strategies = [s['to_strategy'] for s in self.factory.switch_history]
        strategy_usage = pd.Series(switch_strategies).value_counts() if switch_strategies else pd.Series()
        
        # Confidence statistics
        confidences = [r['confidence'] for r in self.factory.regime_history]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        return {
            "regime_distribution": regime_counts.to_dict(),
            "strategy_usage": strategy_usage.to_dict(),
            "total_switches": len(self.factory.switch_history),
            "average_confidence": avg_confidence,
            "current_regime": self.factory.current_regime.value if self.factory.current_regime else "unknown",
            "current_strategy": self.current_strategy_name,
            "switch_history": self.factory.switch_history[-10:],  # Last 10 switches
            "recent_regimes": self.factory.regime_history[-50:]    # Last 50 regime readings
        }