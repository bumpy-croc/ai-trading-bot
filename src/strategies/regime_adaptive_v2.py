"""
Regime Adaptive Strategy V2 - Clean Implementation

A cleaner implementation of regime-adaptive strategy selection that separates
concerns and eliminates code duplication.
"""

import logging
from typing import Any, Dict, Optional
import pandas as pd

from src.strategies.base import BaseStrategy
from src.strategies.regime_factory import RegimeStrategyFactory, RegimeStrategyConfig

logger = logging.getLogger(__name__)


class RegimeAdaptiveV2(BaseStrategy):
    """
    Clean implementation of regime-adaptive strategy that uses the factory pattern
    to separate regime detection from strategy execution.
    """
    
    def __init__(self, name: str = "RegimeAdaptiveV2"):
        super().__init__(name)
        
        # Create factory with default configuration
        self.factory = RegimeStrategyFactory()
        
        # Create the regime-aware strategy
        self.regime_aware_strategy = self.factory.create_regime_aware_strategy(
            name=f"{name}_Core"
        )
        
        logger.info(f"Initialized {self.name} with clean regime-adaptive architecture")
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate indicators using regime-aware strategy"""
        return self.regime_aware_strategy.calculate_indicators(df)
    
    def check_entry_conditions(self, df: pd.DataFrame, index: int) -> bool:
        """Check entry conditions using regime-aware strategy"""
        return self.regime_aware_strategy.check_entry_conditions(df, index)
    
    def check_exit_conditions(self, df: pd.DataFrame, index: int, entry_price: float) -> bool:
        """Check exit conditions using regime-aware strategy"""
        return self.regime_aware_strategy.check_exit_conditions(df, index, entry_price)
    
    def calculate_position_size(self, df: pd.DataFrame, index: int, balance: float) -> float:
        """Calculate position size using regime-aware strategy"""
        return self.regime_aware_strategy.calculate_position_size(df, index, balance)
    
    def calculate_stop_loss(self, df: pd.DataFrame, index: int, price: float, side: str = "long") -> float:
        """Calculate stop loss using regime-aware strategy"""
        return self.regime_aware_strategy.calculate_stop_loss(df, index, price, side)
    
    def get_risk_overrides(self) -> Optional[Dict[str, Any]]:
        """Get risk overrides from regime-aware strategy"""
        return self.regime_aware_strategy.get_risk_overrides()
    
    def get_parameters(self) -> dict:
        """Return strategy parameters"""
        return self.regime_aware_strategy.get_parameters()
    
    def get_regime_analysis(self) -> Dict[str, Any]:
        """Get detailed regime and switching analysis"""
        return self.regime_aware_strategy.get_regime_analysis()
    
    def configure_strategy_mapping(
        self,
        bull_low_vol: str = "momentum_leverage",
        bull_high_vol: str = "ensemble_weighted",
        bear_low_vol: str = "bear",
        bear_high_vol: str = "bear",
        range_low_vol: str = "ml_basic",
        range_high_vol: str = "ml_basic"
    ):
        """Configure strategy mapping for different regimes"""
        
        config = RegimeStrategyConfig(
            bull_low_vol_strategy=bull_low_vol,
            bull_high_vol_strategy=bull_high_vol,
            bear_low_vol_strategy=bear_low_vol,
            bear_high_vol_strategy=bear_high_vol,
            range_low_vol_strategy=range_low_vol,
            range_high_vol_strategy=range_high_vol
        )
        
        # Update factory configuration
        self.factory.strategy_config = config
        logger.info(f"Updated strategy mapping configuration")
    
    def configure_position_sizing(
        self,
        bull_low_vol_multiplier: float = 1.0,
        bull_high_vol_multiplier: float = 0.7,
        bear_low_vol_multiplier: float = 0.6,
        bear_high_vol_multiplier: float = 0.4,
        range_low_vol_multiplier: float = 0.6,
        range_high_vol_multiplier: float = 0.3
    ):
        """Configure position size multipliers for different regimes"""
        
        self.factory.strategy_config.bull_low_vol_multiplier = bull_low_vol_multiplier
        self.factory.strategy_config.bull_high_vol_multiplier = bull_high_vol_multiplier
        self.factory.strategy_config.bear_low_vol_multiplier = bear_low_vol_multiplier
        self.factory.strategy_config.bear_high_vol_multiplier = bear_high_vol_multiplier
        self.factory.strategy_config.range_low_vol_multiplier = range_low_vol_multiplier
        self.factory.strategy_config.range_high_vol_multiplier = range_high_vol_multiplier
        
        logger.info(f"Updated position sizing configuration")
    
    def configure_switching_parameters(
        self,
        min_confidence: float = 0.4,
        min_regime_duration: int = 12,
        switch_cooldown: int = 20
    ):
        """Configure regime switching parameters"""
        
        self.factory.strategy_config.min_confidence = min_confidence
        self.factory.strategy_config.min_regime_duration = min_regime_duration
        self.factory.strategy_config.switch_cooldown = switch_cooldown
        
        logger.info(f"Updated switching parameters: confidence={min_confidence}, "
                   f"duration={min_regime_duration}, cooldown={switch_cooldown}")
    
    def register_custom_strategy(self, name: str, strategy_class):
        """Register a custom strategy for use in regime mapping"""
        self.factory.register_strategy(name, strategy_class)
        logger.info(f"Registered custom strategy: {name}")
    
    def get_current_strategy_name(self) -> str:
        """Get the name of the currently active strategy"""
        return self.regime_aware_strategy.current_strategy_name
    
    def get_current_regime(self) -> str:
        """Get the current market regime"""
        return self.factory.current_regime.value if self.factory.current_regime else "unknown"
    
    def get_regime_confidence(self) -> float:
        """Get the confidence level of the current regime detection"""
        return self.factory.regime_confidence
    
    def force_strategy_switch(self, new_strategy_name: str) -> bool:
        """Force a switch to a specific strategy (for testing/debugging)"""
        return self.regime_aware_strategy._load_strategy(new_strategy_name)