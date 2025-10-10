"""
Regime-Aware Strategy Switcher

This module extends the StrategyManager to automatically switch trading strategies
based on detected market regimes for optimal performance.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable, Optional

import pandas as pd

from src.live.strategy_manager import StrategyManager
from src.regime.detector import RegimeConfig, RegimeDetector, TrendLabel, VolLabel

logger = logging.getLogger(__name__)


@dataclass
class RegimeStrategyMapping:
    """Maps market regimes to optimal strategies"""
    
    # Bull market strategies
    bull_low_vol: str = "momentum_leverage"    # Use aggressive momentum in calm bull markets
    bull_high_vol: str = "ensemble_weighted"   # Use ensemble in volatile bull markets
    
    # Bear market strategies  
    bear_low_vol: str = "ml_basic"             # Use ML in calm bear markets
    bear_high_vol: str = "ml_adaptive"         # Use adaptive ML in volatile bear markets
    
    # Range/sideways market strategies
    range_low_vol: str = "ml_basic"            # Use ML in calm range markets
    range_high_vol: str = "ml_basic"           # Use ML with reduced size in volatile ranges
    
    # Position size adjustments by regime
    bull_low_vol_multiplier: float = 1.0      # Full size in ideal conditions
    bull_high_vol_multiplier: float = 0.7     # Reduced size in volatile bull
    bear_low_vol_multiplier: float = 0.8      # Reduced size in bear market
    bear_high_vol_multiplier: float = 0.5     # Much reduced in volatile bear
    range_low_vol_multiplier: float = 0.6     # Reduced in range market
    range_high_vol_multiplier: float = 0.3    # Very reduced in volatile range


@dataclass
class SwitchingConfig:
    """Configuration for regime-based strategy switching"""
    
    # Switching criteria
    min_regime_confidence: float = 0.4         # Minimum confidence to switch
    min_regime_duration: int = 15              # Minimum bars in regime before switching
    switch_cooldown_minutes: int = 60          # Cooldown between switches
    
    # Enhanced regime detection
    enable_multi_timeframe: bool = True        # Use multiple timeframes for confirmation
    timeframes: list = None                    # Timeframes to analyze ['1h', '4h', '1d']
    require_timeframe_agreement: float = 0.6   # Require 60% agreement across timeframes
    
    # Position management during switches
    close_positions_on_switch: bool = False    # Whether to close positions when switching
    reduce_size_during_transition: bool = True # Reduce position sizes during transitions
    transition_size_multiplier: float = 0.5    # Size multiplier during transitions
    
    # Risk management
    max_drawdown_switch_threshold: float = 0.15  # Switch to defensive if drawdown > 15%
    emergency_strategy: str = "ml_basic"          # Strategy to use in emergencies


class RegimeStrategySwitcher:
    """
    Intelligent strategy switcher that automatically adapts to market regimes
    """
    
    def __init__(
        self,
        strategy_manager: StrategyManager,
        regime_config: Optional[RegimeConfig] = None,
        strategy_mapping: Optional[RegimeStrategyMapping] = None,
        switching_config: Optional[SwitchingConfig] = None
    ):
        self.strategy_manager = strategy_manager
        self.regime_detector = RegimeDetector(regime_config or RegimeConfig())
        self.strategy_mapping = strategy_mapping or RegimeStrategyMapping()
        self.switching_config = switching_config or SwitchingConfig()
        
        # Initialize timeframes for multi-timeframe analysis
        if self.switching_config.timeframes is None:
            self.switching_config.timeframes = ['1h', '4h', '1d']
        
        # State tracking
        self.current_regime: Optional[str] = None
        self.regime_confidence: float = 0.0
        self.regime_start_time: Optional[datetime] = None
        self.regime_start_candle_index: Optional[int] = None
        self.last_switch_time: Optional[datetime] = None
        self.regime_duration: int = 0
        
        # Multi-timeframe regime detectors
        self.timeframe_detectors: dict[str, RegimeDetector] = {}
        if self.switching_config.enable_multi_timeframe:
            for tf in self.switching_config.timeframes:
                # Adjust parameters based on timeframe
                config = self._get_timeframe_config(tf)
                self.timeframe_detectors[tf] = RegimeDetector(config)
        
        # Performance tracking
        self.strategy_performance: dict[str, dict[str, float]] = {}
        self.regime_history: list = []
        
        # Callbacks
        self.on_regime_change: Optional[Callable] = None
        self.on_strategy_switch: Optional[Callable] = None
        
        logger.info("RegimeStrategySwitcher initialized")
    
    def _get_timeframe_config(self, timeframe: str) -> RegimeConfig:
        """Get regime detection config adjusted for timeframe"""
        
        base_config = RegimeConfig()
        
        # Adjust parameters based on timeframe
        if timeframe == '1h':
            return RegimeConfig(
                slope_window=30,
                hysteresis_k=3,
                min_dwell=10,
                trend_threshold=0.001
            )
        elif timeframe == '4h':
            return RegimeConfig(
                slope_window=20,
                hysteresis_k=2,
                min_dwell=5,
                trend_threshold=0.002
            )
        elif timeframe == '1d':
            return RegimeConfig(
                slope_window=15,
                hysteresis_k=2,
                min_dwell=3,
                trend_threshold=0.003
            )
        else:
            return base_config
    
    def analyze_market_regime(self, price_data: dict[str, pd.DataFrame]) -> dict[str, Any]:
        """
        Analyze market regime across multiple timeframes
        
        Args:
            price_data: Dictionary with timeframe keys and price dataframes
            
        Returns:
            Dictionary with regime analysis results
        """
        
        regime_results = {}
        
        # Analyze each timeframe
        for timeframe in self.switching_config.timeframes:
            if timeframe not in price_data:
                continue
                
            df = price_data[timeframe]
            if len(df) < 60:  # Need sufficient data
                continue
            
            detector = self.timeframe_detectors.get(timeframe, self.regime_detector)
            df_with_regime = detector.annotate(df)
            
            # Get current regime for this timeframe
            trend_label, vol_label, confidence = detector.current_labels(df_with_regime)
            regime_label = f"{trend_label}:{vol_label}"
            
            regime_results[timeframe] = {
                'trend_label': trend_label,
                'vol_label': vol_label,
                'regime_label': regime_label,
                'confidence': confidence,
                'trend_score': df_with_regime['trend_score'].iloc[-1],
                'atr_percentile': df_with_regime['atr_percentile'].iloc[-1]
            }
        
        # Determine consensus regime
        consensus_regime = self._determine_consensus_regime(regime_results)
        
        return {
            'timeframe_regimes': regime_results,
            'consensus_regime': consensus_regime,
            'analysis_timestamp': datetime.now()
        }
    
    def _determine_consensus_regime(self, regime_results: dict[str, dict]) -> dict[str, Any]:
        """Determine consensus regime across timeframes"""
        
        if not regime_results:
            return {
                'regime_label': 'unknown:unknown',
                'confidence': 0.0,
                'agreement_score': 0.0,
                'participating_timeframes': []
            }
        
        # Collect regime votes with weights
        regime_votes = {}
        confidence_sum = 0.0
        weight_sum = 0.0
        
        # Weight longer timeframes more heavily
        timeframe_weights = {'1h': 1.0, '4h': 1.5, '1d': 2.0}
        
        for tf, result in regime_results.items():
            regime = result['regime_label']
            confidence = result['confidence']
            weight = timeframe_weights.get(tf, 1.0)
            
            if regime not in regime_votes:
                regime_votes[regime] = 0.0
            
            regime_votes[regime] += weight * confidence
            confidence_sum += confidence * weight
            weight_sum += weight
        
        # Find winning regime
        if not regime_votes:
            return {
                'regime_label': 'unknown:unknown',
                'confidence': 0.0,
                'agreement_score': 0.0,
                'participating_timeframes': []
            }
        
        winning_regime = max(regime_votes.keys(), key=lambda k: regime_votes[k])
        winning_score = regime_votes[winning_regime]
        
        # Calculate agreement score
        total_votes = sum(regime_votes.values())
        agreement_score = winning_score / total_votes if total_votes > 0 else 0.0
        
        # Calculate average confidence
        avg_confidence = confidence_sum / weight_sum if weight_sum > 0 else 0.0
        
        return {
            'regime_label': winning_regime,
            'confidence': avg_confidence,
            'agreement_score': agreement_score,
            'participating_timeframes': list(regime_results.keys()),
            'regime_votes': regime_votes
        }
    
    def should_switch_strategy(self, regime_analysis: dict[str, Any], current_candle_index: Optional[int] = None) -> dict[str, Any]:
        """Determine if strategy should be switched based on regime analysis
        
        Args:
            regime_analysis: Results from analyze_market_regime
            current_candle_index: Current candle index for accurate duration tracking
        """
        
        consensus = regime_analysis['consensus_regime']
        new_regime = consensus['regime_label']
        confidence = consensus['confidence']
        agreement = consensus['agreement_score']
        
        # Get optimal strategy for this regime
        optimal_strategy = self._get_optimal_strategy(new_regime)
        current_strategy_name = self.strategy_manager.current_strategy.name if self.strategy_manager.current_strategy else None
        
        # Decision criteria
        decision = {
            'should_switch': False,
            'reason': '',
            'new_regime': new_regime,
            'optimal_strategy': optimal_strategy,
            'current_strategy': current_strategy_name,
            'confidence': confidence,
            'agreement': agreement
        }
        
        # Check if we should switch
        if confidence < self.switching_config.min_regime_confidence:
            decision['reason'] = f"Low confidence: {confidence:.3f} < {self.switching_config.min_regime_confidence}"
            return decision
        
        if agreement < self.switching_config.require_timeframe_agreement:
            decision['reason'] = f"Low timeframe agreement: {agreement:.3f} < {self.switching_config.require_timeframe_agreement}"
            return decision
        
        # Check cooldown
        if self.last_switch_time:
            time_since_switch = datetime.now() - self.last_switch_time
            cooldown = timedelta(minutes=self.switching_config.switch_cooldown_minutes)
            if time_since_switch < cooldown:
                decision['reason'] = f"Switch cooldown: {time_since_switch} < {cooldown}"
                return decision
        
        # Check if regime has been stable long enough
        if self.current_regime == new_regime:
            # Calculate duration based on actual candle count if index is provided
            if current_candle_index is not None and self.regime_start_candle_index is not None:
                self.regime_duration = current_candle_index - self.regime_start_candle_index + 1
            else:
                # Fallback to old method if candle index not available
                self.regime_duration += 1
        else:
            self.regime_duration = 1
            self.current_regime = new_regime
            self.regime_start_time = datetime.now()
            if current_candle_index is not None:
                self.regime_start_candle_index = current_candle_index
        
        if self.regime_duration < self.switching_config.min_regime_duration:
            decision['reason'] = f"Regime not stable: {self.regime_duration} < {self.switching_config.min_regime_duration}"
            return decision
        
        # Check if optimal strategy is different from current
        if optimal_strategy == current_strategy_name:
            decision['reason'] = f"Already using optimal strategy: {optimal_strategy}"
            return decision
        
        # All checks passed - recommend switch
        decision['should_switch'] = True
        decision['reason'] = "Regime stable, high confidence, different optimal strategy"
        
        return decision
    
    def _get_optimal_strategy(self, regime_label: str) -> str:
        """Get optimal strategy for a given regime"""
        
        if ':' not in regime_label:
            return self.switching_config.emergency_strategy
        
        trend_label, vol_label = regime_label.split(':')
        
        # Map regime to strategy
        if trend_label == TrendLabel.TREND_UP.value:
            if vol_label == VolLabel.LOW.value:
                return self.strategy_mapping.bull_low_vol
            else:
                return self.strategy_mapping.bull_high_vol
        elif trend_label == TrendLabel.TREND_DOWN.value:
            if vol_label == VolLabel.LOW.value:
                return self.strategy_mapping.bear_low_vol
            else:
                return self.strategy_mapping.bear_high_vol
        else:  # Range market
            if vol_label == VolLabel.LOW.value:
                return self.strategy_mapping.range_low_vol
            else:
                return self.strategy_mapping.range_high_vol
    
    def get_position_size_multiplier(self, regime_label: str) -> float:
        """Get position size multiplier for regime"""
        
        if ':' not in regime_label:
            return 0.5  # Conservative default
        
        trend_label, vol_label = regime_label.split(':')
        
        # Map regime to position size multiplier
        if trend_label == TrendLabel.TREND_UP.value:
            if vol_label == VolLabel.LOW.value:
                return self.strategy_mapping.bull_low_vol_multiplier
            else:
                return self.strategy_mapping.bull_high_vol_multiplier
        elif trend_label == TrendLabel.TREND_DOWN.value:
            if vol_label == VolLabel.LOW.value:
                return self.strategy_mapping.bear_low_vol_multiplier
            else:
                return self.strategy_mapping.bear_high_vol_multiplier
        else:  # Range market
            if vol_label == VolLabel.LOW.value:
                return self.strategy_mapping.range_low_vol_multiplier
            else:
                return self.strategy_mapping.range_high_vol_multiplier
    
    def execute_strategy_switch(self, switch_decision: dict[str, Any]) -> bool:
        """Execute the strategy switch"""
        
        if not switch_decision['should_switch']:
            return False
        
        try:
            new_strategy = switch_decision['optimal_strategy']
            
            # For now, just pass the name parameter to avoid constructor issues
            # The regime-specific configuration can be added later when strategies support it
            regime_config = {
                'name': f"{new_strategy}_RegimeAdaptive"
            }
            
            # Execute hot swap
            success = self.strategy_manager.hot_swap_strategy(
                new_strategy_name=new_strategy,
                new_config=regime_config,
                close_existing_positions=self.switching_config.close_positions_on_switch
            )
            
            if success:
                self.last_switch_time = datetime.now()
                
                # Record the switch
                switch_record = {
                    'timestamp': self.last_switch_time,
                    'from_strategy': switch_decision['current_strategy'],
                    'to_strategy': new_strategy,
                    'regime': switch_decision['new_regime'],
                    'confidence': switch_decision['confidence'],
                    'reason': switch_decision['reason']
                }
                self.regime_history.append(switch_record)
                
                # Notify callbacks
                if self.on_strategy_switch:
                    self.on_strategy_switch(switch_record)
                
                logger.info(f"✅ Strategy switched: {switch_decision['current_strategy']} → {new_strategy} "
                           f"(regime: {switch_decision['new_regime']}, confidence: {switch_decision['confidence']:.3f})")
                
                return True
            else:
                logger.error(f"❌ Strategy switch failed: {switch_decision}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Strategy switch execution failed: {e}")
            return False
    
    def get_status(self) -> dict[str, Any]:
        """Get current status of regime-based switching"""
        
        return {
            'current_regime': self.current_regime,
            'regime_confidence': self.regime_confidence,
            'regime_duration': self.regime_duration,
            'current_strategy': self.strategy_manager.current_strategy.name if self.strategy_manager.current_strategy else None,
            'last_switch_time': self.last_switch_time.isoformat() if self.last_switch_time else None,
            'total_switches': len(self.regime_history),
            'switching_enabled': True,
            'multi_timeframe_enabled': self.switching_config.enable_multi_timeframe,
            'timeframes': self.switching_config.timeframes
        }
    
    def get_switching_history(self) -> list:
        """Get history of strategy switches"""
        return [
            {
                **record,
                'timestamp': record['timestamp'].isoformat()
            }
            for record in self.regime_history
        ]