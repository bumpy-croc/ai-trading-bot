"""
Multi-Modal Adaptive Ensemble Strategy

This ensemble strategy combines multiple existing strategies with intelligent
weighting, regime detection, and consensus-based decision making to improve
overall returns and reduce risk.

Key Features:
- Dynamic strategy weighting based on recent performance
- Regime-aware strategy selection (bull/bear/sideways)
- Consensus-based entry/exit decisions
- Multi-timeframe analysis
- Adaptive position sizing based on ensemble confidence
- Risk-adjusted performance optimization

Strategy Components:
1. ML Basic - Core price prediction
2. ML Adaptive - Regime-aware ML predictions
3. ML Sentiment - Market sentiment integration
4. Bull Strategy - Trend-following in uptrends
5. Bear Strategy - Short opportunities in downtrends

Ensemble Methods:
- Weighted voting for entry/exit decisions
- Performance-based strategy weight adjustment
- Regime-based strategy activation/deactivation
- Confidence-weighted position sizing
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.config.config_manager import get_config
from src.regime.detector import RegimeDetector, TrendLabel, VolLabel
from src.strategies.base import BaseStrategy
from src.strategies.ml_basic import MlBasic
from src.strategies.ml_adaptive import MlAdaptive
from src.strategies.ml_sentiment import MlSentiment
from src.strategies.bull import Bull
from src.strategies.bear import BearStrategy


class EnsembleAdaptive(BaseStrategy):
    """
    Multi-Modal Adaptive Ensemble Strategy
    
    Combines multiple strategies with intelligent weighting and consensus-based decisions.
    """
    
    # Ensemble configuration
    MIN_CONSENSUS_THRESHOLD = 0.6  # Minimum agreement for signal
    PERFORMANCE_WINDOW_DAYS = 30  # Days to look back for performance calculation
    REGIME_CONFIDENCE_THRESHOLD = 0.7  # Minimum regime confidence for regime-specific strategies
    
    # Position sizing parameters
    BASE_POSITION_SIZE = 0.15  # 15% base allocation
    MIN_POSITION_SIZE_RATIO = 0.05  # 5% minimum
    MAX_POSITION_SIZE_RATIO = 0.30  # 30% maximum
    CONFIDENCE_MULTIPLIER = 1.5  # Amplify position size with high confidence
    
    # Risk management
    STOP_LOSS_PCT = 0.025  # 2.5% stop loss
    TAKE_PROFIT_PCT = 0.05  # 5% take profit
    
    def __init__(
        self,
        name: str = "EnsembleAdaptive",
        enable_ml_basic: bool = True,
        enable_ml_adaptive: bool = True,
        enable_ml_sentiment: bool = True,
        enable_bull: bool = True,
        enable_bear: bool = True,
        performance_lookback_days: int = 30,
    ):
        super().__init__(name)
        self.trading_pair = "BTCUSDT"
        
        # Strategy components
        self.strategies: Dict[str, BaseStrategy] = {}
        self.strategy_weights: Dict[str, float] = {}
        self.strategy_performance: Dict[str, List[float]] = {}
        self.strategy_enabled: Dict[str, bool] = {}
        
        # Initialize component strategies
        if enable_ml_basic:
            self.strategies["ml_basic"] = MlBasic(name="ML_Basic_Component")
            self.strategy_weights["ml_basic"] = 0.25
            self.strategy_enabled["ml_basic"] = True
            
        if enable_ml_adaptive:
            self.strategies["ml_adaptive"] = MlAdaptive(name="ML_Adaptive_Component")
            self.strategy_weights["ml_adaptive"] = 0.25
            self.strategy_enabled["ml_adaptive"] = True
            
        if enable_ml_sentiment:
            self.strategies["ml_sentiment"] = MlSentiment(name="ML_Sentiment_Component")
            self.strategy_weights["ml_sentiment"] = 0.20
            self.strategy_enabled["ml_sentiment"] = True
            
        if enable_bull:
            self.strategies["bull"] = Bull(name="Bull_Component")
            self.strategy_weights["bull"] = 0.15
            self.strategy_enabled["bull"] = True
            
        if enable_bear:
            self.strategies["bear"] = BearStrategy(name="Bear_Component")
            self.strategy_weights["bear"] = 0.15
            self.strategy_enabled["bear"] = True
        
        # Initialize performance tracking
        for strategy_name in self.strategies.keys():
            self.strategy_performance[strategy_name] = []
        
        # Regime detector for adaptive weighting
        self.regime_detector = RegimeDetector()
        self.performance_lookback_days = performance_lookback_days
        
        # Track ensemble decisions
        self.decision_history: List[Dict[str, Any]] = []
        
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate indicators for all component strategies and ensemble-specific metrics"""
        df = df.copy()
        
        # Calculate indicators for each component strategy
        strategy_dfs = {}
        for name, strategy in self.strategies.items():
            if self.strategy_enabled[name]:
                try:
                    strategy_df = strategy.calculate_indicators(df.copy())
                    strategy_dfs[name] = strategy_df
                except Exception as e:
                    self.logger.warning(f"Failed to calculate indicators for {name}: {e}")
                    continue
        
        # Add regime detection
        try:
            df = self.regime_detector.annotate(df)
        except Exception as e:
            self.logger.warning(f"Regime detection failed: {e}")
            df["trend_label"] = "range"
            df["vol_label"] = "low_vol"
            df["regime_confidence"] = 0.5
        
        # Merge strategy-specific indicators with prefixes
        for name, strategy_df in strategy_dfs.items():
            for col in strategy_df.columns:
                if col not in ["open", "high", "low", "close", "volume"]:
                    df[f"{name}_{col}"] = strategy_df[col]
        
        # Calculate ensemble-specific indicators
        df = self._calculate_ensemble_indicators(df, strategy_dfs)
        
        return df
    
    def _calculate_ensemble_indicators(self, df: pd.DataFrame, strategy_dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Calculate ensemble-specific indicators"""
        
        # Initialize ensemble columns
        df["ensemble_signal_strength"] = 0.0
        df["ensemble_confidence"] = 0.0
        df["strategy_consensus"] = 0.0
        df["regime_adjusted_weights"] = ""
        
        # For each row, calculate ensemble metrics
        for i in range(len(df)):
            if i < 120:  # Skip early rows without enough history
                continue
                
            # Collect signals from enabled strategies
            signals = {}
            confidences = {}
            
            for name, strategy in self.strategies.items():
                if not self.strategy_enabled[name] or name not in strategy_dfs:
                    continue
                    
                strategy_df = strategy_dfs[name]
                
                try:
                    # Get entry signal
                    if hasattr(strategy, 'check_entry_conditions'):
                        signal = strategy.check_entry_conditions(strategy_df, i)
                        signals[name] = 1.0 if signal else 0.0
                    
                    # Get confidence if available
                    if "prediction_confidence" in strategy_df.columns:
                        conf = strategy_df["prediction_confidence"].iloc[i]
                        confidences[name] = conf if not pd.isna(conf) else 0.5
                    else:
                        confidences[name] = 0.5
                        
                except Exception as e:
                    self.logger.debug(f"Error getting signal from {name}: {e}")
                    signals[name] = 0.0
                    confidences[name] = 0.0
            
            # Calculate weighted consensus
            if signals:
                # Get current regime-adjusted weights
                regime_weights = self._get_regime_adjusted_weights(df, i)
                
                # Calculate weighted signal
                weighted_signal = sum(
                    signals.get(name, 0.0) * regime_weights.get(name, 0.0)
                    for name in signals.keys()
                )
                
                # Calculate confidence-weighted consensus
                confidence_weighted_signal = sum(
                    signals.get(name, 0.0) * confidences.get(name, 0.5) * regime_weights.get(name, 0.0)
                    for name in signals.keys()
                )
                
                total_weight = sum(regime_weights.get(name, 0.0) for name in signals.keys())
                total_confidence_weight = sum(
                    confidences.get(name, 0.5) * regime_weights.get(name, 0.0)
                    for name in signals.keys()
                )
                
                if total_weight > 0:
                    df.at[df.index[i], "ensemble_signal_strength"] = weighted_signal / total_weight
                    df.at[df.index[i], "strategy_consensus"] = len([s for s in signals.values() if s > 0]) / len(signals)
                
                if total_confidence_weight > 0:
                    df.at[df.index[i], "ensemble_confidence"] = confidence_weighted_signal / total_confidence_weight
                
                df.at[df.index[i], "regime_adjusted_weights"] = str(regime_weights)
        
        return df
    
    def _get_regime_adjusted_weights(self, df: pd.DataFrame, index: int) -> Dict[str, float]:
        """Get strategy weights adjusted for current market regime"""
        if index >= len(df):
            return self.strategy_weights.copy()
        
        # Get current regime
        trend_label = df["trend_label"].iloc[index] if "trend_label" in df.columns else "range"
        vol_label = df["vol_label"].iloc[index] if "vol_label" in df.columns else "low_vol"
        regime_confidence = df["regime_confidence"].iloc[index] if "regime_confidence" in df.columns else 0.5
        
        # Start with base weights
        adjusted_weights = self.strategy_weights.copy()
        
        # Adjust based on regime with high confidence
        if regime_confidence >= self.REGIME_CONFIDENCE_THRESHOLD:
            if trend_label == TrendLabel.TREND_UP.value:
                # Boost bull strategy, reduce bear strategy
                adjusted_weights["bull"] = adjusted_weights.get("bull", 0.15) * 1.5
                adjusted_weights["bear"] = adjusted_weights.get("bear", 0.15) * 0.3
                adjusted_weights["ml_adaptive"] = adjusted_weights.get("ml_adaptive", 0.25) * 1.2
                
            elif trend_label == TrendLabel.TREND_DOWN.value:
                # Boost bear strategy, reduce bull strategy
                adjusted_weights["bear"] = adjusted_weights.get("bear", 0.15) * 1.5
                adjusted_weights["bull"] = adjusted_weights.get("bull", 0.15) * 0.3
                adjusted_weights["ml_adaptive"] = adjusted_weights.get("ml_adaptive", 0.25) * 1.2
                
            else:  # Range-bound market
                # Boost ML strategies, reduce trend-following
                adjusted_weights["ml_basic"] = adjusted_weights.get("ml_basic", 0.25) * 1.3
                adjusted_weights["ml_sentiment"] = adjusted_weights.get("ml_sentiment", 0.20) * 1.2
                adjusted_weights["bull"] = adjusted_weights.get("bull", 0.15) * 0.7
                adjusted_weights["bear"] = adjusted_weights.get("bear", 0.15) * 0.7
        
        # Adjust for volatility
        if vol_label == VolLabel.HIGH.value:
            # Reduce position in high volatility
            adjusted_weights = {k: v * 0.8 for k, v in adjusted_weights.items()}
            # But boost sentiment strategy which handles volatility well
            adjusted_weights["ml_sentiment"] = adjusted_weights.get("ml_sentiment", 0.16) * 1.25
        
        # Normalize weights
        total_weight = sum(adjusted_weights.values())
        if total_weight > 0:
            adjusted_weights = {k: v / total_weight for k, v in adjusted_weights.items()}
        
        return adjusted_weights
    
    def check_entry_conditions(self, df: pd.DataFrame, index: int) -> bool:
        """Check ensemble entry conditions based on strategy consensus"""
        if index < 120 or index >= len(df):
            return False
        
        # Get ensemble metrics
        signal_strength = df["ensemble_signal_strength"].iloc[index]
        consensus = df["strategy_consensus"].iloc[index]
        confidence = df["ensemble_confidence"].iloc[index]
        
        # Entry conditions
        strong_signal = signal_strength >= self.MIN_CONSENSUS_THRESHOLD
        high_consensus = consensus >= 0.5  # At least half the strategies agree
        sufficient_confidence = confidence >= 0.4
        
        entry_decision = strong_signal and high_consensus and sufficient_confidence
        
        # Log decision
        decision_data = {
            "timestamp": df.index[index],
            "signal_strength": signal_strength,
            "consensus": consensus,
            "confidence": confidence,
            "decision": "entry" if entry_decision else "no_entry",
            "regime": df.get("trend_label", {}).get(index, "unknown"),
        }
        
        self.log_execution(
            signal_type="entry",
            action_taken="entry_signal" if entry_decision else "no_action",
            price=df["close"].iloc[index],
            signal_strength=signal_strength,
            confidence_score=confidence,
            reasons=[
                f"signal_strength_{signal_strength:.3f}",
                f"consensus_{consensus:.3f}",
                f"confidence_{confidence:.3f}",
                "entry_conditions_met" if entry_decision else "entry_conditions_not_met",
            ],
            additional_context=decision_data,
        )
        
        return entry_decision
    
    def check_exit_conditions(self, df: pd.DataFrame, index: int, entry_price: float) -> bool:
        """Check ensemble exit conditions"""
        if index < 1 or index >= len(df):
            return False
        
        current_price = df["close"].iloc[index]
        returns = (current_price - entry_price) / entry_price if entry_price > 0 else 0.0
        
        # Basic stop loss / take profit
        hit_stop_loss = returns <= -self.STOP_LOSS_PCT
        hit_take_profit = returns >= self.TAKE_PROFIT_PCT
        
        # Ensemble-based exit: if consensus turns negative
        signal_strength = df.get("ensemble_signal_strength", pd.Series([0.0])).iloc[index]
        consensus_exit = signal_strength < 0.3  # Weak or negative consensus
        
        exit_decision = hit_stop_loss or hit_take_profit or consensus_exit
        
        self.log_execution(
            signal_type="exit",
            action_taken="exit_signal" if exit_decision else "hold",
            price=current_price,
            reasons=[
                "stop_loss" if hit_stop_loss else "",
                "take_profit" if hit_take_profit else "",
                "consensus_exit" if consensus_exit else "",
                f"returns_{returns:.3f}",
                f"signal_strength_{signal_strength:.3f}",
            ],
        )
        
        return exit_decision
    
    def calculate_position_size(self, df: pd.DataFrame, index: int, balance: float) -> float:
        """Calculate position size based on ensemble confidence and regime"""
        if index >= len(df) or balance <= 0:
            return 0.0
        
        # Get ensemble confidence
        confidence = df.get("ensemble_confidence", pd.Series([0.5])).iloc[index]
        signal_strength = df.get("ensemble_signal_strength", pd.Series([0.5])).iloc[index]
        
        # Base position size
        base_size = self.BASE_POSITION_SIZE
        
        # Adjust for confidence and signal strength
        confidence_factor = max(0.5, min(1.5, confidence * self.CONFIDENCE_MULTIPLIER))
        signal_factor = max(0.5, min(1.5, signal_strength * 1.2))
        
        # Combine factors
        dynamic_size = base_size * confidence_factor * signal_factor
        
        # Apply limits
        position_ratio = max(
            self.MIN_POSITION_SIZE_RATIO,
            min(self.MAX_POSITION_SIZE_RATIO, dynamic_size)
        )
        
        return position_ratio * balance
    
    def calculate_stop_loss(self, df: pd.DataFrame, index: int, price: float, side: str = "long") -> float:
        """Calculate stop loss level"""
        side_str = side.value if hasattr(side, "value") else str(side)
        
        if side_str == "long":
            return price * (1 - self.STOP_LOSS_PCT)
        else:
            return price * (1 + self.STOP_LOSS_PCT)
    
    def get_risk_overrides(self) -> Optional[Dict[str, Any]]:
        """Enhanced risk management for ensemble strategy"""
        return {
            "position_sizer": "confidence_weighted",
            "base_fraction": 0.15,
            "min_fraction": 0.05,
            "max_fraction": 0.30,
            "stop_loss_pct": self.STOP_LOSS_PCT,
            "take_profit_pct": self.TAKE_PROFIT_PCT,
            "dynamic_risk": {
                "enabled": True,
                "performance_window_days": 30,
                "drawdown_thresholds": [0.05, 0.10, 0.20],
                "risk_reduction_factors": [0.8, 0.6, 0.4],
                "recovery_thresholds": [0.02, 0.05],
                "volatility_adjustment_enabled": True,
            },
            "partial_operations": {
                "exit_targets": [0.025, 0.04, 0.06],  # 2.5%, 4%, 6%
                "exit_sizes": [0.30, 0.30, 0.40],     # 30%, 30%, 40%
                "scale_in_thresholds": [0.015, 0.03], # 1.5%, 3%
                "scale_in_sizes": [0.25, 0.25],       # 25%, 25%
                "max_scale_ins": 2,
            },
            "trailing_stop": {
                "activation_threshold": 0.02,   # 2%
                "trailing_distance_pct": 0.008, # 0.8%
                "breakeven_threshold": 0.025,   # 2.5%
                "breakeven_buffer": 0.002,      # 0.2%
            },
        }
    
    def get_parameters(self) -> dict:
        """Return ensemble strategy parameters"""
        return {
            "name": self.name,
            "enabled_strategies": list(self.strategies.keys()),
            "strategy_weights": self.strategy_weights,
            "min_consensus_threshold": self.MIN_CONSENSUS_THRESHOLD,
            "performance_window_days": self.PERFORMANCE_WINDOW_DAYS,
            "regime_confidence_threshold": self.REGIME_CONFIDENCE_THRESHOLD,
            "base_position_size": self.BASE_POSITION_SIZE,
            "stop_loss_pct": self.STOP_LOSS_PCT,
            "take_profit_pct": self.TAKE_PROFIT_PCT,
            "confidence_multiplier": self.CONFIDENCE_MULTIPLIER,
        }
    
    def update_strategy_performance(self, strategy_name: str, performance_score: float):
        """Update performance tracking for a strategy"""
        if strategy_name not in self.strategy_performance:
            self.strategy_performance[strategy_name] = []
        
        self.strategy_performance[strategy_name].append(performance_score)
        
        # Keep only recent performance data
        max_history = self.performance_lookback_days
        if len(self.strategy_performance[strategy_name]) > max_history:
            self.strategy_performance[strategy_name] = self.strategy_performance[strategy_name][-max_history:]
    
    def get_strategy_health_report(self) -> Dict[str, Any]:
        """Generate a health report for the ensemble and its components"""
        report = {
            "ensemble_name": self.name,
            "total_strategies": len(self.strategies),
            "enabled_strategies": sum(self.strategy_enabled.values()),
            "strategy_details": {},
            "current_weights": self.strategy_weights,
        }
        
        for name, strategy in self.strategies.items():
            strategy_info = {
                "enabled": self.strategy_enabled[name],
                "current_weight": self.strategy_weights[name],
                "recent_performance": (
                    np.mean(self.strategy_performance[name][-10:])
                    if len(self.strategy_performance[name]) >= 10
                    else None
                ),
                "performance_samples": len(self.strategy_performance[name]),
            }
            report["strategy_details"][name] = strategy_info
        
        return report