"""
Weighted Ensemble Strategy

A simpler ensemble approach that combines multiple strategies using
performance-based weighting and majority voting for decisions.

Key Features:
- Performance-based dynamic weighting
- Simple majority voting for entry/exit
- Confidence-based position sizing
- Strategy performance tracking and adaptation
- Fallback mechanisms when strategies disagree

This ensemble is designed to be more stable and predictable than complex
adaptive approaches while still capturing the benefits of strategy diversification.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from collections import defaultdict

from src.strategies.base import BaseStrategy
from src.strategies.ml_basic import MlBasic
from src.strategies.ml_adaptive import MlAdaptive
from src.strategies.ml_sentiment import MlSentiment


class EnsembleWeighted(BaseStrategy):
    """
    Weighted Ensemble Strategy using performance-based weighting and majority voting
    """
    
    # Configuration
    MIN_STRATEGIES_FOR_SIGNAL = 2  # Minimum strategies that must agree
    PERFORMANCE_WINDOW = 50  # Number of recent decisions to track
    WEIGHT_UPDATE_FREQUENCY = 20  # Update weights every N decisions
    
    # Position sizing
    BASE_POSITION_SIZE = 0.18  # 18% base allocation
    MIN_POSITION_SIZE_RATIO = 0.06
    MAX_POSITION_SIZE_RATIO = 0.25
    
    # Risk management
    STOP_LOSS_PCT = 0.02  # 2%
    TAKE_PROFIT_PCT = 0.045  # 4.5%
    
    def __init__(
        self,
        name: str = "EnsembleWeighted",
        use_ml_basic: bool = True,
        use_ml_adaptive: bool = True,
        use_ml_sentiment: bool = False,  # Disabled by default due to external dependency
    ):
        super().__init__(name)
        self.trading_pair = "BTCUSDT"
        
        # Initialize component strategies
        self.strategies: Dict[str, BaseStrategy] = {}
        self.strategy_weights: Dict[str, float] = {}
        self.strategy_performance: Dict[str, List[float]] = defaultdict(list)
        self.decision_count = 0
        
        if use_ml_basic:
            self.strategies["ml_basic"] = MlBasic(name="MLBasic_Component")
            self.strategy_weights["ml_basic"] = 0.4
            
        if use_ml_adaptive:
            self.strategies["ml_adaptive"] = MlAdaptive(name="MLAdaptive_Component")
            self.strategy_weights["ml_adaptive"] = 0.4
            
        if use_ml_sentiment:
            self.strategies["ml_sentiment"] = MlSentiment(name="MLSentiment_Component")
            self.strategy_weights["ml_sentiment"] = 0.2
        
        # Normalize initial weights
        self._normalize_weights()
        
        self.logger.info(f"Initialized ensemble with strategies: {list(self.strategies.keys())}")
        self.logger.info(f"Initial weights: {self.strategy_weights}")
    
    def _normalize_weights(self):
        """Normalize strategy weights to sum to 1.0"""
        total_weight = sum(self.strategy_weights.values())
        if total_weight > 0:
            self.strategy_weights = {
                name: weight / total_weight 
                for name, weight in self.strategy_weights.items()
            }
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate indicators for all component strategies"""
        df = df.copy()
        
        # Calculate indicators for each strategy
        strategy_dfs = {}
        for name, strategy in self.strategies.items():
            try:
                strategy_df = strategy.calculate_indicators(df.copy())
                strategy_dfs[name] = strategy_df
                
                # Add strategy-specific columns with prefixes
                for col in strategy_df.columns:
                    if col not in ["open", "high", "low", "close", "volume"]:
                        df[f"{name}_{col}"] = strategy_df[col]
                        
            except Exception as e:
                self.logger.warning(f"Failed to calculate indicators for {name}: {e}")
                continue
        
        # Calculate ensemble-specific indicators
        df = self._calculate_ensemble_signals(df, strategy_dfs)
        
        return df
    
    def _calculate_ensemble_signals(self, df: pd.DataFrame, strategy_dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Calculate ensemble signals and confidence metrics"""
        
        # Initialize ensemble columns
        df["ensemble_entry_score"] = 0.0
        df["ensemble_confidence"] = 0.0
        df["strategy_agreement"] = 0.0
        df["active_strategies"] = 0
        
        # Calculate for each row
        for i in range(len(df)):
            if i < 120:  # Skip early rows
                continue
            
            entry_signals = []
            confidences = []
            weights = []
            
            # Collect signals from each strategy
            for name, strategy in self.strategies.items():
                if name not in strategy_dfs:
                    continue
                
                try:
                    strategy_df = strategy_dfs[name]
                    
                    # Get entry signal
                    has_entry = strategy.check_entry_conditions(strategy_df, i)
                    entry_signals.append(1.0 if has_entry else 0.0)
                    
                    # Get confidence (if available)
                    if "prediction_confidence" in strategy_df.columns:
                        conf = strategy_df["prediction_confidence"].iloc[i]
                        confidences.append(conf if not pd.isna(conf) else 0.5)
                    else:
                        confidences.append(0.5)
                    
                    # Get current weight for this strategy
                    weights.append(self.strategy_weights[name])
                    
                except Exception as e:
                    self.logger.debug(f"Error getting signal from {name} at index {i}: {e}")
                    continue
            
            if len(entry_signals) >= self.MIN_STRATEGIES_FOR_SIGNAL:
                # Calculate weighted entry score
                weighted_score = sum(
                    signal * weight * confidence 
                    for signal, weight, confidence in zip(entry_signals, weights, confidences)
                )
                total_weight = sum(weights)
                
                if total_weight > 0:
                    df.at[df.index[i], "ensemble_entry_score"] = weighted_score / total_weight
                
                # Calculate strategy agreement (percentage that agree)
                agreement = sum(entry_signals) / len(entry_signals)
                df.at[df.index[i], "strategy_agreement"] = agreement
                
                # Calculate average confidence
                avg_confidence = np.mean(confidences)
                df.at[df.index[i], "ensemble_confidence"] = avg_confidence
                
                # Track active strategies
                df.at[df.index[i], "active_strategies"] = len(entry_signals)
        
        return df
    
    def check_entry_conditions(self, df: pd.DataFrame, index: int) -> bool:
        """Check ensemble entry conditions using weighted voting"""
        if index < 120 or index >= len(df):
            return False
        
        # Get ensemble metrics
        entry_score = df["ensemble_entry_score"].iloc[index]
        agreement = df["strategy_agreement"].iloc[index]
        confidence = df["ensemble_confidence"].iloc[index]
        active_strategies = df["active_strategies"].iloc[index]
        
        # Entry criteria
        score_threshold = 0.5  # Weighted score must be > 0.5
        min_agreement = 0.6    # At least 60% of strategies must agree
        min_confidence = 0.4   # Minimum confidence level
        min_active = self.MIN_STRATEGIES_FOR_SIGNAL
        
        entry_decision = (
            entry_score > score_threshold and
            agreement >= min_agreement and
            confidence >= min_confidence and
            active_strategies >= min_active
        )
        
        # Log decision
        self.log_execution(
            signal_type="entry",
            action_taken="entry_signal" if entry_decision else "no_action",
            price=df["close"].iloc[index],
            signal_strength=entry_score,
            confidence_score=confidence,
            reasons=[
                f"entry_score_{entry_score:.3f}",
                f"agreement_{agreement:.3f}",
                f"confidence_{confidence:.3f}",
                f"active_strategies_{int(active_strategies)}",
                "entry_criteria_met" if entry_decision else "entry_criteria_not_met",
            ],
            additional_context={
                "ensemble_method": "weighted_voting",
                "current_weights": dict(self.strategy_weights),
                "score_threshold": score_threshold,
                "min_agreement": min_agreement,
            },
        )
        
        # Track decision for performance updates
        self.decision_count += 1
        if self.decision_count % self.WEIGHT_UPDATE_FREQUENCY == 0:
            self._update_strategy_weights()
        
        return entry_decision
    
    def check_exit_conditions(self, df: pd.DataFrame, index: int, entry_price: float) -> bool:
        """Check exit conditions using ensemble approach"""
        if index < 1 or index >= len(df):
            return False
        
        current_price = df["close"].iloc[index]
        returns = (current_price - entry_price) / entry_price if entry_price > 0 else 0.0
        
        # Basic stop loss / take profit
        hit_stop_loss = returns <= -self.STOP_LOSS_PCT
        hit_take_profit = returns >= self.TAKE_PROFIT_PCT
        
        # Ensemble exit: check if majority of strategies suggest exit
        exit_signals = []
        for name, strategy in self.strategies.items():
            try:
                # Get strategy-specific data
                strategy_cols = [col for col in df.columns if col.startswith(f"{name}_")]
                if not strategy_cols:
                    continue
                
                # Create a mini-df with strategy indicators
                strategy_data = df[["open", "high", "low", "close", "volume"] + strategy_cols].copy()
                # Remove prefix from column names
                strategy_data.columns = [
                    col.replace(f"{name}_", "") if col.startswith(f"{name}_") else col
                    for col in strategy_data.columns
                ]
                
                # Check exit conditions
                has_exit = strategy.check_exit_conditions(strategy_data, index, entry_price)
                exit_signals.append(has_exit)
                
            except Exception as e:
                self.logger.debug(f"Error checking exit for {name}: {e}")
                continue
        
        # Ensemble exit decision
        ensemble_exit = False
        if len(exit_signals) >= 2:  # Need at least 2 strategies
            exit_agreement = sum(exit_signals) / len(exit_signals)
            ensemble_exit = exit_agreement >= 0.6  # 60% must agree to exit
        
        exit_decision = hit_stop_loss or hit_take_profit or ensemble_exit
        
        self.log_execution(
            signal_type="exit",
            action_taken="exit_signal" if exit_decision else "hold",
            price=current_price,
            reasons=[
                "stop_loss" if hit_stop_loss else "",
                "take_profit" if hit_take_profit else "",
                "ensemble_exit" if ensemble_exit else "",
                f"returns_{returns:.3f}",
                f"exit_agreement_{exit_agreement:.3f}" if len(exit_signals) >= 2 else "",
            ],
        )
        
        return exit_decision
    
    def calculate_position_size(self, df: pd.DataFrame, index: int, balance: float) -> float:
        """Calculate position size based on ensemble confidence"""
        if index >= len(df) or balance <= 0:
            return 0.0
        
        # Get ensemble confidence
        confidence = df.get("ensemble_confidence", pd.Series([0.5])).iloc[index]
        entry_score = df.get("ensemble_entry_score", pd.Series([0.5])).iloc[index]
        agreement = df.get("strategy_agreement", pd.Series([0.5])).iloc[index]
        
        # Base position size
        base_size = self.BASE_POSITION_SIZE
        
        # Adjust for ensemble metrics
        confidence_factor = max(0.6, min(1.4, confidence * 1.2))
        score_factor = max(0.7, min(1.3, entry_score * 1.1))
        agreement_factor = max(0.8, min(1.2, agreement))
        
        # Combine factors
        dynamic_size = base_size * confidence_factor * score_factor * agreement_factor
        
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
    
    def _update_strategy_weights(self):
        """Update strategy weights based on recent performance"""
        # This is a placeholder for performance-based weight updates
        # In a full implementation, this would track individual strategy
        # performance and adjust weights accordingly
        
        self.logger.info(f"Weight update triggered at decision #{self.decision_count}")
        # For now, keep weights stable
        pass
    
    def get_risk_overrides(self) -> Optional[Dict[str, Any]]:
        """Risk management overrides for ensemble strategy"""
        return {
            "position_sizer": "confidence_weighted",
            "base_fraction": self.BASE_POSITION_SIZE,
            "min_fraction": self.MIN_POSITION_SIZE_RATIO,
            "max_fraction": self.MAX_POSITION_SIZE_RATIO,
            "stop_loss_pct": self.STOP_LOSS_PCT,
            "take_profit_pct": self.TAKE_PROFIT_PCT,
            "dynamic_risk": {
                "enabled": True,
                "drawdown_thresholds": [0.06, 0.12, 0.20],
                "risk_reduction_factors": [0.8, 0.6, 0.4],
                "recovery_thresholds": [0.03, 0.06],
            },
            "partial_operations": {
                "exit_targets": [0.02, 0.035, 0.05],  # 2%, 3.5%, 5%
                "exit_sizes": [0.25, 0.35, 0.40],     # 25%, 35%, 40%
                "scale_in_thresholds": [0.015, 0.025], # 1.5%, 2.5%
                "scale_in_sizes": [0.3, 0.2],         # 30%, 20%
                "max_scale_ins": 2,
            },
        }
    
    def get_parameters(self) -> dict:
        """Return ensemble parameters"""
        return {
            "name": self.name,
            "strategies": list(self.strategies.keys()),
            "current_weights": dict(self.strategy_weights),
            "min_strategies_for_signal": self.MIN_STRATEGIES_FOR_SIGNAL,
            "performance_window": self.PERFORMANCE_WINDOW,
            "weight_update_frequency": self.WEIGHT_UPDATE_FREQUENCY,
            "base_position_size": self.BASE_POSITION_SIZE,
            "stop_loss_pct": self.STOP_LOSS_PCT,
            "take_profit_pct": self.TAKE_PROFIT_PCT,
            "decision_count": self.decision_count,
        }
    
    def get_ensemble_status(self) -> Dict[str, Any]:
        """Get current ensemble status and health"""
        return {
            "ensemble_name": self.name,
            "active_strategies": len(self.strategies),
            "current_weights": dict(self.strategy_weights),
            "total_decisions": self.decision_count,
            "last_weight_update": self.decision_count // self.WEIGHT_UPDATE_FREQUENCY * self.WEIGHT_UPDATE_FREQUENCY,
            "next_weight_update": ((self.decision_count // self.WEIGHT_UPDATE_FREQUENCY) + 1) * self.WEIGHT_UPDATE_FREQUENCY,
        }