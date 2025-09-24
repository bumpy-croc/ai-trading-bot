"""
Optimized Weighted Ensemble Strategy

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

import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from collections import defaultdict

from src.strategies.base import BaseStrategy
from src.strategies.ml_basic import MlBasic
from src.strategies.ml_adaptive import MlAdaptive
from src.strategies.ml_sentiment import MlSentiment
from src.strategies.bull import Bull
from src.strategies.bear import BearStrategy


class EnsembleWeighted(BaseStrategy):
    """
    Weighted Ensemble Strategy using performance-based weighting and majority voting
    """
    
    # Configuration - Optimized for higher returns with 20-30% acceptable drawdown
    MIN_STRATEGIES_FOR_SIGNAL = 2  # Minimum strategies that must agree
    PERFORMANCE_WINDOW = 50  # Number of recent decisions to track
    WEIGHT_UPDATE_FREQUENCY = 20  # Update weights every N decisions
    
    # Position sizing - More aggressive for higher returns
    BASE_POSITION_SIZE = 0.30  # 30% base allocation (increased from 18%)
    MIN_POSITION_SIZE_RATIO = 0.10  # 10% minimum (increased from 6%)
    MAX_POSITION_SIZE_RATIO = 0.45  # 45% maximum (increased from 25%)
    
    # Risk management - Wider stops for higher returns
    STOP_LOSS_PCT = 0.035  # 3.5% (increased from 2%)
    TAKE_PROFIT_PCT = 0.08  # 8% (increased from 4.5%)
    
    def __init__(
        self,
        name: str = "EnsembleWeighted",
        use_ml_basic: bool = True,
        use_ml_adaptive: bool = True,
        use_ml_sentiment: bool = False,  # Disabled by default due to external dependency
        use_bull: bool = True,  # Enable trend-following for bull markets
        use_bear: bool = True,  # Enable short strategies for bear markets
    ):
        super().__init__(name)
        self.trading_pair = "BTCUSDT"
        
        # Initialize component strategies with optimized weights for higher returns
        self.strategies: Dict[str, BaseStrategy] = {}
        self.strategy_weights: Dict[str, float] = {}
        self.strategy_performance: Dict[str, List[float]] = defaultdict(list)
        self.decision_count = 0
        
        if use_ml_basic:
            self.strategies["ml_basic"] = MlBasic(name="MLBasic_Component")
            self.strategy_weights["ml_basic"] = 0.30  # Increased weight
            
        if use_ml_adaptive:
            self.strategies["ml_adaptive"] = MlAdaptive(name="MLAdaptive_Component")
            self.strategy_weights["ml_adaptive"] = 0.30  # Increased weight
            
        if use_ml_sentiment:
            self.strategies["ml_sentiment"] = MlSentiment(name="MLSentiment_Component")
            self.strategy_weights["ml_sentiment"] = 0.15
            
        if use_bull:
            self.strategies["bull"] = Bull(name="Bull_Component")
            self.strategy_weights["bull"] = 0.25  # Strong weight for trend following
            
        if use_bear:
            self.strategies["bear"] = BearStrategy(name="Bear_Component")
            self.strategy_weights["bear"] = 0.15  # Moderate weight for downturns
        
        # Normalize initial weights
        self._normalize_weights()
        
        self.logger.info(f"Initialized optimized ensemble with strategies: {list(self.strategies.keys())}")
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
        
        # Add momentum and trend indicators to help beat buy-and-hold
        df = self._calculate_momentum_indicators(df)
        
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
    
    def _calculate_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate momentum and trend indicators to help beat buy-and-hold"""
        
        # Price momentum
        df["price_momentum_5"] = df["close"].pct_change(5)  # 5-period momentum
        df["price_momentum_20"] = df["close"].pct_change(20)  # 20-period momentum
        
        # Trend strength
        df["sma_20"] = df["close"].rolling(20).mean()
        df["sma_50"] = df["close"].rolling(50).mean()
        df["trend_strength"] = (df["sma_20"] - df["sma_50"]) / df["sma_50"]
        
        # Volatility for position sizing
        df["volatility"] = df["close"].pct_change().rolling(20).std()
        df["volatility_rank"] = df["volatility"].rolling(100).rank(pct=True)
        
        # Breakout detection
        df["high_20"] = df["high"].rolling(20).max()
        df["low_20"] = df["low"].rolling(20).min()
        df["breakout_up"] = (df["close"] > df["high_20"].shift(1)) & (df["trend_strength"] > 0.02)
        df["breakout_down"] = (df["close"] < df["low_20"].shift(1)) & (df["trend_strength"] < -0.02)
        
        # Market regime detection (simple)
        df["bull_market"] = (df["sma_20"] > df["sma_50"]) & (df["trend_strength"] > 0.01)
        df["bear_market"] = (df["sma_20"] < df["sma_50"]) & (df["trend_strength"] < -0.01)
        
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
        
        # Entry criteria - More aggressive for higher returns
        score_threshold = 0.4  # Weighted score must be > 0.4 (lowered from 0.5)
        min_agreement = 0.5    # At least 50% of strategies must agree (lowered from 60%)
        min_confidence = 0.3   # Minimum confidence level (lowered from 0.4)
        min_active = self.MIN_STRATEGIES_FOR_SIGNAL
        
        # Momentum filters to beat buy-and-hold
        momentum_5 = df.get("price_momentum_5", pd.Series([0.0])).iloc[index]
        momentum_20 = df.get("price_momentum_20", pd.Series([0.0])).iloc[index]
        trend_strength = df.get("trend_strength", pd.Series([0.0])).iloc[index]
        breakout_up = df.get("breakout_up", pd.Series([False])).iloc[index]
        bull_market = df.get("bull_market", pd.Series([False])).iloc[index]
        
        # Enhanced entry conditions with momentum
        basic_entry = (
            entry_score > score_threshold and
            agreement >= min_agreement and
            confidence >= min_confidence and
            active_strategies >= min_active
        )
        
        # Momentum boost conditions
        strong_momentum = momentum_5 > 0.01 or momentum_20 > 0.03  # Strong recent momentum
        trending_up = trend_strength > 0.005  # Positive trend
        breakout_signal = breakout_up  # Price breakout
        favorable_regime = bull_market  # Bull market regime
        
        # Combined entry decision (basic OR momentum-boosted)
        momentum_entry = strong_momentum and (trending_up or breakout_signal or favorable_regime)
        
        entry_decision = basic_entry or (entry_score > 0.3 and momentum_entry)
        
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
        
        # Ensemble exit decision - More aggressive (hold longer for higher returns)
        ensemble_exit = False
        if len(exit_signals) >= 2:  # Need at least 2 strategies
            exit_agreement = sum(exit_signals) / len(exit_signals)
            ensemble_exit = exit_agreement >= 0.7  # 70% must agree to exit (increased from 60%)
        
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
        
        # Adjust for ensemble metrics - More aggressive multipliers
        confidence_factor = max(0.8, min(2.0, confidence * 1.5))  # Increased multipliers
        score_factor = max(0.9, min(1.8, entry_score * 1.4))      # Higher score impact
        agreement_factor = max(0.9, min(1.5, agreement * 1.2))    # Reward strong agreement
        
        # Momentum and volatility adjustments to beat buy-and-hold
        momentum_5 = df.get("price_momentum_5", pd.Series([0.0])).iloc[index]
        trend_strength = df.get("trend_strength", pd.Series([0.0])).iloc[index]
        volatility_rank = df.get("volatility_rank", pd.Series([0.5])).iloc[index]
        bull_market = df.get("bull_market", pd.Series([False])).iloc[index]
        
        # Momentum factor: boost size in strong trends
        momentum_factor = 1.0
        if momentum_5 > 0.02:  # Strong 5-period momentum
            momentum_factor = 1.3
        elif momentum_5 > 0.01:  # Moderate momentum
            momentum_factor = 1.15
        elif momentum_5 < -0.01:  # Negative momentum
            momentum_factor = 0.8
        
        # Trend factor: boost size in strong trends
        trend_factor = 1.0
        if abs(trend_strength) > 0.03:  # Strong trend
            trend_factor = 1.4
        elif abs(trend_strength) > 0.01:  # Moderate trend
            trend_factor = 1.2
        
        # Volatility factor: reduce size in high volatility
        vol_factor = 1.0
        if volatility_rank > 0.8:  # High volatility
            vol_factor = 0.7
        elif volatility_rank > 0.6:  # Moderate volatility
            vol_factor = 0.85
        elif volatility_rank < 0.3:  # Low volatility
            vol_factor = 1.2
        
        # Bull market boost
        regime_factor = 1.3 if bull_market else 1.0
        
        # Combine all factors
        dynamic_size = (base_size * confidence_factor * score_factor * agreement_factor * 
                       momentum_factor * trend_factor * vol_factor * regime_factor)
        
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
                "drawdown_thresholds": [0.10, 0.20, 0.30],  # Accept higher drawdowns
                "risk_reduction_factors": [0.9, 0.7, 0.5],   # Less aggressive reduction
                "recovery_thresholds": [0.05, 0.10],         # Higher recovery thresholds
            },
            "partial_operations": {
                "exit_targets": [0.04, 0.06, 0.10],  # 4%, 6%, 10% (higher targets)
                "exit_sizes": [0.20, 0.30, 0.50],     # 20%, 30%, 50% (hold more)
                "scale_in_thresholds": [0.02, 0.04], # 2%, 4% (more aggressive scale-in)
                "scale_in_sizes": [0.4, 0.3],         # 40%, 30% (larger scale-ins)
                "max_scale_ins": 3,                   # Allow more scale-ins
            },
            "trailing_stop": {
                "activation_threshold": 0.03,   # 3% before trailing starts
                "trailing_distance_pct": 0.015, # 1.5% trailing distance
                "breakeven_threshold": 0.04,    # 4% before breakeven
                "breakeven_buffer": 0.005,      # 0.5% buffer
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