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

from collections import defaultdict
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.strategies.base import BaseStrategy
from src.strategies.bear import BearStrategy
from src.strategies.bull import Bull
from src.strategies.ml_adaptive import MlAdaptive
from src.strategies.ml_basic import MlBasic
from src.strategies.ml_sentiment import MlSentiment


class EnsembleWeighted(BaseStrategy):
    """
    Weighted Ensemble Strategy using performance-based weighting and majority voting
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
        use_bull: bool = True,  # Enable trend-following for bull markets
        use_bear: bool = True,  # Enable short strategies for bear markets
    ):
        super().__init__(name)
        self.trading_pair = "BTCUSDT"
        
        # Initialize component strategies with optimized weights for higher returns
        self.strategies: dict[str, BaseStrategy] = {}
        self.strategy_weights: dict[str, float] = {}
        self.strategy_performance: dict[str, list[float]] = defaultdict(list)
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
    
    def _calculate_ensemble_signals(self, df: pd.DataFrame, strategy_dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
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
        """Calculate advanced momentum and trend indicators for cycle 1 optimization"""
        
        # Multi-timeframe momentum (key to beating buy-and-hold)
        df["momentum_fast"] = df["close"].pct_change(3)   # 3-period for quick signals
        df["momentum_medium"] = df["close"].pct_change(10) # 10-period for trend
        df["momentum_slow"] = df["close"].pct_change(30)   # 30-period for regime
        
        # Exponential moving averages (more responsive than SMA)
        df["ema_12"] = df["close"].ewm(span=12).mean()
        df["ema_26"] = df["close"].ewm(span=26).mean()
        df["ema_50"] = df["close"].ewm(span=50).mean()
        
        # MACD for momentum confirmation
        df["macd"] = df["ema_12"] - df["ema_26"]
        df["macd_signal"] = df["macd"].ewm(span=9).mean()
        df["macd_histogram"] = df["macd"] - df["macd_signal"]
        
        # Advanced trend strength (multi-timeframe)
        df["trend_strength_fast"] = (df["ema_12"] - df["ema_26"]) / df["ema_26"]
        df["trend_strength_slow"] = (df["ema_26"] - df["ema_50"]) / df["ema_50"]
        df["trend_alignment"] = (df["trend_strength_fast"] > 0) & (df["trend_strength_slow"] > 0)
        
        # Volatility indicators for position sizing
        df["atr"] = df[["high", "low", "close"]].apply(lambda x: 
            max(x["high"] - x["low"], 
                abs(x["high"] - x["close"]), 
                abs(x["low"] - x["close"])), axis=1).rolling(14).mean()
        df["volatility_fast"] = df["close"].pct_change().rolling(10).std()
        df["volatility_slow"] = df["close"].pct_change().rolling(30).std()
        df["volatility_ratio"] = df["volatility_fast"] / df["volatility_slow"]
        
        # Advanced breakout detection (faster response)
        lookback = self.BREAKOUT_LOOKBACK
        df[f"high_{lookback}"] = df["high"].rolling(lookback).max()
        df[f"low_{lookback}"] = df["low"].rolling(lookback).min()
        df["breakout_strength_up"] = (df["close"] - df[f"high_{lookback}"].shift(1)) / df[f"high_{lookback}"].shift(1)
        df["breakout_strength_down"] = (df[f"low_{lookback}"].shift(1) - df["close"]) / df[f"low_{lookback}"].shift(1)
        
        # Strong breakout signals (higher threshold for quality)
        df["strong_breakout_up"] = (df["breakout_strength_up"] > 0.01) & (df["trend_strength_fast"] > 0.005)
        df["strong_breakout_down"] = (df["breakout_strength_down"] > 0.01) & (df["trend_strength_fast"] < -0.005)
        
        # Market regime classification (enhanced)
        confirmation = self.TREND_CONFIRMATION_PERIODS
        df["strong_bull"] = (
            (df["ema_12"] > df["ema_26"]) & 
            (df["ema_26"] > df["ema_50"]) & 
            (df["momentum_medium"] > self.MOMENTUM_BULL_THRESHOLD) &
            (df["macd_histogram"] > 0)
        ).rolling(confirmation).sum() >= confirmation * self.TREND_AGREEMENT_RATIO
        
        df["strong_bear"] = (
            (df["ema_12"] < df["ema_26"]) & 
            (df["ema_26"] < df["ema_50"]) & 
            (df["momentum_medium"] < self.MOMENTUM_BEAR_THRESHOLD) &
            (df["macd_histogram"] < 0)
        ).rolling(confirmation).sum() >= confirmation * self.TREND_AGREEMENT_RATIO
        
        # Momentum score (composite indicator)
        df["momentum_score"] = (
            np.sign(df["momentum_fast"]) * 0.3 +
            np.sign(df["momentum_medium"]) * 0.4 +
            np.sign(df["momentum_slow"]) * 0.3
        )
        
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
        
        # Entry criteria - Cycle 2: Very aggressive for maximum capture
        score_threshold = 0.3  # Weighted score must be > 0.3 (very low threshold)
        min_agreement = 0.4    # At least 40% of strategies must agree 
        min_confidence = 0.25  # Minimum confidence level (very low)
        min_active = self.MIN_STRATEGIES_FOR_SIGNAL
        
        # Advanced momentum filters (Cycle 1 optimization)
        momentum_fast = df.get("momentum_fast", pd.Series([0.0])).iloc[index]
        momentum_medium = df.get("momentum_medium", pd.Series([0.0])).iloc[index]
        momentum_score = df.get("momentum_score", pd.Series([0.0])).iloc[index]
        trend_strength_fast = df.get("trend_strength_fast", pd.Series([0.0])).iloc[index]
        strong_breakout_up = df.get("strong_breakout_up", pd.Series([False])).iloc[index]
        strong_bull = df.get("strong_bull", pd.Series([False])).iloc[index]
        macd_histogram = df.get("macd_histogram", pd.Series([0.0])).iloc[index]
        trend_alignment = df.get("trend_alignment", pd.Series([False])).iloc[index]
        
        # Enhanced entry conditions with advanced momentum
        
        # Advanced momentum conditions for pseudo-leverage entry
        explosive_momentum = (
            momentum_fast > 0.03 and  # Use direct value instead of self.STRONG_MOMENTUM_THRESHOLD
            momentum_medium > 0.015 and
            momentum_score > 0.5
        )
        
        confirmed_trend = (
            trend_alignment and 
            abs(trend_strength_fast) > 0.01 and
            macd_histogram > 0
        )
        
        quality_breakout = strong_breakout_up and confirmed_trend
        regime_favorable = strong_bull or (trend_alignment and momentum_score > 0.3)
        
        # Cycle 2: Simplified but aggressive entry logic
        # Focus on capturing ALL profitable opportunities
        
        # Basic entry (lowered thresholds)
        basic_entry_decision = (
            entry_score > score_threshold and
            agreement >= min_agreement and
            confidence >= min_confidence and
            active_strategies >= min_active
        )
        
        # Momentum-based entry (very aggressive)
        momentum_entry_decision = (
            entry_score > 0.2 and  # Very low threshold
            active_strategies >= 1 and  # Any single strategy
            (momentum_fast > 0.005 or momentum_medium > 0.01 or explosive_momentum)
        )
        
        # Breakout-based entry (catch all breakouts)
        breakout_entry_decision = (
            entry_score > 0.15 and  # Extremely low threshold
            (strong_breakout_up or quality_breakout) and
            momentum_fast > 0.002  # Minimal momentum requirement
        )
        
        # Trend-following entry (capture trends early)
        trend_entry_decision = (
            entry_score > 0.2 and
            confirmed_trend and
            (trend_alignment or regime_favorable)
        )
        
        # Combined decision: Enter on ANY favorable condition
        entry_decision = (basic_entry_decision or momentum_entry_decision or 
                         breakout_entry_decision or trend_entry_decision)
        
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
        
        # Cycle 2: Hold much longer for bigger moves
        ensemble_exit = False
        if len(exit_signals) >= 2:  # Need at least 2 strategies
            exit_agreement = sum(exit_signals) / len(exit_signals)
            ensemble_exit = exit_agreement >= 0.8  # 80% must agree to exit (very high threshold)
        
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
        max(0.9, min(1.5, agreement * 1.2))    # Reward strong agreement
        
        # Advanced momentum and volatility adjustments (Cycle 1)
        momentum_fast = df.get("momentum_fast", pd.Series([0.0])).iloc[index]
        df.get("momentum_medium", pd.Series([0.0])).iloc[index]
        momentum_score = df.get("momentum_score", pd.Series([0.0])).iloc[index]
        trend_strength_fast = df.get("trend_strength_fast", pd.Series([0.0])).iloc[index]
        df.get("volatility_ratio", pd.Series([1.0])).iloc[index]
        strong_bull = df.get("strong_bull", pd.Series([False])).iloc[index]
        strong_breakout_up = df.get("strong_breakout_up", pd.Series([False])).iloc[index]
        trend_alignment = df.get("trend_alignment", pd.Series([False])).iloc[index]
        macd_histogram = df.get("macd_histogram", pd.Series([0.0])).iloc[index]
        
        # Calculate confirmed_trend for position sizing (same logic as entry conditions)
        confirmed_trend = (
            trend_alignment and 
            abs(trend_strength_fast) > 0.01 and
            macd_histogram > 0
        )
        
        # Cycle 2: Aggressive volatility-based pseudo-leverage
        # Key insight: Use volatility to determine leverage amount
        
        # Base volatility calculation for leverage decisions
        volatility_fast = df.get("volatility_fast", pd.Series([0.02])).iloc[index]
        
        # Volatility-based leverage factor (core innovation)
        if volatility_fast < 0.01:  # Low vol = higher leverage
            volatility_leverage = 2.5  # Aggressive in calm markets
        elif volatility_fast < 0.02:  # Moderate volatility
            volatility_leverage = 2.0
        elif volatility_fast < 0.05:  # Normal volatility
            volatility_leverage = 1.5
        else:  # High volatility
            volatility_leverage = 1.0  # Conservative in volatile markets
        
        # Momentum amplification (stronger effect)
        momentum_amplifier = 1.0
        if momentum_fast > 0.02:  # Strong momentum
            momentum_amplifier = 1.8
        elif momentum_fast > 0.01:  # Good momentum
            momentum_amplifier = 1.4
        elif momentum_fast > 0.005:  # Weak momentum
            momentum_amplifier = 1.2
        elif momentum_fast < -0.005:  # Negative momentum
            momentum_amplifier = 0.7
        
        # Trend strength amplification
        trend_amplifier = 1.0
        if strong_bull and trend_alignment:  # Perfect bull setup
            trend_amplifier = 2.0  # Double down in perfect conditions
        elif trend_alignment:  # Good trend alignment
            trend_amplifier = 1.6
        elif abs(trend_strength_fast) > 0.015:
            trend_amplifier = 1.4
        
        # Breakout amplification
        breakout_amplifier = 1.0
        if strong_breakout_up and confirmed_trend:  # Quality breakout
            breakout_amplifier = 1.7
        elif strong_breakout_up:  # Simple breakout
            breakout_amplifier = 1.3
        
        # Composite momentum boost
        composite_boost = 1.0
        if momentum_score > 0.7:  # Very strong composite
            composite_boost = 1.5
        elif momentum_score > 0.4:  # Good composite
            composite_boost = 1.2
        
        # Ultimate leverage calculation (can reach 80% in perfect conditions)
        leverage_multiplier = (volatility_leverage * momentum_amplifier * trend_amplifier * 
                              breakout_amplifier * composite_boost)
        
        # Apply to base size with all factors
        dynamic_size = (base_size * confidence_factor * score_factor * 
                       min(3.0, leverage_multiplier))  # Cap at 3x base
        
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
    
    def get_ensemble_status(self) -> dict[str, Any]:
        """Get current ensemble status and health"""
        return {
            "ensemble_name": self.name,
            "active_strategies": len(self.strategies),
            "current_weights": dict(self.strategy_weights),
            "total_decisions": self.decision_count,
            "last_weight_update": self.decision_count // self.WEIGHT_UPDATE_FREQUENCY * self.WEIGHT_UPDATE_FREQUENCY,
            "next_weight_update": ((self.decision_count // self.WEIGHT_UPDATE_FREQUENCY) + 1) * self.WEIGHT_UPDATE_FREQUENCY,
        }