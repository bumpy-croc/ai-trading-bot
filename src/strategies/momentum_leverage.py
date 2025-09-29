"""
Momentum Leverage Strategy

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

import numpy as np
import pandas as pd

from src.strategies.base import BaseStrategy
from src.strategies.bull import Bull
from src.strategies.ml_basic import MlBasic


class MomentumLeverage(BaseStrategy):
    """
    Pure momentum strategy with pseudo-leverage to beat buy-and-hold
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
        super().__init__(name)
        self.trading_pair = "BTCUSDT"
        
        # Core components (simplified)
        self.ml_strategy = MlBasic(name="ML_Component")
        self.bull_strategy = Bull(name="Bull_Component")
        
        self.logger.info("Initialized momentum leverage strategy")
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate momentum and trend indicators"""
        df = df.copy()
        
        # Get indicators from component strategies
        df = self.ml_strategy.calculate_indicators(df)
        
        # Calculate our momentum indicators
        # Multi-timeframe momentum
        df["momentum_3"] = df["close"].pct_change(3)
        df["momentum_7"] = df["close"].pct_change(7)
        df["momentum_20"] = df["close"].pct_change(20)
        
        # Exponential moving averages for trend
        df["ema_12"] = df["close"].ewm(span=12).mean()
        df["ema_26"] = df["close"].ewm(span=26).mean()
        df["ema_50"] = df["close"].ewm(span=50).mean()
        
        # Trend strength
        df["trend_strength"] = (df["ema_12"] - df["ema_26"]) / df["ema_26"]
        df["long_trend"] = (df["ema_26"] - df["ema_50"]) / df["ema_50"]
        
        # Volatility for position sizing
        df["volatility"] = df["close"].pct_change().rolling(20).std()
        
        # Bull market detection
        df["bull_confirmed"] = (
            (df["ema_12"] > df["ema_26"]) &
            (df["ema_26"] > df["ema_50"]) &
            (df["momentum_7"] > 0.01)
        )
        
        # Strong momentum signal
        df["strong_momentum"] = (
            (df["momentum_3"] > self.MOMENTUM_ENTRY_THRESHOLD) &
            (df["momentum_7"] > 0.01) &
            (df["trend_strength"] > 0.005)
        )
        
        # Breakout detection
        df["high_20"] = df["high"].rolling(self.TREND_LOOKBACK).max()
        df["breakout"] = df["close"] > df["high_20"].shift(1)
        
        return df
    
    def check_entry_conditions(self, df: pd.DataFrame, index: int) -> bool:
        """Aggressive entry conditions to capture all profitable moves"""
        if index < 50 or index >= len(df):
            return False
        
        # Get current conditions
        momentum_3 = df["momentum_3"].iloc[index]
        momentum_7 = df["momentum_7"].iloc[index]
        trend_strength = df["trend_strength"].iloc[index]
        bull_confirmed = df["bull_confirmed"].iloc[index]
        strong_momentum = df["strong_momentum"].iloc[index]
        breakout = df["breakout"].iloc[index]
        
        # ML prediction (if available)
        ml_pred = df.get("onnx_pred", pd.Series([np.nan])).iloc[index]
        current_price = df["close"].iloc[index]
        ml_bullish = False
        if not pd.isna(ml_pred) and current_price > 0:
            ml_return = (ml_pred - current_price) / current_price
            ml_bullish = ml_return > 0.005  # ML predicts 0.5%+ gain
        
        # Ultra-aggressive entry conditions (Cycle 3)
        conditions = [
            # Condition 1: Any positive momentum in bull market
            bull_confirmed and momentum_3 > 0.005,
            
            # Condition 2: Breakout (any size)
            breakout and momentum_3 > 0.003,
            
            # Condition 3: ML prediction (any positive)
            ml_bullish and momentum_7 > 0.002,
            
            # Condition 4: Pure momentum (lower threshold)
            momentum_3 > self.MOMENTUM_ENTRY_THRESHOLD and trend_strength > 0.005,
            
            # Condition 5: Strong momentum regardless of trend
            momentum_3 > self.STRONG_MOMENTUM_THRESHOLD,
            
            # Condition 6: Bull market with any momentum
            bull_confirmed and momentum_7 > 0.003,
            
            # Condition 7: Trend alignment with weak momentum
            trend_strength > 0.01 and momentum_7 > 0.005
        ]
        
        entry_decision = any(conditions)
        
        # Log decision
        self.log_execution(
            signal_type="entry",
            action_taken="entry_signal" if entry_decision else "no_action",
            price=current_price,
            signal_strength=abs(momentum_3) if entry_decision else 0.0,
            confidence_score=min(1.0, abs(momentum_3) * 10),
            reasons=[
                f"momentum_3_{momentum_3:.4f}",
                f"momentum_7_{momentum_7:.4f}",
                f"trend_strength_{trend_strength:.4f}",
                f"bull_confirmed_{bull_confirmed}",
                f"strong_momentum_{strong_momentum}",
                f"breakout_{breakout}",
                f"ml_bullish_{ml_bullish}",
                "entry_conditions_met" if entry_decision else "no_entry",
            ],
        )
        
        return entry_decision
    
    def check_exit_conditions(self, df: pd.DataFrame, index: int, entry_price: float) -> bool:
        """Hold positions longer to capture full moves"""
        if index < 1 or index >= len(df):
            return False
        
        current_price = df["close"].iloc[index]
        returns = (current_price - entry_price) / entry_price if entry_price > 0 else 0.0
        
        # Basic stops
        hit_stop_loss = returns <= -self.STOP_LOSS_PCT
        hit_take_profit = returns >= self.TAKE_PROFIT_PCT
        
        # Momentum-based exit (only exit if momentum clearly turns negative)
        momentum_3 = df["momentum_3"].iloc[index]
        trend_strength = df["trend_strength"].iloc[index]
        
        # More aggressive holding - only exit on very strong negative signals
        momentum_exit = (momentum_3 < -self.STRONG_MOMENTUM_THRESHOLD and trend_strength < -0.015 and returns < 0.05)
        
        exit_decision = hit_stop_loss or hit_take_profit or momentum_exit
        
        self.log_execution(
            signal_type="exit",
            action_taken="exit_signal" if exit_decision else "hold",
            price=current_price,
            reasons=[
                "stop_loss" if hit_stop_loss else "",
                "take_profit" if hit_take_profit else "",
                "momentum_exit" if momentum_exit else "",
                f"returns_{returns:.3f}",
                f"momentum_3_{momentum_3:.4f}",
            ],
        )
        
        return exit_decision
    
    def calculate_position_size(self, df: pd.DataFrame, index: int, balance: float) -> float:
        """Aggressive position sizing based on momentum strength"""
        if index >= len(df) or balance <= 0:
            return 0.0
        
        # Get momentum and volatility
        momentum_3 = df.get("momentum_3", pd.Series([0.0])).iloc[index]
        momentum_7 = df.get("momentum_7", pd.Series([0.0])).iloc[index]
        trend_strength = df.get("trend_strength", pd.Series([0.0])).iloc[index]
        volatility = df.get("volatility", pd.Series([0.02])).iloc[index]
        bull_confirmed = df.get("bull_confirmed", pd.Series([False])).iloc[index]
        strong_momentum = df.get("strong_momentum", pd.Series([False])).iloc[index]
        
        # Base size
        base_size = self.BASE_POSITION_SIZE
        
        # Ultra-aggressive momentum sizing (Cycle 3 - key to beating buy-and-hold)
        momentum_multiplier = 1.0
        if strong_momentum and bull_confirmed:
            momentum_multiplier = 1.6  # Maximum aggression
        elif momentum_3 > self.STRONG_MOMENTUM_THRESHOLD:
            momentum_multiplier = 1.5
        elif momentum_3 > self.MOMENTUM_ENTRY_THRESHOLD:
            momentum_multiplier = 1.3
        elif momentum_3 > 0.005:
            momentum_multiplier = 1.2
        else:
            momentum_multiplier = 1.0  # Still aggressive baseline
        
        # Volatility adjustment (leverage in low vol)
        vol_multiplier = 1.0
        if volatility < 0.01:   # Very low volatility - maximum leverage
            vol_multiplier = 1.4
        elif volatility < 0.02: # Low volatility - high leverage
            vol_multiplier = 1.25
        elif volatility > 0.06: # High volatility - reduce slightly
            vol_multiplier = 0.9
        
        # Bull market leverage boost
        regime_multiplier = 1.0
        if bull_confirmed and momentum_7 > 0.02:  # Strong bull + momentum
            regime_multiplier = 1.5
        elif bull_confirmed:  # Just bull market
            regime_multiplier = 1.3
        elif momentum_7 > 0.015:  # Strong momentum alone
            regime_multiplier = 1.2
        
        # Trend confirmation boost
        trend_multiplier = 1.0
        if trend_strength > 0.02:  # Very strong trend
            trend_multiplier = 1.3
        elif trend_strength > 0.01:  # Good trend
            trend_multiplier = 1.15
        
        # Calculate dynamic size with all multipliers
        dynamic_size = (base_size * momentum_multiplier * vol_multiplier * 
                       regime_multiplier * trend_multiplier)
        
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