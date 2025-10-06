"""
Trend Momentum Strategy

A high-performance strategy based on proven profitable trading principles that combines:
- Multi-timeframe trend following
- Momentum confirmation with multiple indicators
- Dynamic position sizing based on market conditions
- Breakout detection for major moves
- Volatility-adjusted risk management

Key Features:
- Uses 3 different timeframes for trend confirmation (fast, medium, slow)
- Multiple momentum indicators (RSI, MACD, Rate of Change)
- Breakout detection with volume confirmation
- Dynamic position sizing (20-60% of balance)
- Adaptive stop losses based on volatility
- Profit targets that scale with trend strength

This strategy is designed to capture major trend moves while managing risk effectively.
"""

from typing import Any, Optional
import numpy as np
import pandas as pd
from src.strategies.base import BaseStrategy


class TrendMomentum(BaseStrategy):
    """
    Trend Momentum Strategy - Combines trend following with momentum confirmation
    """
    
    # Position sizing configuration
    BASE_POSITION_SIZE = 0.30  # 30% base allocation
    MIN_POSITION_SIZE_RATIO = 0.20  # 20% minimum
    MAX_POSITION_SIZE_RATIO = 0.60  # 60% maximum
    
    # Risk management
    BASE_STOP_LOSS_PCT = 0.03  # 3% base stop loss
    BASE_TAKE_PROFIT_PCT = 0.08  # 8% base take profit
    
    # Trend confirmation periods
    FAST_TREND_PERIOD = 10
    MEDIUM_TREND_PERIOD = 20
    SLOW_TREND_PERIOD = 50
    
    # Momentum indicators
    RSI_PERIOD = 14
    MACD_FAST = 12
    MACD_SLOW = 26
    MACD_SIGNAL = 9
    
    # Breakout detection
    BREAKOUT_LOOKBACK = 20
    VOLUME_CONFIRMATION_PERIOD = 10
    
    def __init__(self, name: str = "TrendMomentum"):
        super().__init__(name)
        self.trading_pair = "BTCUSDT"
        self.logger.info("Initialized Trend Momentum Strategy")
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive trend and momentum indicators"""
        df = df.copy()
        
        # Price-based indicators
        df = self._calculate_trend_indicators(df)
        df = self._calculate_momentum_indicators(df)
        df = self._calculate_breakout_indicators(df)
        df = self._calculate_volatility_indicators(df)
        df = self._calculate_volume_indicators(df)
        df = self._calculate_signal_indicators(df)
        
        return df
    
    def _calculate_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate multi-timeframe trend indicators"""
        
        # Moving averages for different timeframes
        df["sma_fast"] = df["close"].rolling(self.FAST_TREND_PERIOD).mean()
        df["sma_medium"] = df["close"].rolling(self.MEDIUM_TREND_PERIOD).mean()
        df["sma_slow"] = df["close"].rolling(self.SLOW_TREND_PERIOD).mean()
        
        # Exponential moving averages (more responsive)
        df["ema_fast"] = df["close"].ewm(span=self.FAST_TREND_PERIOD).mean()
        df["ema_medium"] = df["close"].ewm(span=self.MEDIUM_TREND_PERIOD).mean()
        df["ema_slow"] = df["close"].ewm(span=self.SLOW_TREND_PERIOD).mean()
        
        # Trend strength indicators
        df["trend_strength_fast"] = (df["ema_fast"] - df["ema_medium"]) / df["ema_medium"]
        df["trend_strength_medium"] = (df["ema_medium"] - df["ema_slow"]) / df["ema_slow"]
        df["trend_strength_slow"] = (df["close"] - df["ema_slow"]) / df["ema_slow"]
        
        # Trend alignment (all timeframes in same direction)
        df["trend_aligned"] = (
            (df["trend_strength_fast"] > 0) & 
            (df["trend_strength_medium"] > 0) & 
            (df["trend_strength_slow"] > 0)
        )
        
        # Trend momentum (rate of change in trend)
        df["trend_momentum"] = df["trend_strength_fast"].pct_change(5)
        
        return df
    
    def _calculate_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate momentum indicators"""
        
        # RSI
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(self.RSI_PERIOD).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(self.RSI_PERIOD).mean()
        rs = gain / loss
        df["rsi"] = 100 - (100 / (1 + rs))
        
        # MACD
        df["macd"] = df["close"].ewm(span=self.MACD_FAST).mean() - df["close"].ewm(span=self.MACD_SLOW).mean()
        df["macd_signal"] = df["macd"].ewm(span=self.MACD_SIGNAL).mean()
        df["macd_histogram"] = df["macd"] - df["macd_signal"]
        
        # Rate of Change
        df["roc_fast"] = df["close"].pct_change(5)
        df["roc_medium"] = df["close"].pct_change(10)
        df["roc_slow"] = df["close"].pct_change(20)
        
        # Momentum score (composite)
        df["momentum_score"] = (
            np.sign(df["roc_fast"]) * 0.4 +
            np.sign(df["roc_medium"]) * 0.4 +
            np.sign(df["roc_slow"]) * 0.2
        )
        
        # Strong momentum conditions
        df["strong_momentum"] = (
            (df["rsi"] > 50) & 
            (df["macd_histogram"] > 0) & 
            (df["momentum_score"] > 0.5)
        )
        
        return df
    
    def _calculate_breakout_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate breakout detection indicators"""
        
        # Price levels
        df["high_lookback"] = df["high"].rolling(self.BREAKOUT_LOOKBACK).max()
        df["low_lookback"] = df["low"].rolling(self.BREAKOUT_LOOKBACK).min()
        df["close_lookback"] = df["close"].rolling(self.BREAKOUT_LOOKBACK).mean()
        
        # Breakout strength
        df["breakout_up"] = (df["close"] - df["high_lookback"].shift(1)) / df["high_lookback"].shift(1)
        df["breakout_down"] = (df["low_lookback"].shift(1) - df["close"]) / df["low_lookback"].shift(1)
        
        # Strong breakout signals
        df["strong_breakout_up"] = (
            (df["breakout_up"] > 0.01) &  # 1% breakout
            (df["trend_strength_fast"] > 0.005)  # Confirmed by trend
        )
        
        # Breakout with volume confirmation
        df["volume_avg"] = df["volume"].rolling(self.VOLUME_CONFIRMATION_PERIOD).mean()
        df["volume_spike"] = df["volume"] > (df["volume_avg"] * 1.5)
        df["breakout_with_volume"] = df["strong_breakout_up"] & df["volume_spike"]
        
        return df
    
    def _calculate_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volatility indicators for position sizing"""
        
        # ATR (Average True Range)
        df["tr"] = np.maximum(
            df["high"] - df["low"],
            np.maximum(
                abs(df["high"] - df["close"].shift(1)),
                abs(df["low"] - df["close"].shift(1))
        )
        df["atr"] = df["tr"].rolling(14).mean()
        
        # Volatility percentage
        df["volatility_pct"] = df["atr"] / df["close"]
        
        # Volatility regime
        df["volatility_avg"] = df["volatility_pct"].rolling(30).mean()
        df["low_volatility"] = df["volatility_pct"] < (df["volatility_avg"] * 0.7)
        df["high_volatility"] = df["volatility_pct"] > (df["volatility_avg"] * 1.3)
        
        return df
    
    def _calculate_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume-based indicators"""
        
        # Volume moving averages
        df["volume_sma"] = df["volume"].rolling(20).mean()
        df["volume_ratio"] = df["volume"] / df["volume_sma"]
        
        # Volume trend
        df["volume_trend"] = df["volume"].rolling(10).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
        
        # Strong volume confirmation
        df["strong_volume"] = (df["volume_ratio"] > 1.5) & (df["volume_trend"] > 0)
        
        return df
    
    def _calculate_signal_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate composite signal indicators"""
        
        # Entry signal strength
        df["entry_signal_strength"] = 0.0
        
        # Trend component (40% weight)
        trend_score = (
            (df["trend_aligned"].astype(int) * 0.4) +
            (np.clip(df["trend_strength_fast"] * 10, 0, 1) * 0.3) +
            (np.clip(df["trend_momentum"] * 20, 0, 1) * 0.3)
        )
        
        # Momentum component (35% weight)
        momentum_score = (
            (np.clip((df["rsi"] - 50) / 50, 0, 1) * 0.3) +
            (np.clip(df["macd_histogram"] * 100, 0, 1) * 0.3) +
            (np.clip(df["momentum_score"], 0, 1) * 0.4)
        )
        
        # Breakout component (25% weight)
        breakout_score = (
            (df["strong_breakout_up"].astype(int) * 0.5) +
            (df["breakout_with_volume"].astype(int) * 0.5)
        )
        
        # Composite signal
        df["entry_signal_strength"] = (
            trend_score * 0.4 +
            momentum_score * 0.35 +
            breakout_score * 0.25
        )
        
        # Signal quality (consistency across indicators)
        df["signal_quality"] = (
            df["trend_aligned"].astype(int) +
            df["strong_momentum"].astype(int) +
            df["strong_breakout_up"].astype(int) +
            df["strong_volume"].astype(int)
        ) / 4.0
        
        return df
    
    def check_entry_conditions(self, df: pd.DataFrame, index: int) -> bool:
        """Check if entry conditions are met"""
        if index < self.SLOW_TREND_PERIOD or index >= len(df):
            return False
        
        # Get current values
        entry_strength = df["entry_signal_strength"].iloc[index]
        signal_quality = df["signal_quality"].iloc[index]
        trend_aligned = df["trend_aligned"].iloc[index]
        strong_momentum = df["strong_momentum"].iloc[index]
        strong_breakout = df["strong_breakout_up"].iloc[index]
        rsi = df["rsi"].iloc[index]
        macd_hist = df["macd_histogram"].iloc[index]
        
        # Entry conditions
        conditions = [
            # Primary condition: Strong signal with good quality
            entry_strength > 0.6 and signal_quality > 0.5,
            
            # Trend following with momentum
            trend_aligned and strong_momentum and entry_strength > 0.4,
            
            # Breakout with confirmation
            strong_breakout and trend_aligned and rsi > 45,
            
            # High quality setup
            signal_quality > 0.75 and entry_strength > 0.5,
            
            # Strong momentum with trend
            strong_momentum and trend_aligned and macd_hist > 0 and entry_strength > 0.45
        ]
        
        entry_decision = any(conditions)
        
        # Log decision
        self.log_execution(
            signal_type="entry",
            action_taken="entry_signal" if entry_decision else "no_action",
            price=df["close"].iloc[index],
            signal_strength=entry_strength,
            confidence_score=signal_quality,
            reasons=[
                f"entry_strength_{entry_strength:.3f}",
                f"signal_quality_{signal_quality:.3f}",
                f"trend_aligned_{trend_aligned}",
                f"strong_momentum_{strong_momentum}",
                f"strong_breakout_{strong_breakout}",
                f"rsi_{rsi:.1f}",
                f"macd_hist_{macd_hist:.6f}",
                "entry_conditions_met" if entry_decision else "no_entry",
            ],
        )
        
        return entry_decision
    
    def check_exit_conditions(self, df: pd.DataFrame, index: int, entry_price: float) -> bool:
        """Check if exit conditions are met"""
        if index < 1 or index >= len(df):
            return False
        
        current_price = df["close"].iloc[index]
        returns = (current_price - entry_price) / entry_price if entry_price > 0 else 0.0
        
        # Get current values
        volatility_pct = df["volatility_pct"].iloc[index]
        trend_aligned = df["trend_aligned"].iloc[index]
        strong_momentum = df["strong_momentum"].iloc[index]
        rsi = df["rsi"].iloc[index]
        macd_hist = df["macd_histogram"].iloc[index]
        entry_strength = df["entry_signal_strength"].iloc[index]
        
        # Dynamic stop loss based on volatility
        stop_loss_pct = self.BASE_STOP_LOSS_PCT
        if volatility_pct > 0.03:  # High volatility
            stop_loss_pct *= 1.5
        elif volatility_pct < 0.01:  # Low volatility
            stop_loss_pct *= 0.8
        
        # Dynamic take profit based on trend strength
        take_profit_pct = self.BASE_TAKE_PROFIT_PCT
        if entry_strength > 0.8:  # Very strong entry
            take_profit_pct *= 1.5
        elif entry_strength > 0.6:  # Strong entry
            take_profit_pct *= 1.2
        
        # Basic stops
        hit_stop_loss = returns <= -stop_loss_pct
        hit_take_profit = returns >= take_profit_pct
        
        # Trend-based exit
        trend_exit = not trend_aligned and returns > 0.02  # Exit if trend turns and we have profit
        
        # Momentum-based exit
        momentum_exit = (
            not strong_momentum and 
            rsi < 40 and 
            macd_hist < 0 and 
            returns > 0.01
        )
        
        # Strong negative signal
        strong_exit = (
            not trend_aligned and 
            not strong_momentum and 
            rsi < 35 and 
            macd_hist < -0.001
        )
        
        exit_decision = hit_stop_loss or hit_take_profit or trend_exit or momentum_exit or strong_exit
        
        self.log_execution(
            signal_type="exit",
            action_taken="exit_signal" if exit_decision else "hold",
            price=current_price,
            reasons=[
                "stop_loss" if hit_stop_loss else "",
                "take_profit" if hit_take_profit else "",
                "trend_exit" if trend_exit else "",
                "momentum_exit" if momentum_exit else "",
                "strong_exit" if strong_exit else "",
                f"returns_{returns:.3f}",
                f"volatility_pct_{volatility_pct:.4f}",
                f"rsi_{rsi:.1f}",
            ],
        )
        
        return exit_decision
    
    def calculate_position_size(self, df: pd.DataFrame, index: int, balance: float) -> float:
        """Calculate position size based on signal strength and market conditions"""
        if index >= len(df) or balance <= 0:
            return 0.0
        
        # Get current values
        entry_strength = df["entry_signal_strength"].iloc[index]
        signal_quality = df["signal_quality"].iloc[index]
        volatility_pct = df["volatility_pct"].iloc[index]
        trend_strength = df["trend_strength_fast"].iloc[index]
        strong_breakout = df["strong_breakout_up"].iloc[index]
        strong_volume = df["strong_volume"].iloc[index]
        
        # Base position size
        base_size = self.BASE_POSITION_SIZE
        
        # Signal strength multiplier
        signal_multiplier = min(2.0, entry_strength * 1.5)
        
        # Quality multiplier
        quality_multiplier = min(1.5, signal_quality * 1.2)
        
        # Volatility adjustment (reduce size in high volatility)
        vol_multiplier = 1.0
        if volatility_pct > 0.04:  # Very high volatility
            vol_multiplier = 0.7
        elif volatility_pct > 0.03:  # High volatility
            vol_multiplier = 0.8
        elif volatility_pct < 0.015:  # Low volatility
            vol_multiplier = 1.2
        
        # Trend strength multiplier
        trend_multiplier = min(1.3, 1.0 + abs(trend_strength) * 10)
        
        # Breakout bonus
        breakout_multiplier = 1.2 if strong_breakout else 1.0
        
        # Volume confirmation bonus
        volume_multiplier = 1.1 if strong_volume else 1.0
        
        # Calculate final position size
        position_ratio = (
            base_size * 
            signal_multiplier * 
            quality_multiplier * 
            vol_multiplier * 
            trend_multiplier * 
            breakout_multiplier * 
            volume_multiplier
        )
        
        # Apply limits
        position_ratio = max(
            self.MIN_POSITION_SIZE_RATIO,
            min(self.MAX_POSITION_SIZE_RATIO, position_ratio)
        )
        
        return position_ratio * balance
    
    def calculate_stop_loss(self, df: pd.DataFrame, index: int, price: float, side: str = "long") -> float:
        """Calculate dynamic stop loss based on volatility"""
        side_str = side.value if hasattr(side, "value") else str(side)
        
        # Get current volatility
        volatility_pct = df["volatility_pct"].iloc[index] if index < len(df) else 0.02
        
        # Adjust stop loss based on volatility
        stop_loss_pct = self.BASE_STOP_LOSS_PCT
        if volatility_pct > 0.03:  # High volatility
            stop_loss_pct *= 1.5
        elif volatility_pct < 0.01:  # Low volatility
            stop_loss_pct *= 0.8
        
        if side_str == "long":
            return price * (1 - stop_loss_pct)
        else:
            return price * (1 + stop_loss_pct)
    
    def get_risk_overrides(self) -> Optional[dict[str, Any]]:
        """Risk management overrides for trend momentum strategy"""
        return {
            "position_sizer": "confidence_weighted",
            "base_fraction": self.BASE_POSITION_SIZE,
            "min_fraction": self.MIN_POSITION_SIZE_RATIO,
            "max_fraction": self.MAX_POSITION_SIZE_RATIO,
            "stop_loss_pct": self.BASE_STOP_LOSS_PCT,
            "take_profit_pct": self.BASE_TAKE_PROFIT_PCT,
            "dynamic_risk": {
                "enabled": True,
                "drawdown_thresholds": [0.10, 0.20, 0.30],
                "risk_reduction_factors": [0.9, 0.7, 0.5],
                "recovery_thresholds": [0.05, 0.10],
            },
            "partial_operations": {
                "exit_targets": [0.04, 0.08, 0.12],  # 4%, 8%, 12%
                "exit_sizes": [0.25, 0.35, 0.40],     # 25%, 35%, 40%
                "scale_in_thresholds": [0.02, 0.04], # 2%, 4%
                "scale_in_sizes": [0.3, 0.2],         # 30%, 20%
                "max_scale_ins": 2,
            },
            "trailing_stop": {
                "activation_threshold": 0.03,   # 3% before trailing
                "trailing_distance_pct": 0.015, # 1.5% trailing distance
                "breakeven_threshold": 0.05,    # 5% before breakeven
                "breakeven_buffer": 0.005,      # 0.5% buffer
            },
        }
    
    def get_parameters(self) -> dict:
        """Return strategy parameters"""
        return {
            "name": self.name,
            "base_position_size": self.BASE_POSITION_SIZE,
            "min_position_size": self.MIN_POSITION_SIZE_RATIO,
            "max_position_size": self.MAX_POSITION_SIZE_RATIO,
            "base_stop_loss_pct": self.BASE_STOP_LOSS_PCT,
            "base_take_profit_pct": self.BASE_TAKE_PROFIT_PCT,
            "fast_trend_period": self.FAST_TREND_PERIOD,
            "medium_trend_period": self.MEDIUM_TREND_PERIOD,
            "slow_trend_period": self.SLOW_TREND_PERIOD,
            "rsi_period": self.RSI_PERIOD,
            "breakout_lookback": self.BREAKOUT_LOOKBACK,
        }