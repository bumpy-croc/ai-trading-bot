from strategies.base import BaseStrategy
import pandas as pd
import numpy as np
import talib
import logging

logger = logging.getLogger(__name__)

class AdaptiveStrategy2(BaseStrategy):
    def __init__(self, name="AdaptiveStrategy2"):
        super().__init__(name)
        
        # Risk parameters
        self.stop_loss_pct = 0.015  # 1.5% stop loss
        self.take_profit_pct = 0.025  # 2.5% take profit
        self.base_position_size = 0.15  # 15% of balance
        self.max_position_size = 0.25  # 25% of balance
        
        # Technical indicator parameters
        self.short_window = 9
        self.long_window = 21
        self.signal_window = 9
        self.rsi_period = 14
        self.rsi_oversold = 30
        self.rsi_overbought = 70
        
        # Volatility parameters
        self.atr_period = 14
        self.volatility_window = 20
        self.atr_multiplier = 2.0
        self.min_stop_loss = 0.01
        self.max_stop_loss = 0.03

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate strategy-specific indicators"""
        df = df.copy()
        
        # MACD
        df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(
            df['close'], 
            fastperiod=self.short_window, 
            slowperiod=self.long_window, 
            signalperiod=self.signal_window
        )
        
        # RSI
        df['rsi'] = talib.RSI(df['close'], timeperiod=self.rsi_period)
        
        # ATR for volatility
        df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=self.atr_period)
        
        # Volatility ratio
        df['volatility'] = df['atr'] / df['close']
        df['volatility_ma'] = df['volatility'].rolling(window=self.volatility_window).mean()
        
        return df

    def calculate_position_size(self, df: pd.DataFrame, index: int, balance: float) -> float:
        """Calculate dynamic position size based on volatility"""
        if index >= len(df):
            return 0.0
            
        if balance <= 0:
            return 0.0
            
        volatility = df['volatility'].iloc[index]
        volatility_ma = df['volatility_ma'].iloc[index]
        
        # Base position size as percentage of balance
        base_size = min(balance * self.base_position_size, balance * 0.15)  # Never risk more than 15%
        
        # Adjust position size based on volatility
        if volatility > volatility_ma * 1.5:
            position_size = base_size * 0.5  # Reduce position in high volatility
        elif volatility < volatility_ma * 0.5:
            position_size = min(base_size * 1.2, balance * self.max_position_size)  # Less aggressive scaling
        else:
            position_size = base_size
            
        # Additional safety checks
        max_allowed = balance * self.max_position_size
        position_size = min(position_size, max_allowed)
        
        # Ensure minimum position size
        if position_size < balance * 0.01:  # Minimum 1% of balance
            return 0.0  # Don't trade if position would be too small
            
        return position_size

    def calculate_stop_loss(self, price: float, side: str = 'long') -> float:
        """Calculate stop loss price"""
        stop_loss_pct = self.stop_loss_pct
        
        if side == 'long':
            return price * (1 - stop_loss_pct)
        else:  # short
            return price * (1 + stop_loss_pct)

    def check_entry_conditions(self, df: pd.DataFrame, index: int) -> bool:
        """Check if entry conditions are met at the given index"""
        if index < 1 or index >= len(df):
            return False
            
        price = df['close'].iloc[index]
        rsi = df['rsi'].iloc[index]
        macd = df['macd'].iloc[index]
        macd_signal = df['macd_signal'].iloc[index]
        prev_macd = df['macd'].iloc[index-1]
        prev_macd_signal = df['macd_signal'].iloc[index-1]
        
        # MACD crossover
        macd_cross_up = prev_macd < prev_macd_signal and macd > macd_signal
        
        # RSI conditions
        rsi_buy_zone = rsi < self.rsi_oversold
        
        # Volatility check
        volatility = df['volatility'].iloc[index]
        volatility_ma = df['volatility_ma'].iloc[index]
        acceptable_volatility = volatility <= volatility_ma * 2.0  # Don't enter in extreme volatility
        
        # Combined entry conditions
        return acceptable_volatility and (macd_cross_up or rsi_buy_zone)

    def check_exit_conditions(self, df: pd.DataFrame, index: int, entry_price: float) -> bool:
        """Check if exit conditions are met at the given index"""
        if index < 1 or index >= len(df):
            return False
            
        current_price = df['close'].iloc[index]
        rsi = df['rsi'].iloc[index]
        
        # Calculate returns
        returns = (current_price - entry_price) / entry_price
        
        # Exit conditions
        hit_stop_loss = returns <= -self.stop_loss_pct
        hit_take_profit = returns >= self.take_profit_pct
        rsi_overbought = rsi > self.rsi_overbought
        
        # MACD reversal
        macd = df['macd'].iloc[index]
        macd_signal = df['macd_signal'].iloc[index]
        prev_macd = df['macd'].iloc[index-1]
        prev_macd_signal = df['macd_signal'].iloc[index-1]
        macd_cross_down = prev_macd > prev_macd_signal and macd < macd_signal
        
        return hit_stop_loss or hit_take_profit or rsi_overbought or macd_cross_down

    def get_parameters(self) -> dict:
        """Return strategy parameters for logging"""
        return {
            'name': self.name,
            'stop_loss_pct': self.stop_loss_pct,
            'take_profit_pct': self.take_profit_pct,
            'base_position_size': self.base_position_size,
            'max_position_size': self.max_position_size,
            'short_window': self.short_window,
            'long_window': self.long_window,
            'signal_window': self.signal_window,
            'rsi_period': self.rsi_period,
            'rsi_oversold': self.rsi_oversold,
            'rsi_overbought': self.rsi_overbought,
            'atr_period': self.atr_period,
            'volatility_window': self.volatility_window,
            'atr_multiplier': self.atr_multiplier
        } 