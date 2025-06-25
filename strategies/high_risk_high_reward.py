from strategies.base import BaseStrategy
import pandas as pd
import numpy as np
import talib
import logging

logger = logging.getLogger(__name__)

class HighRiskHighRewardStrategy(BaseStrategy):
    def __init__(self, name="HighRiskHighRewardStrategy"):
        super().__init__(name)
        
        # Set strategy-specific trading pair - this strategy focuses on ETH
        self.trading_pair = 'ETHUSDT'
        
        # Risk parameters
        self.stop_loss_pct = 0.0005  # 0.05% stop loss
        self.take_profit_pct = 0.002  # 0.2% take profit
        self.base_position_size = 0.7  # 70% of balance
        self.max_position_size = 1.0  # 100% of balance
        
        # Technical indicator parameters
        self.short_window = 5
        self.long_window = 13
        self.signal_window = 5
        self.rsi_period = 7
        self.rsi_oversold = 20
        self.rsi_overbought = 80
        
        # Volatility parameters
        self.atr_period = 7
        self.volatility_window = 10
        self.atr_multiplier = 3.0

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
        
        # ATR for volatility-based position sizing
        df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=self.atr_period)
        
        # Moving Averages for trend confirmation
        df['50_MA'] = df['close'].rolling(window=50).mean()
        df['200_MA'] = df['close'].rolling(window=200).mean()
        
        return df

    def calculate_position_size(self, df: pd.DataFrame, index: int, balance: float) -> float:
        """Calculate dynamic position size based on volatility"""
        atr = df['atr'].iloc[index]
        volatility_adjusted_size = self.base_position_size * (1 + atr / df['close'].iloc[index])
        return min(volatility_adjusted_size, self.max_position_size) * balance

    def calculate_stop_loss(self, df: pd.DataFrame, index: int, price: float, side: str = 'long') -> float:
        """Calculate stop loss price"""
        # Implement trailing stop logic
        if side == 'long':
            return max(price * (1 - self.stop_loss_pct), price - df['atr'].iloc[index] * self.atr_multiplier)
        else:  # short
            return min(price * (1 + self.stop_loss_pct), price + df['atr'].iloc[index] * self.atr_multiplier)

    def check_entry_conditions(self, df: pd.DataFrame, index: int) -> bool:
        """Check if entry conditions are met at the given index"""
        if index < 1 or index >= len(df):
            return False
            
        price = df['close'].iloc[index]
        rsi = df['rsi'].iloc[index]
        macd_hist = df['macd_hist'].iloc[index]
        
        # Trend confirmation
        trend_up = df['50_MA'].iloc[index] > df['200_MA'].iloc[index]
        
        # Entry condition: MACD histogram positive, RSI oversold, and trend up
        return macd_hist > 0 and rsi < self.rsi_oversold and trend_up

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
        
        # Trailing stop logic
        trailing_stop = current_price <= self.calculate_stop_loss(df, index, current_price, 'long')
        
        return trailing_stop or hit_stop_loss or hit_take_profit or rsi_overbought or macd_cross_down

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