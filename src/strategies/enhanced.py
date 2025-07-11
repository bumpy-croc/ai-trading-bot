from strategies.base import BaseStrategy
import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

class EnhancedStrategy(BaseStrategy):
    def __init__(self, name: str = "EnhancedStrategy"):
        super().__init__(name)
        
        # Set strategy-specific trading pair
        self.trading_pair = 'BTCUSDT'
        
        # Risk Management Parameters
        self.risk_per_trade = 0.01  # 1% risk per trade
        self.max_position_size = 0.20  # Maximum 20% of balance per position
        self.base_stop_loss_pct = 0.02  # Base stop loss percentage
        self.stop_loss_atr_multiplier = 2.0  # ATR multiplier for dynamic stop loss
        self.take_profit_multiplier = 0.05  # 5% take profit target
        self.trailing_activation = 0.02  # Activate trailing stop after 2% profit
        self.trailing_distance = 0.015  # Trailing stop distance
        
        # Technical Indicators Parameters
        self.min_volume_ma = 1000  # Minimum volume moving average
        self.trend_ma_period = 20  # Trend MA period
        self.short_ma_period = 10  # Short MA period
        self.long_ma_period = 20   # Long MA period
        self.volume_ma_period = 20  # Volume MA period
        self.rsi_period = 14
        self.rsi_overbought = 80  # Made less conservative (was 75)
        self.rsi_oversold = 20    # Made less conservative (was 25)
        self.atr_period = 14
        self.min_conditions = 4  # Reduced from 6 to 4 conditions needed for entry

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators with enhanced metrics"""
        df = df.copy()
        
        # Previous indicators
        df['volume_ma'] = df['volume'].rolling(window=self.min_volume_ma).mean()
        df['trend_ma'] = df['close'].rolling(window=self.trend_ma_period).mean()
        df['short_ma'] = df['close'].rolling(window=self.short_ma_period).mean()
        
        # Enhanced RSI calculation
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # ATR calculation with safety checks
        if len(df) >= self.atr_period:
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)  # Use pandas max instead of numpy
            df['atr'] = true_range.rolling(window=self.atr_period).mean()
            # Fill NaN values with reasonable defaults
            df['atr'] = df['atr'].fillna(df['close'] * 0.01)
        else:
            # If insufficient data, use 1% of close price as ATR estimate
            df['atr'] = df['close'] * 0.01
        
        # Volume trend
        df['volume_trend'] = df['volume'].rolling(window=self.volume_ma_period).mean().pct_change()
        
        # Trend strength
        df['trend_strength'] = (df['close'] - df['trend_ma']) / df['trend_ma']
        
        # Price action confirmation
        df['body_size'] = np.abs(df['close'] - df['open']) / df['open']
        df['upper_wick'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['open']
        df['lower_wick'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['open']
        
        return df

    def calculate_position_size(self, df: pd.DataFrame, index: int, balance: float) -> float:
        """Calculate position size based on risk and volatility"""
        if index >= len(df) or balance <= 0:
            return 0.0
            
        price = df['close'].iloc[index]
        atr = df['atr'].iloc[index]
        
        # Calculate position size based on ATR and base stop loss
        stop_distance = max(
            self.base_stop_loss_pct,
            (atr * self.stop_loss_atr_multiplier) / price
        )
        
        # Calculate position size based on risk
        position_size = (self.risk_per_trade * balance) / (price * stop_distance)
        
        # Ensure position size doesn't exceed maximum
        return min(position_size, balance * self.max_position_size)

    def calculate_stop_loss(self, df: pd.DataFrame, index: int, price: float, side: str = 'long') -> float:
        """Calculate stop loss price"""
        if side == 'long':
            return price * (1 - self.base_stop_loss_pct)
        else:  # short
            return price * (1 + self.base_stop_loss_pct)

    def check_entry_conditions(self, df: pd.DataFrame, index: int) -> bool:
        """Check if entry conditions are met at the given index"""
        if index < 2:  # Need at least 2 previous candles
            return False
            
        # Get current and previous values
        close = df['close'].iloc[index]
        trend_ma = df['trend_ma'].iloc[index]
        trend_strength = df['trend_strength'].iloc[index]
        volume = df['volume'].iloc[index]
        volume_ma = df['volume_ma'].iloc[index]
        rsi = df['rsi'].iloc[index]
        short_ma = df['short_ma'].iloc[index]
        prev_high = df['high'].iloc[index-1]
        
        # Calculate changes
        volume_change = df['volume'].pct_change(3).iloc[index]
        price_change = df['close'].pct_change(3).iloc[index]
        
        # Check all conditions
        conditions = {
            'above_trend': close > trend_ma,
            'trend_strength': abs(trend_strength) > 0.01,  # Reduced from 0.02
            'volume_above_avg': volume > volume_ma * 1.1,  # Reduced from 1.2
            'volume_trend_positive': volume_change > -0.05,  # Allow slight volume decline
            'rsi_not_extreme': 25 < rsi < 75,  # Widened RSI range
            'price_momentum': price_change > 0.005,  # Reduced from 0.01
            'ma_alignment': short_ma > trend_ma * 0.995,  # Added 0.5% tolerance
            'price_action': close > prev_high * 0.995  # Added 0.5% tolerance
        }
        
        # Count how many conditions are met
        met_conditions = sum(conditions.values())
        entry_signal = met_conditions >= self.min_conditions
        
        # Log detailed condition analysis
        indicators = {
            'close': close,
            'trend_ma': trend_ma,
            'trend_strength': trend_strength,
            'volume': volume,
            'volume_ma': volume_ma,
            'rsi': rsi,
            'short_ma': short_ma,
            'volume_change': volume_change,
            'price_change': price_change
        }
        
        # Create detailed reasons list
        reasons = [
            f'conditions_met_{met_conditions}_of_{len(conditions)}_min_{self.min_conditions}',
            'entry_signal_met' if entry_signal else 'entry_signal_not_met'
        ]
        
        # Add condition-specific details
        for condition_name, condition_met in conditions.items():
            reasons.append(f'{condition_name}_{condition_met}')
        
        # Add threshold details
        reasons.extend([
            f'trend_strength_{trend_strength:.4f}_vs_0.01',
            f'volume_ratio_{volume/volume_ma:.2f}_vs_1.1',
            f'rsi_{rsi:.1f}_in_range_25_75',
            f'price_momentum_{price_change:.4f}_vs_0.005'
        ])
        
        self.log_execution(
            signal_type='entry',
            action_taken='entry_signal' if entry_signal else 'no_action',
            price=close,
            signal_strength=met_conditions / len(conditions),  # Strength as percentage of conditions met
            confidence_score=met_conditions / len(conditions),
            indicators=indicators,
            reasons=reasons,
            volume=volume,
            volatility=indicators.get('atr', 0) / close if 'atr' in df.columns else None,
            additional_context={
                'strategy_type': 'enhanced_multi_condition',
                'total_conditions': len(conditions),
                'min_conditions_required': self.min_conditions,
                'conditions_met': met_conditions
            }
        )
        
        return entry_signal

    def check_exit_conditions(self, df: pd.DataFrame, index: int, entry_price: float) -> bool:
        """Check if exit conditions are met at the given index"""
        if index < 1:
            return False
            
        # Get current values
        close = df['close'].iloc[index]
        trend_ma = df['trend_ma'].iloc[index]
        atr = df['atr'].iloc[index]
        rsi = df['rsi'].iloc[index]
        volume = df['volume'].iloc[index]
        volume_ma = df['volume_ma'].iloc[index]
        
        # Calculate returns and changes
        returns = (close - entry_price) / entry_price
        price_change = df['close'].pct_change(3).iloc[index]
        
        # Calculate dynamic stop loss
        stop_distance = max(
            self.base_stop_loss_pct,
            (atr * self.stop_loss_atr_multiplier) / close
        )
        
        # Check stop loss
        hit_stop_loss = returns <= -stop_distance
        
        # Check take profit
        hit_take_profit = returns >= self.take_profit_multiplier
        
        # Check trailing stop if activated
        if returns > self.trailing_activation:
            trailing_stop_price = close * (1 - self.trailing_distance)
            hit_trailing_stop = close <= trailing_stop_price
        else:
            hit_trailing_stop = False
        
        # Check trend reversal
        trend_reversal = (
            close < trend_ma and 
            price_change < -0.01 and 
            volume > volume_ma
        )
        
        # Check RSI extreme
        rsi_extreme = rsi > self.rsi_overbought
        
        return hit_stop_loss or hit_take_profit or hit_trailing_stop or trend_reversal or rsi_extreme

    def get_parameters(self) -> Dict:
        """Return strategy parameters for logging"""
        return {
            'name': self.name,
            'risk_per_trade': self.risk_per_trade,
            'max_position_size': self.max_position_size,
            'base_stop_loss_pct': self.base_stop_loss_pct,
            'stop_loss_atr_multiplier': self.stop_loss_atr_multiplier,
            'take_profit_multiplier': self.take_profit_multiplier,
            'trailing_activation': self.trailing_activation,
            'trailing_distance': self.trailing_distance,
            'min_volume_ma': self.min_volume_ma,
            'trend_ma_period': self.trend_ma_period,
            'short_ma_period': self.short_ma_period,
            'volume_ma_period': self.volume_ma_period,
            'rsi_period': self.rsi_period,
            'rsi_overbought': self.rsi_overbought,
            'rsi_oversold': self.rsi_oversold,
            'atr_period': self.atr_period,
            'min_conditions': self.min_conditions
        } 