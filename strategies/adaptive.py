from strategies.base import BaseStrategy
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from sklearn.preprocessing import StandardScaler

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AdaptiveStrategy(BaseStrategy):
    def __init__(self, name="AdaptiveStrategy", **config):
        super().__init__(name)
        
        # Set strategy-specific trading pair
        self.trading_pair = 'BTCUSDT'
        
        # Dynamic Risk Management - More adaptive
        self.base_risk_per_trade = config.get('base_risk_per_trade', 0.015)   # Increased from 1% to 1.5%
        self.max_risk_per_trade = config.get('max_risk_per_trade', 0.025)    # Increased from 2% to 2.5%
        self.position_size_atr_multiplier = config.get('position_size_atr_multiplier', 1.0)  # More aggressive position sizing
        self.max_position_size = config.get('max_position_size', 0.30)      # Increased from 25% to 30%
        self.max_daily_risk = config.get('max_daily_risk', 0.06)         # Increased from 5% to 6%
        
        # Adaptive Stop Loss and Take Profit - More dynamic
        self.base_stop_loss_atr = config.get('base_stop_loss_atr', 2.0)     # Increased from 1.5 to 2.0
        self.min_stop_loss_pct = config.get('min_stop_loss_pct', 0.012)    # Increased from 0.8% to 1.2%
        self.max_stop_loss_pct = config.get('max_stop_loss_pct', 0.03)     # Increased from 2.5% to 3%
        self.take_profit_levels = config.get('take_profit_levels', [        # Optimized take profit levels
            {'size': 0.3, 'target': 2.0},  # Exit 30% at 2.0x stop distance
            {'size': 0.4, 'target': 3.0},  # Exit 40% at 3.0x stop distance
            {'size': 0.3, 'target': 5.0}   # Exit 30% at 5.0x stop distance
        ])
        
        # Market Regime Detection - More sensitive
        self.volatility_lookback = config.get('volatility_lookback', 15)      # Reduced from 20 to be more responsive
        self.trend_lookback = config.get('trend_lookback', 40)           # Reduced from 50 to be more responsive
        self.regime_threshold = config.get('regime_threshold', 0.012)      # Reduced from 0.015 for earlier trend detection
        
        # Technical Indicators - Optimized periods
        self.fast_ma = config.get('fast_ma', 8)                   # Reduced from 10 for faster signals
        self.slow_ma = config.get('slow_ma', 21)                  # Kept at 21 for stability
        self.long_ma = config.get('long_ma', 50)                  # Kept at 50 for trend confirmation
        self.rsi_period = config.get('rsi_period', 14)
        self.rsi_thresholds = config.get('rsi_thresholds', {
            'low_vol': {'oversold': 40, 'overbought': 60},  # More lenient
            'high_vol': {'oversold': 30, 'overbought': 70}  # More lenient
        })
        self.volume_ma_period = config.get('volume_ma_period', 20)
        self.min_volume_multiplier = config.get('min_volume_multiplier', 1.1)   # Reduced from 1.2 for more trade opportunities
        
        # Machine Learning Features
        self.feature_periods = config.get('feature_periods', [5, 10, 20])
        self.scaler = StandardScaler()

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical and adaptive indicators"""
        df = df.copy()
        
        # Basic indicators
        for period in [self.fast_ma, self.slow_ma, self.long_ma]:
            df[f'ma_{period}'] = df['close'].rolling(window=period).mean()
        
        # Enhanced RSI with adaptive thresholds
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Volatility and trend metrics
        df['atr'] = self.calculate_atr(df)
        df['volatility'] = df['atr'] / df['close']
        df['trend_strength'] = self.calculate_trend_strength(df)
        
        # Volume analysis
        df['volume_ma'] = df['volume'].rolling(window=self.volume_ma_period).mean()
        df['volume_trend'] = df['volume'].pct_change(5)
        
        # Market regime features
        df['regime'] = self.detect_market_regime(df)
        
        # Price action features
        df['body_size'] = (df['close'] - df['open']).abs() / df['open']
        df['upper_wick'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['open']
        df['lower_wick'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['open']
        
        # Momentum features
        for period in self.feature_periods:
            df[f'momentum_{period}'] = df['close'].pct_change(period)
            df[f'volatility_{period}'] = df['close'].rolling(period).std() / df['close']
        
        return df

    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        if len(df) < period:
            # If insufficient data, return a series with reasonable defaults
            return pd.Series([df['close'].iloc[-1] * 0.01] * len(df), index=df.index)
        
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(window=period).mean()
        
        # Fill NaN values with reasonable defaults
        atr = atr.fillna(df['close'] * 0.01)  # Use 1% of close price as fallback
        
        return atr

    def calculate_trend_strength(self, df: pd.DataFrame) -> pd.Series:
        """Calculate trend strength using multiple timeframes"""
        ma_diffs = []
        for fast, slow in [(self.fast_ma, self.slow_ma), (self.slow_ma, self.long_ma)]:
            ma_diff = (df[f'ma_{fast}'] - df[f'ma_{slow}']) / df[f'ma_{slow}']
            ma_diffs.append(ma_diff)
        return pd.concat(ma_diffs, axis=1).mean(axis=1)

    def detect_market_regime(self, df: pd.DataFrame) -> pd.Series:
        """Detect market regime (trending, ranging, volatile)"""
        volatility = df['close'].rolling(self.volatility_lookback).std()
        vol_ratio = volatility / volatility.rolling(50).mean()
        trend_strength = self.calculate_trend_strength(df)
        
        conditions = [
            (vol_ratio > 1.3) & (abs(trend_strength) < self.regime_threshold),  # Volatile
            (abs(trend_strength) > self.regime_threshold * 0.5)  # Trending
        ]
        choices = ['volatile', 'trending']
        return pd.Series(np.select(conditions, choices, default='ranging'), index=df.index)

    def calculate_position_size(self, df: pd.DataFrame, index: int, balance: float) -> float:
        """Calculate adaptive position size based on market conditions"""
        if index >= len(df):
            return 0.0
            
        # Adjust risk based on market regime
        regime = df['regime'].iloc[index]
        base_risk = self.base_risk_per_trade
        
        if regime == 'trending':
            risk = base_risk * 1.2
        elif regime == 'volatile':
            risk = base_risk * 0.7
        else:  # ranging
            risk = base_risk
        
        # Ensure we don't exceed maximum risk limits
        risk = min(risk, self.max_risk_per_trade)
        
        # Calculate position size based on ATR
        atr = df['atr'].iloc[index]
        price = df['close'].iloc[index]
        position_size = (risk * balance) / (atr * self.position_size_atr_multiplier)
        
        # Ensure position size doesn't exceed maximum
        position_size = min(position_size, balance * self.max_position_size)
        
        return position_size

    def calculate_stop_loss(self, df: pd.DataFrame, index: int, price: float, side: str = 'long') -> float:
        """Calculate adaptive stop loss level"""
        if index >= len(df):
            stop_pct = self.min_stop_loss_pct
        else:
            # Use ATR for adaptive stop loss calculation
            atr = df['atr'].iloc[index] if index < len(df) and 'atr' in df.columns else price * 0.02
            stop_pct = max(self.min_stop_loss_pct, min(atr / price * self.base_stop_loss_atr, self.max_stop_loss_pct))
        
        if side == 'long':
            return price * (1 - stop_pct)
        else:  # short
            return price * (1 + stop_pct)

    def check_entry_conditions(self, df: pd.DataFrame, index: int) -> bool:
        """Check if entry conditions are met at the given index"""
        if index < 1 or index >= len(df):
            return False
            
        # Get current market conditions
        regime = df['regime'].iloc[index] if 'regime' in df.columns else 'trending'
        rsi = df['rsi'].iloc[index] if 'rsi' in df.columns else 50
        volume = df['volume'].iloc[index] if 'volume' in df.columns else 1000
        volume_ma = df['volume_ma'].iloc[index] if 'volume_ma' in df.columns else 1000
        trend_strength = df['trend_strength'].iloc[index] if 'trend_strength' in df.columns else 0.01
        
        # Check for NaN values and use defaults
        if pd.isna(regime):
            regime = 'trending'
        if pd.isna(rsi):
            rsi = 50
        if pd.isna(volume):
            volume = 1000
        if pd.isna(volume_ma):
            volume_ma = 1000
        if pd.isna(trend_strength):
            trend_strength = 0.01
        
        # Get RSI thresholds based on volatility
        rsi_thresholds = (
            self.rsi_thresholds['high_vol'] 
            if regime == 'volatile' 
            else self.rsi_thresholds['low_vol']
        )
        
        # Check trend conditions (more permissive for bull markets)
        trend_conditions = {
            'trending': trend_strength > self.regime_threshold * 0.3,  # Reduced threshold
            'ranging': abs(trend_strength) < self.regime_threshold * 2,  # More permissive
            'volatile': True  # We trade volatility breakouts
        }
        
        # Check if volume is sufficient (more permissive)
        volume_condition = volume > volume_ma * max(0.8, self.min_volume_multiplier * 0.7)
        
        # Check RSI conditions (more permissive)
        rsi_condition = rsi < rsi_thresholds['oversold'] * 1.2 or rsi > 40  # Allow more entries
        
        # Check moving average crossovers
        fast_ma = df[f'ma_{self.fast_ma}'].iloc[index] if f'ma_{self.fast_ma}' in df.columns else df['close'].iloc[index]
        slow_ma = df[f'ma_{self.slow_ma}'].iloc[index] if f'ma_{self.slow_ma}' in df.columns else df['close'].iloc[index-1]
        prev_fast_ma = df[f'ma_{self.fast_ma}'].iloc[index-1] if f'ma_{self.fast_ma}' in df.columns else df['close'].iloc[index-1]
        prev_slow_ma = df[f'ma_{self.slow_ma}'].iloc[index-1] if f'ma_{self.slow_ma}' in df.columns else df['close'].iloc[index-2] if index > 1 else df['close'].iloc[index-1]
        
        # Check for NaN values in moving averages
        if pd.isna(fast_ma) or pd.isna(slow_ma) or pd.isna(prev_fast_ma) or pd.isna(prev_slow_ma):
            # Use price-based trend detection as fallback
            price_trend = df['close'].iloc[index] > df['close'].iloc[index-1]
            ma_crossover = price_trend
        else:
            ma_crossover = (prev_fast_ma <= prev_slow_ma) and (fast_ma > slow_ma)
        
        # Combined entry conditions based on regime
        trend_ok = trend_conditions.get(regime, True)  # Default to True if regime unknown
        
        # More permissive entry logic - require fewer conditions
        conditions_met = 0
        if trend_ok:
            conditions_met += 1
        if volume_condition:
            conditions_met += 1
        if rsi_condition:
            conditions_met += 1
        if ma_crossover:
            conditions_met += 1
        
        # Adjust minimum conditions based on price trend
        # If price is declining (bear market), require more conditions
        current_price = df['close'].iloc[index]
        prev_price = df['close'].iloc[index-1] if index > 0 else current_price
        
        # Check for sustained decline over multiple periods
        sustained_decline = False
        if index >= 3:
            price_3_ago = df['close'].iloc[index-3]
            sustained_decline = current_price < price_3_ago * 0.97  # 3% decline over 3 periods
        
        price_declining = current_price < prev_price * 0.995  # 0.5% decline (more sensitive)
        
        # Be very conservative in bear markets
        if sustained_decline:
            min_conditions = 4  # Require all conditions in sustained bear market
        elif price_declining:
            min_conditions = 3  # Require most conditions in declining market
        else:
            min_conditions = 2  # Normal conditions in stable/rising market
        
        # Return True if minimum conditions are met
        return conditions_met >= min_conditions

    def check_exit_conditions(self, df: pd.DataFrame, index: int, entry_price: float) -> bool:
        """Check if exit conditions are met at the given index"""
        if index < 1 or index >= len(df):
            return False
            
        current_price = df['close'].iloc[index]
        regime = df['regime'].iloc[index]
        rsi = df['rsi'].iloc[index]
        
        # Get RSI thresholds based on volatility
        rsi_thresholds = (
            self.rsi_thresholds['high_vol'] 
            if regime == 'volatile' 
            else self.rsi_thresholds['low_vol']
        )
        
        # Calculate profit/loss
        pnl = (current_price - entry_price) / entry_price
        
        # Check take profit levels
        for level in self.take_profit_levels:
            if pnl >= level['target'] * self.base_stop_loss_atr:
                return True
        
        # Exit on RSI overbought
        if rsi > rsi_thresholds['overbought']:
            return True
        
        # Exit on trend reversal
        fast_ma = df[f'ma_{self.fast_ma}'].iloc[index]
        slow_ma = df[f'ma_{self.slow_ma}'].iloc[index]
        prev_fast_ma = df[f'ma_{self.fast_ma}'].iloc[index-1]
        prev_slow_ma = df[f'ma_{self.slow_ma}'].iloc[index-1]
        
        trend_reversal = (prev_fast_ma > prev_slow_ma) and (fast_ma < slow_ma)
        if trend_reversal:
            return True
        
        return False

    def get_parameters(self) -> dict:
        """Return strategy parameters for logging"""
        return {
            'name': self.name,
            'base_risk_per_trade': self.base_risk_per_trade,
            'max_risk_per_trade': self.max_risk_per_trade,
            'position_size_atr_multiplier': self.position_size_atr_multiplier,
            'max_position_size': self.max_position_size,
            'base_stop_loss_atr': self.base_stop_loss_atr,
            'fast_ma': self.fast_ma,
            'slow_ma': self.slow_ma,
            'long_ma': self.long_ma,
            'rsi_period': self.rsi_period,
            'volume_ma_period': self.volume_ma_period,
            'min_volume_multiplier': self.min_volume_multiplier,
            'regime_threshold': self.regime_threshold
        } 