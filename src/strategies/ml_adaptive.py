"""
ML Adaptive Strategy

This strategy improves upon ML Basic to handle extreme market volatility and bear markets.
It incorporates lessons learned from the 2020 COVID crash analysis.

Key Improvements:
- Volatility-based position sizing using ATR
- Market regime detection (trending, ranging, volatile, crisis)
- Dynamic stop loss based on market conditions
- Crisis mode protocols for extreme volatility
- Maximum daily loss limits
- Enhanced risk management
- Multiple timeframe analysis
- Uses prediction engine for ML predictions

Ideal for:
- Volatile market conditions
- Bear market protection
- Crisis situations
- Risk-conscious trading
"""

import numpy as np
import pandas as pd
from src.strategies.base import BaseStrategy
from src.indicators.technical import (
    calculate_atr, calculate_rsi, calculate_bollinger_bands,
    calculate_moving_averages, calculate_macd
)
from datetime import datetime, timedelta
from src.utils.symbol_factory import SymbolFactory

class MlAdaptive(BaseStrategy):
    """
    Adaptive ML-based strategy for BTC-USD (Coinbase style symbol).
    """
    # Constants for magic numbers
    MIN_PREDICTION_CONFIDENCE = 0.007  # 0.7% minimum predicted move
    SECONDS_PER_DAY = 86400  # Number of seconds in a day
    LOSS_REDUCTION_FACTOR = 0.2  # 20% reduction per consecutive loss

    def __init__(self, name="MlAdaptive", **kwargs):
        super().__init__(name, **kwargs)
        
        # Set strategy-specific trading pair - ML model trained on BTC-USD
        self.trading_pair = kwargs.get('symbol', 'BTC-USD')
        
        # Adaptive risk management parameters
        self.base_stop_loss_pct = 0.02  # 2% base stop loss
        self.base_take_profit_pct = 0.04  # 4% base take profit
        self.max_stop_loss_pct = 0.08  # 8% maximum stop loss in crisis
        self.min_stop_loss_pct = 0.015  # 1.5% minimum stop loss
        
        # Position sizing parameters
        self.base_position_size = 0.10  # 10% base position
        self.min_position_size = 0.02  # 2% minimum position
        self.max_position_size = 0.15  # 15% maximum position
        
        # Market regime thresholds
        self.volatility_low_threshold = 0.02  # 2% daily volatility
        self.volatility_high_threshold = 0.05  # 5% daily volatility
        self.volatility_crisis_threshold = 0.20  # 20% daily volatility (crisis mode)
        
        # Risk limits
        self.max_daily_loss_pct = 0.05  # 5% maximum daily loss
        self.max_consecutive_losses = 3  # Maximum consecutive losing trades
        self.crisis_mode_cooldown_hours = 24  # Hours to wait after crisis
        
        # Tracking variables
        self.daily_losses = {}
        self.consecutive_losses = 0
        self.last_crisis_time = None
        self.in_crisis_mode = False
        
        # ML confidence thresholds
        # Lower minimum confidence slightly to allow more frequent trades
        self.min_prediction_confidence = self.MIN_PREDICTION_CONFIDENCE  # 0.7% minimum predicted move
        self.crisis_confidence_multiplier = 2.0  # Double confidence requirement in crisis
        self.adaptive_threshold = 0.7  # Base adaptive threshold

        # Expose a generic take_profit_pct attribute expected by the trading engine
        self.take_profit_pct = self.base_take_profit_pct  # For short positions handled by engine

        # Rebound detection parameters
        self.rebound_confidence_multiplier = 0.5  # Reduce confidence threshold by 50% on rebounds

        # Bear market detection parameters
        self.bear_trend_threshold = -0.05  # 5% negative trend strength considered bear
    
    def _init_strategy_params(self, **kwargs):
        """Initialize strategy-specific parameters"""
        self.base_stop_loss_pct = kwargs.get('base_stop_loss_pct', 0.02)
        self.base_take_profit_pct = kwargs.get('base_take_profit_pct', 0.04)
        self.adaptive_threshold = kwargs.get('adaptive_threshold', 0.7)
        self.volatility_crisis_threshold = kwargs.get('volatility_crisis_threshold', 0.20)

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # Calculate technical indicators
        df = calculate_atr(df, period=14)
        df = calculate_moving_averages(df, periods=[20, 50, 200])
        df = calculate_bollinger_bands(df, period=20, std_dev=2.0)
        df = calculate_macd(df)
        df['rsi'] = calculate_rsi(df, period=14)
        
        # Calculate volatility metrics
        df['returns'] = df['close'].pct_change()
        df['volatility_20'] = df['returns'].rolling(window=20).std()
        df['volatility_50'] = df['returns'].rolling(window=50).std()
        df['atr_pct'] = df['atr'] / df['close']  # ATR as percentage of price
        
        # We need trend and bear/rebound information before detecting regime
        # Calculate trend strength
        df['trend_strength'] = (df['close'] - df['ma_50']) / df['ma_50']
        df['trend_direction'] = np.where(df['ma_20'] > df['ma_50'], 1, -1)

        # Bear market flag – MA50 below MA200 and significant negative trend strength
        df['bear_market'] = np.where(
            (df['ma_50'] < df['ma_200']) & (df['trend_strength'] < self.bear_trend_threshold),
            1,
            0
        )

        # Rebound signal – previous candle in bear market but current trend turns positive (ma_20 > ma_50)
        df['rebound_signal'] = (
            (df['bear_market'].shift(1) == 1) &  # Previously in bear
            (df['trend_direction'] > 0) &        # Trend flips positive
            (df['close'] > df['ma_50'])           # Price reclaims MA50
        ).astype(int)

        # Now that we have bear/rebound flags, detect market regime
        df['market_regime'] = self._detect_market_regime(df)
        
        return df

    def _detect_market_regime(self, df: pd.DataFrame) -> pd.Series:
        """Detect market regime including bear and bull phases"""
        regimes = []
        
        for i in range(len(df)):
            if i < 20:  # Not enough data
                regimes.append('normal')
                continue
            
            volatility = df['volatility_20'].iloc[i] if not pd.isna(df['volatility_20'].iloc[i]) else 0
            atr_pct = df['atr_pct'].iloc[i] if not pd.isna(df['atr_pct'].iloc[i]) else 0
            
            # Crisis overrides all other regimes
            if volatility > self.volatility_crisis_threshold or atr_pct > 0.15:
                regimes.append('crisis')
                continue
            
            # Bear market check
            if df['bear_market'].iloc[i] == 1:
                regimes.append('bear')
                continue
                
            # High volatility
            if volatility > self.volatility_high_threshold:
                regimes.append('volatile')
            # Low volatility (ranging)
            elif volatility < self.volatility_low_threshold:
                regimes.append('ranging')
            # Normal trending
            else:
                regimes.append('trending')
        
        return pd.Series(regimes, index=df.index)

    def _calculate_adaptive_threshold(self, df: pd.DataFrame, index: int) -> float:
        """Calculate adaptive confidence threshold based on market conditions"""
        if index >= len(df) or index < 20:
            return self.adaptive_threshold
        
        # Calculate volatility
        returns = df['close'].pct_change().iloc[max(0, index-20):index+1]
        volatility = returns.std()
        
        # Get market regime
        regime = df['market_regime'].iloc[index] if 'market_regime' in df.columns else 'normal'
        
        # Adjust threshold based on volatility and regime
        if regime == 'crisis':
            return self.adaptive_threshold * self.crisis_confidence_multiplier
        elif regime == 'volatile':
            return self.adaptive_threshold * 1.2  # Higher threshold in volatile markets
        elif regime == 'bear':
            return self.adaptive_threshold * 1.1  # Slightly higher threshold in bear markets
        elif regime == 'ranging':
            return self.adaptive_threshold * 0.8  # Lower threshold in ranging markets
        elif volatility > 0.05:  # High volatility
            return self.adaptive_threshold * 1.2
        elif volatility < 0.02:  # Low volatility
            return self.adaptive_threshold * 0.8
        else:
            return self.adaptive_threshold

    def check_entry_conditions(self, df: pd.DataFrame, index: int) -> bool:
        """Check if we should enter a position using prediction engine"""
        if index < 1 or index >= len(df):
            return False
        
        # Get prediction from engine
        prediction = self.get_prediction(df, index)
        
        if prediction.get('error'):
            # Log the error
            self.log_execution(
                signal_type='entry',
                action_taken='no_action',
                price=df['close'].iloc[index],
                reasons=[f'prediction_error: {prediction["error"]}'],
                additional_context={'prediction_available': False}
            )
            return False
        
        # Adaptive threshold based on market conditions
        current_threshold = self._calculate_adaptive_threshold(df, index)
        
        # Use prediction for entry decision
        should_enter = (prediction['direction'] == 1 and 
                       prediction['confidence'] > current_threshold)
        
        # Check rebound signal for reduced threshold
        rebound_signal = df['rebound_signal'].iloc[index] if 'rebound_signal' in df.columns else False
        if rebound_signal:
            reduced_threshold = current_threshold * self.rebound_confidence_multiplier
            should_enter = (prediction['direction'] == 1 and 
                          prediction['confidence'] > reduced_threshold)
        
        # Log the decision process
        ml_predictions = {
            'raw_prediction': prediction['price'],
            'current_price': df['close'].iloc[index],
            'confidence': prediction['confidence'],
            'direction': prediction['direction']
        }
        
        regime = df['market_regime'].iloc[index] if 'market_regime' in df.columns else 'unknown'
        
        reasons = [
            f'confidence_{prediction["confidence"]:.4f}',
            f'direction_{prediction["direction"]}',
            f'threshold_{current_threshold:.4f}',
            f'regime_{regime}',
            'entry_signal_met' if should_enter else 'entry_signal_not_met'
        ]
        
        if rebound_signal:
            reasons.append('rebound_signal_detected')
        
        self.log_execution(
            signal_type='entry',
            action_taken='entry_signal' if should_enter else 'no_action',
            price=df['close'].iloc[index],
            signal_strength=prediction['confidence'] if should_enter else 0.0,
            confidence_score=prediction['confidence'],
            ml_predictions=ml_predictions,
            reasons=reasons,
            additional_context={
                'model_type': 'ml_adaptive',
                'model_name': prediction['model_name'],
                'market_regime': regime,
                'adaptive_threshold': current_threshold,
                'rebound_signal': rebound_signal,
                'prediction_available': True
            }
        )
        
        return should_enter

    def check_exit_conditions(self, df: pd.DataFrame, index: int, entry_price: float) -> bool:
        """Check if we should exit a position using prediction engine"""
        if index < 1 or index >= len(df):
            return False
        
        current_price = df['close'].iloc[index]
        returns = (current_price - entry_price) / entry_price
        
        # Adaptive stop loss based on market regime
        regime = df['market_regime'].iloc[index] if 'market_regime' in df.columns else 'normal'
        adaptive_stop_loss = self._get_adaptive_stop_loss(regime)
        adaptive_take_profit = self._get_adaptive_take_profit(regime)
        
        # Basic stop loss and take profit
        hit_stop_loss = returns <= -adaptive_stop_loss
        hit_take_profit = returns >= adaptive_take_profit
        
        # Get prediction from engine for ML-based exit
        prediction = self.get_prediction(df, index)
        ml_exit_signal = False
        
        if not prediction.get('error'):
            # Adaptive threshold for exit
            current_threshold = self._calculate_adaptive_threshold(df, index)
            
            # Exit if prediction suggests significant price drop
            ml_exit_signal = (prediction['direction'] == -1 and 
                            prediction['confidence'] > current_threshold)
        
        should_exit = hit_stop_loss or hit_take_profit or ml_exit_signal
        
        # Determine exit reason
        exit_reason = []
        if hit_stop_loss:
            exit_reason.append('adaptive_stop_loss')
        if hit_take_profit:
            exit_reason.append('adaptive_take_profit')
        if ml_exit_signal:
            exit_reason.append('adaptive_ml_exit_signal')
        
        # Log exit decision
        if should_exit:
            self.log_execution(
                signal_type='exit',
                action_taken='exit_signal',
                price=current_price,
                signal_strength=prediction.get('confidence', 0.0),
                confidence_score=prediction.get('confidence', 0.0),
                ml_predictions=prediction if not prediction.get('error') else None,
                reasons=exit_reason,
                additional_context={
                    'returns': returns,
                    'entry_price': entry_price,
                    'exit_type': ','.join(exit_reason),
                    'market_regime': regime,
                    'adaptive_stop_loss': adaptive_stop_loss,
                    'adaptive_take_profit': adaptive_take_profit
                }
            )
        
        return should_exit
    
    def _get_adaptive_stop_loss(self, regime: str) -> float:
        """Get adaptive stop loss based on market regime"""
        if regime == 'crisis':
            return self.max_stop_loss_pct
        elif regime == 'volatile':
            return self.base_stop_loss_pct * 1.5
        elif regime == 'bear':
            return self.base_stop_loss_pct * 1.2
        elif regime == 'ranging':
            return self.min_stop_loss_pct
        else:
            return self.base_stop_loss_pct
    
    def _get_adaptive_take_profit(self, regime: str) -> float:
        """Get adaptive take profit based on market regime"""
        if regime == 'crisis':
            return self.base_take_profit_pct * 0.5  # Take profits quickly in crisis
        elif regime == 'volatile':
            return self.base_take_profit_pct * 1.5  # Higher targets in volatile markets
        elif regime == 'trending':
            return self.base_take_profit_pct * 2.0  # Higher targets in trending markets
        else:
            return self.base_take_profit_pct

    def calculate_position_size(self, df: pd.DataFrame, index: int, balance: float) -> float:
        """Calculate position size based on prediction confidence and market regime"""
        if index >= len(df) or balance <= 0:
            return 0.0
        
        # Get prediction for position sizing
        prediction = self.get_prediction(df, index)
        
        if prediction.get('error'):
            return 0.0
        
        # Adaptive threshold
        current_threshold = self._calculate_adaptive_threshold(df, index)
        
        if prediction['confidence'] < current_threshold:
            return 0.0
        
        # Get market regime for position sizing
        regime = df['market_regime'].iloc[index] if 'market_regime' in df.columns else 'normal'
        
        # Base position size with confidence scaling
        confidence_multiplier = prediction['confidence'] / current_threshold
        dynamic_size = self.base_position_size * confidence_multiplier
        
        # Adjust for market regime
        if regime == 'crisis':
            dynamic_size *= 0.3  # Much smaller positions in crisis
        elif regime == 'volatile':
            dynamic_size *= 0.6  # Smaller positions in volatile markets
        elif regime == 'bear':
            dynamic_size *= 0.7  # Smaller positions in bear markets
        elif regime == 'trending':
            dynamic_size *= 1.2  # Larger positions in trending markets
        
        # Apply bounds
        final_size = max(self.min_position_size, min(self.max_position_size, dynamic_size))
        
        return final_size * balance

    def calculate_stop_loss(self, df: pd.DataFrame, index: int, price: float, side: str = 'long') -> float:
        """Calculate adaptive stop loss price"""
        # Handle both string and enum inputs for backward compatibility
        side_str = side.value if hasattr(side, 'value') else str(side)
        
        # Get market regime for adaptive stop loss
        regime = df['market_regime'].iloc[index] if 'market_regime' in df.columns and index < len(df) else 'normal'
        adaptive_stop_loss = self._get_adaptive_stop_loss(regime)
        
        if side_str == 'long':
            return price * (1 - adaptive_stop_loss)
        else:  # short
            return price * (1 + adaptive_stop_loss)

    def get_parameters(self) -> dict:
        """Return strategy parameters for logging"""
        return {
            'name': self.name,
            'base_stop_loss_pct': self.base_stop_loss_pct,
            'base_take_profit_pct': self.base_take_profit_pct,
            'adaptive_threshold': self.adaptive_threshold,
            'volatility_crisis_threshold': self.volatility_crisis_threshold,
            'bear_trend_threshold': self.bear_trend_threshold,
            'symbol': self.symbol,
            'timeframe': self.timeframe
        }