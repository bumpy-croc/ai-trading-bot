"""
ML Basic Strategy

This strategy uses a machine learning model trained exclusively on price data (OHLCV).
It provides simple, reliable predictions without external dependencies like sentiment data.

Key Features:
- Price-only predictions using prediction engine
- Simple entry/exit logic based on ML predictions
- 2% stop loss, 4% take profit risk management
- No external API dependencies

Ideal for:
- Consistent, reliable trading signals
- Backtesting historical periods
- Environments where sentiment data is unavailable
- Simple deployment scenarios
"""

import numpy as np
import pandas as pd
from src.strategies.base import BaseStrategy

class MlBasic(BaseStrategy):
    """Basic ML strategy using prediction engine"""
    
    def __init__(self, name="MlBasic", **kwargs):
        super().__init__(name, **kwargs)
        
        # Set strategy-specific trading pair - ML model trained on BTC
        self.trading_pair = kwargs.get('symbol', 'BTCUSDT')
        
        # Trading parameters
        self.stop_loss_pct = 0.02  # 2% stop loss
        self.take_profit_pct = 0.04  # 4% take profit
        
        # ML prediction thresholds
        self.min_confidence_threshold = 0.6  # Minimum confidence for entry
        
    def _init_strategy_params(self, **kwargs):
        """Initialize strategy-specific parameters"""
        self.stop_loss_pct = kwargs.get('stop_loss_pct', 0.02)
        self.take_profit_pct = kwargs.get('take_profit_pct', 0.04)
        self.min_confidence_threshold = kwargs.get('min_confidence_threshold', 0.6)

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate any additional indicators needed for the strategy"""
        df = df.copy()
        
        # For ML Basic, we rely primarily on the prediction engine
        # Add any simple technical indicators if needed
        return df

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
        
        # Use prediction for entry decision
        should_enter = (prediction['direction'] == 1 and 
                       prediction['confidence'] > self.min_confidence_threshold)
        
        # Log the decision process
        ml_predictions = {
            'raw_prediction': prediction['price'],
            'current_price': df['close'].iloc[index],
            'confidence': prediction['confidence'],
            'direction': prediction['direction']
        }
        
        reasons = [
            f'confidence_{prediction["confidence"]:.4f}',
            f'direction_{prediction["direction"]}',
            f'threshold_{self.min_confidence_threshold}',
            'entry_signal_met' if should_enter else 'entry_signal_not_met'
        ]
        
        self.log_execution(
            signal_type='entry',
            action_taken='entry_signal' if should_enter else 'no_action',
            price=df['close'].iloc[index],
            signal_strength=prediction['confidence'] if should_enter else 0.0,
            confidence_score=prediction['confidence'],
            ml_predictions=ml_predictions,
            reasons=reasons,
            additional_context={
                'model_type': 'ml_basic',
                'model_name': prediction['model_name'],
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
        
        # Basic stop loss and take profit
        hit_stop_loss = returns <= -self.stop_loss_pct
        hit_take_profit = returns >= self.take_profit_pct
        
        # Get prediction from engine for ML-based exit
        prediction = self.get_prediction(df, index)
        ml_exit_signal = False
        
        if not prediction.get('error'):
            # Exit if prediction suggests significant price drop
            ml_exit_signal = (prediction['direction'] == -1 and 
                            prediction['confidence'] > self.min_confidence_threshold)
        
        should_exit = hit_stop_loss or hit_take_profit or ml_exit_signal
        
        # Determine exit reason
        exit_reason = []
        if hit_stop_loss:
            exit_reason.append('stop_loss')
        if hit_take_profit:
            exit_reason.append('take_profit')
        if ml_exit_signal:
            exit_reason.append('ml_exit_signal')
        
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
                    'exit_type': ','.join(exit_reason)
                }
            )
        
        return should_exit

    def calculate_position_size(self, df: pd.DataFrame, index: int, balance: float) -> float:
        """Calculate position size based on prediction confidence"""
        if index >= len(df) or balance <= 0:
            return 0.0
        
        # Get prediction for position sizing
        prediction = self.get_prediction(df, index)
        
        if prediction.get('error') or prediction['confidence'] < self.min_confidence_threshold:
            return 0.0
        
        # Base position size with confidence scaling
        base_position_size = 0.1  # 10% of balance
        min_position_size = 0.05  # 5% minimum
        max_position_size = 0.2   # 20% maximum
        
        # Scale position size based on confidence
        confidence_multiplier = prediction['confidence'] / self.min_confidence_threshold
        dynamic_size = base_position_size * confidence_multiplier
        
        # Apply bounds
        final_size = max(min_position_size, min(max_position_size, dynamic_size))
        
        return final_size * balance

    def calculate_stop_loss(self, df: pd.DataFrame, index: int, price: float, side: str = 'long') -> float:
        """Calculate stop loss price"""
        # Handle both string and enum inputs for backward compatibility
        side_str = side.value if hasattr(side, 'value') else str(side)
        
        if side_str == 'long':
            return price * (1 - self.stop_loss_pct)
        else:  # short
            return price * (1 + self.stop_loss_pct)

    def get_parameters(self) -> dict:
        """Return strategy parameters for logging"""
        return {
            'name': self.name,
            'stop_loss_pct': self.stop_loss_pct,
            'take_profit_pct': self.take_profit_pct,
            'min_confidence_threshold': self.min_confidence_threshold,
            'symbol': self.symbol,
            'timeframe': self.timeframe
        } 