"""
ML Basic Strategy

This strategy uses a machine learning model trained exclusively on price data (OHLCV).
It provides simple, reliable predictions without external dependencies like sentiment data.

Key Features:
- Price-only predictions using the unified prediction engine
- Simple risk management with 2% stop loss, 4% take profit
- No external API dependencies
- Graceful fallback when prediction engine is unavailable

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
    # Strategy configuration constants
    SHORT_ENTRY_THRESHOLD = -0.0005  # -0.05% threshold for short entries
    CONFIDENCE_MULTIPLIER = 10  # Multiplier for confidence calculation
    BASE_POSITION_SIZE = 0.1  # Base position size (10% of balance)
    MIN_POSITION_SIZE_RATIO = 0.05  # Minimum position size (5% of balance)
    MAX_POSITION_SIZE_RATIO = 0.2  # Maximum position size (20% of balance)
    
    def __init__(self, name="MlBasic", prediction_engine=None, **kwargs):
        super().__init__(name, prediction_engine=prediction_engine)
        
        # Set strategy-specific trading pair - ML model trained on BTC
        self.trading_pair = 'BTCUSDT'
        
        # Trading parameters
        self.stop_loss_pct = 0.02  # 2% stop loss
        self.take_profit_pct = 0.04  # 4% take profit

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate indicators - simplified since prediction engine handles feature engineering"""
        df = df.copy()
        
        # No need for complex indicator calculations - prediction engine handles this
        # Just add placeholder columns for backward compatibility
        df['ml_prediction'] = np.nan
        df['prediction_confidence'] = np.nan
        
        return df

    def check_entry_conditions(self, df: pd.DataFrame, index: int) -> bool:
        """Check if we should enter a position using prediction engine"""
        if index < 1 or index >= len(df):
            return False
        
        # Get prediction from engine
        prediction = self.get_prediction(df, index)
        
        if prediction.get('error'):
            # Log the prediction error
            self.log_execution(
                signal_type='entry',
                action_taken='no_action',
                price=df['close'].iloc[index],
                reasons=[f'prediction_error: {prediction["error"]}'],
                additional_context={'prediction_available': False}
            )
            return False
        
        current_price = df['close'].iloc[index]
        predicted_price = prediction['price']
        confidence = prediction['confidence']
        
        if predicted_price is None:
            # Log missing prediction
            self.log_execution(
                signal_type='entry',
                action_taken='no_action',
                price=current_price,
                reasons=['missing_ml_prediction'],
                additional_context={'prediction_available': False}
            )
            return False
        
        # Calculate predicted return
        predicted_return = (predicted_price - current_price) / current_price if current_price > 0 else 0
        
        # Entry signal: simple price comparison (original behavior)
        # Enter if predicted price is higher than current price
        entry_signal = predicted_price > current_price
        
        # Log the decision process
        ml_predictions = {
            'raw_prediction': predicted_price,
            'current_price': current_price,
            'predicted_return': predicted_return,
            'confidence': confidence,
            'model_name': prediction['model_name']
        }
        
        reasons = [
            f'predicted_return_{predicted_return:.4f}',
            f'prediction_{predicted_price:.2f}_vs_current_{current_price:.2f}',
            f'confidence_{confidence:.4f}',
            f'price_comparison_{predicted_price:.2f}_>{current_price:.2f}' if entry_signal else f'price_comparison_{predicted_price:.2f}_<={current_price:.2f}',
            'entry_signal_met' if entry_signal else 'entry_signal_not_met'
        ]
        
        self.log_execution(
            signal_type='entry',
            action_taken='entry_signal' if entry_signal else 'no_action',
            price=current_price,
            signal_strength=abs(predicted_return) if entry_signal else 0.0,
            confidence_score=confidence,
            ml_predictions=ml_predictions,
            reasons=reasons,
            additional_context={
                'model_type': 'ml_basic',
                'prediction_available': True,
                'inference_time': prediction.get('inference_time', 0.0)
            }
        )
        
        return entry_signal

    def check_short_entry_conditions(self, df: pd.DataFrame, index: int) -> bool:
        """Check short entry conditions using prediction engine"""
        if index < 1 or index >= len(df):
            return False
        
        # Get prediction from engine
        prediction = self.get_prediction(df, index)
        
        if prediction.get('error') or prediction['price'] is None:
            return False
        
        current_price = df['close'].iloc[index]
        predicted_price = prediction['price']
        predicted_return = (predicted_price - current_price) / current_price if current_price > 0 else 0
        
        # Short entry: require significant predicted price drop below threshold
        # Enter short if predicted return is below the threshold
        return predicted_return < self.SHORT_ENTRY_THRESHOLD

    def check_exit_conditions(self, df: pd.DataFrame, index: int, entry_price: float) -> bool:
        """Check exit conditions using basic risk management and ML signals"""
        if index < 1 or index >= len(df):
            return False
        
        current_price = df['close'].iloc[index]
        returns = (current_price - entry_price) / entry_price
        
        # Basic exit conditions (stop loss and take profit)
        hit_stop_loss = returns <= -self.stop_loss_pct
        hit_take_profit = returns >= self.take_profit_pct
        basic_exit = hit_stop_loss or hit_take_profit
        
        # ML-based exit signal for unfavorable predictions
        prediction = self.get_prediction(df, index)
        if not prediction.get('error') and prediction['price'] is not None:
            # For long positions: exit if prediction suggests significant price drop
            predicted_return = (prediction['price'] - current_price) / current_price if current_price > 0 else 0
            significant_unfavorable_prediction = (
                predicted_return < -0.02  # 2% threshold - simple price comparison
            )
            
            return basic_exit or significant_unfavorable_prediction
        
        return basic_exit

    def calculate_position_size(self, df: pd.DataFrame, index: int, balance: float) -> float:
        """Calculate position size based on prediction confidence"""
        if index >= len(df) or balance <= 0:
            return 0.0
        
        # Get prediction from engine
        prediction = self.get_prediction(df, index)
        
        if prediction.get('error') or prediction['price'] is None:
            return 0.0
        
        current_price = df['close'].iloc[index]
        predicted_price = prediction['price']
        
        # Calculate predicted return
        predicted_return = (predicted_price - current_price) / current_price if current_price > 0 else 0
        
        # Use absolute value of predicted return for confidence to handle both bullish and bearish predictions
        # This ensures consistent position sizing regardless of prediction direction
        confidence = min(1.0, abs(predicted_return) * self.CONFIDENCE_MULTIPLIER)
        
        # Scale position size by confidence
        dynamic_size = self.BASE_POSITION_SIZE * confidence
        final_size = max(self.MIN_POSITION_SIZE_RATIO, min(self.MAX_POSITION_SIZE_RATIO, dynamic_size))
        
        return balance * final_size

    def get_parameters(self) -> dict:
        """Get strategy parameters"""
        return {
            'name': self.name,
            'stop_loss_pct': self.stop_loss_pct,
            'take_profit_pct': self.take_profit_pct,
            'base_position_size': self.BASE_POSITION_SIZE,
            'prediction_engine_available': self.prediction_engine is not None
        }

    def calculate_stop_loss(self, df, index, price, side) -> float:
        """Calculate stop loss price"""
        # Handle both string and enum inputs for backward compatibility
        side_str = side.value if hasattr(side, 'value') else str(side)
        
        if side_str == 'long':
            return price * (1 - self.stop_loss_pct)
        else:  # short
            return price * (1 + self.stop_loss_pct) 