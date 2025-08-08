"""
ML Basic Strategy

This strategy uses a machine learning model trained exclusively on price data (OHLCV).
It provides simple, reliable predictions without external dependencies like sentiment data.

Key Features:
- Price-only predictions using LSTM neural network
- 120-day sequence length for pattern recognition
- Normalized price inputs for better model performance
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
import onnxruntime as ort
from src.strategies.base import BaseStrategy
from src.config.feature_flags import is_enabled

class MlBasic(BaseStrategy):
    # * Strategy configuration constants
    SHORT_ENTRY_THRESHOLD = -0.0005  # -0.05% threshold for short entries
    CONFIDENCE_MULTIPLIER = 10  # Multiplier for confidence calculation
    BASE_POSITION_SIZE = 0.1  # Base position size (10% of balance)
    MIN_POSITION_SIZE_RATIO = 0.05  # Minimum position size (5% of balance)
    MAX_POSITION_SIZE_RATIO = 0.2  # Maximum position size (20% of balance)
    
    def __init__(self, name="MlBasic", model_path="src/ml/btcusdt_price.onnx", sequence_length=120):
        super().__init__(name)
        
        # Set strategy-specific trading pair - ML model trained on BTC
        self.trading_pair = 'BTCUSDT'
        
        self.model_path = model_path
        self.sequence_length = sequence_length
        self.ort_session = ort.InferenceSession(self.model_path)
        self.input_name = self.ort_session.get_inputs()[0].name
        self.stop_loss_pct = 0.02  # 2% stop loss
        self.take_profit_pct = 0.04  # 4% take profit

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # * Gate prediction engine usage via feature flag
        use_prediction_engine = is_enabled("use_prediction_engine", default=True)

        # Normalize price features (same as training)
        price_features = ['close', 'volume', 'high', 'low', 'open']
        for feature in price_features:
            if feature in df.columns:
                # Simple min-max normalization within the sequence window
                df[f'{feature}_normalized'] = df[feature].rolling(
                    window=self.sequence_length, min_periods=1
                ).apply(
                    lambda x: (x[-1] - np.min(x)) / (np.max(x) - np.min(x)) if np.max(x) != np.min(x) else 0.5,
                    raw=True
                )
        
        # Prepare predictions column
        df['onnx_pred'] = np.nan
        
        # Generate predictions for each row that has enough history
        if use_prediction_engine:
            for i in range(self.sequence_length, len(df)):
                # Prepare input features
                feature_columns = [f'{feature}_normalized' for feature in price_features]
                input_data = df[feature_columns].iloc[i-self.sequence_length:i].values
                
                # Reshape for ONNX model: (batch_size, sequence_length, features)
                input_data = input_data.astype(np.float32)
                input_data = np.expand_dims(input_data, axis=0)  # Add batch dimension
                
                # Run prediction
                try:
                    output = self.ort_session.run(None, {self.input_name: input_data})
                    pred = output[0][0][0]  # Extract scalar prediction
                    
                    # Denormalize prediction back to actual price scale
                    recent_close = df['close'].iloc[i-self.sequence_length:i].values
                    min_close = np.min(recent_close)
                    max_close = np.max(recent_close)
                    
                    if max_close != min_close:
                        pred_denormalized = pred * (max_close - min_close) + min_close
                    else:
                        pred_denormalized = df['close'].iloc[i-1]  # Use previous close if no range
                    
                    df.at[df.index[i], 'onnx_pred'] = pred_denormalized
                    
                except Exception as e:
                    print(f"Prediction error at index {i}: {e}")
                    df.at[df.index[i], 'onnx_pred'] = df['close'].iloc[i-1]  # Fallback to previous close
        else:
            # * If prediction engine is disabled, keep 'onnx_pred' as NaN to disable ML-driven entries/size
            pass
        
        return df

    def check_entry_conditions(self, df: pd.DataFrame, index: int) -> bool:
        # Go long if the predicted price for the next bar is higher than the current close
        if index < 1 or index >= len(df):
            return False
        
        pred = df['onnx_pred'].iloc[index]
        close = df['close'].iloc[index]
        
        # Check if we have a valid prediction
        if pd.isna(pred):
            # Log the missing prediction
            self.log_execution(
                signal_type='entry',
                action_taken='no_action',
                price=close,
                reasons=['missing_ml_prediction'],
                additional_context={'prediction_available': False}
            )
            return False
        
        # Calculate predicted return
        predicted_return = (pred - close) / close if close > 0 else 0
        
        # Determine entry signal
        entry_signal = pred > close
        
        # Log the decision process
        ml_predictions = {
            'raw_prediction': pred,
            'current_price': close,
            'predicted_return': predicted_return
        }
        
        reasons = [
            f'predicted_return_{predicted_return:.4f}',
            f'prediction_{pred:.2f}_vs_current_{close:.2f}',
            'entry_signal_met' if entry_signal else 'entry_signal_not_met'
        ]
        
        self.log_execution(
            signal_type='entry',
            action_taken='entry_signal' if entry_signal else 'no_action',
            price=close,
            signal_strength=abs(predicted_return) if entry_signal else 0.0,
            confidence_score=min(1.0, abs(predicted_return) * 10),  # Scale confidence based on predicted return
            ml_predictions=ml_predictions,
            reasons=reasons,
            additional_context={
                'model_type': 'ml_basic',
                'sequence_length': self.sequence_length,
                'prediction_available': True
            }
        )
        
        return entry_signal

    def check_short_entry_conditions(self, df: pd.DataFrame, index: int) -> bool:
        if index < 1 or index >= len(df):
            return False
        pred = df['onnx_pred'].iloc[index]
        close = df['close'].iloc[index]
        if pd.isna(pred):
            return False
        predicted_return = (pred - close) / close if close > 0 else 0
        return predicted_return < self.SHORT_ENTRY_THRESHOLD

    def check_exit_conditions(self, df: pd.DataFrame, index: int, entry_price: float) -> bool:
        if index < 1 or index >= len(df):
            return False
        current_price = df['close'].iloc[index]
        returns = (current_price - entry_price) / entry_price
        hit_stop_loss = returns <= -self.stop_loss_pct
        hit_take_profit = returns >= self.take_profit_pct
        
        # * Basic exit conditions (stop loss and take profit)
        basic_exit = hit_stop_loss or hit_take_profit
        
        # * ML-based exit signal for unfavorable predictions
        pred = df['onnx_pred'].iloc[index]
        if not pd.isna(pred):
            # * For long positions: exit if prediction suggests significant price drop
            # * Only exit if prediction is significantly unfavorable (>2% drop predicted)
            predicted_return = (pred - current_price) / current_price if current_price > 0 else 0
            significant_unfavorable_prediction = predicted_return < -0.02  # 2% threshold
            
            return basic_exit or significant_unfavorable_prediction
        
        return basic_exit

    def calculate_position_size(self, df: pd.DataFrame, index: int, balance: float) -> float:
        if index >= len(df) or balance <= 0:
            return 0.0
        pred = df['onnx_pred'].iloc[index]
        close = df['close'].iloc[index]
        if pd.isna(pred):
            return 0.0
        predicted_return = abs(pred - close) / close if close > 0 else 0
        confidence = min(1.0, predicted_return * self.CONFIDENCE_MULTIPLIER)
        dynamic_size = self.BASE_POSITION_SIZE * confidence
        return max(self.MIN_POSITION_SIZE_RATIO, min(self.MAX_POSITION_SIZE_RATIO, dynamic_size)) * balance

    def get_parameters(self) -> dict:
        return {
            'name': self.name,
            'model_path': self.model_path,
            'sequence_length': self.sequence_length,
            'stop_loss_pct': self.stop_loss_pct,
            'take_profit_pct': self.take_profit_pct
        }

    def calculate_stop_loss(self, df, index, price, side) -> float:
        """Calculate stop loss price"""
        # * Handle both string and enum inputs for backward compatibility
        side_str = side.value if hasattr(side, 'value') else str(side)
        
        if side_str == 'long':
            return price * (1 - self.stop_loss_pct)
        else:  # short
            return price * (1 + self.stop_loss_pct)
    
    def _load_model(self):
        """Load or reload the ONNX model"""
        try:
            self.ort_session = ort.InferenceSession(self.model_path)
            self.input_name = self.ort_session.get_inputs()[0].name
        except Exception as e:
            print(f"Failed to load model {self.model_path}: {e}")
            raise 