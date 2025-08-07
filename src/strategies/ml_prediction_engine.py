"""
MlPredictionEngine Strategy

This strategy is a copy of MlBasic, but fully integrated with the centralized
prediction engine. It does not load ONNX internally; instead, it relies on
BaseStrategy.get_prediction for signals and confidence.
"""

from typing import Optional
import pandas as pd
from src.strategies.base import BaseStrategy


class MlPredictionEngine(BaseStrategy):
    SHORT_ENTRY_THRESHOLD = -0.0005
    BASE_POSITION_SIZE = 0.1
    MIN_POSITION_SIZE_RATIO = 0.05
    MAX_POSITION_SIZE_RATIO = 0.2

    def __init__(self, name: str = "MlPredictionEngine", prediction_engine: Optional["PredictionEngine"] = None):
        super().__init__(name, prediction_engine=prediction_engine)
        self.trading_pair = 'BTCUSDT'
        self.stop_loss_pct = 0.02
        self.take_profit_pct = 0.04

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # Minimal indicators; engine handles features
        return df

    def check_entry_conditions(self, df: pd.DataFrame, index: int) -> bool:
        if index < 1 or index >= len(df):
            return False
        prediction = self.get_prediction(df, index)
        if prediction.get('error') or prediction.get('price') is None:
            return False
        current_price = df['close'].iloc[index]
        predicted_price = prediction['price']
        entry_signal = bool(predicted_price > current_price)
        self.log_execution(
            signal_type='entry',
            action_taken='entry_signal' if entry_signal else 'no_action',
            price=current_price,
            signal_strength=prediction.get('confidence', 0.0) if entry_signal else 0.0,
            confidence_score=prediction.get('confidence', 0.0),
            ml_predictions={
                'raw_prediction': predicted_price,
                'current_price': current_price,
                'predicted_return': ((predicted_price - current_price) / current_price) if current_price > 0 else 0.0,
                'model_name': prediction.get('model_name')
            },
            reasons=[
                f"prediction_{predicted_price:.2f}_vs_current_{current_price:.2f}",
                'entry_signal_met' if entry_signal else 'entry_signal_not_met'
            ],
            additional_context={'prediction_available': True, 'inference_time': prediction.get('inference_time', 0.0)}
        )
        return entry_signal

    def check_short_entry_conditions(self, df: pd.DataFrame, index: int) -> bool:
        if index < 1 or index >= len(df):
            return False
        prediction = self.get_prediction(df, index)
        if prediction.get('error') or prediction.get('price') is None:
            return False
        current_price = df['close'].iloc[index]
        predicted_price = prediction['price']
        predicted_return = (predicted_price - current_price) / current_price if current_price > 0 else 0.0
        return bool(predicted_return < self.SHORT_ENTRY_THRESHOLD)

    def check_exit_conditions(self, df: pd.DataFrame, index: int, entry_price: float) -> bool:
        if index < 1 or index >= len(df):
            return False
        current_price = df['close'].iloc[index]
        returns = (current_price - entry_price) / entry_price
        basic_exit = returns <= -self.stop_loss_pct or returns >= self.take_profit_pct
        prediction = self.get_prediction(df, index)
        if prediction.get('error') or prediction.get('price') is None:
            return bool(basic_exit)
        predicted_return = (prediction['price'] - current_price) / current_price if current_price > 0 else 0.0
        significant_unfavorable_prediction = predicted_return < -0.02
        return bool(basic_exit or significant_unfavorable_prediction)

    def calculate_position_size(self, df: pd.DataFrame, index: int, balance: float) -> float:
        if index >= len(df) or balance <= 0:
            return 0.0
        prediction = self.get_prediction(df, index)
        if prediction.get('error'):
            return 0.0
        raw_conf = float(prediction.get('confidence', 0.0))
        confidence = max(0.0, min(1.0, raw_conf))
        dynamic_ratio = self.BASE_POSITION_SIZE * confidence
        return float(max(self.MIN_POSITION_SIZE_RATIO, min(self.MAX_POSITION_SIZE_RATIO, dynamic_ratio)))

    def calculate_stop_loss(self, df, index, price, side) -> float:
        side_str = side.value if hasattr(side, 'value') else str(side)
        return price * (1 - self.stop_loss_pct) if side_str == 'long' else price * (1 + self.stop_loss_pct)

    def get_parameters(self) -> dict:
        return {
            'name': self.name,
            'stop_loss_pct': self.stop_loss_pct,
            'take_profit_pct': self.take_profit_pct,
            'base_position_size': self.BASE_POSITION_SIZE,
            'prediction_engine_available': self.prediction_engine is not None
        }


