import numpy as np
import pandas as pd
import onnx
import onnxruntime as ort
from strategies.base import BaseStrategy

class MlModelStrategy(BaseStrategy):
    def __init__(self, name="MlModelStrategy", model_path="ml/model_ethusdt.onnx", sequence_length=120):
        super().__init__(name)
        self.model_path = model_path
        self.sequence_length = sequence_length
        self.ort_session = ort.InferenceSession(self.model_path)
        self.input_name = self.ort_session.get_inputs()[0].name
        self.stop_loss_pct = 0.02  # 2% stop loss
        self.take_profit_pct = 0.04  # 4% take profit

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # Calculate rolling min-max normalization for close price
        df['close_normalized'] = df['close'].rolling(window=self.sequence_length, min_periods=1).apply(
            lambda x: (x[-1] - np.min(x)) / (np.max(x) - np.min(x)) if np.max(x) != np.min(x) else 0.5,
            raw=True
        )
        # Prepare predictions column
        df['onnx_pred'] = np.nan
        for i in range(self.sequence_length, len(df)):
            window_normalized = df['close_normalized'].values[i-self.sequence_length:i]
            window_actual = df['close'].values[i-self.sequence_length:i]
            min_close = np.min(window_actual)
            max_close = np.max(window_actual)
            input_window = np.array(window_normalized).astype(np.float32)
            input_window = np.expand_dims(input_window, axis=0)
            input_window = np.expand_dims(input_window, axis=2)
            output = self.ort_session.run(None, {self.input_name: input_window})
            pred = output[0][0][0]
            # Denormalize
            pred = pred * (max_close - min_close) + min_close
            df.at[df.index[i], 'onnx_pred'] = pred
        return df

    def check_entry_conditions(self, df: pd.DataFrame, index: int) -> bool:
        # Go long if the predicted price for the next bar is higher than the current close
        if index < 1 or index >= len(df):
            return False
        pred = df['onnx_pred'].iloc[index]
        close = df['close'].iloc[index]
        return pred > close

    def check_exit_conditions(self, df: pd.DataFrame, index: int, entry_price: float) -> bool:
        if index < 1 or index >= len(df):
            return False
        current_price = df['close'].iloc[index]
        returns = (current_price - entry_price) / entry_price
        hit_stop_loss = returns <= -self.stop_loss_pct
        hit_take_profit = returns >= self.take_profit_pct
        return hit_stop_loss or hit_take_profit

    def calculate_position_size(self, df: pd.DataFrame, index: int, balance: float) -> float:
        # Simple fixed position size: 10% of balance
        if index >= len(df) or balance <= 0:
            return 0.0
        return balance * 0.10

    def get_parameters(self) -> dict:
        return {
            'name': self.name,
            'model_path': self.model_path,
            'sequence_length': self.sequence_length,
            'stop_loss_pct': self.stop_loss_pct,
            'take_profit_pct': self.take_profit_pct
        }

    def calculate_stop_loss(self, df, index, price, side: str = 'long') -> float:
        """Calculate stop loss price"""
        if side == 'long':
            return price * (1 - self.stop_loss_pct)
        else:  # short
            return price * (1 + self.stop_loss_pct) 