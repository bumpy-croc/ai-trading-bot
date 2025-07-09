from strategies.base import BaseStrategy
import pandas as pd

class TestHighFrequencyStrategy(BaseStrategy):
    def __init__(self, name: str = "test_high_frequency"):
        super().__init__(name)
        self.trading_pair = 'BTCUSDT'
        self.position_size_pct = 0.01  # 1% of balance
        self.stop_loss_pct = 0.01      # 1% stop loss

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        # No indicators needed for this test strategy
        return df

    def check_entry_conditions(self, df: pd.DataFrame, index: int) -> bool:
        # Enter on every even index (bar)
        return index % 2 == 0

    def check_exit_conditions(self, df: pd.DataFrame, index: int, entry_price: float) -> bool:
        # Exit on every odd index (bar)
        return index % 2 == 1

    def calculate_position_size(self, df: pd.DataFrame, index: int, balance: float) -> float:
        # Always use 1% of balance
        price = df['close'].iloc[index]
        return (balance * self.position_size_pct) / price if price > 0 else 0.0

    def calculate_stop_loss(self, df: pd.DataFrame, index: int, price: float, side: str = 'long') -> float:
        # Fixed stop loss 1% below entry for long, above for short
        if side == 'long':
            return price * (1 - self.stop_loss_pct)
        else:
            return price * (1 + self.stop_loss_pct)

    def get_parameters(self) -> dict:
        return {
            'name': self.name,
            'position_size_pct': self.position_size_pct,
            'stop_loss_pct': self.stop_loss_pct
        } 