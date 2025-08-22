import pandas as pd

from src.strategies.base import BaseStrategy


class TestHighFrequencyStrategy(BaseStrategy):
    def __init__(self, name: str = "test_high_frequency"):
        super().__init__(name)
        self.trading_pair = "BTCUSDT"
        self.position_size_pct = 0.01  # 1% of balance
        self.stop_loss_pct = 0.01  # 1% stop loss
        self._last_timestamp = None
        self._next_action_is_entry = True

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        # No indicators needed for this test strategy
        return df

    def check_entry_conditions(self, df: pd.DataFrame, index: int) -> bool:
        # Alternate entry/exit on every new bar (timestamp)
        ts = df.index[index]
        if self._last_timestamp != ts and self._next_action_is_entry:
            self._last_timestamp = ts
            self._next_action_is_entry = False
            return True
        return False

    def check_exit_conditions(self, df: pd.DataFrame, index: int, entry_price: float) -> bool:
        ts = df.index[index]
        if self._last_timestamp != ts and not self._next_action_is_entry:
            self._last_timestamp = ts
            self._next_action_is_entry = True
            return True
        return False

    def calculate_position_size(self, df: pd.DataFrame, index: int, balance: float) -> float:
        price = df["close"].iloc[index]
        return (balance * self.position_size_pct) / price if price > 0 else 0.0

    def calculate_stop_loss(
        self, df: pd.DataFrame, index: int, price: float, side: str = "long"
    ) -> float:
        if side == "long":
            return price * (1 - self.stop_loss_pct)
        else:
            return price * (1 + self.stop_loss_pct)

    def get_parameters(self) -> dict:
        return {
            "name": self.name,
            "position_size_pct": self.position_size_pct,
            "stop_loss_pct": self.stop_loss_pct,
        }
