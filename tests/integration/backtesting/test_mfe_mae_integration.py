from datetime import datetime, timedelta

import pandas as pd

from src.backtesting.engine import Backtester
from src.data_providers.mock_data_provider import MockDataProvider
from src.strategies.base import BaseStrategy


class SimpleStrategy(BaseStrategy):
    def __init__(self):
        super().__init__("SimpleStrategy")

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def check_entry_conditions(self, df: pd.DataFrame, index: int) -> bool:
        return index == 10

    def check_exit_conditions(self, df: pd.DataFrame, index: int, entry_price: float) -> bool:
        return index >= 20

    def calculate_position_size(self, df: pd.DataFrame, index: int, balance: float) -> float:
        return 0.1

    def calculate_stop_loss(self, df, index, price, side="long") -> float:
        return price * 0.95

    def get_parameters(self) -> dict:
        return {}


def test_backtester_records_mfe_mae(monkeypatch):
    strategy = SimpleStrategy()
    provider = MockDataProvider(interval_seconds=1, num_candles=200)
    start = datetime.now() - timedelta(hours=200)
    end = datetime.now()

    bt = Backtester(strategy=strategy, data_provider=provider, log_to_database=False)
    result = bt.run(symbol="BTCUSDT", timeframe="1h", start=start, end=end)

    # Expect at least one trade with MFE/MAE attributes
    assert isinstance(bt.trades, list)
    assert len(bt.trades) >= 1
    t = bt.trades[0]
    assert hasattr(t, 'mfe') and hasattr(t, 'mae')
    assert t.mfe is not None
    assert t.mae is not None