import os
import pandas as pd
from datetime import datetime, timedelta
from src.backtesting.engine import Backtester
from src.data_providers.mock_data_provider import MockDataProvider
from src.strategies.base import BaseStrategy

class DummyStrategy(BaseStrategy):
    def __init__(self):
        super().__init__("DummyStrategy")
        self.take_profit_pct = 0.02
        self.stop_loss_pct = 0.01
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['onnx_pred'] = df['close']
        df['prediction_confidence'] = 0.1
        return df
    def check_entry_conditions(self, df: pd.DataFrame, index: int) -> bool:
        return index % 15 == 0 and index > 0
    def check_exit_conditions(self, df: pd.DataFrame, index: int, entry_price: float) -> bool:
        return False
    def calculate_position_size(self, df: pd.DataFrame, index: int, balance: float) -> float:
        return 0.05
    def calculate_stop_loss(self, df, index, price, side='long') -> float:
        return price * (1 - self.stop_loss_pct)
    def get_parameters(self) -> dict:
        return {}


def test_backtester_regime_annotation(monkeypatch):
    monkeypatch.setenv("FEATURE_ENABLE_REGIME_DETECTION", "true")
    strategy = DummyStrategy()
    provider = MockDataProvider(interval_seconds=1, num_candles=500)
    start = datetime.now() - timedelta(hours=400)
    end = datetime.now()
    backtester = Backtester(
        strategy=strategy,
        data_provider=provider,
        initial_balance=1000,
        database_url='sqlite:///:memory:',
        log_to_database=False,
        enable_short_trading=False
    )
    result = backtester.run(symbol='BTCUSDT', timeframe='1h', start=start, end=end)
    # Detector should be initialized and run without error
    assert backtester.regime_detector is not None
    assert 'total_trades' in result