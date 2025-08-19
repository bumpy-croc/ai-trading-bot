import pandas as pd

from src.data_providers.mock_data_provider import MockDataProvider
from src.live.trading_engine import LiveTradingEngine
from src.strategies.base import BaseStrategy


class DummyStrategy(BaseStrategy):
    def __init__(self):
        super().__init__("DummyStrategy")
        self.take_profit_pct = 0.02
        self.stop_loss_pct = 0.01

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["onnx_pred"] = df["close"]  # neutral prediction
        df["prediction_confidence"] = 0.1
        return df

    def check_entry_conditions(self, df: pd.DataFrame, index: int) -> bool:
        return index % 10 == 0 and index > 0

    def check_exit_conditions(self, df: pd.DataFrame, index: int, entry_price: float) -> bool:
        return False

    def calculate_position_size(self, df: pd.DataFrame, index: int, balance: float) -> float:
        return 0.05

    def calculate_stop_loss(self, df, index, price, side="long") -> float:
        return price * (1 - self.stop_loss_pct)

    def get_parameters(self) -> dict:
        return {}


def test_live_engine_regime_annotation(monkeypatch):
    # Enable regime detection via feature flag gate
    monkeypatch.setenv("FEATURE_ENABLE_REGIME_DETECTION", "true")
    strategy = DummyStrategy()
    provider = MockDataProvider(interval_seconds=1, num_candles=120)
    engine = LiveTradingEngine(
        strategy=strategy,
        data_provider=provider,
        enable_live_trading=False,
        check_interval=1,
        initial_balance=1000,
        max_consecutive_errors=2,
        enable_hot_swapping=False,
        account_snapshot_interval=0,
        database_url="sqlite:///:memory:",
    )
    # Run only a few steps to pass through loop and annotate
    engine._trading_loop(symbol="BTCUSDT", timeframe="1h", max_steps=2)
    assert engine.regime_detector is not None
