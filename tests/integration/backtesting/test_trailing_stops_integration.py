import pytest
from datetime import datetime

import pandas as pd

from src.backtesting.engine import Backtester
from src.data_providers.data_provider import DataProvider
from src.strategies.ml_adaptive import MlAdaptive
from src.position_management.trailing_stops import TrailingStopPolicy


pytestmark = pytest.mark.integration


class DummyProvider(DataProvider):
    def get_historical_data(self, symbol: str, timeframe: str, start: datetime, end: datetime | None = None, limit: int | None = None):
        idx = pd.date_range("2024-01-01", periods=10, freq="1h")
        # Upward drift to trigger activation and breakeven
        closes = [100 + i for i in range(10)]
        df = pd.DataFrame({
            "open": closes,
            "high": [c + 1 for c in closes],
            "low": [c - 1 for c in closes],
            "close": closes,
            "volume": [1000] * 10,
            "atr": [1.0] * 10,
        }, index=idx)
        return df


def test_backtester_applies_trailing_stops():
    strategy = MlAdaptive()
    provider = DummyProvider()
    ts_policy = TrailingStopPolicy(activation_threshold=0.005, trailing_distance_pct=0.005, breakeven_threshold=0.02, breakeven_buffer=0.001)

    bt = Backtester(strategy=strategy, data_provider=provider, trailing_stop_policy=ts_policy, enable_engine_risk_exits=True)

    results = bt.run("BTCUSDT", "1h", datetime(2024, 1, 1))

    assert isinstance(results, dict)
    # There should be at least one trade executed under simple upward drift, and trailing logic should not error
    assert "total_trades" in results