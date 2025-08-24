from datetime import datetime, timedelta

import pandas as pd
import pytest

from src.backtesting.engine import Backtester
from src.data_providers.mock_data_provider import MockDataProvider
from src.position_management.partial_manager import PartialExitPolicy
from src.strategies.ml_adaptive import MlAdaptive

pytestmark = pytest.mark.integration


class SimpleMockProvider(MockDataProvider):
    def __init__(self, prices):
        self._prices = prices

    def get_historical_data(self, symbol, timeframe, start=None, end=None):
        idx = pd.date_range(start=datetime.utcnow() - timedelta(minutes=len(self._prices)), periods=len(self._prices), freq="T")
        df = pd.DataFrame({
            "open": self._prices,
            "high": self._prices,
            "low": self._prices,
            "close": self._prices,
            "volume": [1.0] * len(self._prices),
        }, index=idx)
        df["onnx_pred"] = df["close"] * 1.001
        df["prediction_confidence"] = 0.8
        return df


def test_backtester_partial_ops_flow():
    prices = [100, 101, 102, 103, 104, 105, 106, 107]
    provider = SimpleMockProvider(prices)
    strategy = MlAdaptive()
    pem = PartialExitPolicy(
        exit_targets=[0.03, 0.06],
        exit_sizes=[0.25, 0.25],
        scale_in_thresholds=[0.02],
        scale_in_sizes=[0.25],
        max_scale_ins=1,
    )
    bt = Backtester(strategy=strategy, data_provider=provider, partial_manager=pem)
    res = bt.run("BTCUSDT", "1m", start=datetime.utcnow() - timedelta(minutes=10))
    # Basic sanity: produces results without error
    assert "total_trades" in res