from datetime import datetime

import pandas as pd

from src.engines.backtest.engine import Backtester
from src.strategies.ml_basic import create_ml_basic_strategy


class DummyProvider:
    def get_historical_data(self, symbol, timeframe, start, end=None):
        idx = pd.date_range(start, periods=200, freq="1h")
        return pd.DataFrame(
            {
                "open": 100 + pd.Series(range(len(idx))).values * 0.1,
                "high": 100 + pd.Series(range(len(idx))).values * 0.1 + 0.2,
                "low": 100 + pd.Series(range(len(idx))).values * 0.1 - 0.2,
                "close": 100 + pd.Series(range(len(idx))).values * 0.1,
                "volume": 1000,
            },
            index=idx,
        )


def test_backtester_reports_prediction_metrics(monkeypatch):
    strategy = create_ml_basic_strategy()
    provider = DummyProvider()
    bt = Backtester(
        strategy=strategy, data_provider=provider, initial_balance=1000, log_to_database=False
    )
    res = bt.run("BTCUSDT", "1h", datetime(2024, 1, 1))
    assert "prediction_metrics" in res
    pm = res["prediction_metrics"]
    for key in ["directional_accuracy_pct", "mae", "mape_pct", "brier_score_direction"]:
        assert key in pm
