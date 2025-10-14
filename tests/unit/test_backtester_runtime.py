from datetime import timedelta

import pandas as pd

from src.backtesting.engine import Backtester
from src.strategies.components.signal_generator import Signal, SignalDirection
from src.strategies.ml_basic import MlBasic


class DummyDataProvider:
    def __init__(self, frame: pd.DataFrame):
        self._frame = frame

    def get_historical_data(self, symbol, timeframe, start, end):
        return self._frame


class DummySentimentProvider:
    def get_sentiment(self, *args, **kwargs):
        return None

    def get_historical_sentiment(self, symbol, start, end):
        return pd.DataFrame()

    def aggregate_sentiment(self, df, window):
        return df


def build_price_data(rows: int = 50) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=rows, freq="1h")
    base = pd.Series(range(rows), index=index, dtype=float)
    data = {
        "open": base + 100.0,
        "high": base + 101.0,
        "low": base + 99.0,
        "close": base + 100.5,
        "volume": pd.Series([1_000.0] * rows, index=index, dtype=float),
    }
    return pd.DataFrame(data, index=index)


def test_backtester_uses_runtime(monkeypatch):
    df = build_price_data()
    strategy = MlBasic()

    def fake_signal(df, index, regime):
        direction = SignalDirection.BUY if index % 5 == 0 else SignalDirection.HOLD
        return Signal(direction=direction, strength=0.9, confidence=0.8, metadata={})

    monkeypatch.setattr(strategy.signal_generator, "generate_signal", fake_signal)
    monkeypatch.setattr(
        strategy.risk_manager,
        "calculate_position_size",
        lambda signal, balance, regime: balance * (0.1 if signal.direction == SignalDirection.BUY else 0.0),
    )
    monkeypatch.setattr(
        strategy.position_sizer,
        "calculate_size",
        lambda signal, balance, risk_amount, regime: risk_amount,
    )

    provider = DummyDataProvider(df)
    backtester = Backtester(
        strategy=strategy,
        data_provider=provider,
        sentiment_provider=DummySentimentProvider(),
        enable_dynamic_risk=False,
        log_to_database=False,
    )

    end = df.index[-1]
    start = end - timedelta(hours=len(df))
    results = backtester.run(symbol="BTCUSDT", timeframe="1h", start=start, end=end)

    assert results["final_balance"] != 0
    assert isinstance(backtester.trades, list)
    assert results["total_trades"] >= 0
