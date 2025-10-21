import pandas as pd

from src.data_providers.mock_data_provider import MockDataProvider
from src.live.trading_engine import LiveTradingEngine
from src.strategies.ml_basic import create_ml_basic_strategy


def test_live_engine_regime_annotation(monkeypatch):
    # Enable regime detection via feature flag gate
    monkeypatch.setenv("FEATURE_ENABLE_REGIME_DETECTION", "true")
    strategy = create_ml_basic_strategy()
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
    )
    # Run only a few steps to pass through loop and annotate
    engine._trading_loop(symbol="BTCUSDT", timeframe="1h", max_steps=2)
    assert engine.regime_detector is not None
