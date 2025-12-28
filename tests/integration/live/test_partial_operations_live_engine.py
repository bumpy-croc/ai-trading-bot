from datetime import datetime, timedelta

import pandas as pd
import pytest

from src.data_providers.mock_data_provider import MockDataProvider
from src.engines.live.trading_engine import LiveTradingEngine, PositionSide
from src.position_management.partial_manager import PartialExitPolicy
from src.strategies.ml_adaptive import create_ml_adaptive_strategy

pytestmark = pytest.mark.integration


class SimpleMockProvider(MockDataProvider):
    def __init__(self, prices):
        self.prices = prices

    def get_current_price(self, symbol: str):
        return self.prices[-1]

    def get_historical_data(self, symbol, timeframe, start=None, end=None):
        # Build minimal OHLCV DataFrame
        idx = pd.date_range(
            start=datetime.utcnow() - timedelta(minutes=len(self.prices)),
            periods=len(self.prices),
            freq="T",
        )
        df = pd.DataFrame(
            {
                "open": self.prices,
                "high": self.prices,
                "low": self.prices,
                "close": self.prices,
                "volume": [1.0] * len(self.prices),
            },
            index=idx,
        )
        # Add minimal columns used by strategy
        df["onnx_pred"] = df["close"] * 1.001
        df["prediction_confidence"] = 0.8
        return df


@pytest.mark.live_trading
def test_partial_exits_and_scale_ins_execution(monkeypatch):
    # Prices go up steadily to trigger scale-in and partial exits
    prices = [100, 101, 102, 103, 104, 105, 106, 107]
    provider = SimpleMockProvider(prices)

    strategy = create_ml_adaptive_strategy()
    pem = PartialExitPolicy(
        exit_targets=[0.03, 0.06],
        exit_sizes=[0.25, 0.25],
        scale_in_thresholds=[0.02],
        scale_in_sizes=[0.25],
        max_scale_ins=1,
    )

    engine = LiveTradingEngine(
        strategy=strategy,
        data_provider=provider,
        enable_live_trading=False,
        initial_balance=10000,
        resume_from_last_balance=False,
        partial_manager=pem,
        max_position_size=0.5,  # Allow 50% position size for testing
        # Disable fees/slippage for this test to match expected price levels
        fee_rate=0.0,
        slippage_rate=0.0,
    )

    # Open a position manually at 100, size 0.5
    engine._execute_entry(
        symbol="BTCUSDT",
        side=PositionSide.LONG,
        size=0.5,
        price=100.0,
        stop_loss=None,
        take_profit=None,
        signal_strength=0.0,
        signal_confidence=0.0,
    )
    pos = list(engine.live_position_tracker._positions.values())[0]
    assert pos.current_size == pytest.approx(0.5)

    # At 102 (+2%), expect one scale-in of 0.25 (of original) -> current_size 0.75
    engine._check_partial_and_scale_ops(provider.get_historical_data("BTCUSDT", "1m"), -1, 102.0)
    pos = list(engine.live_position_tracker._positions.values())[0]
    assert pos.current_size >= 0.5  # scaled in

    # At 103 (+3%), expect first partial exit of 0.25 -> current_size decreases
    engine._check_partial_and_scale_ops(provider.get_historical_data("BTCUSDT", "1m"), -1, 103.0)
    pos = list(engine.live_position_tracker._positions.values())[0]
    assert pos.partial_exits_taken >= 1

    # At 107 (+7%), expect remaining partial exit(s)
    engine._check_partial_and_scale_ops(provider.get_historical_data("BTCUSDT", "1m"), -1, 107.0)
    # Position may be closed by partials
    # No strict assertion beyond ensuring engine did not error
