from datetime import UTC, datetime, timedelta

import pandas as pd

from src.data_providers.mock_data_provider import MockDataProvider
from src.engines.backtest.engine import Backtester
from src.strategies.components.position_sizer import PositionSizer
from src.strategies.components.risk_manager import RiskManager
from src.strategies.components.signal_generator import Signal, SignalDirection, SignalGenerator
from src.strategies.components.strategy import Strategy


class SimpleSignalGenerator(SignalGenerator):
    """Simple signal generator that signals BUY at index 10"""

    def __init__(self):
        super().__init__(name="simple_signal")

    def generate_signal(self, df: pd.DataFrame, index: int, regime=None) -> Signal:
        if index == 10:
            return Signal(
                direction=SignalDirection.BUY,
                confidence=0.8,
                strength=1.0,
                metadata={"timestamp": df.index[index]},
            )
        elif index >= 20:
            return Signal(
                direction=SignalDirection.SELL,
                confidence=0.8,
                strength=1.0,
                metadata={"timestamp": df.index[index]},
            )
        return Signal(
            direction=SignalDirection.HOLD,
            confidence=0.0,
            strength=0.0,
            metadata={"timestamp": df.index[index]},
        )

    def get_confidence(self, df: pd.DataFrame, index: int) -> float:
        if index == 10 or index >= 20:
            return 0.8
        return 0.0


class SimpleRiskManager(RiskManager):
    """Simple risk manager with fixed risk"""

    def __init__(self):
        super().__init__(name="simple_risk")

    def calculate_position_size(self, signal: Signal, balance: float, regime=None) -> float:
        return 0.1 * balance

    def should_exit(self, position, current_data, regime=None) -> bool:
        return False

    def get_stop_loss(self, entry_price: float, signal: Signal, regime=None) -> float:
        return entry_price * 0.95


class SimplePositionSizer(PositionSizer):
    """Simple position sizer with fixed fraction"""

    def __init__(self):
        super().__init__(name="simple_sizer")

    def calculate_size(
        self, signal: Signal, balance: float, risk_amount: float, regime=None
    ) -> float:
        return 0.1


def create_simple_strategy() -> Strategy:
    """Create a simple component-based strategy for testing"""
    return Strategy(
        name="SimpleStrategy",
        signal_generator=SimpleSignalGenerator(),
        risk_manager=SimpleRiskManager(),
        position_sizer=SimplePositionSizer(),
    )


def test_backtester_records_mfe_mae(monkeypatch):
    strategy = create_simple_strategy()
    provider = MockDataProvider(interval_seconds=1, num_candles=200)
    start = datetime.now(UTC) - timedelta(hours=200)
    end = datetime.now(UTC)

    bt = Backtester(strategy=strategy, data_provider=provider, log_to_database=False)
    result = bt.run(symbol="BTCUSDT", timeframe="1h", start=start, end=end)

    # Expect at least one trade with MFE/MAE attributes
    assert isinstance(bt.trades, list)
    assert len(bt.trades) >= 1
    t = bt.trades[0]
    assert hasattr(t, "mfe") and hasattr(t, "mae")
    assert t.mfe is not None
    assert t.mae is not None

    # Validate that TradingDecision objects were used
    assert result is not None
    assert "total_trades" in result
