from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import pytest

from src.backtesting.engine import Backtester
from src.data_providers.data_provider import DataProvider
from src.strategies.components import (
    Signal,
    SignalDirection,
    SignalGenerator,
    Strategy,
    EnhancedRegimeDetector,
)
from src.strategies.components.risk_manager import RiskManager
from src.strategies.components.position_sizer import PositionSizer

pytestmark = pytest.mark.unit


class DummyDataProvider(DataProvider):
    def __init__(self, df):
        super().__init__()
        self._df = df

    def get_historical_data(
        self, symbol: str, timeframe: str, start: datetime, end: Optional[datetime] = None
    ):
        return self._df.copy()

    def get_live_data(self, symbol: str, timeframe: str, limit: int = 100):
        raise NotImplementedError

    def update_live_data(self, symbol: str, timeframe: str):
        raise NotImplementedError

    def get_current_price(self, symbol: str) -> float:
        return float(self._df["close"].iloc[-1])


class YearlyCycleSignalGenerator(SignalGenerator):
    def __init__(self):
        super().__init__("yearly_cycle_generator")

    def generate_signal(self, df: pd.DataFrame, index: int, regime=None) -> Signal:
        self.validate_inputs(df, index)
        current_year = df.index[index].year

        if index == 0 or df.index[index - 1].year != current_year:
            return Signal(SignalDirection.BUY, strength=1.0, confidence=1.0, metadata={})

        is_last_in_year = index == len(df) - 1 or df.index[index + 1].year != current_year
        if is_last_in_year:
            return Signal(SignalDirection.SELL, strength=1.0, confidence=1.0, metadata={})

        return Signal(SignalDirection.HOLD, strength=0.0, confidence=0.0, metadata={})

    def get_confidence(self, df: pd.DataFrame, index: int) -> float:
        self.validate_inputs(df, index)
        return 1.0


class FullAllocationRiskManager(RiskManager):
    def __init__(self):
        super().__init__("full_allocation_risk")

    def calculate_position_size(self, signal: Signal, balance: float, regime=None) -> float:
        if balance <= 0 or signal.direction == SignalDirection.HOLD:
            return 0.0
        return balance

    def should_exit(self, position, current_data, regime=None) -> bool:
        return False

    def get_parameters(self) -> dict[str, float]:
        params = super().get_parameters()
        params.update({"allocation": 1.0})
        return params

    def get_stop_loss(self, entry_price: float, signal: Signal, regime=None) -> float:
        if entry_price <= 0:
            return entry_price
        if signal.direction == SignalDirection.BUY:
            return entry_price * 0.9
        if signal.direction == SignalDirection.SELL:
            return entry_price * 1.1
        return entry_price


class PassThroughSizer(PositionSizer):
    def __init__(self):
        super().__init__("pass_through_sizer")

    def calculate_size(self, signal: Signal, balance: float, risk_amount: float, regime=None) -> float:
        self.validate_inputs(balance, risk_amount)
        if signal.direction == SignalDirection.HOLD:
            return 0.0
        return self.apply_bounds_checking(risk_amount, balance, min_fraction=0.0, max_fraction=1.0)

    def get_parameters(self) -> dict[str, str]:
        params = super().get_parameters()
        params.update({"mode": "pass_through"})
        return params


def create_yearly_cycle_strategy() -> Strategy:
    return Strategy(
        name="YearlyCycleStrategy",
        signal_generator=YearlyCycleSignalGenerator(),
        risk_manager=FullAllocationRiskManager(),
        position_sizer=PassThroughSizer(),
        regime_detector=EnhancedRegimeDetector(),
    )


def generate_test_dataframe():
    timestamps = [
        datetime(2020, 1, 1, 0),
        datetime(2020, 12, 31, 23),
        datetime(2021, 1, 1, 0),
        datetime(2021, 12, 31, 23),
    ]
    closes = [100.0, 200.0, 200.0, 100.0]
    df = pd.DataFrame(
        {
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": np.ones(len(closes)),
        },
        index=pd.DatetimeIndex(timestamps),
    )
    return df


def test_yearly_returns_positive_and_negative():
    df = generate_test_dataframe()
    provider = DummyDataProvider(df)
    strategy = create_yearly_cycle_strategy()
    backtester = Backtester(
        strategy=strategy,
        data_provider=provider,
        sentiment_provider=None,
        risk_parameters=None,
        initial_balance=1000,
        log_to_database=False,
        # Disable realistic execution for this test (legacy behavior)
        fee_rate=0.0,
        slippage_rate=0.0,
        use_next_bar_execution=False,
    )
    result = backtester.run(symbol="TEST", timeframe="1h", start=df.index[0], end=df.index[-1])
    yr = result["yearly_returns"]
    assert "2020" in yr and "2021" in yr
    assert yr["2020"] > 0
    assert yr["2021"] < 0
    total_return_factor = (1 + yr["2020"] / 100) * (1 + yr["2021"] / 100)
    expected_total = (total_return_factor - 1) * 100
    assert abs(expected_total - result["total_return"]) < 1e-6
