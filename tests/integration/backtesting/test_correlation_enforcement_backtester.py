from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from src.engines.backtest.engine import Backtester
from src.risk.risk_manager import RiskParameters
from src.strategies.components.position_sizer import PositionSizer
from src.strategies.components.risk_manager import RiskManager
from src.strategies.components.signal_generator import Signal, SignalDirection, SignalGenerator
from src.strategies.components.strategy import Strategy

pytestmark = pytest.mark.integration


class AlwaysBuySignalGenerator(SignalGenerator):
    """Signal generator that always emits a BUY signal with high confidence."""

    def __init__(self):
        super().__init__(name="always_buy")

    def generate_signal(self, df: pd.DataFrame, index: int, regime=None) -> Signal:
        return Signal(
            direction=SignalDirection.BUY,
            confidence=1.0,
            strength=1.0,
            metadata={
                "timestamp": df.index[index] if len(df.index) > index else pd.Timestamp.now(tz="UTC")
            },
        )

    def get_confidence(self, df: pd.DataFrame, index: int) -> float:
        return 1.0


class FixedRiskManager(RiskManager):
    """Risk manager that allocates a fixed fraction of balance for testing."""

    def __init__(self, fraction: float = 0.08):
        super().__init__(name="fixed_risk")
        self.fraction = fraction

    def calculate_position_size(self, signal: Signal, balance: float, regime=None) -> float:
        if signal.direction == SignalDirection.HOLD:
            return 0.0
        return balance * self.fraction

    def should_exit(self, position, current_data, regime=None) -> bool:
        return False

    def get_stop_loss(self, entry_price: float, signal: Signal, regime=None) -> float:
        return entry_price * 0.98


class PassThroughSizer(PositionSizer):
    """Position sizer that returns the risk amount unchanged."""

    def __init__(self):
        super().__init__(name="pass_through")

    def calculate_size(
        self, signal: Signal, balance: float, risk_amount: float, regime=None
    ) -> float:
        return risk_amount


class DummyComponentStrategy(Strategy):
    """Component-based strategy used to exercise correlation enforcement."""

    def __init__(self):
        super().__init__(
            name="DummyBacktest",
            signal_generator=AlwaysBuySignalGenerator(),
            risk_manager=FixedRiskManager(),
            position_sizer=PassThroughSizer(),
        )

    def get_risk_overrides(self) -> dict:
        return {
            "position_sizer": "fixed_fraction",
            "base_fraction": 0.08,
            "correlation_control": {"max_correlated_exposure": 0.1},
        }


def _df(prices: np.ndarray) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=len(prices), freq="h")
    return pd.DataFrame(
        {
            "open": prices,
            "high": prices,
            "low": prices,
            "close": prices,
            "volume": np.ones(len(prices)),
        },
        index=idx,
    )


def test_backtester_correlation_reduces_size(monkeypatch):
    candidate_prices = np.linspace(100, 120, 50)
    df = _df(candidate_prices)

    provider = Mock()
    rng = np.random.default_rng(42)

    def _hist(sym, timeframe, start=None, end=None):
        if sym == "ETHUSDT":
            return df
        if sym == "BTCUSDT":
            correlated_prices = candidate_prices + rng.normal(0, 0.01, len(candidate_prices))
            return _df(correlated_prices)
        return _df(rng.uniform(100, 120, len(candidate_prices)))

    provider.get_historical_data.side_effect = _hist

    risk_params = RiskParameters(
        base_risk_per_trade=0.2,
        max_risk_per_trade=0.2,
        max_position_size=0.5,
        max_daily_risk=1.0,
    )

    bt = Backtester(
        strategy=DummyComponentStrategy(), data_provider=provider, risk_parameters=risk_params
    )
    bt.balance = 10_000

    # Seed an existing correlated open position
    bt.risk_manager.positions = {"BTCUSDT": {"size": 0.06, "entry_price": 110.0, "side": "long"}}
    bt.positions = [SimpleNamespace(symbol="BTCUSDT")]

    original_apply = bt.correlation_handler.apply_correlation_control
    captures: dict[str, float] = {}

    def _wrap(self, *args, **kwargs):
        candidate_fraction = kwargs.get("candidate_fraction")
        if candidate_fraction is None and len(args) >= 5:
            candidate_fraction = args[4]
        captures.setdefault("before", candidate_fraction)
        result = original_apply(*args, **kwargs)
        captures["after"] = result
        return result

    monkeypatch.setattr(bt.correlation_handler, "apply_correlation_control", MethodType(_wrap, bt.correlation_handler))

    start = df.index[0]
    end = df.index[-1]
    result = bt.run(symbol="ETHUSDT", timeframe="1h", start=start, end=end)

    assert result is not None
    assert "before" in captures and "after" in captures
    assert captures["before"] > 0
    assert captures["after"] >= 0
    assert captures["after"] <= captures["before"]
    # Correlation cap (0.1) minus existing exposure (0.06) leaves at most 0.04 for the new trade.
    assert captures["before"] == pytest.approx(0.08, abs=1e-4)
    assert captures["after"] < captures["before"]
    assert captures["after"] <= 0.04 + 1e-6
