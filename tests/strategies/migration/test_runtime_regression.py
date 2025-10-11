"""Tests for the runtime regression harness."""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd

from src.data_providers.data_provider import DataProvider
from src.strategies.base import BaseStrategy
from src.strategies.components.position_sizer import PositionSizer
from src.strategies.components.regime_context import RegimeContext
from src.strategies.components.risk_manager import MarketData, Position, RiskManager
from src.strategies.components.signal_generator import Signal, SignalDirection, SignalGenerator
from src.strategies.components.strategy import Strategy
from src.strategies.migration.runtime_regression import compare_backtest_results


class _StaticDataProvider(DataProvider):
    """Simple data provider returning a pre-built DataFrame."""

    def __init__(self, frame: pd.DataFrame):
        self._frame = frame

    def get_historical_data(self, symbol: str, timeframe: str, start: datetime, end: datetime | None = None):  # type: ignore[override]
        return self._frame.copy()

    def get_current_price(self, symbol: str):  # type: ignore[override]
        return float(self._frame["close"].iloc[-1])

    def get_live_data(self, symbol: str, timeframe: str, limit: int = 500):  # type: ignore[override]
        return self._frame.tail(limit).copy()

    def update_live_data(self, symbol: str, timeframe: str):  # type: ignore[override]
        return self._frame.copy()


class _TrendingSignalGenerator(SignalGenerator):
    def __init__(self) -> None:
        super().__init__("trending_signal")

    def generate_signal(self, df: pd.DataFrame, index: int, regime: RegimeContext | None = None) -> Signal:  # type: ignore[override]
        if index == 0:
            direction = SignalDirection.HOLD
        elif df["close"].iloc[index] > df["close"].iloc[index - 1]:
            direction = SignalDirection.BUY
        elif df["close"].iloc[index] < df["close"].iloc[index - 1]:
            direction = SignalDirection.SELL
        else:
            direction = SignalDirection.HOLD
        return Signal(direction=direction, strength=1.0, confidence=0.9, metadata={})

    def get_confidence(self, df: pd.DataFrame, index: int) -> float:  # type: ignore[override]
        return 0.9


class _FixedRiskManager(RiskManager):
    def __init__(self) -> None:
        super().__init__("fixed_risk")

    def calculate_position_size(self, signal: Signal, balance: float, regime: RegimeContext | None = None) -> float:  # type: ignore[override]
        return balance * 0.1 if signal.direction != SignalDirection.HOLD else 0.0

    def should_exit(self, position: Position, current_data: MarketData, regime: RegimeContext | None = None) -> bool:  # type: ignore[override]
        if position.side == "long":
            return current_data.price < position.entry_price
        return current_data.price > position.entry_price

    def get_stop_loss(self, entry_price: float, signal: Signal, regime: RegimeContext | None = None) -> float:  # type: ignore[override]
        return entry_price * (0.97 if signal.direction == SignalDirection.BUY else 1.03)


class _IdentitySizer(PositionSizer):
    def __init__(self) -> None:
        super().__init__("identity_sizer")

    def calculate_size(self, signal: Signal, balance: float, risk_amount: float, regime: RegimeContext | None = None) -> float:  # type: ignore[override]
        return risk_amount


class _LegacyTrendingStrategy(BaseStrategy):
    def __init__(self) -> None:
        super().__init__("legacy_trend")

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:  # type: ignore[override]
        df = df.copy()
        df["trend"] = df["close"].diff().fillna(0)
        return df

    def check_entry_conditions(self, df: pd.DataFrame, index: int) -> bool:  # type: ignore[override]
        return bool(df["trend"].iloc[index] > 0)

    def check_exit_conditions(self, df: pd.DataFrame, index: int, entry_price: float) -> bool:  # type: ignore[override]
        return bool(df["trend"].iloc[index] < 0)

    def calculate_position_size(self, df: pd.DataFrame, index: int, balance: float) -> float:  # type: ignore[override]
        return 0.1

    def calculate_stop_loss(self, df: pd.DataFrame, index: int, price: float, side: str = "long") -> float:  # type: ignore[override]
        if side == "long":
            return price * 0.97
        return price * 1.03

    def get_parameters(self) -> dict:  # type: ignore[override]
        return {"type": "legacy_trend"}


class _ComponentTrendingStrategy(Strategy):
    def __init__(self) -> None:
        super().__init__(
            name="component_trend",
            signal_generator=_TrendingSignalGenerator(),
            risk_manager=_FixedRiskManager(),
            position_sizer=_IdentitySizer(),
        )


def _build_test_frame() -> pd.DataFrame:
    start = datetime(2024, 1, 1)
    dates = [start + timedelta(hours=i) for i in range(20)]
    closes = [100 + i * 0.5 for i in range(20)]
    frame = pd.DataFrame(
        {
            "open": closes,
            "high": [c * 1.01 for c in closes],
            "low": [c * 0.99 for c in closes],
            "close": closes,
            "volume": [1_000 for _ in closes],
        },
        index=dates,
    )
    return frame


def test_compare_backtest_results_matches_metrics():
    frame = _build_test_frame()
    provider = _StaticDataProvider(frame)

    legacy_strategy = _LegacyTrendingStrategy()
    component_strategy = _ComponentTrendingStrategy()

    comparison = compare_backtest_results(
        legacy_strategy,
        component_strategy,
        provider,
        symbol="TESTUSDT",
        timeframe="1h",
        start=frame.index[0],
        end=frame.index[-1],
        backtester_kwargs={"log_to_database": False, "enable_dynamic_risk": False},
    )

    assert comparison.matches
    for diff in comparison.metric_differences.values():
        assert abs(diff) <= 1e-6


def test_compare_backtest_results_detects_differences():
    frame = _build_test_frame()
    provider = _StaticDataProvider(frame)

    legacy_strategy = _LegacyTrendingStrategy()
    component_strategy = _ComponentTrendingStrategy()

    class _BiasedSizer(_IdentitySizer):
        def calculate_size(self, signal: Signal, balance: float, risk_amount: float, regime: RegimeContext | None = None) -> float:  # type: ignore[override]
            return risk_amount * 1.5

    component_strategy.position_sizer = _BiasedSizer()

    comparison = compare_backtest_results(
        legacy_strategy,
        component_strategy,
        provider,
        symbol="TESTUSDT",
        timeframe="1h",
        start=frame.index[0],
        end=frame.index[-1],
        backtester_kwargs={"log_to_database": False, "enable_dynamic_risk": False},
    )

    assert not comparison.matches
    assert any(abs(diff) > 0 for diff in comparison.metric_differences.values())
