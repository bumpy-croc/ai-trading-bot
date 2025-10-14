"""Tests for the composed Strategy runtime orchestration."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from datetime import datetime
from typing import Any

import pandas as pd
import pytest

from src.regime.detector import TrendLabel, VolLabel
from src.strategies.components.position_sizer import PositionSizer
from src.strategies.components.regime_context import (
    EnhancedRegimeDetector,
    RegimeContext,
)
from src.strategies.components.risk_manager import MarketData, Position, RiskManager
from src.strategies.components.signal_generator import Signal, SignalDirection, SignalGenerator
from src.strategies.components.strategy import Strategy, TradingDecision


class _StubSignalGenerator(SignalGenerator):
    def __init__(self, *, warmup_period: int = 5) -> None:
        super().__init__("stub_signal")
        self._warmup = warmup_period
        self.invocations: list[tuple[int, RegimeContext | None]] = []

    @property
    def warmup_period(self) -> int:  # type: ignore[override]
        return self._warmup

    def generate_signal(
        self, df: pd.DataFrame, index: int, regime: RegimeContext | None = None
    ) -> Signal:
        self.validate_inputs(df, index)
        self.invocations.append((index, regime))
        direction = SignalDirection.BUY if index % 2 == 0 else SignalDirection.HOLD
        return Signal(
            direction=direction,
            strength=1.0 if direction is SignalDirection.BUY else 0.0,
            confidence=0.9,
            metadata={"index": index},
        )


    def get_confidence(self, df: pd.DataFrame, index: int) -> float:
        self.validate_inputs(df, index)
        return 0.9


class _StubRiskManager(RiskManager):
    def __init__(self, *, warmup_period: int = 10) -> None:
        super().__init__("stub_risk")
        self._warmup = warmup_period
        self.exit_calls: list[tuple[Position, MarketData, RegimeContext | None]] = []

    @property
    def warmup_period(self) -> int:  # type: ignore[override]
        return self._warmup

    def calculate_position_size(
        self, signal: Signal, balance: float, regime: RegimeContext | None = None
    ) -> float:
        if signal.direction is SignalDirection.HOLD:
            return 0.0
        return balance * 0.1

    def should_exit(
        self,
        position: Position,
        current_data: MarketData,
        regime: RegimeContext | None = None,
    ) -> bool:
        self.exit_calls.append((position, current_data, regime))
        threshold = position.entry_price * 0.95
        return current_data.price <= threshold

    def get_stop_loss(
        self,
        entry_price: float,
        signal: Signal,
        regime: RegimeContext | None = None,
    ) -> float:
        return entry_price * (0.95 if signal.direction is SignalDirection.BUY else 1.05)


class _StubPositionSizer(PositionSizer):
    def __init__(self, *, warmup_period: int = 3) -> None:
        super().__init__("stub_sizer")
        self._warmup = warmup_period
        self.call_history: list[dict[str, Any]] = []

    @property
    def warmup_period(self) -> int:  # type: ignore[override]
        return self._warmup

    def calculate_size(
        self,
        signal: Signal,
        balance: float,
        risk_amount: float,
        regime: RegimeContext | None = None,
    ) -> float:
        payload = {
            "direction": signal.direction,
            "risk_amount": risk_amount,
            "regime": regime.trend if regime else None,
        }
        self.call_history.append(payload)
        return risk_amount * 0.5 if signal.direction is SignalDirection.BUY else 0.0


class _StubRegimeDetector(EnhancedRegimeDetector):
    def __init__(self, *, warmup_period: int = 8) -> None:
        super().__init__()
        self._warmup = warmup_period

    @property
    def warmup_period(self) -> int:  # type: ignore[override]
        return self._warmup

    def detect_regime(self, df: pd.DataFrame, index: int) -> RegimeContext:  # type: ignore[override]
        return RegimeContext(
            trend=TrendLabel.TREND_UP,
            volatility=VolLabel.LOW,
            confidence=0.85,
            duration=10,
            strength=0.8,
        )


def _frame(rows: int = 10) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=rows, freq="h")
    data = {
        "open": pd.Series(range(rows), index=index, dtype=float) + 100.0,
        "high": pd.Series(range(rows), index=index, dtype=float) + 101.0,
        "low": pd.Series(range(rows), index=index, dtype=float) + 99.0,
        "close": pd.Series(range(rows), index=index, dtype=float) + 100.5,
        "volume": pd.Series([1_000.0] * rows, index=index, dtype=float),
    }
    return pd.DataFrame(data, index=index)


@pytest.fixture()
def runtime_strategy() -> Strategy:
    return Strategy(
        name="runtime_stub",
        signal_generator=_StubSignalGenerator(),
        risk_manager=_StubRiskManager(),
        position_sizer=_StubPositionSizer(),
        regime_detector=_StubRegimeDetector(),
    )


def test_process_candle_returns_enriched_decision(runtime_strategy: Strategy) -> None:
    frame = _frame()
    decision = runtime_strategy.process_candle(frame, index=6, balance=1_000.0)

    assert isinstance(decision, TradingDecision)
    assert decision.signal.direction is SignalDirection.BUY
    assert decision.position_size == pytest.approx(50.0)
    assert decision.regime and decision.regime.trend is TrendLabel.TREND_UP
    assert decision.risk_metrics["balance_risk_pct"] == pytest.approx(5.0)
    assert decision.metadata["market_data"]["close"] == pytest.approx(frame["close"].iloc[6])
    assert runtime_strategy.signal_generator.invocations  # type: ignore[attr-defined]


def test_warmup_period_respects_component_max(runtime_strategy: Strategy) -> None:
    assert runtime_strategy.warmup_period == 10
    runtime_strategy.set_warmup_period(2)
    assert runtime_strategy.warmup_period == 2


def test_should_exit_delegates_to_risk_manager(runtime_strategy: Strategy) -> None:
    position = Position(
        symbol="BTCUSDT",
        side="long",
        size=1.0,
        entry_price=100.0,
        current_price=100.0,
        entry_time=datetime.utcnow(),
    )
    current = MarketData(symbol="BTCUSDT", price=94.5, volume=1_000.0, timestamp=datetime.utcnow())

    assert runtime_strategy.should_exit_position(position, current) is True
    exit_calls = runtime_strategy.risk_manager.exit_calls  # type: ignore[attr-defined]
    assert exit_calls and exit_calls[0][0] == position


def test_process_candle_returns_safe_decision_on_error(runtime_strategy: Strategy, monkeypatch: pytest.MonkeyPatch) -> None:
    frame = _frame()
    monkeypatch.setattr(
        runtime_strategy.signal_generator,
        "generate_signal",
        lambda df, index, regime: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    decision = runtime_strategy.process_candle(frame, index=0, balance=1_000.0)

    assert decision.signal.direction is SignalDirection.HOLD
    assert decision.position_size == 0.0
    assert decision.signal.metadata.get("component") == "signal_generator"


def test_get_component_info_exposes_configuration(runtime_strategy: Strategy) -> None:
    info = runtime_strategy.get_component_info()

    assert set(info) == {"signal_generator", "risk_manager", "position_sizer", "regime_detector"}
    assert info["signal_generator"]["name"] == "stub_signal"


def test_get_feature_generators_collects_from_components() -> None:
    generators: list[list[int]] = [[1], [2, 3]]

    class _FeatureComponent(_StubSignalGenerator):
        def get_feature_generators(self) -> Iterable[Iterator[int]]:  # type: ignore[override]
            return (iter(item) for item in generators)

    strategy = Strategy(
        name="features",
        signal_generator=_FeatureComponent(),
        risk_manager=_StubRiskManager(),
        position_sizer=_StubPositionSizer(),
        regime_detector=_StubRegimeDetector(),
    )

    collected = strategy.get_feature_generators()
    assert len(collected) == len(generators)
    assert [list(iterator) for iterator in collected] == generators
