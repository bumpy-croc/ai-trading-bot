from __future__ import annotations

from datetime import datetime
from math import isfinite

import pandas as pd
import pytest

from src.live.trading_engine import LiveTradingEngine, Position, PositionSide
from src.strategies.components.position_sizer import PositionSizer
from src.strategies.components.regime_context import RegimeContext
from src.strategies.components.risk_manager import MarketData
from src.strategies.components.risk_manager import Position as ComponentPosition
from src.strategies.components.risk_manager import RiskManager as ComponentRiskManager
from src.strategies.components.signal_generator import Signal, SignalDirection, SignalGenerator
from src.strategies.components.strategy import Strategy, TradingDecision
from tests.mocks import MockDatabaseManager


class DummyDataProvider:
    """Minimal live data provider for unit tests."""

    def __init__(self, frame: pd.DataFrame):
        self._frame = frame
        self._current_price = float(frame["close"].iloc[-1])

    def get_current_price(self, symbol: str) -> float:
        return self._current_price

    def get_live_data(self, symbol: str, timeframe: str, limit: int = 500) -> pd.DataFrame:
        return self._frame


class StubSignalGenerator(SignalGenerator):
    """Signal generator stub that always returns HOLD when invoked."""

    def __init__(self):
        super().__init__("stub_signal")

    def generate_signal(self, df: pd.DataFrame, index: int, regime: RegimeContext | None = None) -> Signal:
        self.validate_inputs(df, index)
        return Signal(
            direction=SignalDirection.HOLD,
            strength=0.0,
            confidence=1.0,
            metadata={},
        )

    def get_confidence(self, df: pd.DataFrame, index: int) -> float:
        self.validate_inputs(df, index)
        return 1.0


class StubRiskManager(ComponentRiskManager):
    """Component risk manager with deterministic sizing and exit behaviour."""

    def __init__(self, fraction: float = 0.1):
        super().__init__("stub_risk")
        self.size_fraction = fraction
        self.exit_next = False

    def calculate_position_size(
        self,
        signal: Signal,
        balance: float,
        regime: RegimeContext | None = None,
    ) -> float:
        if signal.direction == SignalDirection.HOLD:
            return 0.0
        return balance * self.size_fraction

    def should_exit(
        self,
        position: ComponentPosition,
        current_data: MarketData,
        regime: RegimeContext | None = None,
    ) -> bool:
        return self.exit_next

    def get_stop_loss(
        self,
        entry_price: float,
        signal: Signal,
        regime: RegimeContext | None = None,
    ) -> float:
        if signal.direction == SignalDirection.BUY:
            return entry_price * 0.9
        if signal.direction == SignalDirection.SELL:
            return entry_price * 1.1
        return entry_price


class StubPositionSizer(PositionSizer):
    """Component position sizer that forwards the risk amount."""

    def __init__(self):
        super().__init__("stub_sizer")

    def calculate_size(
        self,
        signal: Signal,
        balance: float,
        risk_amount: float,
        regime: RegimeContext | None = None,
    ) -> float:
        self.validate_inputs(balance, max(risk_amount, 0.0))
        return risk_amount


def build_price_frame(rows: int = 5) -> pd.DataFrame:
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


def build_engine(monkeypatch, *, max_position_size: float = 0.3) -> tuple[LiveTradingEngine, pd.DataFrame]:
    frame = build_price_frame()
    monkeypatch.setattr("src.live.trading_engine.DatabaseManager", MockDatabaseManager)

    strategy = Strategy(
        name="stub_strategy",
        signal_generator=StubSignalGenerator(),
        risk_manager=StubRiskManager(),
        position_sizer=StubPositionSizer(),
        enable_logging=False,
    )

    engine = LiveTradingEngine(
        strategy=strategy,
        data_provider=DummyDataProvider(frame),
        enable_live_trading=False,
        enable_hot_swapping=False,
        enable_dynamic_risk=False,
        database_url="mock://memory",
        max_position_size=max_position_size,
    )
    engine.current_balance = 1_000.0
    engine.initial_balance = 1_000.0
    engine.db_manager = None
    return engine, frame


def make_decision(
    direction: SignalDirection,
    *,
    balance: float,
    fraction: float,
    metadata: dict | None = None,
) -> TradingDecision:
    position_size = balance * fraction
    decision_metadata = dict(metadata) if metadata else {}
    return TradingDecision(
        timestamp=datetime.utcnow(),
        signal=Signal(direction=direction, strength=1.0, confidence=1.0, metadata=decision_metadata.copy()),
        position_size=position_size,
        regime=None,
        risk_metrics={},
        execution_time_ms=0.5,
        metadata=decision_metadata,
    )


def extract_single_position(engine: LiveTradingEngine) -> Position:
    assert engine.positions, "Expected at least one open position"
    return next(iter(engine.positions.values()))


def test_check_entry_conditions_opens_long_position(monkeypatch):
    engine, frame = build_engine(monkeypatch)
    price = float(frame["close"].iloc[-1])
    decision = make_decision(SignalDirection.BUY, balance=engine.current_balance, fraction=0.1)

    engine._check_entry_conditions(frame, len(frame) - 1, "BTCUSDT", price, runtime_decision=decision)

    position = extract_single_position(engine)
    assert position.side == PositionSide.LONG
    assert position.size == pytest.approx(0.1)
    assert isfinite(position.stop_loss)
    assert position.stop_loss == pytest.approx(price * 0.9)


def test_check_entry_conditions_allows_short_with_metadata(monkeypatch):
    engine, frame = build_engine(monkeypatch)
    price = float(frame["close"].iloc[-1])
    decision = make_decision(
        SignalDirection.SELL,
        balance=engine.current_balance,
        fraction=0.15,
        metadata={"enter_short": True},
    )

    engine._check_entry_conditions(frame, len(frame) - 1, "BTCUSDT", price, runtime_decision=decision)

    position = extract_single_position(engine)
    assert position.side == PositionSide.SHORT
    assert position.size == pytest.approx(0.15)
    assert isfinite(position.stop_loss)
    assert position.stop_loss == pytest.approx(price * 1.1)


def test_check_exit_conditions_closes_on_signal_reversal(monkeypatch):
    engine, frame = build_engine(monkeypatch)
    price = float(frame["close"].iloc[-1])

    # Seed an open long position
    engine._open_position("BTCUSDT", PositionSide.LONG, 0.1, price, price * 0.9, price * 1.1)
    assert engine.positions

    decision = make_decision(
        SignalDirection.SELL,
        balance=engine.current_balance,
        fraction=0.0,
        metadata={"enter_short": True},
    )

    engine._check_exit_conditions(
        frame,
        len(frame) - 1,
        price,
        runtime_decision=decision,
        candle=frame.iloc[-1],
    )

    assert not engine.positions
