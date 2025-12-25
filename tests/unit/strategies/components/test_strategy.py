import logging
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from src.engines.live.trading_engine import LiveTradingEngine
from src.risk.risk_manager import RiskParameters
from src.strategies.components import Strategy
from src.strategies.components.position_sizer import PositionSizer
from src.strategies.components.regime_context import RegimeContext
from src.strategies.components.risk_manager import MarketData, Position, RiskManager
from src.strategies.components.signal_generator import Signal, SignalDirection, SignalGenerator


class StaticSignalGenerator(SignalGenerator):
    """Signal generator returning a constant BUY signal."""

    def __init__(self) -> None:
        super().__init__("static-signal")

    def generate_signal(
        self, df: pd.DataFrame, index: int, regime: RegimeContext | None = None
    ) -> Signal:
        self.validate_inputs(df, index)
        return Signal(
            direction=SignalDirection.BUY,
            strength=1.0,
            confidence=0.8,
            metadata={},
        )

    def get_confidence(self, df: pd.DataFrame, index: int) -> float:
        self.validate_inputs(df, index)
        return 0.8


class PassthroughPositionSizer(PositionSizer):
    """Position sizer that returns the provided risk amount."""

    def __init__(self) -> None:
        super().__init__("passthrough-sizer")

    def calculate_size(
        self,
        signal: Signal,
        balance: float,
        risk_amount: float,
        regime: RegimeContext | None = None,
    ) -> float:
        self.validate_inputs(balance, risk_amount)
        return risk_amount


class RecordingRiskManager(RiskManager):
    """Risk manager that records the context supplied for sizing."""

    def __init__(self) -> None:
        super().__init__("recording-risk")
        self.calls: list[dict[str, Any]] = []

    def calculate_position_size(
        self,
        signal: Signal,
        balance: float,
        regime: RegimeContext | None = None,
        **context: Any,
    ) -> float:
        self.validate_inputs(balance)
        self.calls.append(dict(context))
        if signal.direction is SignalDirection.HOLD:
            return 0.0
        return balance * 0.1

    def should_exit(
        self,
        position: Position,
        current_data: MarketData,
        regime: RegimeContext | None = None,
        **context: Any,
    ) -> bool:
        return False

    def get_stop_loss(
        self,
        entry_price: float,
        signal: Signal,
        regime: RegimeContext | None = None,
        **context: Any,
    ) -> float:
        return entry_price * 0.95


class DummyRegimeDetector:
    """Regime detector stub that always returns ``None``."""

    warmup_period = 0

    def get_feature_generators(self) -> list[Any]:
        return []

    def detect_regime(
        self, df: pd.DataFrame, index: int
    ) -> RegimeContext | None:  # pragma: no cover - simple stub
        return None


class FakeCorrelationEngine:
    """Correlation engine stub returning an identity matrix."""

    def calculate_position_correlations(self, price_series: dict[str, pd.Series]) -> pd.DataFrame:
        symbols = sorted(price_series)
        return pd.DataFrame(
            data=[
                [1.0 if i == j else 0.0 for j in range(len(symbols))] for i in range(len(symbols))
            ],
            index=symbols,
            columns=symbols,
        )


class FakeDataProvider:
    """Data provider stub returning a cached frame."""

    def __init__(self, frame: pd.DataFrame) -> None:
        self._frame = frame

    def get_historical_data(
        self,
        symbol: str,
        timeframe: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> pd.DataFrame:
        _ = (symbol, timeframe, start, end)
        return self._frame


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=5, freq="h")
    return pd.DataFrame(
        {
            "open": [100, 101, 102, 103, 104],
            "high": [101, 102, 103, 104, 105],
            "low": [99, 100, 101, 102, 103],
            "close": [100, 101, 102, 103, 104],
            "volume": [10, 11, 12, 13, 14],
        },
        index=index,
    )


def _build_strategy(risk_manager: RecordingRiskManager) -> Strategy:
    return Strategy(
        name="test-strategy",
        signal_generator=StaticSignalGenerator(),
        risk_manager=risk_manager,
        position_sizer=PassthroughPositionSizer(),
        regime_detector=DummyRegimeDetector(),
        enable_logging=False,
    )


def test_process_candle_merges_additional_risk_context(sample_dataframe: pd.DataFrame) -> None:
    risk_manager = RecordingRiskManager()
    strategy = _build_strategy(risk_manager)

    extra_context = {"correlation_ctx": {"enabled": True}}
    strategy.set_additional_risk_context_provider(lambda df, index, signal: extra_context)

    decision = strategy.process_candle(sample_dataframe, len(sample_dataframe) - 1, 1000.0)

    assert decision.position_size == pytest.approx(100.0)
    assert risk_manager.calls, "Risk manager should receive sizing context"
    assert risk_manager.calls[-1]["correlation_ctx"] == {"enabled": True}


def test_live_engine_supplies_correlation_context(sample_dataframe: pd.DataFrame) -> None:
    risk_manager = RecordingRiskManager()
    strategy = _build_strategy(risk_manager)

    engine = LiveTradingEngine.__new__(LiveTradingEngine)
    engine.logger = logging.getLogger("live-engine-test")
    engine.positions = {}
    engine.correlation_engine = FakeCorrelationEngine()
    engine.data_provider = FakeDataProvider(sample_dataframe)
    engine.risk_manager = SimpleNamespace(params=RiskParameters())
    engine.timeframe = "1h"
    engine._partial_operations_opt_in = False
    engine.partial_manager = None
    engine.trailing_stop_policy = None
    engine._trailing_stop_opt_in = False
    engine.dynamic_risk_manager = None
    engine.enable_dynamic_risk = False
    engine.db_manager = None
    engine.enable_live_trading = False
    engine._active_symbol = "BTCUSDT"
    engine.strategy_manager = None
    engine._runtime = None

    engine._configure_strategy(strategy)

    decision = strategy.process_candle(sample_dataframe, len(sample_dataframe) - 1, 1000.0)

    assert decision.position_size == pytest.approx(100.0)
    assert risk_manager.calls, "Risk manager should be invoked"
    correlation_ctx = risk_manager.calls[-1].get("correlation_ctx")
    assert correlation_ctx is not None
    assert correlation_ctx["candidate_symbol"] == "BTCUSDT"
    assert correlation_ctx["engine"] is engine.correlation_engine
