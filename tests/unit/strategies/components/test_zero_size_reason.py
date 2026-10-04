"""Every zero-size BUY/SELL decision names the component that zeroed it (#1045)."""

import logging
from typing import Any

import pandas as pd
import pytest

from src.strategies.components import Strategy
from src.strategies.components.position_sizer import (
    ConfidenceWeightedSizer,
    LeveragedPositionSizer,
    PositionSizer,
)
from src.strategies.components.regime_context import RegimeContext, TrendLabel, VolLabel
from src.strategies.components.risk_manager import RiskManager
from src.strategies.components.signal_generator import Signal, SignalDirection, SignalGenerator
from src.strategies.hyper_growth import FlatRiskManager

pytestmark = pytest.mark.unit


class _FixedSignal(SignalGenerator):
    def __init__(self, direction: SignalDirection, confidence: float) -> None:
        super().__init__("fixed-signal")
        self._direction = direction
        self._confidence = confidence

    def generate_signal(
        self, df: pd.DataFrame, index: int, regime: RegimeContext | None = None
    ) -> Signal:
        return Signal(
            direction=self._direction, strength=0.5, confidence=self._confidence, metadata={}
        )

    def get_confidence(self, df: pd.DataFrame, index: int) -> float:
        return self._confidence


class _PassthroughSizer(PositionSizer):
    def __init__(self) -> None:
        super().__init__("passthrough")

    def calculate_size(self, signal, balance, risk_amount, regime=None) -> float:
        return risk_amount


class _ZeroSizer(PositionSizer):
    def __init__(self) -> None:
        super().__init__("zero-sizer")

    def calculate_size(self, signal, balance, risk_amount, regime=None) -> float:
        return 0.0


class _NoExplanationRiskManager(FlatRiskManager):
    """Zeroes every directional signal and cannot say why."""

    def calculate_position_size(self, signal, balance, regime=None, **context: Any) -> float:
        return 0.0

    def explain_zero_size(self, signal, balance, regime=None, **context: Any) -> str | None:
        return None


class _ExplodingExplainRiskManager(_NoExplanationRiskManager):
    def explain_zero_size(self, signal, balance, regime=None, **context: Any) -> str | None:
        raise RuntimeError("boom")


class _NullRegimeDetector:
    warmup_period = 0

    def get_feature_generators(self) -> list[Any]:
        return []

    def detect_regime(self, df: pd.DataFrame, index: int) -> RegimeContext | None:
        return None


@pytest.fixture
def frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": [100.0] * 3,
            "high": [101.0] * 3,
            "low": [99.0] * 3,
            "close": [100.0] * 3,
            "volume": [10.0] * 3,
        },
        index=pd.date_range("2024-01-01", periods=3, freq="h"),
    )


def _strategy(
    risk_manager: RiskManager,
    sizer: PositionSizer,
    *,
    direction: SignalDirection = SignalDirection.BUY,
    confidence: float = 0.02,
) -> Strategy:
    return Strategy(
        name="zero-size-test",
        signal_generator=_FixedSignal(direction, confidence),
        risk_manager=risk_manager,
        position_sizer=sizer,
        regime_detector=_NullRegimeDetector(),
        enable_logging=True,
    )


def test_risk_manager_confidence_floor_is_named_in_metadata_and_log(frame, caplog):
    strategy = _strategy(FlatRiskManager(min_confidence=0.05), _PassthroughSizer())

    with caplog.at_level(logging.INFO):
        decision = strategy.process_candle(frame, 2, 1000.0)

    reason = decision.metadata["size_zero_reason"]
    assert reason == "risk_manager(flat_risk_manager):confidence_0.020_below_min_0.050"
    assert decision.position_size == 0.0
    assert f"ZeroSizeReason: {reason}" in caplog.text


def test_sizer_zero_is_attributed_to_the_sizer(frame):
    strategy = _strategy(FlatRiskManager(min_confidence=0.0), _ZeroSizer(), confidence=0.5)

    decision = strategy.process_candle(frame, 2, 1000.0)

    assert decision.metadata["size_zero_reason"] == "position_sizer(zero-sizer):returned_zero"


def test_sizer_specific_detail_is_surfaced(frame):
    sizer = ConfidenceWeightedSizer(base_fraction=0.1, min_confidence=0.3)
    strategy = _strategy(FlatRiskManager(min_confidence=0.0), sizer, confidence=0.1)

    decision = strategy.process_candle(frame, 2, 1000.0)

    assert decision.metadata["size_zero_reason"] == (
        f"position_sizer({sizer.name}):confidence_0.100_below_min_0.300"
    )


def test_unexplained_zero_still_names_the_component(frame):
    strategy = _strategy(_NoExplanationRiskManager(), _PassthroughSizer())

    decision = strategy.process_candle(frame, 2, 1000.0)

    assert decision.metadata["size_zero_reason"] == "risk_manager(flat_risk_manager):returned_zero"


def test_failing_explanation_never_breaks_the_decision(frame):
    strategy = _strategy(_ExplodingExplainRiskManager(), _PassthroughSizer())

    decision = strategy.process_candle(frame, 2, 1000.0)

    assert decision.position_size == 0.0
    assert decision.metadata["size_zero_reason"] == "unknown"


def test_sized_decision_carries_no_zero_reason(frame, caplog):
    strategy = _strategy(FlatRiskManager(min_confidence=0.05), _PassthroughSizer(), confidence=0.5)

    with caplog.at_level(logging.INFO):
        decision = strategy.process_candle(frame, 2, 1000.0)

    assert decision.position_size > 0
    assert "size_zero_reason" not in decision.metadata
    assert "ZeroSizeReason" not in caplog.text


def test_hold_decision_carries_no_zero_reason(frame):
    strategy = _strategy(
        FlatRiskManager(min_confidence=0.05),
        _PassthroughSizer(),
        direction=SignalDirection.HOLD,
    )

    decision = strategy.process_candle(frame, 2, 1000.0)

    assert decision.position_size == 0.0
    assert "size_zero_reason" not in decision.metadata


class _CashLeverage:
    """Leverage manager that sits out every regime."""

    def get_leverage_multiplier(self, regime) -> float:
        return 0.0


def test_zero_leverage_regime_is_named(frame):
    regime = RegimeContext(
        trend=TrendLabel.TREND_DOWN,
        volatility=VolLabel.HIGH,
        confidence=0.9,
        duration=5,
        strength=0.8,
    )
    detector = _NullRegimeDetector()
    detector.detect_regime = lambda df, index: regime  # type: ignore[method-assign]
    sizer = LeveragedPositionSizer(_PassthroughSizer(), _CashLeverage())
    strategy = Strategy(
        name="zero-size-test",
        signal_generator=_FixedSignal(SignalDirection.BUY, 0.5),
        risk_manager=FlatRiskManager(min_confidence=0.0),
        position_sizer=sizer,
        regime_detector=detector,
        enable_logging=False,
    )

    decision = strategy.process_candle(frame, 2, 1000.0)

    assert decision.metadata["size_zero_reason"] == (
        f"position_sizer({sizer.name}):leverage_multiplier_zero_for_regime"
    )
