"""Unit tests for the StrategyRuntime orchestration layer."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pytest

from src.strategies.components import (
    FeatureGeneratorSpec,
    RuntimeContext,
    Strategy,
    StrategyRuntime,
)
from src.strategies.components.position_sizer import PositionSizer
from src.strategies.components.regime_context import RegimeContext
from src.strategies.components.risk_manager import MarketData, Position, RiskManager
from src.strategies.components.signal_generator import Signal, SignalDirection, SignalGenerator


class DummySignalGenerator(SignalGenerator):
    """Signal generator used for runtime tests."""

    def __init__(self):
        super().__init__("dummy_signal")

    def generate_signal(
        self, df: pd.DataFrame, index: int, regime: RegimeContext | None = None
    ) -> Signal:
        self.validate_inputs(df, index)
        return Signal(direction=SignalDirection.BUY, strength=1.0, confidence=0.9, metadata={})

    def get_confidence(self, df: pd.DataFrame, index: int) -> float:
        self.validate_inputs(df, index)
        return 0.9

    @property
    def warmup_period(self) -> int:
        return 5

    def get_feature_generators(self) -> list[FeatureGeneratorSpec]:
        def _generate_features(frame: pd.DataFrame) -> pd.DataFrame:
            return pd.DataFrame(
                {"dummy_feature": frame["close"].rolling(window=2, min_periods=1).mean()}
            )

        def _incremental(frame: pd.DataFrame, new_row: pd.Series) -> pd.Series:
            return pd.Series({"dummy_feature": float(new_row["close"])})

        return [
            FeatureGeneratorSpec(
                name="dummy_signal_features",
                required_columns=("close",),
                warmup_period=3,
                generate=_generate_features,
                incremental=_incremental,
            )
        ]


class DummyRiskManager(RiskManager):
    """Risk manager returning a fixed fraction of balance."""

    def __init__(self):
        super().__init__("dummy_risk")

    @property
    def warmup_period(self) -> int:
        return 2

    def calculate_position_size(
        self, signal: Signal, balance: float, regime: RegimeContext | None = None
    ) -> float:
        self.validate_inputs(balance)
        if signal.direction is SignalDirection.HOLD:
            return 0.0
        return balance * 0.1

    def should_exit(
        self, position: Position, current_data: MarketData, regime: RegimeContext | None = None
    ) -> bool:
        return False

    def get_stop_loss(
        self, entry_price: float, signal: Signal, regime: RegimeContext | None = None
    ) -> float:
        return entry_price * 0.9


class DummyPositionSizer(PositionSizer):
    """Position sizer that passes through the risk amount."""

    def __init__(self):
        super().__init__("dummy_sizer")

    def calculate_size(
        self,
        signal: Signal,
        balance: float,
        risk_amount: float,
        regime: RegimeContext | None = None,
    ) -> float:
        self.validate_inputs(balance, risk_amount)
        return risk_amount


@dataclass
class DummyRegimeDetector:
    """Minimal regime detector that opts out of regime analysis."""

    warmup_period: int = 0

    def get_feature_generators(self) -> list[FeatureGeneratorSpec]:
        return []

    def detect_regime(self, df: pd.DataFrame, index: int) -> RegimeContext | None:
        return None


class RecordingStrategy(Strategy):
    """Strategy subclass that records runtime hook invocations."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prepared_with = None
        self.finalized = False

    def prepare_runtime(self, dataset):  # type: ignore[override]
        super().prepare_runtime(dataset)
        self.prepared_with = dataset

    def finalize_runtime(self) -> None:  # type: ignore[override]
        self.finalized = True
        super().finalize_runtime()


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": [100, 101, 102, 103, 104],
            "high": [101, 102, 103, 104, 105],
            "low": [99, 100, 101, 102, 103],
            "close": [100, 101, 102, 103, 104],
            "volume": [10, 11, 12, 13, 14],
        }
    )


@pytest.fixture
def runtime_strategy() -> RecordingStrategy:
    return RecordingStrategy(
        name="runtime_test",
        signal_generator=DummySignalGenerator(),
        risk_manager=DummyRiskManager(),
        position_sizer=DummyPositionSizer(),
        regime_detector=DummyRegimeDetector(),
    )


def test_prepare_data_enriches_dataset(
    runtime_strategy: RecordingStrategy, sample_dataframe: pd.DataFrame
) -> None:
    runtime = StrategyRuntime(runtime_strategy)
    dataset = runtime.prepare_data(sample_dataframe)

    assert runtime_strategy.prepared_with is dataset
    assert "dummy_feature" in dataset.data.columns
    assert dataset.feature_caches["dummy_signal_features"].supports_incremental()
    assert dataset.warmup_period == runtime_strategy.warmup_period


def test_process_produces_trading_decision(
    runtime_strategy: RecordingStrategy, sample_dataframe: pd.DataFrame
) -> None:
    runtime = StrategyRuntime(runtime_strategy)
    runtime.prepare_data(sample_dataframe)

    context = RuntimeContext(balance=1_000.0)
    decision = runtime.process(index=2, context=context)

    assert decision.signal.direction is SignalDirection.BUY
    assert decision.position_size > 0
    assert (
        decision.metadata["components"]["signal_generator"]
        == runtime_strategy.signal_generator.name
    )


def test_finalize_clears_dataset(
    runtime_strategy: RecordingStrategy, sample_dataframe: pd.DataFrame
) -> None:
    runtime = StrategyRuntime(runtime_strategy)
    runtime.prepare_data(sample_dataframe)
    runtime.finalize()

    assert runtime_strategy.finalized is True
    with pytest.raises(RuntimeError):
        _ = runtime.dataset


def test_prepare_data_validates_required_columns(sample_dataframe: pd.DataFrame) -> None:
    class MissingColumnSignalGenerator(DummySignalGenerator):
        def get_feature_generators(self) -> list[FeatureGeneratorSpec]:
            def _generate_features(frame: pd.DataFrame) -> pd.DataFrame:
                return pd.DataFrame({"feature": frame["missing"]})

            return [
                FeatureGeneratorSpec(
                    name="missing_column",
                    required_columns=("missing",),
                    generate=_generate_features,
                )
            ]

    strategy = Strategy(
        name="invalid",
        signal_generator=MissingColumnSignalGenerator(),
        risk_manager=DummyRiskManager(),
        position_sizer=DummyPositionSizer(),
        regime_detector=DummyRegimeDetector(),
    )
    runtime = StrategyRuntime(strategy)

    with pytest.raises(ValueError, match="missing column"):
        runtime.prepare_data(sample_dataframe)
