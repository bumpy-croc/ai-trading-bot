import pandas as pd
import pytest

from src.strategies.components.position_sizer import PositionSizer
from src.strategies.components.regime_context import EnhancedRegimeDetector
from src.strategies.components.risk_manager import MarketData, Position, RiskManager
from src.strategies.components.signal_generator import (
    Signal,
    SignalDirection,
    SignalGenerator,
)
from src.strategies.components.strategy import Strategy
from src.strategies.runtime import (
    FeatureGenerator,
    FeatureGeneratorResult,
    RuntimeContext,
    StrategyDataset,
    StrategyRuntime,
)


class DummySignalGenerator(SignalGenerator):
    def __init__(self) -> None:
        super().__init__("dummy_signal")

    def generate_signal(self, df: pd.DataFrame, index: int, regime=None) -> Signal:  # type: ignore[override]
        self.validate_inputs(df, index)
        return Signal(
            direction=SignalDirection.BUY,
            strength=1.0,
            confidence=1.0,
            metadata={"index": index},
        )

    def get_confidence(self, df: pd.DataFrame, index: int) -> float:  # type: ignore[override]
        self.validate_inputs(df, index)
        return 1.0


class DummyRiskManager(RiskManager):
    def __init__(self) -> None:
        super().__init__("dummy_risk")

    def calculate_position_size(self, signal: Signal, balance: float, regime=None) -> float:  # type: ignore[override]
        self.validate_inputs(balance)
        return balance * 0.1 if signal.direction != SignalDirection.HOLD else 0.0

    def should_exit(self, position: Position, current_data: MarketData, regime=None) -> bool:  # type: ignore[override]
        return False

    def get_stop_loss(self, entry_price: float, signal: Signal, regime=None) -> float:  # type: ignore[override]
        return entry_price * 0.95


class DummyPositionSizer(PositionSizer):
    def __init__(self) -> None:
        super().__init__("dummy_sizer")

    def calculate_size(self, signal: Signal, balance: float, risk_amount: float, regime=None) -> float:  # type: ignore[override]
        budget = balance * 0.1 if risk_amount <= 0 else min(risk_amount, balance)
        self.validate_inputs(balance, budget)
        return budget


class DummyFeatureGenerator(FeatureGenerator):
    def __init__(self) -> None:
        super().__init__("dummy_feature", required_columns={"close"}, warmup_period=2)

    def generate(self, df: pd.DataFrame) -> FeatureGeneratorResult:
        features = pd.DataFrame({"close_mean": df["close"].rolling(2).mean()})
        return FeatureGeneratorResult(features=features, cache={"window": 2})


class DummyStrategy(Strategy):
    def __init__(self) -> None:
        super().__init__(
            name="dummy",
            signal_generator=DummySignalGenerator(),
            risk_manager=DummyRiskManager(),
            position_sizer=DummyPositionSizer(),
            regime_detector=EnhancedRegimeDetector(),
            enable_logging=False,
        )
        self._prepared: StrategyDataset | None = None
        self._finalized: StrategyDataset | None = None

    @property
    def warmup_period(self) -> int:
        return 3

    def get_feature_generators(self):  # type: ignore[override]
        return [DummyFeatureGenerator()]

    def prepare_runtime(self, dataset: StrategyDataset) -> None:  # type: ignore[override]
        self._prepared = dataset

    def finalize_runtime(self, dataset: StrategyDataset) -> None:  # type: ignore[override]
        self._finalized = dataset


@pytest.fixture()
def sample_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": [1, 2, 3, 4, 5],
            "high": [1, 2, 3, 4, 5],
            "low": [1, 2, 3, 4, 5],
            "close": [1, 2, 3, 4, 5],
            "volume": [1, 2, 3, 4, 5],
        }
    )


def test_prepare_data_runs_feature_generators(sample_dataframe: pd.DataFrame) -> None:
    strategy = DummyStrategy()
    runtime = StrategyRuntime(strategy)

    dataset = runtime.prepare_data(sample_dataframe)

    assert "close_mean" in dataset.data.columns
    assert dataset.feature_caches["dummy_feature"]["window"] == 2
    assert dataset.warmup_period == 3
    assert strategy._prepared is dataset


def test_process_uses_prepared_dataset(sample_dataframe: pd.DataFrame) -> None:
    strategy = DummyStrategy()
    runtime = StrategyRuntime(strategy)
    runtime.prepare_data(sample_dataframe)

    context = RuntimeContext(balance=1000.0, current_positions=[])

    decision = runtime.process(3, context)

    assert decision.signal.direction == SignalDirection.BUY
    assert decision.position_size > 0


def test_finalize_returns_dataset(sample_dataframe: pd.DataFrame) -> None:
    strategy = DummyStrategy()
    runtime = StrategyRuntime(strategy)
    dataset = runtime.prepare_data(sample_dataframe)

    returned = runtime.finalize()

    assert returned is dataset
    assert runtime.dataset is None
    assert strategy._finalized is dataset
