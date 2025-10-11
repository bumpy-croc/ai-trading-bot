"""Runtime orchestration layer for component strategies."""
from __future__ import annotations

from collections.abc import MutableMapping
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from ..components.risk_manager import Position
from ..components.strategy import Strategy
from .feature_generator import FeatureGenerator, FeatureGeneratorResult


@dataclass(slots=True)
class StrategyDataset:
    """Container describing the prepared market data for a strategy run."""

    data: pd.DataFrame
    warmup_period: int
    feature_caches: MutableMapping[str, MutableMapping[str, Any]] = field(
        default_factory=dict
    )


@dataclass(slots=True)
class RuntimeContext:
    """Mutable context passed to :meth:`StrategyRuntime.process` calls."""

    balance: float
    current_positions: list[Position] | None = None
    metadata: MutableMapping[str, Any] = field(default_factory=dict)


class StrategyRuntime:
    """Prepare and execute component strategies over a dataset."""

    def __init__(self, strategy: Strategy) -> None:
        self.strategy = strategy
        self._dataset: StrategyDataset | None = None
        self._feature_generators: list[FeatureGenerator] = []

    @property
    def dataset(self) -> StrategyDataset | None:
        """Return the currently prepared dataset, if any."""

        return self._dataset

    def prepare_data(self, df: pd.DataFrame) -> StrategyDataset:
        """Prepare the dataframe and return the runtime dataset."""

        if not isinstance(df, pd.DataFrame):
            raise TypeError("StrategyRuntime.prepare_data expects a pandas DataFrame")

        prepared = df.copy(deep=False)
        feature_caches: MutableMapping[str, MutableMapping[str, Any]] = {}

        generators = list(self.strategy.get_feature_generators())
        self._feature_generators = generators

        warmup = max(
            [self.strategy.warmup_period, *[g.warmup_period for g in generators]]
            or [0]
        )

        for generator in generators:
            generator.validate(prepared)
            result = generator.generate(prepared)
            if not isinstance(result, FeatureGeneratorResult):
                raise TypeError(
                    f"Feature generator '{generator.name}' must return FeatureGeneratorResult"
                )

            if not isinstance(result.features, pd.DataFrame):
                raise TypeError(
                    f"Feature generator '{generator.name}' returned non-DataFrame features"
                )

            for column in result.features.columns:
                prepared[column] = result.features[column]

            if result.cache is not None:
                feature_caches[generator.name] = result.cache

        dataset = StrategyDataset(prepared, warmup, feature_caches)
        self._dataset = dataset

        if hasattr(self.strategy, "prepare_runtime"):
            self.strategy.prepare_runtime(dataset)

        return dataset

    def process(self, index: int, context: RuntimeContext) -> Any:
        """Process a single index using the prepared dataset."""

        if self._dataset is None:
            raise RuntimeError("StrategyRuntime.process called before prepare_data")

        if index < 0 or index >= len(self._dataset.data):
            raise IndexError(
                f"Index {index} out of bounds for dataset of length {len(self._dataset.data)}"
            )

        return self.strategy.process_candle(
            self._dataset.data,
            index,
            balance=context.balance,
            current_positions=context.current_positions,
        )

    def finalize(self) -> StrategyDataset | None:
        """Finalize the run and allow the strategy to clean up state."""

        dataset = self._dataset
        if dataset is None:
            return None

        if hasattr(self.strategy, "finalize_runtime"):
            self.strategy.finalize_runtime(dataset)

        self._dataset = None
        self._feature_generators = []
        return dataset


__all__ = [
    "RuntimeContext",
    "StrategyDataset",
    "StrategyRuntime",
]
