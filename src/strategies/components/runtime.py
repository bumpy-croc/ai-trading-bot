"""Runtime orchestration utilities for component-based strategies."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

import pandas as pd
from pandas import Series

if TYPE_CHECKING:
    from .risk_manager import Position
    from .strategy import TradingDecision


@dataclass(frozen=True)
class FeatureGeneratorSpec:
    """Descriptor for a vectorised feature generator.

    Attributes:
        name: Human readable identifier for the generator.
        generate: Callable that receives the working DataFrame and returns the
            computed feature columns as a DataFrame aligned to the input index.
        required_columns: Columns that must exist on the DataFrame before
            `generate` is invoked.
        warmup_period: Minimum number of rows required before the generated
            features are considered reliable.
        incremental: Optional callable that can update the feature set for a
            new row without recomputing the entire batch.
        metadata: Free-form metadata describing the generator configuration.
    """

    name: str
    generate: Callable[[pd.DataFrame], pd.DataFrame]
    required_columns: Sequence[str] = ()
    warmup_period: int = 0
    incremental: Callable[[pd.DataFrame, Series], Series] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FeatureCache:
    """Cached information about features produced during preparation."""

    name: str
    columns: Sequence[str]
    incremental: Callable[[pd.DataFrame, Series], Series] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def supports_incremental(self) -> bool:
        """Return True when incremental updates are available."""

        return self.incremental is not None


@dataclass
class StrategyDataset:
    """Dataset prepared for runtime execution."""

    data: pd.DataFrame
    warmup_period: int
    feature_caches: dict[str, FeatureCache] = field(default_factory=dict)


@dataclass
class RuntimeContext:
    """Per-candle execution context provided to the runtime."""

    balance: float
    current_positions: list[Position] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class SupportsRuntimeHooks(Protocol):
    """Protocol describing the methods required by :class:`StrategyRuntime`."""

    name: str

    @property
    def warmup_period(self) -> int:  # pragma: no cover - protocol definition
        ...

    def get_feature_generators(self) -> Sequence[FeatureGeneratorSpec]:  # pragma: no cover - protocol definition
        ...

    def prepare_runtime(self, dataset: StrategyDataset) -> None:  # pragma: no cover - protocol definition
        ...

    def process_candle(
        self,
        df: pd.DataFrame,
        index: int,
        balance: float,
        current_positions: list[Position] | None = None,
    ) -> TradingDecision:  # pragma: no cover - protocol definition
        ...

    def finalize_runtime(self) -> None:  # pragma: no cover - protocol definition
        ...


class StrategyRuntime:
    """Orchestrates dataset preparation and per-candle strategy execution."""

    def __init__(self, strategy: SupportsRuntimeHooks):
        self._strategy = strategy
        self._dataset: StrategyDataset | None = None

    @property
    def dataset(self) -> StrategyDataset:
        """Return the prepared dataset or raise if preparation has not happened."""

        if self._dataset is None:
            raise RuntimeError("Strategy data has not been prepared. Call prepare_data first.")
        return self._dataset

    def prepare_data(self, df: pd.DataFrame) -> StrategyDataset:
        """Enrich the provided DataFrame with component-declared features."""

        working_df = df.copy(deep=True)
        feature_caches: dict[str, FeatureCache] = {}
        warmup = max(0, int(self._strategy.warmup_period))

        specs = list(self._strategy.get_feature_generators() or [])
        for spec in specs:
            missing = [col for col in spec.required_columns if col not in working_df.columns]
            if missing:
                raise ValueError(
                    f"Feature generator '{spec.name}' missing column(s): {missing}"
                )

            generated = spec.generate(working_df)
            if not isinstance(generated, pd.DataFrame):
                raise TypeError(
                    f"Feature generator '{spec.name}' must return a pandas DataFrame, "
                    f"got {type(generated)!r}"
                )

            for column in generated.columns:
                working_df[column] = generated[column]

            feature_caches[spec.name] = FeatureCache(
                name=spec.name,
                columns=tuple(generated.columns),
                incremental=spec.incremental,
                metadata=dict(spec.metadata),
            )
            warmup = max(warmup, int(spec.warmup_period))

        dataset = StrategyDataset(data=working_df, warmup_period=warmup, feature_caches=feature_caches)
        self._dataset = dataset
        self._strategy.prepare_runtime(dataset)
        return dataset

    def process(self, index: int, context: RuntimeContext) -> TradingDecision:
        """Process a single candle using the prepared dataset."""

        dataset = self.dataset
        return self._strategy.process_candle(
            dataset.data,
            index,
            context.balance,
            context.current_positions,
        )

    def finalize(self) -> None:
        """Finalize runtime execution and release dataset references."""

        try:
            self._strategy.finalize_runtime()
        finally:
            self._dataset = None
