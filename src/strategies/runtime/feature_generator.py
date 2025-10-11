"""Feature generator interfaces for the strategy runtime."""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, MutableMapping
from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(slots=True)
class FeatureGeneratorResult:
    """Result returned by :class:`FeatureGenerator.generate`.

    The runtime expects feature generators to return a frame containing the
    derived columns. Implementations may optionally provide a cache payload
    that will be stored on the :class:`StrategyDataset` so incremental updates
    during live trading can reuse expensive intermediate values.
    """

    features: pd.DataFrame
    cache: MutableMapping[str, Any] | None = None


@dataclass(slots=True)
class IncrementalFeatureUpdate:
    """Represents the outcome of an incremental feature update."""

    row: Mapping[str, Any]
    cache: MutableMapping[str, Any] | None = None


class FeatureGenerator(ABC):
    """Base interface for declaring strategy feature requirements.

    Feature generators run once during :meth:`StrategyRuntime.prepare_data`
    to extend the shared market data frame with derived columns. Implementations
    can optionally override :meth:`update` to support constant time updates for
    streaming data.
    """

    def __init__(
        self,
        name: str,
        required_columns: Iterable[str] | None = None,
        warmup_period: int = 0,
    ) -> None:
        self.name = name
        self.required_columns = frozenset(required_columns or [])
        self._warmup_period = warmup_period

    @property
    def warmup_period(self) -> int:
        """Return the minimum history length required by the generator."""

        return self._warmup_period

    def validate(self, df: pd.DataFrame) -> None:
        """Validate that the incoming frame contains the required columns."""

        missing = self.required_columns.difference(df.columns)
        if missing:
            raise ValueError(
                f"Feature generator '{self.name}' missing required columns: {sorted(missing)}"
            )

    @abstractmethod
    def generate(self, df: pd.DataFrame) -> FeatureGeneratorResult:
        """Produce vectorised feature columns for the provided data frame."""

    def update(
        self,
        df: pd.DataFrame,
        index: int,
        previous_cache: MutableMapping[str, Any] | None = None,
    ) -> IncrementalFeatureUpdate | None:
        """Compute the next row of features for streaming contexts.

        Implementations may override this method when incremental updates can be
        computed more efficiently than recomputing the full feature frame. The
        default implementation returns ``None`` signalling that the runtime
        should fall back to batch generation if incremental updates are not
        supported.
        """

        return None


__all__ = [
    "FeatureGenerator",
    "FeatureGeneratorResult",
    "IncrementalFeatureUpdate",
]
