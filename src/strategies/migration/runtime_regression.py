"""Runtime regression harness comparing legacy and component backtests."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Mapping, Sequence

from src.backtesting.engine import Backtester
from src.data_providers.data_provider import DataProvider
from src.strategies.base import BaseStrategy
from src.strategies.components.strategy import Strategy as ComponentStrategy


_DEFAULT_METRICS: Sequence[str] = (
    "total_trades",
    "final_balance",
    "total_return",
    "max_drawdown",
    "win_rate",
)


@dataclass(slots=True)
class BacktestComparison:
    """Summary of a legacy versus runtime backtest comparison."""

    matches: bool
    metric_differences: dict[str, float]
    legacy_metrics: dict[str, float]
    runtime_metrics: dict[str, float]
    legacy_duration: float
    runtime_duration: float


def compare_backtest_results(
    legacy_strategy: BaseStrategy,
    component_strategy: ComponentStrategy,
    data_provider: DataProvider,
    *,
    symbol: str,
    timeframe: str,
    start: datetime,
    end: datetime | None = None,
    metrics: Iterable[str] = _DEFAULT_METRICS,
    tolerance: float = 1e-6,
    backtester_kwargs: Mapping[str, object] | None = None,
) -> BacktestComparison:
    """Run legacy and runtime backtests and compare key metrics.

    Parameters
    ----------
    legacy_strategy:
        Strategy implementing the legacy :class:`BaseStrategy` contract.
    component_strategy:
        Component-based strategy evaluated via :class:`StrategyRuntime`.
    data_provider:
        Data source supplying historical candles for both runs.
    symbol:
        Trading symbol passed to the backtester.
    timeframe:
        Candle timeframe string (e.g. ``"1h"``).
    start:
        Beginning of the backtest window.
    end:
        Optional end of the backtest window.
    metrics:
        Metrics compared between runs. Defaults to ``_DEFAULT_METRICS``.
    tolerance:
        Relative tolerance used when deciding if metrics match.
    backtester_kwargs:
        Extra keyword arguments forwarded to the :class:`Backtester` constructor.

    Returns
    -------
    BacktestComparison
        Dataclass describing metric differences and execution times.
    """

    extra_kwargs = dict(backtester_kwargs or {})

    legacy_backtester = Backtester(legacy_strategy, data_provider, **extra_kwargs)
    legacy_start = datetime.utcnow().timestamp()
    legacy_result = legacy_backtester.run(symbol, timeframe, start, end)
    legacy_duration = datetime.utcnow().timestamp() - legacy_start

    runtime_backtester = Backtester(component_strategy, data_provider, **extra_kwargs)
    runtime_start = datetime.utcnow().timestamp()
    runtime_result = runtime_backtester.run(symbol, timeframe, start, end)
    runtime_duration = datetime.utcnow().timestamp() - runtime_start

    legacy_metrics: dict[str, float] = {}
    runtime_metrics: dict[str, float] = {}
    differences: dict[str, float] = {}
    matches = True

    for metric in metrics:
        legacy_value = legacy_result.get(metric)
        runtime_value = runtime_result.get(metric)
        if legacy_value is None or runtime_value is None:
            continue

        legacy_float = float(legacy_value)
        runtime_float = float(runtime_value)
        diff = runtime_float - legacy_float

        legacy_metrics[metric] = legacy_float
        runtime_metrics[metric] = runtime_float
        differences[metric] = diff

        allowed = tolerance * max(1.0, abs(legacy_float))
        if abs(diff) > allowed:
            matches = False

    return BacktestComparison(
        matches=matches,
        metric_differences=differences,
        legacy_metrics=legacy_metrics,
        runtime_metrics=runtime_metrics,
        legacy_duration=legacy_duration,
        runtime_duration=runtime_duration,
    )
