"""Declarative experiment suites — baseline plus N variants in a single run."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from src.experiments.runner import ExperimentRunner
from src.experiments.schemas import ExperimentConfig, ExperimentResult, ParameterSet


@dataclass(frozen=True)
class VariantSpec:
    """A single variant within a suite.

    ``overrides`` keys use the dotted ``<strategy>.<attr>`` form accepted by
    ``ExperimentRunner._apply_parameter_overrides``. An empty overrides dict
    yields a run equivalent to the baseline.
    """

    name: str
    overrides: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ComparisonSettings:
    """Controls used by the reporter when ranking variants."""

    target_metric: str = "sharpe_ratio"
    min_trades: int = 0
    significance_level: float = 0.05

    def __post_init__(self) -> None:
        allowed = {
            "sharpe_ratio",
            "annualized_return",
            "total_return",
            "calmar",
            "final_balance",
            "win_rate",
        }
        if self.target_metric not in allowed:
            raise ValueError(
                f"target_metric must be one of {sorted(allowed)}, got {self.target_metric!r}"
            )
        if self.min_trades < 0:
            raise ValueError(f"min_trades must be >= 0, got {self.min_trades}")
        if not 0.0 < self.significance_level < 1.0:
            raise ValueError(f"significance_level must be in (0, 1), got {self.significance_level}")


@dataclass
class BacktestSettings:
    """Shared backtest settings every variant inherits."""

    strategy: str
    symbol: str = "BTCUSDT"
    timeframe: str = "1h"
    days: int = 30
    initial_balance: float = 1000.0
    provider: str = "binance"
    use_cache: bool = True
    random_seed: int | None = None

    def resolve_window(self, now: datetime | None = None) -> tuple[datetime, datetime]:
        end = now or datetime.now(UTC)
        start = end - timedelta(days=self.days)
        return start, end


@dataclass
class SuiteConfig:
    """Fully-validated suite definition."""

    id: str
    description: str
    backtest: BacktestSettings
    baseline: VariantSpec
    variants: list[VariantSpec]
    comparison: ComparisonSettings = field(default_factory=ComparisonSettings)

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("suite id must be non-empty")
        seen = {self.baseline.name}
        for variant in self.variants:
            if variant.name in seen:
                raise ValueError(f"duplicate variant name: {variant.name}")
            seen.add(variant.name)

    def all_variants(self) -> list[VariantSpec]:
        """Return baseline followed by every variant in declaration order."""
        return [self.baseline, *self.variants]


@dataclass
class SuiteResult:
    """Outcome of executing a suite."""

    suite_id: str
    config: SuiteConfig
    baseline: ExperimentResult
    variants: list[ExperimentResult]
    started_at: datetime
    finished_at: datetime

    def all_results(self) -> list[ExperimentResult]:
        return [self.baseline, *self.variants]


def _build_experiment_config(
    suite: SuiteConfig,
    variant: VariantSpec,
    start: datetime,
    end: datetime,
) -> ExperimentConfig:
    strategy_name = suite.backtest.strategy
    parameters: ParameterSet | None = None
    if variant.overrides:
        parameters = ParameterSet(name=variant.name, values=dict(variant.overrides))

    return ExperimentConfig(
        strategy_name=strategy_name,
        symbol=suite.backtest.symbol,
        timeframe=suite.backtest.timeframe,
        start=start,
        end=end,
        initial_balance=suite.backtest.initial_balance,
        parameters=parameters,
        use_cache=suite.backtest.use_cache,
        provider=suite.backtest.provider,
        random_seed=suite.backtest.random_seed,
    )


class ExperimentSuiteRunner:
    """Expands a :class:`SuiteConfig` into experiments and executes them."""

    def __init__(self, runner: ExperimentRunner | None = None):
        self.runner = runner or ExperimentRunner()
        self.logger = logging.getLogger(self.__class__.__name__)

    def build_configs(
        self, suite: SuiteConfig, *, now: datetime | None = None
    ) -> list[ExperimentConfig]:
        start, end = suite.backtest.resolve_window(now=now)
        return [_build_experiment_config(suite, v, start, end) for v in suite.all_variants()]

    def run(self, suite: SuiteConfig, *, now: datetime | None = None) -> SuiteResult:
        configs = self.build_configs(suite, now=now)
        variants = suite.all_variants()
        assert len(configs) == len(variants), "config/variant mismatch"

        started_at = datetime.now(UTC)
        results: list[ExperimentResult] = []
        for variant, cfg in zip(variants, configs, strict=True):
            self.logger.info("Running variant %s", variant.name)
            results.append(self.runner.run(cfg))
        finished_at = datetime.now(UTC)

        return SuiteResult(
            suite_id=suite.id,
            config=suite,
            baseline=results[0],
            variants=results[1:],
            started_at=started_at,
            finished_at=finished_at,
        )


__all__ = [
    "BacktestSettings",
    "ComparisonSettings",
    "ExperimentSuiteRunner",
    "SuiteConfig",
    "SuiteResult",
    "VariantSpec",
]
