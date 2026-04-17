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
    """Shared backtest settings every variant inherits.

    ``start`` / ``end`` are optional absolute UTC datetimes. When either is
    set, it anchors the backtest window so reruns produce the same data
    slice (critical for reproducing promoted results). When both are unset,
    the window is ``now() - days`` → ``now()`` and drifts with wall-clock
    time — use that mode only for ad-hoc exploration, not promotion.
    """

    strategy: str
    symbol: str = "BTCUSDT"
    timeframe: str = "1h"
    days: int = 30
    initial_balance: float = 1000.0
    provider: str = "binance"
    use_cache: bool = True
    random_seed: int | None = None
    start: datetime | None = None
    end: datetime | None = None

    def resolve_window(self, now: datetime | None = None) -> tuple[datetime, datetime]:
        if self.end is not None and self.start is not None:
            return self.start, self.end
        end = self.end or now or datetime.now(UTC)
        start = self.start or (end - timedelta(days=self.days))
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
    """Outcome of executing a suite.

    ``errors`` maps variant name → stringified exception for variants whose
    backtest raised. Baseline failures raise immediately (no SuiteResult is
    produced), so the baseline never appears in ``errors``.
    """

    suite_id: str
    config: SuiteConfig
    baseline: ExperimentResult
    variants: list[ExperimentResult]
    started_at: datetime
    finished_at: datetime
    errors: dict[str, str] = field(default_factory=dict)

    def all_results(self) -> list[ExperimentResult]:
        return [self.baseline, *self.variants]


def _errored_result(cfg: ExperimentConfig) -> ExperimentResult:
    """Sentinel ExperimentResult for a variant whose backtest raised."""
    return ExperimentResult(
        config=cfg,
        total_trades=0,
        win_rate=0.0,
        total_return=0.0,
        annualized_return=0.0,
        max_drawdown=0.0,
        sharpe_ratio=0.0,
        final_balance=cfg.initial_balance,
    )


def _build_experiment_config(
    suite: SuiteConfig,
    variant: VariantSpec,
    start: datetime,
    end: datetime,
) -> ExperimentConfig:
    """Build a single variant's ExperimentConfig.

    Variants inherit the suite baseline's overrides so a promoted patch YAML
    (whose baseline carries the promoted state) remains the comparison
    reference when the next suite layers new variants on top. Variant keys
    win when both baseline and variant set the same override — the variant
    is the thing being *tested against* the promoted baseline.
    """
    strategy_name = suite.backtest.strategy
    is_baseline = variant is suite.baseline

    merged_overrides: dict[str, object] = {}
    if not is_baseline:
        # Baseline overrides are the reference state; variant overrides layer on top.
        merged_overrides.update(suite.baseline.overrides)
    merged_overrides.update(variant.overrides)

    parameters: ParameterSet | None = None
    if merged_overrides:
        parameters = ParameterSet(name=variant.name, values=merged_overrides)

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
        errors: dict[str, str] = {}
        baseline_failed = False
        for variant, cfg in zip(variants, configs, strict=True):
            self.logger.info("Running variant %s", variant.name)
            try:
                results.append(self.runner.run(cfg))
            except Exception as exc:
                # Baseline failures are fatal — there's nothing to compare
                # variants against. Variant failures are recorded and the
                # suite continues so a single typo doesn't burn an overnight
                # run.
                is_baseline = variant is suite.baseline
                self.logger.exception(
                    "Variant %r raised %s; baseline=%s",
                    variant.name,
                    type(exc).__name__,
                    is_baseline,
                )
                if is_baseline:
                    baseline_failed = True
                    raise
                errors[variant.name] = f"{type(exc).__name__}: {exc}"
                results.append(_errored_result(cfg))
        finished_at = datetime.now(UTC)

        if baseline_failed:  # pragma: no cover — raise above prevents this path
            raise RuntimeError("baseline variant failed; suite aborted")

        return SuiteResult(
            suite_id=suite.id,
            config=suite,
            baseline=results[0],
            variants=results[1:],
            started_at=started_at,
            finished_at=finished_at,
            errors=errors,
        )


__all__ = [
    "BacktestSettings",
    "ComparisonSettings",
    "ExperimentSuiteRunner",
    "SuiteConfig",
    "SuiteResult",
    "VariantSpec",
]
