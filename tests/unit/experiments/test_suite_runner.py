"""Unit tests for ExperimentSuiteRunner with a mocked ExperimentRunner."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest

from src.experiments.schemas import ExperimentConfig, ExperimentResult
from src.experiments.suite import (
    BacktestSettings,
    ComparisonSettings,
    ExperimentSuiteRunner,
    SuiteConfig,
    VariantSpec,
)

pytestmark = pytest.mark.fast


def _fake_result(cfg: ExperimentConfig, *, total_return: float = 0.0) -> ExperimentResult:
    return ExperimentResult(
        config=cfg,
        total_trades=100,
        win_rate=55.0,
        total_return=total_return,
        annualized_return=total_return,
        max_drawdown=5.0,
        sharpe_ratio=1.2,
        final_balance=1000.0 * (1 + total_return / 100.0),
    )


@pytest.fixture
def suite() -> SuiteConfig:
    return SuiteConfig(
        id="sample_suite",
        description="desc",
        backtest=BacktestSettings(
            strategy="ml_basic",
            symbol="BTCUSDT",
            timeframe="1h",
            days=30,
            initial_balance=1000.0,
            provider="mock",
            use_cache=False,
            random_seed=42,
        ),
        baseline=VariantSpec(name="baseline"),
        variants=[
            VariantSpec(
                name="tight_stops",
                overrides={"ml_basic.stop_loss_pct": 0.02},
            ),
            VariantSpec(
                name="wide_long_thr",
                overrides={"ml_basic.long_entry_threshold": 0.001},
            ),
        ],
        comparison=ComparisonSettings(target_metric="sharpe_ratio", min_trades=0),
    )


def test_build_configs_emits_baseline_then_variants(suite: SuiteConfig) -> None:
    now = datetime(2026, 4, 1, tzinfo=UTC)
    sut = ExperimentSuiteRunner(runner=MagicMock())
    configs = sut.build_configs(suite, now=now)

    assert [c.parameters.name if c.parameters is not None else None for c in configs] == [
        None,
        "tight_stops",
        "wide_long_thr",
    ]
    assert all(c.strategy_name == "ml_basic" for c in configs)
    assert all(c.symbol == "BTCUSDT" for c in configs)
    assert all(c.timeframe == "1h" for c in configs)
    assert all(c.provider == "mock" for c in configs)
    assert all(c.random_seed == 42 for c in configs)


def test_run_calls_runner_once_per_variant(suite: SuiteConfig) -> None:
    mock_runner = MagicMock()
    mock_runner.run.side_effect = [
        _fake_result(MagicMock(spec=ExperimentConfig), total_return=1.0),
        _fake_result(MagicMock(spec=ExperimentConfig), total_return=2.0),
        _fake_result(MagicMock(spec=ExperimentConfig), total_return=1.5),
    ]

    sut = ExperimentSuiteRunner(runner=mock_runner)
    result = sut.run(suite, now=datetime(2026, 4, 1, tzinfo=UTC))

    assert mock_runner.run.call_count == 3
    assert result.suite_id == "sample_suite"
    assert result.baseline.total_return == 1.0
    assert [v.total_return for v in result.variants] == [2.0, 1.5]
    assert result.started_at <= result.finished_at


def test_duplicate_variant_name_raises() -> None:
    with pytest.raises(ValueError, match="duplicate variant name"):
        SuiteConfig(
            id="dup",
            description="",
            backtest=BacktestSettings(strategy="ml_basic"),
            baseline=VariantSpec(name="dup_name"),
            variants=[VariantSpec(name="dup_name")],
        )


def test_comparison_validates_target_metric() -> None:
    with pytest.raises(ValueError, match="target_metric"):
        ComparisonSettings(target_metric="not_a_metric")


def test_comparison_validates_significance_level() -> None:
    with pytest.raises(ValueError, match="significance_level"):
        ComparisonSettings(significance_level=1.5)


def test_baseline_with_no_overrides_yields_no_parameters(suite: SuiteConfig) -> None:
    sut = ExperimentSuiteRunner(runner=MagicMock())
    configs = sut.build_configs(suite, now=datetime(2026, 4, 1, tzinfo=UTC))
    assert configs[0].parameters is None
