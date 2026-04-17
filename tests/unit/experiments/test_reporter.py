"""Unit tests for ExperimentReporter."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from src.experiments.reporter import ExperimentReporter, Verdict
from src.experiments.schemas import ExperimentConfig, ExperimentResult
from src.experiments.suite import (
    BacktestSettings,
    ComparisonSettings,
    SuiteConfig,
    SuiteResult,
    VariantSpec,
)

pytestmark = pytest.mark.fast


def _cfg(name: str) -> ExperimentConfig:
    now = datetime.now(UTC)
    return ExperimentConfig(
        strategy_name="ml_basic",
        symbol="BTCUSDT",
        timeframe="1h",
        start=now,
        end=now,
        initial_balance=1000.0,
    )


def _result(
    name: str,
    *,
    total_return: float,
    sharpe: float,
    trades: int = 100,
    annualized: float = 0.0,
    drawdown: float = 5.0,
) -> ExperimentResult:
    return ExperimentResult(
        config=_cfg(name),
        total_trades=trades,
        win_rate=55.0,
        total_return=total_return,
        annualized_return=annualized or total_return,
        max_drawdown=drawdown,
        sharpe_ratio=sharpe,
        final_balance=1000.0 * (1 + total_return / 100),
    )


def _suite(variants: list[VariantSpec], min_trades: int = 0) -> SuiteConfig:
    return SuiteConfig(
        id="suite_x",
        description="",
        backtest=BacktestSettings(strategy="ml_basic"),
        baseline=VariantSpec(name="baseline"),
        variants=variants,
        comparison=ComparisonSettings(
            target_metric="sharpe_ratio", min_trades=min_trades, significance_level=0.05
        ),
    )


def _suite_result(
    suite: SuiteConfig, baseline: ExperimentResult, variants: list[ExperimentResult]
) -> SuiteResult:
    now = datetime.now(UTC)
    return SuiteResult(
        suite_id=suite.id,
        config=suite,
        baseline=baseline,
        variants=variants,
        started_at=now,
        finished_at=now,
    )


def test_report_ranks_by_target_metric() -> None:
    suite = _suite(
        [
            VariantSpec(name="better", overrides={"ml_basic.stop_loss_pct": 0.02}),
            VariantSpec(name="worse", overrides={"ml_basic.stop_loss_pct": 0.10}),
        ]
    )
    baseline = _result("baseline", total_return=2.0, sharpe=1.0, trades=200)
    variants = [
        _result("better", total_return=4.0, sharpe=1.5, trades=200),
        _result("worse", total_return=1.0, sharpe=0.5, trades=200),
    ]
    suite_res = _suite_result(suite, baseline, variants)
    report = ExperimentReporter().render(suite_res)

    names = [row.name for row in report.rows]
    assert names[0] == "better", f"expected 'better' first, got {names}"
    assert names[-1] == "worse"


def test_verdict_insufficient_data_when_trades_below_threshold() -> None:
    suite = _suite(
        [VariantSpec(name="few_trades", overrides={"ml_basic.stop_loss_pct": 0.03})],
        min_trades=50,
    )
    baseline = _result("baseline", total_return=2.0, sharpe=1.0, trades=200)
    variants = [_result("few_trades", total_return=10.0, sharpe=2.0, trades=10)]
    suite_res = _suite_result(suite, baseline, variants)
    report = ExperimentReporter().render(suite_res)

    row = next(r for r in report.rows if r.name == "few_trades")
    assert row.verdict == Verdict.INSUFFICIENT_DATA


def test_reject_when_variant_worse_than_baseline() -> None:
    suite = _suite(
        [VariantSpec(name="worse", overrides={"ml_basic.stop_loss_pct": 0.10})],
        min_trades=0,
    )
    baseline = _result("baseline", total_return=5.0, sharpe=1.5, trades=200)
    variants = [_result("worse", total_return=2.0, sharpe=0.8, trades=200)]
    suite_res = _suite_result(suite, baseline, variants)
    report = ExperimentReporter().render(suite_res)

    row = next(r for r in report.rows if r.name == "worse")
    assert row.verdict == Verdict.REJECT


def test_text_report_contains_header_and_verdict_column() -> None:
    suite = _suite([VariantSpec(name="v1", overrides={"ml_basic.stop_loss_pct": 0.02})])
    baseline = _result("baseline", total_return=2.0, sharpe=1.0, trades=200)
    variants = [_result("v1", total_return=3.0, sharpe=1.3, trades=200)]
    suite_res = _suite_result(suite, baseline, variants)
    report = ExperimentReporter().render(suite_res)

    text = ExperimentReporter().render_text(report)
    assert "Variant" in text
    assert "Verdict" in text
    assert "v1" in text


def test_csv_report_includes_rows_and_headers() -> None:
    suite = _suite([VariantSpec(name="v1")])
    baseline = _result("baseline", total_return=2.0, sharpe=1.0, trades=200)
    variants = [_result("v1", total_return=3.0, sharpe=1.3, trades=200)]
    suite_res = _suite_result(suite, baseline, variants)
    report = ExperimentReporter().render(suite_res)

    csv_text = ExperimentReporter().render_csv(report)
    first_line = csv_text.strip().splitlines()[0]
    assert "variant" in first_line and "sharpe_ratio" in first_line
    assert "v1" in csv_text
    assert "baseline" in csv_text


def test_delta_and_confidence_populated_for_variants() -> None:
    suite = _suite(
        [VariantSpec(name="v1", overrides={"ml_basic.stop_loss_pct": 0.03})], min_trades=0
    )
    baseline = _result("baseline", total_return=2.0, sharpe=1.0, trades=200)
    variants = [_result("v1", total_return=5.0, sharpe=2.0, trades=200)]
    suite_res = _suite_result(suite, baseline, variants)
    report = ExperimentReporter().render(suite_res)

    v1 = next(r for r in report.rows if r.name == "v1")
    assert pytest.approx(v1.delta_vs_baseline, rel=1e-9) == 1.0
    assert v1.ranking_confidence is not None
    assert 0.0 <= v1.ranking_confidence <= 1.0
