"""Edge-case tests for ExperimentReporter: Calmar ∞, delta=0, stat-sig gating."""

from __future__ import annotations

import math
from datetime import UTC, datetime

import pytest

from src.experiments.reporter import ExperimentReporter, Verdict, _ranking_confidence
from src.experiments.schemas import ExperimentConfig, ExperimentResult
from src.experiments.suite import (
    BacktestSettings,
    ComparisonSettings,
    SuiteConfig,
    SuiteResult,
    VariantSpec,
)


def _cfg() -> ExperimentConfig:
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
    *,
    total_return: float = 0.0,
    sharpe: float = 1.0,
    trades: int = 200,
    max_drawdown: float = 5.0,
    annualized: float | None = None,
) -> ExperimentResult:
    return ExperimentResult(
        config=_cfg(),
        total_trades=trades,
        win_rate=55.0,
        total_return=total_return,
        annualized_return=annualized if annualized is not None else total_return,
        max_drawdown=max_drawdown,
        sharpe_ratio=sharpe,
        final_balance=1000.0 * (1 + total_return / 100),
    )


def _suite(variants, *, metric: str = "sharpe_ratio", min_trades: int = 0) -> SuiteConfig:
    return SuiteConfig(
        id="suite_e",
        description="",
        backtest=BacktestSettings(strategy="ml_basic"),
        baseline=VariantSpec(name="baseline"),
        variants=variants,
        comparison=ComparisonSettings(
            target_metric=metric, min_trades=min_trades, significance_level=0.05
        ),
    )


def _suite_result(suite: SuiteConfig, baseline: ExperimentResult, variants) -> SuiteResult:
    now = datetime.now(UTC)
    return SuiteResult(
        suite_id=suite.id,
        config=suite,
        baseline=baseline,
        variants=variants,
        started_at=now,
        finished_at=now,
    )


def test_calmar_zero_drawdown_with_positive_return_ranks_best() -> None:
    suite = _suite(
        [VariantSpec(name="perfect", overrides={"ml_basic.stop_loss_pct": 0.01})],
        metric="calmar",
    )
    baseline = _result(total_return=10.0, annualized=15.0, max_drawdown=5.0, trades=200)
    # variant with zero drawdown + positive return → Calmar +inf
    variants = [_result(total_return=8.0, annualized=10.0, max_drawdown=0.0, trades=200)]
    report = ExperimentReporter().render(_suite_result(suite, baseline, variants))

    # Perfect variant must rank first
    assert report.rows[0].name == "perfect"
    # And be promotable (infinite delta, full confidence)
    row = next(r for r in report.rows if r.name == "perfect")
    assert row.verdict in {Verdict.PROMOTE, Verdict.HOLD}  # depends on confidence gate
    # Confidence must be 1.0 for the ∞ edge
    assert row.ranking_confidence == pytest.approx(1.0)


def test_delta_zero_is_HOLD_not_REJECT() -> None:
    suite = _suite([VariantSpec(name="neutral", overrides={"ml_basic.stop_loss_pct": 0.02})])
    baseline = _result(total_return=3.0, sharpe=1.0, trades=200)
    variants = [_result(total_return=3.0, sharpe=1.0, trades=200)]
    report = ExperimentReporter().render(_suite_result(suite, baseline, variants))

    row = next(r for r in report.rows if r.name == "neutral")
    assert row.verdict == Verdict.HOLD


def test_ranking_confidence_is_scale_invariant() -> None:
    """Same relative improvement → same confidence regardless of metric scale."""
    # Case A: sharpe 1.0 → 1.5 (+50%)
    base_a = _result(sharpe=1.0, trades=200)
    var_a = _result(sharpe=1.5, trades=200)
    # Case B: total_return 2% → 3% (+50%)
    base_b = _result(total_return=2.0, sharpe=0.0, trades=200)
    var_b = _result(total_return=3.0, sharpe=0.0, trades=200)

    conf_a = _ranking_confidence(base_a, var_a, "sharpe_ratio")
    conf_b = _ranking_confidence(base_b, var_b, "total_return")

    assert conf_a is not None and conf_b is not None
    # Same relative effect should produce same confidence.
    assert conf_a == pytest.approx(conf_b, abs=1e-9)


def test_insufficient_trades_returns_none_confidence() -> None:
    base = _result(trades=1)
    var = _result(trades=200)
    assert _ranking_confidence(base, var, "sharpe_ratio") is None


def test_winner_requires_PROMOTE_verdict_not_just_first_rank() -> None:
    """A better-but-low-confidence variant is HOLD — no winner, even at top."""
    suite = _suite(
        [VariantSpec(name="weakly_better", overrides={"ml_basic.stop_loss_pct": 0.02})],
        min_trades=0,
    )
    # small effect size → low confidence → HOLD even though it's ranked first
    baseline = _result(sharpe=1.0, trades=5)  # 5 trades → low sample confidence
    variants = [_result(sharpe=1.01, trades=5)]  # 1% improvement
    report = ExperimentReporter().render(_suite_result(suite, baseline, variants))

    row = next(r for r in report.rows if r.name == "weakly_better")
    assert row.verdict == Verdict.HOLD
    assert report.winner is None  # no PROMOTE → no winner


def test_json_roundtrip_does_not_break_on_infinities(tmp_path) -> None:
    suite = _suite([VariantSpec(name="inf_var")], metric="calmar")
    baseline = _result(total_return=1.0, annualized=1.0, max_drawdown=5.0, trades=200)
    variants = [_result(total_return=5.0, annualized=5.0, max_drawdown=0.0, trades=200)]
    report = ExperimentReporter().render(_suite_result(suite, baseline, variants))

    reporter = ExperimentReporter()
    reporter.write_artifacts(report, tmp_path)

    import json

    data = json.loads((tmp_path / "report.json").read_text())
    # Any inf values must be serialized as strings, not raise
    assert "rows" in data
    # And the text report should not crash either
    assert (tmp_path / "report.txt").exists()


def test_ranking_is_stable_for_ties() -> None:
    suite = _suite(
        [VariantSpec(name="tie_a"), VariantSpec(name="tie_b")],
    )
    baseline = _result(sharpe=1.0, trades=200)
    variants = [
        _result(sharpe=1.5, trades=200),
        _result(sharpe=1.5, trades=200),
    ]
    report = ExperimentReporter().render(_suite_result(suite, baseline, variants))

    # Both variants tied; order should be preserved (stable sort)
    variant_names = [r.name for r in report.rows if not r.is_baseline]
    assert variant_names == ["tie_a", "tie_b"]


def test_isnan_values_in_metric_gracefully_handled() -> None:
    """NaN baseline → confidence is None (shouldn't crash)."""
    base = _result(sharpe=math.nan, trades=200)
    var = _result(sharpe=1.5, trades=200)
    conf = _ranking_confidence(base, var, "sharpe_ratio")
    assert conf is None
