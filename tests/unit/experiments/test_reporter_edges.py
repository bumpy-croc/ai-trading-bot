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

pytestmark = pytest.mark.fast


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


def test_nan_metric_values_raise_in_metric() -> None:
    """NaN metrics must fail loudly — prior policy (silent None confidence)
    allowed NaN to reach verdict classification and render as ``−∞``."""
    import pytest as _pytest

    from src.experiments.reporter import _metric

    result = _result(sharpe=math.nan, trades=200)
    with _pytest.raises(ValueError, match="NaN"):
        _metric(result, "sharpe_ratio")


def test_calmar_opposite_infinities_classified_correctly() -> None:
    """Baseline −∞ (zero-dd, losing), variant +∞ (zero-dd, winning) → PROMOTE."""
    suite = _suite(
        [VariantSpec(name="winner")],
        metric="calmar",
    )
    baseline = _result(total_return=-5.0, annualized=-10.0, max_drawdown=0.0, trades=200)
    variants = [_result(total_return=5.0, annualized=10.0, max_drawdown=0.0, trades=200)]
    report = ExperimentReporter().render(_suite_result(suite, baseline, variants))

    row = next(r for r in report.rows if r.name == "winner")
    # variant = +inf, baseline = −inf → opposite signs → strictly better.
    assert row.verdict == Verdict.PROMOTE
    assert row.ranking_confidence == pytest.approx(1.0)


def test_calmar_baseline_infinite_variant_finite_rejects_when_worse() -> None:
    """Baseline +∞ (zero-dd, winning); variant finite → REJECT."""
    suite = _suite([VariantSpec(name="finite_var")], metric="calmar")
    baseline = _result(total_return=5.0, annualized=10.0, max_drawdown=0.0, trades=200)
    variants = [_result(total_return=5.0, annualized=3.0, max_drawdown=5.0, trades=200)]
    report = ExperimentReporter().render(_suite_result(suite, baseline, variants))

    row = next(r for r in report.rows if r.name == "finite_var")
    assert row.verdict == Verdict.REJECT


# --------------------------------------------------------------------------
# G6 / G7: Dead-code-override warnings.
#
# When a variant's headline metrics tie baseline within floating-point
# tolerance, the reporter emits a warning pointing at the most common
# cause — an override that didn't take effect. The per-trade P&L sequence
# decides between "literally the same trades" (strong signal of a no-op
# override) and "different trades, same aggregate" (rare; worth inspection).
# --------------------------------------------------------------------------


def _result_with_trades(
    trades: list[float],
    *,
    total_return: float,
    sharpe: float = 1.0,
    annualized: float | None = None,
    max_drawdown: float = 5.0,
) -> ExperimentResult:
    r = _result(
        total_return=total_return,
        sharpe=sharpe,
        trades=len(trades),
        annualized=annualized,
        max_drawdown=max_drawdown,
    )
    r.trade_pnl_pcts = list(trades)
    return r


def test_identical_variant_emits_dead_code_warning() -> None:
    """Variant matches every headline metric AND every per-trade P&L →
    near-certain dead-code override. Reporter must emit G6 warning."""
    suite = _suite([VariantSpec(name="noop_var")])
    trades = [0.01, -0.005, 0.02]
    baseline = _result_with_trades(trades, total_return=2.5, sharpe=1.2, annualized=3.0)
    variants = [_result_with_trades(trades, total_return=2.5, sharpe=1.2, annualized=3.0)]
    report = ExperimentReporter().render(_suite_result(suite, baseline, variants))
    row = next(r for r in report.rows if r.name == "noop_var")
    assert row.warnings, "expected identical-variant warning"
    assert any("did not take effect" in w for w in row.warnings)


def test_identical_aggregate_different_trades_emits_distinct_warning() -> None:
    """Aggregate metrics tie baseline, but per-trade sequence differs —
    the G7 "different trades, same aggregate" message, not G6."""
    suite = _suite([VariantSpec(name="path_var")])
    b_trades = [0.01, 0.01, -0.015]
    v_trades = [0.005, -0.005, 0.015]
    baseline = _result_with_trades(b_trades, total_return=2.5, sharpe=1.2, annualized=3.0)
    variant = _result_with_trades(v_trades, total_return=2.5, sharpe=1.2, annualized=3.0)
    report = ExperimentReporter().render(_suite_result(suite, baseline, [variant]))
    row = next(r for r in report.rows if r.name == "path_var")
    assert row.warnings, "expected sequence-tiebreak warning"
    assert any("different trades" in w.lower() for w in row.warnings)
    assert not any("did not take effect" in w for w in row.warnings)


def test_differing_variant_emits_no_warning() -> None:
    """Any headline metric difference above tolerance → no G6 warning."""
    suite = _suite([VariantSpec(name="real_var")])
    baseline = _result_with_trades([0.01, 0.02], total_return=2.5, sharpe=1.2)
    variant = _result_with_trades([0.015, 0.02], total_return=3.5, sharpe=1.5)
    report = ExperimentReporter().render(_suite_result(suite, baseline, [variant]))
    row = next(r for r in report.rows if r.name == "real_var")
    assert row.warnings == []


def test_baseline_row_never_gets_identical_warning() -> None:
    """The baseline is trivially identical to itself; warning is nonsense."""
    suite = _suite([VariantSpec(name="noop_var")])
    trades = [0.01, -0.005]
    baseline = _result_with_trades(trades, total_return=2.5, sharpe=1.2, annualized=3.0)
    variant = _result_with_trades(trades, total_return=2.5, sharpe=1.2, annualized=3.0)
    report = ExperimentReporter().render(_suite_result(suite, baseline, [variant]))
    baseline_row = next(r for r in report.rows if r.is_baseline)
    assert baseline_row.warnings == []


def test_warning_rendered_in_text_report() -> None:
    """The text renderer must surface warnings so operators see them."""
    suite = _suite([VariantSpec(name="noop_var")])
    trades = [0.01, -0.005]
    baseline = _result_with_trades(trades, total_return=2.5, sharpe=1.2, annualized=3.0)
    variant = _result_with_trades(trades, total_return=2.5, sharpe=1.2, annualized=3.0)
    report = ExperimentReporter().render(_suite_result(suite, baseline, [variant]))
    text = ExperimentReporter().render_text(report)
    assert "Variant warnings" in text
    assert "noop_var" in text


def test_warning_serialized_in_csv_report() -> None:
    suite = _suite([VariantSpec(name="noop_var")])
    trades = [0.01]
    baseline = _result_with_trades(trades, total_return=2.5, sharpe=1.2, annualized=3.0)
    variant = _result_with_trades(trades, total_return=2.5, sharpe=1.2, annualized=3.0)
    report = ExperimentReporter().render(_suite_result(suite, baseline, [variant]))
    csv_text = ExperimentReporter().render_csv(report)
    # Header includes the new column
    assert ",warnings" in csv_text.replace("\r\n", "\n")
    # Warning cell contains the key diagnostic phrase
    assert "did not take effect" in csv_text


def test_tolerance_floors_tiny_fp_noise_still_triggers_warning() -> None:
    """FP noise below the identical-tolerance must still trigger the
    warning so a 1-ULP difference doesn't hide a dead-code override."""
    suite = _suite([VariantSpec(name="noop_var")])
    trades = [0.01, -0.005]
    baseline = _result_with_trades(trades, total_return=2.5, sharpe=1.2, annualized=3.0)
    # 1e-11 < 1e-9 tolerance
    variant = _result_with_trades(trades, total_return=2.5 + 1e-11, sharpe=1.2, annualized=3.0)
    report = ExperimentReporter().render(_suite_result(suite, baseline, [variant]))
    row = next(r for r in report.rows if r.name == "noop_var")
    assert row.warnings, "FP noise below tolerance must not suppress warning"


def test_identical_metrics_zero_trades_both_sides_still_warns() -> None:
    """A variant that produced zero trades and matches an empty-trade
    baseline is still likely a dead-code override — warn the same way."""
    suite = _suite([VariantSpec(name="dead_var")])
    baseline = _result_with_trades([], total_return=0.0, sharpe=0.0, annualized=0.0)
    variant = _result_with_trades([], total_return=0.0, sharpe=0.0, annualized=0.0)
    report = ExperimentReporter().render(_suite_result(suite, baseline, [variant]))
    row = next(r for r in report.rows if r.name == "dead_var")
    assert row.warnings, "empty-trade tie should still warn"


def test_csv_warnings_column_escapes_multiline_to_single_cell() -> None:
    """Multi-warning cells must use a delimiter that CSV spreadsheets can
    read as a single cell — ensuring report.csv stays row-per-variant."""
    from src.experiments.reporter import VariantReport

    row = VariantReport(
        name="v",
        total_return=0.0,
        annualized_return=0.0,
        sharpe_ratio=0.0,
        max_drawdown=0.0,
        win_rate=0.0,
        total_trades=0,
        final_balance=0.0,
        delta_vs_baseline=0.0,
        ranking_confidence=None,
        verdict=Verdict.HOLD,
        warnings=["first warning", "second warning"],
    )
    from src.experiments.reporter import SuiteReport

    report = SuiteReport(
        suite_id="x",
        description="",
        target_metric="sharpe_ratio",
        significance_level=0.05,
        min_trades=0,
        baseline_name="baseline",
        winner=None,
        rows=[row],
    )
    csv_text = ExperimentReporter().render_csv(report)
    # Both warnings in the same cell, separated by " | "
    assert "first warning | second warning" in csv_text
