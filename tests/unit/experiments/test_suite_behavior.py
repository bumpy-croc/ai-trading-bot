"""Behavior tests for baseline-inheritance, ERRORED verdict, pinned windows."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest

from src.experiments.reporter import ExperimentReporter, Verdict
from src.experiments.schemas import ExperimentConfig, ExperimentResult
from src.experiments.suite import (
    BacktestSettings,
    ComparisonSettings,
    ExperimentSuiteRunner,
    SuiteConfig,
    VariantSpec,
)


def _fake_result(cfg: ExperimentConfig) -> ExperimentResult:
    return ExperimentResult(
        config=cfg,
        total_trades=100,
        win_rate=55.0,
        total_return=1.0,
        annualized_return=2.0,
        max_drawdown=5.0,
        sharpe_ratio=1.2,
        final_balance=1010.0,
    )


def test_variants_inherit_baseline_overrides() -> None:
    """Patch YAML round-trip: variants see baseline.overrides merged in."""
    suite = SuiteConfig(
        id="inherit_test",
        description="",
        backtest=BacktestSettings(strategy="ml_basic", provider="mock"),
        baseline=VariantSpec(
            name="promoted_state",
            overrides={"ml_basic.long_entry_threshold": 0.0007},
        ),
        variants=[
            VariantSpec(
                name="add_tight_stop",
                overrides={"ml_basic.stop_loss_pct": 0.02},
            )
        ],
    )

    runner = ExperimentSuiteRunner(runner=MagicMock(run=MagicMock(side_effect=_fake_result)))
    configs = runner.build_configs(suite, now=datetime(2026, 4, 1, tzinfo=UTC))

    # Baseline config carries its own overrides
    assert configs[0].parameters is not None
    assert configs[0].parameters.values == {"ml_basic.long_entry_threshold": 0.0007}

    # Variant config merges baseline + variant (variant wins conflicts)
    assert configs[1].parameters is not None
    assert configs[1].parameters.values == {
        "ml_basic.long_entry_threshold": 0.0007,  # inherited from baseline
        "ml_basic.stop_loss_pct": 0.02,  # variant's own
    }


def test_variant_override_wins_on_conflict() -> None:
    suite = SuiteConfig(
        id="conflict_test",
        description="",
        backtest=BacktestSettings(strategy="ml_basic"),
        baseline=VariantSpec(
            name="baseline",
            overrides={"ml_basic.long_entry_threshold": 0.0003},
        ),
        variants=[
            VariantSpec(
                name="tighter",
                overrides={"ml_basic.long_entry_threshold": 0.001},
            )
        ],
    )
    runner = ExperimentSuiteRunner(runner=MagicMock(run=MagicMock(side_effect=_fake_result)))
    configs = runner.build_configs(suite, now=datetime(2026, 4, 1, tzinfo=UTC))
    # Variant's 0.001 wins
    assert configs[1].parameters is not None
    assert configs[1].parameters.values["ml_basic.long_entry_threshold"] == 0.001


def test_one_variant_failure_does_not_abort_suite() -> None:
    suite = SuiteConfig(
        id="partial_failure",
        description="",
        backtest=BacktestSettings(strategy="ml_basic"),
        baseline=VariantSpec(name="baseline"),
        variants=[
            VariantSpec(name="good", overrides={"ml_basic.stop_loss_pct": 0.03}),
            VariantSpec(name="broken", overrides={"ml_basic.stop_loss_pct": 0.04}),
            VariantSpec(name="still_good", overrides={"ml_basic.stop_loss_pct": 0.05}),
        ],
    )

    def run_side_effect(cfg: ExperimentConfig) -> ExperimentResult:
        if cfg.parameters is not None and cfg.parameters.name == "broken":
            raise RuntimeError("simulated backtest failure")
        return _fake_result(cfg)

    mock_runner = MagicMock(run=MagicMock(side_effect=run_side_effect))
    result = ExperimentSuiteRunner(runner=mock_runner).run(suite)

    # Baseline + 3 variants = 4 run calls
    assert mock_runner.run.call_count == 4
    assert "broken" in result.errors
    assert "simulated backtest failure" in result.errors["broken"]
    # The bad variant still appears in results (as a sentinel zero-trade row)
    assert len(result.variants) == 3
    good_names = {"good", "still_good"}
    assert {
        spec.name for spec in suite.variants if spec.name in good_names
    } == good_names  # surface check


def test_baseline_failure_aborts_suite() -> None:
    suite = SuiteConfig(
        id="baseline_failure",
        description="",
        backtest=BacktestSettings(strategy="ml_basic"),
        baseline=VariantSpec(name="baseline"),
        variants=[VariantSpec(name="v1")],
    )

    mock_runner = MagicMock(run=MagicMock(side_effect=RuntimeError("baseline broke")))
    with pytest.raises(RuntimeError, match="baseline broke"):
        ExperimentSuiteRunner(runner=mock_runner).run(suite)


def test_errored_verdict_renders_in_report() -> None:
    from src.experiments.suite import SuiteResult

    suite = SuiteConfig(
        id="erred",
        description="",
        backtest=BacktestSettings(strategy="ml_basic"),
        baseline=VariantSpec(name="baseline"),
        variants=[VariantSpec(name="v1"), VariantSpec(name="v2_broken")],
        comparison=ComparisonSettings(min_trades=0),
    )

    cfg = ExperimentConfig(
        strategy_name="ml_basic",
        symbol="BTCUSDT",
        timeframe="1h",
        start=datetime.now(UTC),
        end=datetime.now(UTC),
        initial_balance=1000.0,
    )
    baseline = _fake_result(cfg)
    good = _fake_result(cfg)
    # Sentinel zero-trade result for the broken variant
    errored = ExperimentResult(
        config=cfg,
        total_trades=0,
        win_rate=0.0,
        total_return=0.0,
        annualized_return=0.0,
        max_drawdown=0.0,
        sharpe_ratio=0.0,
        final_balance=1000.0,
    )

    suite_result = SuiteResult(
        suite_id=suite.id,
        config=suite,
        baseline=baseline,
        variants=[good, errored],
        started_at=datetime.now(UTC),
        finished_at=datetime.now(UTC),
        errors={"v2_broken": "RuntimeError: simulated"},
    )
    report = ExperimentReporter().render(suite_result)

    errored_row = next(r for r in report.rows if r.name == "v2_broken")
    assert errored_row.verdict == Verdict.ERRORED


def test_pinned_start_end_in_backtest_settings_avoids_drift() -> None:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    end = datetime(2025, 4, 1, tzinfo=UTC)
    settings = BacktestSettings(strategy="ml_basic", start=start, end=end)

    # `now` is ignored when start/end are pinned
    actual_start, actual_end = settings.resolve_window(now=datetime(2026, 12, 31, tzinfo=UTC))
    assert actual_start == start
    assert actual_end == end


def test_pinned_start_only_still_drifts_end() -> None:
    settings = BacktestSettings(
        strategy="ml_basic",
        start=datetime(2025, 1, 1, tzinfo=UTC),
        days=60,
    )
    now = datetime(2025, 6, 1, tzinfo=UTC)
    actual_start, actual_end = settings.resolve_window(now=now)
    assert actual_start == datetime(2025, 1, 1, tzinfo=UTC)
    assert actual_end == now
