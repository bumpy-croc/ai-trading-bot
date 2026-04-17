"""A→B workflow test: promote from suite A, reload patch as suite B's config."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.experiments.ledger import Ledger
from src.experiments.promotion import PromotionManager
from src.experiments.reporter import ExperimentReporter, Verdict
from src.experiments.schemas import ExperimentConfig, ExperimentResult
from src.experiments.suite import (
    BacktestSettings,
    ComparisonSettings,
    ExperimentSuiteRunner,
    SuiteConfig,
    VariantSpec,
)
from src.experiments.suite_loader import load_suite

pytestmark = pytest.mark.fast


def _result_with_sharpe(cfg: ExperimentConfig, sharpe: float) -> ExperimentResult:
    return ExperimentResult(
        config=cfg,
        total_trades=200,
        win_rate=55.0,
        total_return=sharpe * 2.0,
        annualized_return=sharpe * 4.0,
        max_drawdown=5.0,
        sharpe_ratio=sharpe,
        final_balance=1000.0 * (1 + sharpe / 10.0),
    )


def test_patch_yaml_reuse_preserves_promoted_state(tmp_path: Path) -> None:
    # ---- Suite A: baseline + 2 variants, promote the winner -----------------
    suite_a = SuiteConfig(
        id="suite_a",
        description="first pass",
        backtest=BacktestSettings(
            strategy="ml_basic",
            provider="mock",
            days=30,
            random_seed=42,
            # Pin window to remove wall-clock drift
            start=datetime(2025, 1, 1, tzinfo=UTC),
            end=datetime(2025, 4, 1, tzinfo=UTC),
        ),
        baseline=VariantSpec(name="default"),
        variants=[
            VariantSpec(
                name="winner",
                overrides={"ml_basic.long_entry_threshold": 0.0003},
            ),
            VariantSpec(
                name="loser",
                overrides={"ml_basic.long_entry_threshold": 0.01},
            ),
        ],
        comparison=ComparisonSettings(min_trades=0, significance_level=0.5),
    )

    def a_runner(cfg: ExperimentConfig) -> ExperimentResult:
        if cfg.parameters is None:
            return _result_with_sharpe(cfg, 1.0)
        if cfg.parameters.name == "winner":
            return _result_with_sharpe(cfg, 2.0)
        return _result_with_sharpe(cfg, 0.5)

    mock_a = MagicMock(run=MagicMock(side_effect=a_runner))
    suite_result_a = ExperimentSuiteRunner(runner=mock_a).run(suite_a)
    report_a = ExperimentReporter().render(suite_result_a)
    assert report_a.winner == "winner"

    ledger = Ledger(root=tmp_path / "history")
    manager = PromotionManager(
        ledger=ledger,
        versions_root=tmp_path / ".versions",
        lineage_root=tmp_path / ".lineage",
        promoted_root=tmp_path / "promoted",
    )
    artifacts_a = ledger.artifacts_dir("suite_a", "run_a")
    manager.record_run(suite_result_a, report_a, artifacts_a)
    outcome = manager.promote_with_outcome(suite_id="suite_a", variant_name="winner")

    # Patch YAML should carry the promoted state as its baseline.
    # ---- Suite B: reload patch, add a new variant --------------------------
    patch_cfg = load_suite(outcome.patch_yaml_path)

    # Patch YAML baseline carries the promoted overrides
    assert patch_cfg.baseline.overrides == {"ml_basic.long_entry_threshold": 0.0003}
    # Window pin round-trips through the patch
    assert patch_cfg.backtest.start == datetime(2025, 1, 1, tzinfo=UTC)
    assert patch_cfg.backtest.end == datetime(2025, 4, 1, tzinfo=UTC)
    assert patch_cfg.backtest.random_seed == 42

    # Layer a new variant on top and verify inheritance.
    suite_b = SuiteConfig(
        id=patch_cfg.id + "_with_variants",
        description=patch_cfg.description,
        backtest=patch_cfg.backtest,
        baseline=patch_cfg.baseline,
        variants=[
            VariantSpec(
                name="tight_stop_on_promoted",
                overrides={"ml_basic.stop_loss_pct": 0.02},
            )
        ],
        comparison=patch_cfg.comparison,
    )

    configs_b = ExperimentSuiteRunner(runner=MagicMock()).build_configs(
        suite_b, now=datetime(2025, 12, 31, tzinfo=UTC)  # irrelevant — pinned
    )

    # Baseline inherits promoted state.
    assert configs_b[0].parameters is not None
    assert configs_b[0].parameters.values == {"ml_basic.long_entry_threshold": 0.0003}

    # Variant carries promoted state + its own override.
    assert configs_b[1].parameters is not None
    assert configs_b[1].parameters.values == {
        "ml_basic.long_entry_threshold": 0.0003,  # inherited from promoted baseline
        "ml_basic.stop_loss_pct": 0.02,  # variant's own
    }

    # Pinned window unchanged across suites.
    assert configs_b[0].start == datetime(2025, 1, 1, tzinfo=UTC)
    assert configs_b[0].end == datetime(2025, 4, 1, tzinfo=UTC)


def test_patch_yaml_freezes_window_even_when_source_was_relative(tmp_path: Path) -> None:
    """Suite A used relative `days` window; promoted patch must still pin."""
    suite_a = SuiteConfig(
        id="unpinned_a",
        description="",
        backtest=BacktestSettings(strategy="ml_basic", provider="mock", days=30),
        baseline=VariantSpec(name="default"),
        variants=[
            VariantSpec(
                name="winner",
                overrides={"ml_basic.long_entry_threshold": 0.0003},
            )
        ],
        comparison=ComparisonSettings(min_trades=0, significance_level=0.5),
    )

    def a_runner(cfg: ExperimentConfig) -> ExperimentResult:
        return _result_with_sharpe(cfg, 2.0 if cfg.parameters else 1.0)

    now = datetime(2025, 6, 1, tzinfo=UTC)
    mock_runner = MagicMock(run=MagicMock(side_effect=a_runner))
    suite_result = ExperimentSuiteRunner(runner=mock_runner).run(suite_a, now=now)
    report = ExperimentReporter().render(suite_result)
    assert report.winner == "winner"

    ledger = Ledger(root=tmp_path / "history")
    manager = PromotionManager(
        ledger=ledger,
        versions_root=tmp_path / ".versions",
        lineage_root=tmp_path / ".lineage",
        promoted_root=tmp_path / "promoted",
    )
    artifacts = ledger.artifacts_dir("unpinned_a", "run1")
    manager.record_run(suite_result, report, artifacts)
    outcome = manager.promote_with_outcome(suite_id="unpinned_a", variant_name="winner")

    patch = load_suite(outcome.patch_yaml_path)
    # Patch has resolved start/end even though source didn't pin.
    assert patch.backtest.start is not None
    assert patch.backtest.end is not None
    assert patch.backtest.end == now  # matches the run-time window


def test_errored_variant_context_is_persisted_and_rendered(tmp_path: Path) -> None:
    suite = SuiteConfig(
        id="broken",
        description="",
        backtest=BacktestSettings(
            strategy="ml_basic",
            start=datetime(2025, 1, 1, tzinfo=UTC),
            end=datetime(2025, 4, 1, tzinfo=UTC),
        ),
        baseline=VariantSpec(name="default"),
        variants=[
            VariantSpec(name="good", overrides={"ml_basic.stop_loss_pct": 0.03}),
            VariantSpec(name="broken", overrides={"ml_basic.stop_loss_pct": 0.04}),
        ],
        comparison=ComparisonSettings(min_trades=0),
    )

    def a_runner(cfg: ExperimentConfig) -> ExperimentResult:
        if cfg.parameters is not None and cfg.parameters.name == "broken":
            raise RuntimeError("simulated backtest failure")
        return _result_with_sharpe(cfg, 1.0)

    mock_runner = MagicMock(run=MagicMock(side_effect=a_runner))
    suite_result = ExperimentSuiteRunner(runner=mock_runner).run(suite)
    report = ExperimentReporter().render(suite_result)

    # Report carries errors map
    assert "broken" in report.errors
    assert "simulated backtest failure" in report.errors["broken"]

    # Text report surfaces errors so the user doesn't need scrollback
    text = ExperimentReporter().render_text(report)
    assert "Variant errors" in text
    assert "simulated backtest failure" in text

    # Ranking pushes ERRORED to bottom
    assert report.rows[-1].verdict == Verdict.ERRORED
    assert report.rows[-1].name == "broken"

    # Persisted artifacts capture the errors
    ledger = Ledger(root=tmp_path / "history")
    manager = PromotionManager(
        ledger=ledger,
        versions_root=tmp_path / ".versions",
        lineage_root=tmp_path / ".lineage",
        promoted_root=tmp_path / "promoted",
    )
    artifacts = ledger.artifacts_dir("broken", "run1")
    manager.record_run(suite_result, report, artifacts)

    import json

    snapshot = json.loads((artifacts / "suite.json").read_text())
    assert snapshot["errors"]["broken"] == report.errors["broken"]


def test_post_override_base_fraction_out_of_bounds_raises(tmp_path: Path) -> None:
    from src.experiments.runner import ExperimentRunner
    from src.experiments.schemas import ParameterSet

    runner = ExperimentRunner()
    strategy = runner._load_strategy("ml_basic")
    cfg = ExperimentConfig(
        strategy_name="ml_basic",
        symbol="BTCUSDT",
        timeframe="1h",
        start=datetime.now(UTC),
        end=datetime.now(UTC),
        initial_balance=1000.0,
        parameters=ParameterSet(
            name="bad_frac",
            values={"ml_basic.base_fraction": 2.0},  # out of [0.001, 0.5]
        ),
    )
    runner._apply_parameter_overrides(strategy, cfg)
    with pytest.raises(ValueError, match="base_fraction"):
        runner._validate_post_override_invariants(strategy)


def test_errored_rows_sort_below_losing_baseline() -> None:
    """A crashed variant must not visually outrank a loss-making baseline."""
    cfg = ExperimentConfig(
        strategy_name="ml_basic",
        symbol="BTCUSDT",
        timeframe="1h",
        start=datetime.now(UTC),
        end=datetime.now(UTC),
        initial_balance=1000.0,
    )
    baseline = ExperimentResult(
        config=cfg,
        total_trades=200,
        win_rate=45.0,
        total_return=-5.0,
        annualized_return=-10.0,
        max_drawdown=8.0,
        sharpe_ratio=-0.5,
        final_balance=950.0,
    )
    errored_stub = ExperimentResult(
        config=cfg,
        total_trades=0,
        win_rate=0.0,
        total_return=0.0,
        annualized_return=0.0,
        max_drawdown=0.0,
        sharpe_ratio=0.0,
        final_balance=1000.0,
    )

    from src.experiments.suite import SuiteResult

    suite = SuiteConfig(
        id="mixed",
        description="",
        backtest=BacktestSettings(strategy="ml_basic"),
        baseline=VariantSpec(name="baseline"),
        variants=[VariantSpec(name="broke")],
        comparison=ComparisonSettings(min_trades=0),
    )
    result = SuiteResult(
        suite_id=suite.id,
        config=suite,
        baseline=baseline,
        variants=[errored_stub],
        started_at=datetime.now(UTC),
        finished_at=datetime.now(UTC),
        errors={"broke": "boom"},
    )
    report = ExperimentReporter().render(result)

    # baseline (sharpe=-0.5) must rank above errored — errored goes to the bottom
    assert report.rows[0].name == "baseline"
    assert report.rows[-1].name == "broke"
    assert report.rows[-1].verdict == Verdict.ERRORED
