"""Integration test: run a small suite end-to-end and check artifacts."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.experiments.ledger import Ledger
from src.experiments.promotion import PromotionManager
from src.experiments.reporter import ExperimentReporter
from src.experiments.schemas import ExperimentConfig, ExperimentResult
from src.experiments.suite import ExperimentSuiteRunner
from src.experiments.suite_loader import parse_suite

pytestmark = pytest.mark.integration


def _fake_result(cfg: ExperimentConfig, *, sharpe: float, trades: int = 120) -> ExperimentResult:
    return ExperimentResult(
        config=cfg,
        total_trades=trades,
        win_rate=55.0,
        total_return=sharpe * 2.0,
        annualized_return=sharpe * 4.0,
        max_drawdown=5.0,
        sharpe_ratio=sharpe,
        final_balance=1000.0 * (1 + sharpe / 10.0),
    )


def _stubbed_runner() -> MagicMock:
    runner = MagicMock()
    runner.run.side_effect = lambda cfg: _fake_result(cfg, sharpe=_variant_sharpe(cfg), trades=150)
    return runner


def _variant_sharpe(cfg: ExperimentConfig) -> float:
    if cfg.parameters is None:
        return 1.0
    if cfg.parameters.name == "boost":
        return 2.0
    if cfg.parameters.name == "regression":
        return 0.5
    return 1.1


def test_suite_round_trip_with_promotion(tmp_path: Path) -> None:
    raw = {
        "id": "integration_suite",
        "description": "integration",
        "backtest": {
            "strategy": "ml_basic",
            "symbol": "BTCUSDT",
            "timeframe": "1h",
            "days": 30,
            "initial_balance": 1000,
            "provider": "mock",
            "random_seed": 42,
        },
        "baseline": {"name": "baseline", "overrides": {}},
        "variants": [
            {"name": "boost", "overrides": {"ml_basic.long_entry_threshold": 0.0003}},
            {"name": "regression", "overrides": {"ml_basic.long_entry_threshold": 0.01}},
        ],
        "comparison": {
            "target_metric": "sharpe_ratio",
            "min_trades": 50,
            "significance_level": 0.5,
        },
    }
    suite = parse_suite(raw)

    runner = _stubbed_runner()
    suite_runner = ExperimentSuiteRunner(runner=runner)
    suite_result = suite_runner.run(suite)

    reporter = ExperimentReporter()
    report = reporter.render(suite_result)
    assert report.baseline_name == "baseline"
    assert {r.name for r in report.rows} == {"baseline", "boost", "regression"}

    ledger_root = tmp_path / "history"
    ledger = Ledger(root=ledger_root)
    promotion_manager = PromotionManager(
        ledger=ledger,
        versions_root=tmp_path / ".versions",
        lineage_root=tmp_path / ".lineage",
        promoted_root=tmp_path / "promoted",
        reporter=reporter,
    )

    artifacts_path = ledger.artifacts_dir(suite.id, "run1")
    promotion_manager.record_run(suite_result, report, artifacts_path)

    assert (artifacts_path / "report.json").exists()
    assert (artifacts_path / "report.csv").exists()
    assert (artifacts_path / "report.txt").exists()
    assert (artifacts_path / "suite.json").exists()

    entries = ledger.list()
    assert len(entries) == 1
    assert entries[0].suite_id == "integration_suite"

    # Winner exists → promotion should succeed without --force.
    assert report.winner == "boost"
    outcome = promotion_manager.promote_with_outcome(suite_id=suite.id, variant_name="boost")
    assert outcome.patch_yaml_path.exists()
    assert outcome.version_record_path.exists()
    assert outcome.change_record_path.exists()

    # Force-promoting the regression variant must still work.
    reg_outcome = promotion_manager.promote_with_outcome(
        suite_id=suite.id, variant_name="regression", force=True
    )
    assert reg_outcome.patch_yaml_path.exists()
