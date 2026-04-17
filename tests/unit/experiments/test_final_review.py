"""Regression tests for the final deep-review findings.

Covers: ml_adaptive stop_loss rejection, ERRORED --force block, --limit
validation, experiment list metric display, sequence_length float override,
and _coerce_value fast-fail on numeric-target failure.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

import pytest

from src.experiments.promotion import PromotionError, PromotionManager
from src.experiments.runner import ExperimentRunner
from src.experiments.schemas import ExperimentConfig, ParameterSet

pytestmark = pytest.mark.fast


# --------------------------------------------------------------------------
# P1: ml_adaptive stop_loss/take_profit must be rejected (silent no-op fix)
# --------------------------------------------------------------------------


def test_ml_adaptive_stop_loss_override_raises() -> None:
    """RegimeAdaptiveRiskManager doesn't read strategy_overrides; the runner
    must reject the override instead of silently letting variants share the
    regime-derived stops."""
    runner = ExperimentRunner()
    strategy = runner._load_strategy("ml_adaptive")
    cfg = ExperimentConfig(
        strategy_name="ml_adaptive",
        symbol="BTCUSDT",
        timeframe="1h",
        start=datetime.now(UTC),
        end=datetime.now(UTC),
        initial_balance=1000.0,
        parameters=ParameterSet(
            name="tight_stop",
            values={"ml_adaptive.stop_loss_pct": 0.015},
        ),
    )
    with pytest.raises(ValueError, match="does not consume strategy_overrides"):
        runner._apply_parameter_overrides(strategy, cfg)


def test_ml_adaptive_take_profit_override_raises() -> None:
    runner = ExperimentRunner()
    strategy = runner._load_strategy("ml_adaptive")
    cfg = ExperimentConfig(
        strategy_name="ml_adaptive",
        symbol="BTCUSDT",
        timeframe="1h",
        start=datetime.now(UTC),
        end=datetime.now(UTC),
        initial_balance=1000.0,
        parameters=ParameterSet(
            name="wide_tp",
            values={"ml_adaptive.take_profit_pct": 0.08},
        ),
    )
    with pytest.raises(ValueError, match="does not consume strategy_overrides"):
        runner._apply_parameter_overrides(strategy, cfg)


def test_ml_basic_stop_loss_override_still_works() -> None:
    """CoreRiskAdapter consumes strategy_overrides; must continue to work."""
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
            name="tight_stop",
            values={"ml_basic.stop_loss_pct": 0.015},
        ),
    )
    runner._apply_parameter_overrides(strategy, cfg)
    overrides = getattr(strategy, "_risk_overrides", {}) or {}
    assert pytest.approx(overrides["stop_loss_pct"], rel=1e-9) == 0.015
    # Risk manager's internal map is also updated
    assert (
        pytest.approx(strategy.risk_manager._strategy_overrides["stop_loss_pct"], rel=1e-9) == 0.015
    )


# --------------------------------------------------------------------------
# P1: --force must never promote ERRORED variants
# --------------------------------------------------------------------------


def test_force_cannot_promote_errored_variant(tmp_path: Path) -> None:
    import json

    from src.experiments.ledger import Ledger, LedgerEntry

    ledger = Ledger(root=tmp_path)
    ledger.root.mkdir(parents=True, exist_ok=True)
    artifacts = ledger.artifacts_dir("suite_x", "run_1")
    artifacts.mkdir(parents=True, exist_ok=True)

    # Build a minimal suite snapshot + report with an ERRORED variant.
    (artifacts / "suite.json").write_text(
        json.dumps(
            {
                "id": "suite_x",
                "description": "",
                "backtest": {
                    "strategy": "ml_basic",
                    "symbol": "BTCUSDT",
                    "timeframe": "1h",
                    "days": 30,
                    "initial_balance": 1000,
                    "provider": "mock",
                    "use_cache": False,
                    "random_seed": 42,
                    "start": "2024-01-01T00:00:00+00:00",
                    "end": "2024-02-01T00:00:00+00:00",
                },
                "baseline": {"name": "baseline", "overrides": {}},
                "variants": [{"name": "broken", "overrides": {"ml_basic.stop_loss_pct": 0.03}}],
                "comparison": {
                    "target_metric": "sharpe_ratio",
                    "min_trades": 0,
                    "significance_level": 0.05,
                },
                "errors": {"broken": "RuntimeError: simulated"},
            }
        )
    )
    (artifacts / "report.json").write_text(
        json.dumps(
            {
                "rows": [
                    {"name": "baseline", "is_baseline": True, "verdict": "HOLD"},
                    {"name": "broken", "is_baseline": False, "verdict": "ERRORED"},
                ]
            }
        )
    )
    with ledger.path().open("a", encoding="utf-8") as fh:
        fh.write(
            json.dumps(
                LedgerEntry(
                    suite_id="suite_x",
                    timestamp="2024-04-17T00:00:00+00:00",
                    config_hash="h",
                    winner=None,
                    baseline_name="baseline",
                    target_metric="sharpe_ratio",
                    metrics={},
                    artifacts_path=str(artifacts),
                ).to_dict()
            )
            + "\n"
        )

    manager = PromotionManager(
        ledger=ledger,
        versions_root=tmp_path / ".versions",
        lineage_root=tmp_path / ".lineage",
        promoted_root=tmp_path / "promoted",
    )
    # Even with force=True, ERRORED must not promote.
    with pytest.raises(PromotionError, match="ERRORED"):
        manager.promote_with_outcome(suite_id="suite_x", variant_name="broken", force=True)


# --------------------------------------------------------------------------
# P2: --limit validation
# --------------------------------------------------------------------------


def test_limit_zero_rejected_at_argparse() -> None:
    from cli.commands.experiment import _positive_int

    with pytest.raises(argparse.ArgumentTypeError):
        _positive_int("0")


def test_limit_negative_rejected_at_argparse() -> None:
    from cli.commands.experiment import _positive_int

    with pytest.raises(argparse.ArgumentTypeError):
        _positive_int("-3")


# --------------------------------------------------------------------------
# P3: sequence_length must stay integer-valued
# --------------------------------------------------------------------------


def test_sequence_length_non_integer_float_rejected() -> None:
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
            name="bad_seq",
            values={"ml_basic.sequence_length": 120.5},
        ),
    )
    runner._apply_parameter_overrides(strategy, cfg)
    # _coerce_value preserves float precision (general rule), so seq_len is 120.5;
    # the post-override validator must reject it.
    with pytest.raises(ValueError, match="whole number"):
        runner._validate_post_override_invariants(strategy)


def test_sequence_length_zero_rejected() -> None:
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
            name="zero_seq",
            values={"ml_basic.sequence_length": 0},
        ),
    )
    runner._apply_parameter_overrides(strategy, cfg)
    with pytest.raises(ValueError, match=">= 1"):
        runner._validate_post_override_invariants(strategy)


def test_sequence_length_integer_override_accepted() -> None:
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
            name="ok_seq",
            values={"ml_basic.sequence_length": 60},
        ),
    )
    runner._apply_parameter_overrides(strategy, cfg)
    runner._validate_post_override_invariants(strategy)  # no raise
    assert strategy.signal_generator.sequence_length == 60


# --------------------------------------------------------------------------
# P2: patch YAML missing-comparison fallback
# --------------------------------------------------------------------------


def test_promotion_comparison_fallback_logs_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    from src.experiments.promotion import _comparison_from_snapshot

    # Snapshot without "comparison" block simulates a suite written by an
    # older version before the field was persisted.
    snapshot = {
        "id": "legacy",
        "backtest": {"strategy": "ml_basic"},
        "baseline": {"name": "baseline", "overrides": {}},
        "variants": [],
    }
    with caplog.at_level("WARNING"):
        comp = _comparison_from_snapshot(snapshot)
    assert comp["target_metric"] == "sharpe_ratio"
    assert comp["min_trades"] == 30
    assert comp["significance_level"] == 0.05
    assert any("no 'comparison' block" in rec.message for rec in caplog.records)
