"""Tests for PromotionManager error paths and changed_components mapping."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.experiments.ledger import Ledger, LedgerEntry
from src.experiments.promotion import (
    PromotionError,
    PromotionManager,
    _components_touched,
)

pytestmark = pytest.mark.fast


def _write_suite_snapshot(path: Path, *, variants: list[dict]) -> None:
    path.write_text(
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
                },
                "baseline": {"name": "baseline", "overrides": {}},
                "variants": variants,
                "comparison": {
                    "target_metric": "calmar",
                    "min_trades": 100,
                    "significance_level": 0.01,
                },
                "winner": "promo_variant",
            }
        )
    )


def _write_report(path: Path, *, rows: list[dict]) -> None:
    path.write_text(json.dumps({"rows": rows}))


def _prep_ledger(tmp_path: Path, *, variant_verdict: str, winner: str | None) -> Ledger:
    ledger = Ledger(root=tmp_path)
    ledger.root.mkdir(parents=True, exist_ok=True)
    artifacts = tmp_path / "suite_x" / "run1"
    artifacts.mkdir(parents=True, exist_ok=True)
    _write_suite_snapshot(
        artifacts / "suite.json",
        variants=[
            {
                "name": "promo_variant",
                "overrides": {"ml_basic.long_entry_threshold": 0.0003},
            },
            {
                "name": "reject_variant",
                "overrides": {"ml_basic.long_entry_threshold": 0.01},
            },
        ],
    )
    _write_report(
        artifacts / "report.json",
        rows=[
            {
                "name": "baseline",
                "is_baseline": True,
                "sharpe_ratio": 1.0,
                "total_return": 2.0,
                "verdict": "HOLD",
            },
            {
                "name": "promo_variant",
                "is_baseline": False,
                "sharpe_ratio": 1.6,
                "total_return": 4.0,
                "verdict": variant_verdict,
            },
            {
                "name": "reject_variant",
                "is_baseline": False,
                "sharpe_ratio": 0.5,
                "total_return": 1.0,
                "verdict": "REJECT",
            },
        ],
    )
    # Inject a ledger entry pointing at those artifacts.
    with ledger.path().open("a", encoding="utf-8") as fh:
        fh.write(
            json.dumps(
                LedgerEntry(
                    suite_id="suite_x",
                    timestamp="2026-04-17T00:00:00+00:00",
                    config_hash="h",
                    winner=winner,
                    baseline_name="baseline",
                    target_metric="calmar",
                    metrics={},
                    artifacts_path=str(artifacts),
                ).to_dict()
            )
            + "\n"
        )
    return ledger


def test_promote_raises_without_force_for_reject(tmp_path: Path) -> None:
    ledger = _prep_ledger(tmp_path, variant_verdict="REJECT", winner=None)
    manager = PromotionManager(
        ledger=ledger,
        versions_root=tmp_path / ".versions",
        lineage_root=tmp_path / ".lineage",
        promoted_root=tmp_path / "promoted",
    )
    with pytest.raises(PromotionError, match="REJECT"):
        manager.promote_with_outcome(suite_id="suite_x", variant_name="promo_variant")


def test_promote_raises_for_unknown_variant(tmp_path: Path) -> None:
    ledger = _prep_ledger(tmp_path, variant_verdict="PROMOTE", winner="promo_variant")
    manager = PromotionManager(
        ledger=ledger,
        versions_root=tmp_path / ".versions",
        lineage_root=tmp_path / ".lineage",
        promoted_root=tmp_path / "promoted",
    )
    with pytest.raises(PromotionError, match="not in suite"):
        manager.promote_with_outcome(suite_id="suite_x", variant_name="not_a_variant")


def test_promote_raises_for_missing_suite(tmp_path: Path) -> None:
    ledger = Ledger(root=tmp_path)
    manager = PromotionManager(
        ledger=ledger,
        versions_root=tmp_path / ".versions",
        lineage_root=tmp_path / ".lineage",
        promoted_root=tmp_path / "promoted",
    )
    with pytest.raises(PromotionError, match="not found"):
        manager.promote_with_outcome(suite_id="nonexistent", variant_name="x")


def test_promote_preserves_comparison_in_patch_yaml(tmp_path: Path) -> None:
    ledger = _prep_ledger(tmp_path, variant_verdict="PROMOTE", winner="promo_variant")
    manager = PromotionManager(
        ledger=ledger,
        versions_root=tmp_path / ".versions",
        lineage_root=tmp_path / ".lineage",
        promoted_root=tmp_path / "promoted",
    )
    outcome = manager.promote_with_outcome(suite_id="suite_x", variant_name="promo_variant")

    import yaml

    patch = yaml.safe_load(outcome.patch_yaml_path.read_text())
    # Should inherit original suite's comparison (calmar, min_trades=100, α=0.01)
    assert patch["comparison"]["target_metric"] == "calmar"
    assert patch["comparison"]["min_trades"] == 100
    assert patch["comparison"]["significance_level"] == 0.01
    # random_seed preserved
    assert patch["backtest"]["random_seed"] == 42


def test_changed_components_maps_attrs_to_components() -> None:
    components = _components_touched(
        {
            "ml_basic.long_entry_threshold": 0.0003,
            "ml_basic.short_entry_threshold": -0.0005,
            "ml_basic.stop_loss_pct": 0.02,
            "ml_basic.base_fraction": 0.1,
        }
    )
    assert components == {"signal_generator", "risk_manager", "position_sizer"}


def test_changed_components_unknown_attr_maps_to_strategy() -> None:
    components = _components_touched({"ml_basic.some_future_knob": 1.0})
    assert components == {"strategy"}
