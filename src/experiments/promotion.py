"""Promote a winning variant from an experiment suite.

Promotion is auditable and reversible — the framework never mutates strategy
defaults in source. Instead it writes a :class:`StrategyVersionRecord` and a
:class:`ChangeRecord` next to the strategy store, and emits a patch YAML that
can be used as the baseline of the next experiment suite.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import yaml

from src.experiments.ledger import Ledger
from src.experiments.reporter import ExperimentReporter, SuiteReport, Verdict
from src.experiments.suite import SuiteResult
from src.strategies.components.strategy_lineage import ChangeRecord, ChangeType, ImpactLevel
from src.strategies.versioning import StrategyVersionRecord


class PromotionError(RuntimeError):
    pass


@dataclass
class PromotionOutcome:
    suite_id: str
    variant_name: str
    strategy_name: str
    version_record_path: Path
    change_record_path: Path
    patch_yaml_path: Path


class PromotionManager:
    """Persist the outcome of a suite run for reproducibility."""

    def __init__(
        self,
        *,
        ledger: Ledger | None = None,
        versions_root: Path = Path("src/strategies/.versions"),
        lineage_root: Path = Path("src/strategies/.lineage"),
        promoted_root: Path = Path("experiments/promoted"),
        reporter: ExperimentReporter | None = None,
    ):
        self.ledger = ledger or Ledger()
        self.versions_root = Path(versions_root)
        self.lineage_root = Path(lineage_root)
        self.promoted_root = Path(promoted_root)
        self.reporter = reporter or ExperimentReporter()

    def promote(
        self,
        *,
        suite_id: str,
        variant_name: str,
        force: bool = False,
    ) -> Path:
        """Promote ``variant_name`` from the most recent run of ``suite_id``.

        Returns the path to the emitted patch YAML. The full outcome is
        available via :meth:`promote_with_outcome`.
        """
        outcome = self.promote_with_outcome(
            suite_id=suite_id, variant_name=variant_name, force=force
        )
        return outcome.patch_yaml_path

    def promote_with_outcome(
        self,
        *,
        suite_id: str,
        variant_name: str,
        force: bool = False,
    ) -> PromotionOutcome:
        entry = self.ledger.find(suite_id)
        if entry is None:
            raise PromotionError(f"Suite {suite_id!r} not found in ledger")

        artifacts_path = Path(entry.artifacts_path)
        report_path = artifacts_path / "report.json"
        if not report_path.exists():
            raise PromotionError(f"Report missing at {report_path}")
        report_data = json.loads(report_path.read_text())

        suite_path = artifacts_path / "suite.json"
        if not suite_path.exists():
            raise PromotionError(f"Suite snapshot missing at {suite_path}")
        suite_snapshot = json.loads(suite_path.read_text())

        variant_row = _find_variant_row(report_data, variant_name)
        if variant_row is None:
            raise PromotionError(f"Variant {variant_name!r} not in suite {suite_id!r}")

        verdict = Verdict(variant_row["verdict"])
        if verdict != Verdict.PROMOTE and not force:
            raise PromotionError(
                f"Variant {variant_name!r} has verdict {verdict.value}; use --force to override."
            )

        variant_overrides = _find_variant_overrides(suite_snapshot, variant_name)
        strategy_name = suite_snapshot["backtest"]["strategy"]

        version_path = self._write_version_record(
            strategy_name=strategy_name,
            variant_name=variant_name,
            overrides=variant_overrides,
            metrics=variant_row,
            suite_id=suite_id,
        )
        change_path = self._write_change_record(
            strategy_name=strategy_name,
            variant_name=variant_name,
            overrides=variant_overrides,
            variant_row=variant_row,
            baseline_row=_find_baseline_row(report_data),
        )
        patch_path = self._write_patch_yaml(
            suite_id=suite_id,
            variant_name=variant_name,
            strategy_name=strategy_name,
            overrides=variant_overrides,
            suite_snapshot=suite_snapshot,
        )

        return PromotionOutcome(
            suite_id=suite_id,
            variant_name=variant_name,
            strategy_name=strategy_name,
            version_record_path=version_path,
            change_record_path=change_path,
            patch_yaml_path=patch_path,
        )

    def snapshot_suite(self, suite_result: SuiteResult, report: SuiteReport) -> dict:
        """Return the snapshot that :meth:`record_run` will persist alongside the report."""
        return {
            "id": suite_result.config.id,
            "description": suite_result.config.description,
            "backtest": {
                "strategy": suite_result.config.backtest.strategy,
                "symbol": suite_result.config.backtest.symbol,
                "timeframe": suite_result.config.backtest.timeframe,
                "days": suite_result.config.backtest.days,
                "initial_balance": suite_result.config.backtest.initial_balance,
                "provider": suite_result.config.backtest.provider,
                "use_cache": suite_result.config.backtest.use_cache,
                "random_seed": suite_result.config.backtest.random_seed,
            },
            "baseline": {
                "name": suite_result.config.baseline.name,
                "overrides": suite_result.config.baseline.overrides,
            },
            "variants": [
                {"name": v.name, "overrides": v.overrides} for v in suite_result.config.variants
            ],
            "comparison": {
                "target_metric": suite_result.config.comparison.target_metric,
                "min_trades": suite_result.config.comparison.min_trades,
                "significance_level": suite_result.config.comparison.significance_level,
            },
            "winner": report.winner,
        }

    def record_run(
        self,
        suite_result: SuiteResult,
        report: SuiteReport,
        artifacts_path: Path,
    ) -> None:
        """Emit the auxiliary files the promotion flow relies on."""
        artifacts_path.mkdir(parents=True, exist_ok=True)
        self.reporter.write_artifacts(report, artifacts_path)
        snapshot = self.snapshot_suite(suite_result, report)
        (artifacts_path / "suite.json").write_text(json.dumps(snapshot, indent=2, default=str))
        self.ledger.append(suite_result, report, artifacts_path)

    # --- internals ---------------------------------------------------------

    def _write_version_record(
        self,
        *,
        strategy_name: str,
        variant_name: str,
        overrides: dict,
        metrics: dict,
        suite_id: str,
    ) -> Path:
        record_dir = self.versions_root / strategy_name
        record_dir.mkdir(parents=True, exist_ok=True)
        version_id = f"{strategy_name}-{variant_name}-{_timestamp()}"
        record = StrategyVersionRecord(
            version_id=version_id,
            created_at=datetime.now(UTC),
            name=variant_name,
            description=f"Promoted from suite {suite_id!r}",
            strategy_name=strategy_name,
            version=version_id,
            config={"overrides": overrides},
            parameters=overrides,
            performance_metrics={
                k: metrics.get(k)
                for k in (
                    "total_return",
                    "annualized_return",
                    "sharpe_ratio",
                    "max_drawdown",
                    "win_rate",
                    "total_trades",
                    "delta_vs_baseline",
                    "ranking_confidence",
                    "verdict",
                )
            },
            is_active=False,
        )
        path = record_dir / f"{version_id}.json"
        path.write_text(json.dumps(record.to_dict(), indent=2, default=str))
        return path

    def _write_change_record(
        self,
        *,
        strategy_name: str,
        variant_name: str,
        overrides: dict,
        variant_row: dict,
        baseline_row: dict | None,
    ) -> Path:
        record_dir = self.lineage_root / strategy_name
        record_dir.mkdir(parents=True, exist_ok=True)
        impact = _impact_from_sharpe_delta(
            variant_row.get("sharpe_ratio", 0.0)
            - (baseline_row.get("sharpe_ratio", 0.0) if baseline_row else 0.0)
        )
        record = ChangeRecord(
            change_id=str(uuid.uuid4()),
            change_type=ChangeType.PARAMETER_CHANGE,
            description=f"Promoted variant {variant_name}",
            impact_level=impact,
            changed_components=sorted(_components_touched(overrides)),
            parameter_changes=overrides,
            performance_impact={
                "delta_sharpe": float(
                    variant_row.get("sharpe_ratio", 0.0)
                    - (baseline_row.get("sharpe_ratio", 0.0) if baseline_row else 0.0)
                ),
                "delta_return": float(
                    variant_row.get("total_return", 0.0)
                    - (baseline_row.get("total_return", 0.0) if baseline_row else 0.0)
                ),
            },
            created_at=datetime.now(UTC),
            created_by="experiment_suite",
        )
        path = record_dir / f"{record.change_id}.json"
        path.write_text(json.dumps(record.to_dict(), indent=2))
        return path

    def _write_patch_yaml(
        self,
        *,
        suite_id: str,
        variant_name: str,
        strategy_name: str,
        overrides: dict,
        suite_snapshot: dict,
    ) -> Path:
        self.promoted_root.mkdir(parents=True, exist_ok=True)
        patch_path = self.promoted_root / f"{suite_id}_{variant_name}.yaml"
        patch_doc = {
            "id": f"{suite_id}_promoted_{variant_name}",
            "description": (
                f"Baseline derived from promoted variant {variant_name!r} of "
                f"suite {suite_id!r}."
            ),
            "backtest": suite_snapshot["backtest"],
            "baseline": {
                "name": f"promoted_{variant_name}",
                "overrides": overrides,
            },
            "variants": [],
            "comparison": suite_snapshot.get(
                "comparison",
                {
                    "target_metric": "sharpe_ratio",
                    "min_trades": 30,
                    "significance_level": 0.05,
                },
            ),
        }
        patch_path.write_text(yaml.safe_dump(patch_doc, sort_keys=False))
        return patch_path


def _find_variant_row(report_data: dict, variant_name: str) -> dict | None:
    for row in report_data.get("rows", []):
        if row.get("name") == variant_name:
            return row
    return None


def _find_baseline_row(report_data: dict) -> dict | None:
    for row in report_data.get("rows", []):
        if row.get("is_baseline"):
            return row
    return None


def _find_variant_overrides(suite_snapshot: dict, variant_name: str) -> dict:
    if suite_snapshot.get("baseline", {}).get("name") == variant_name:
        return dict(suite_snapshot["baseline"].get("overrides", {}))
    for v in suite_snapshot.get("variants", []):
        if v.get("name") == variant_name:
            return dict(v.get("overrides", {}))
    raise PromotionError(f"Variant {variant_name!r} not found in suite snapshot")


# Mirrors the ``component_targets`` map in ``ExperimentRunner._apply_strategy_attribute``.
# Used to describe *which component* was tuned when recording lineage.
_ATTR_TO_COMPONENT: dict[str, str] = {
    "stop_loss_pct": "risk_manager",
    "take_profit_pct": "risk_manager",
    "risk_per_trade": "risk_manager",
    "trailing_stop_pct": "risk_manager",
    "atr_multiplier": "risk_manager",
    "base_fraction": "position_sizer",
    "min_confidence": "position_sizer",
    "min_confidence_floor": "position_sizer",
    "sequence_length": "signal_generator",
    "model_path": "signal_generator",
    "use_prediction_engine": "signal_generator",
    "model_name": "signal_generator",
    "model_type": "signal_generator",
    "timeframe": "signal_generator",
    "long_entry_threshold": "signal_generator",
    "short_entry_threshold": "signal_generator",
    "confidence_multiplier": "signal_generator",
    "short_threshold_trend_up": "signal_generator",
    "short_threshold_trend_down": "signal_generator",
    "short_threshold_range": "signal_generator",
    "short_threshold_high_vol": "signal_generator",
    "short_threshold_low_vol": "signal_generator",
    "short_threshold_confidence_multiplier": "signal_generator",
}


def _components_touched(overrides: dict) -> set[str]:
    """Map dotted override keys → set of component names touched."""
    components: set[str] = set()
    for key in overrides:
        attr = key.split(".", 1)[1] if "." in key else key
        components.add(_ATTR_TO_COMPONENT.get(attr, "strategy"))
    return components


def _impact_from_sharpe_delta(delta_sharpe: float) -> ImpactLevel:
    abs_d = abs(delta_sharpe)
    if abs_d >= 0.5:
        return ImpactLevel.HIGH
    if abs_d >= 0.2:
        return ImpactLevel.MEDIUM
    return ImpactLevel.LOW


def _timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


__all__ = ["PromotionError", "PromotionManager", "PromotionOutcome"]
