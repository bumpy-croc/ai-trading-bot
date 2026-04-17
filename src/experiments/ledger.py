"""Append-only JSONL ledger of completed experiment suites."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.experiments.reporter import SuiteReport
from src.experiments.suite import SuiteResult

logger = logging.getLogger(__name__)

DEFAULT_ROOT = Path("experiments/.history")
LEDGER_FILE = "ledger.jsonl"


def _config_hash(suite_result: SuiteResult) -> str:
    payload = {
        "id": suite_result.config.id,
        "strategy": suite_result.config.backtest.strategy,
        "symbol": suite_result.config.backtest.symbol,
        "timeframe": suite_result.config.backtest.timeframe,
        "days": suite_result.config.backtest.days,
        "baseline": suite_result.config.baseline.overrides,
        "variants": {v.name: v.overrides for v in suite_result.config.variants},
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


@dataclass
class LedgerEntry:
    suite_id: str
    timestamp: str
    config_hash: str
    winner: str | None
    baseline_name: str
    target_metric: str
    metrics: dict[str, float]
    artifacts_path: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "suite_id": self.suite_id,
            "timestamp": self.timestamp,
            "config_hash": self.config_hash,
            "winner": self.winner,
            "baseline_name": self.baseline_name,
            "target_metric": self.target_metric,
            "metrics": self.metrics,
            "artifacts_path": self.artifacts_path,
        }


class Ledger:
    """Append-only store at ``experiments/.history/ledger.jsonl`` by default."""

    def __init__(self, root: Path = DEFAULT_ROOT):
        self.root = Path(root)

    def path(self) -> Path:
        return self.root / LEDGER_FILE

    def artifacts_dir(self, suite_id: str, run_id: str) -> Path:
        return self.root / suite_id / run_id

    def append(
        self,
        suite_result: SuiteResult,
        report: SuiteReport,
        artifacts_path: Path,
    ) -> LedgerEntry:
        self.root.mkdir(parents=True, exist_ok=True)
        entry = LedgerEntry(
            suite_id=suite_result.suite_id,
            timestamp=datetime.now(UTC).isoformat(),
            config_hash=_config_hash(suite_result),
            winner=report.winner,
            baseline_name=report.baseline_name,
            target_metric=report.target_metric,
            metrics={
                "baseline_sharpe": suite_result.baseline.sharpe_ratio,
                "baseline_return": suite_result.baseline.total_return,
            },
            artifacts_path=str(artifacts_path),
        )
        with self.path().open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry.to_dict()) + "\n")
        return entry

    def list_entries(self, limit: int | None = None) -> list[LedgerEntry]:
        """Return ledger entries newest-first, skipping malformed JSON lines."""
        if not self.path().exists():
            return []
        entries: list[LedgerEntry] = []
        with self.path().open("r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    data = json.loads(stripped)
                    entries.append(LedgerEntry(**data))
                except (json.JSONDecodeError, TypeError) as exc:
                    logger.warning(
                        "ledger: skipping malformed line %d in %s: %s",
                        line_no,
                        self.path(),
                        exc,
                    )
        entries.sort(key=lambda e: e.timestamp, reverse=True)
        if limit is not None:
            entries = entries[:limit]
        return entries

    def find(self, suite_id: str) -> LedgerEntry | None:
        for entry in self.list_entries():
            if entry.suite_id == suite_id:
                return entry
        return None


__all__ = ["DEFAULT_ROOT", "Ledger", "LedgerEntry"]
