"""Tests for Ledger — malformed JSONL handling, find semantics."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

from src.experiments.ledger import Ledger, LedgerEntry
from src.experiments.reporter import SuiteReport


def _stub_suite_result() -> MagicMock:
    """A minimal SuiteResult stub good enough for Ledger.append."""
    result = MagicMock()
    result.suite_id = "suite_x"
    result.config.id = "suite_x"
    result.config.backtest.strategy = "ml_basic"
    result.config.backtest.symbol = "BTCUSDT"
    result.config.backtest.timeframe = "1h"
    result.config.backtest.days = 30
    result.config.baseline.overrides = {}
    result.config.variants = []
    result.baseline.sharpe_ratio = 1.0
    result.baseline.total_return = 5.0
    return result


def _stub_report(winner: str | None = None) -> SuiteReport:
    return SuiteReport(
        suite_id="suite_x",
        description="",
        target_metric="sharpe_ratio",
        significance_level=0.05,
        min_trades=0,
        baseline_name="baseline",
        winner=winner,
    )


def test_malformed_line_is_skipped_not_fatal(tmp_path: Path) -> None:
    ledger = Ledger(root=tmp_path)
    suite_result = _stub_suite_result()
    report = _stub_report(winner="best")
    ledger.append(suite_result, report, tmp_path / "run1")

    # Corrupt the ledger with a half-written line
    with ledger.path().open("a", encoding="utf-8") as fh:
        fh.write("{not valid json\n")

    entries = ledger.list_entries()
    assert len(entries) == 1
    assert entries[0].suite_id == "suite_x"


def test_empty_ledger_returns_empty_list(tmp_path: Path) -> None:
    ledger = Ledger(root=tmp_path)
    assert ledger.list_entries() == []
    assert ledger.find("anything") is None


def test_find_returns_most_recent(tmp_path: Path) -> None:
    ledger = Ledger(root=tmp_path)
    # Write two entries manually with specific timestamps.
    ledger.root.mkdir(parents=True, exist_ok=True)
    with ledger.path().open("a", encoding="utf-8") as fh:
        fh.write(
            json.dumps(
                LedgerEntry(
                    suite_id="suite_x",
                    timestamp="2026-01-01T00:00:00+00:00",
                    config_hash="a",
                    winner=None,
                    baseline_name="baseline",
                    target_metric="sharpe_ratio",
                    metrics={},
                    artifacts_path=str(tmp_path / "a"),
                ).to_dict()
            )
            + "\n"
        )
        fh.write(
            json.dumps(
                LedgerEntry(
                    suite_id="suite_x",
                    timestamp="2026-02-01T00:00:00+00:00",
                    config_hash="b",
                    winner="v1",
                    baseline_name="baseline",
                    target_metric="sharpe_ratio",
                    metrics={},
                    artifacts_path=str(tmp_path / "b"),
                ).to_dict()
            )
            + "\n"
        )

    entry = ledger.find("suite_x")
    assert entry is not None
    assert entry.config_hash == "b"  # most recent by timestamp


def test_limit_trims_returned_entries(tmp_path: Path) -> None:
    ledger = Ledger(root=tmp_path)
    ledger.root.mkdir(parents=True, exist_ok=True)
    for i in range(5):
        with ledger.path().open("a", encoding="utf-8") as fh:
            fh.write(
                json.dumps(
                    LedgerEntry(
                        suite_id=f"suite_{i}",
                        timestamp=f"2026-01-{i+1:02d}T00:00:00+00:00",
                        config_hash=str(i),
                        winner=None,
                        baseline_name="baseline",
                        target_metric="sharpe_ratio",
                        metrics={},
                        artifacts_path=str(tmp_path),
                    ).to_dict()
                )
                + "\n"
            )

    entries = ledger.list_entries(limit=2)
    assert len(entries) == 2
