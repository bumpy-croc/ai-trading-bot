"""`atb experiment` — run, list, and show experimentation suites."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

from src.infrastructure.runtime.paths import get_project_root

PROJECT_ROOT = get_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _positive_int(value: str) -> int:
    """argparse type: accept only strictly-positive integers."""
    try:
        coerced = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value!r}") from exc
    if coerced <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {coerced}")
    return coerced


def _safe_artifacts_dir(root: Path, suite_id: str, run_id: str) -> Path:
    """Resolve the artifact directory, rejecting traversal outside ``root``."""
    candidate = (root / suite_id / run_id).resolve()
    root_resolved = root.resolve()
    if root_resolved not in candidate.parents and candidate != root_resolved:
        raise ValueError(
            f"Resolved artifact path {candidate} is outside {root_resolved}; suite id or run id"
            " must not contain path separators."
        )
    return candidate


def _handle_run(ns: argparse.Namespace) -> int:
    try:
        from src.experiments.ledger import Ledger
        from src.experiments.promotion import PromotionManager
        from src.experiments.reporter import ExperimentReporter
        from src.experiments.suite import ExperimentSuiteRunner
        from src.experiments.suite_loader import load_suite

        suite = load_suite(ns.config)

        # Allow ad-hoc CLI overrides for backtest settings
        if ns.provider:
            suite.backtest.provider = ns.provider
        # `if ns.days:` silently dropped `--days 0` and accepted negatives —
        # explicit None + positive-int guard covers both (argparse validator
        # below also rejects non-positive values at parse time).
        if ns.days is not None:
            if ns.days <= 0:
                raise ValueError(f"--days must be a positive integer, got {ns.days}")
            if suite.backtest.start is not None and suite.backtest.end is not None:
                logging.getLogger(__name__).warning(
                    "--days=%d is ignored because the suite pins start=%s end=%s",
                    ns.days,
                    suite.backtest.start.isoformat(),
                    suite.backtest.end.isoformat(),
                )
            else:
                suite.backtest.days = ns.days
        if ns.no_cache:
            suite.backtest.use_cache = False

        suite_runner = ExperimentSuiteRunner()
        result = suite_runner.run(suite)

        reporter = ExperimentReporter()
        report = reporter.render(result)

        print(reporter.render_text(report))

        history_root = Path(ns.history_dir) if ns.history_dir else Path("experiments/.history")
        ledger = Ledger(root=history_root)
        promotion_manager = PromotionManager(ledger=ledger, reporter=reporter)
        # Microsecond precision + random suffix prevent collisions when two
        # runs finish within the same second (CI parallelism, mocked suites).
        now = datetime.now(UTC)
        run_id = f"{now.strftime('%Y%m%dT%H%M%S_%f')}Z_{uuid.uuid4().hex[:8]}"
        artifacts_path = _safe_artifacts_dir(ledger.root, suite.id, run_id)
        promotion_manager.record_run(result, report, artifacts_path)
        print(f"\nArtifacts: {artifacts_path}")
        return 0
    except Exception as exc:
        if getattr(ns, "debug", False):
            raise
        logging.exception("Experiment run failed")
        print(f"Error: {exc}", file=sys.stderr)
        return 1


def _handle_list(ns: argparse.Namespace) -> int:
    try:
        from src.experiments.ledger import Ledger

        ledger = Ledger(
            root=Path(ns.history_dir) if ns.history_dir else Path("experiments/.history")
        )
        entries = ledger.list_entries(limit=ns.limit)
        if not entries:
            print("(no suites recorded)")
            return 0

        header = f"{'Timestamp':<26} {'Suite':<30} {'Winner':<20} {'Metric':<18}"
        print(header)
        print("-" * len(header))
        for e in entries:
            print(
                f"{e.timestamp:<26} {e.suite_id:<30} "
                f"{(e.winner or '—'):<20} {e.target_metric:<18}"
            )
        return 0
    except Exception as exc:
        if getattr(ns, "debug", False):
            raise
        print(f"Error: {exc}", file=sys.stderr)
        return 1


def _handle_show(ns: argparse.Namespace) -> int:
    try:
        from src.experiments.ledger import Ledger

        ledger = Ledger(
            root=Path(ns.history_dir) if ns.history_dir else Path("experiments/.history")
        )
        entry = ledger.find(ns.suite_id)
        if entry is None:
            print(f"Suite {ns.suite_id!r} not found", file=sys.stderr)
            return 1

        report_path = Path(entry.artifacts_path) / "report.txt"
        if report_path.exists():
            print(report_path.read_text())
        else:
            print(json.dumps(entry.to_dict(), indent=2))
        return 0
    except Exception as exc:
        if getattr(ns, "debug", False):
            raise
        print(f"Error: {exc}", file=sys.stderr)
        return 1


def _handle_promote(ns: argparse.Namespace) -> int:
    try:
        from src.experiments.ledger import Ledger
        from src.experiments.promotion import PromotionManager

        ledger = Ledger(
            root=Path(ns.history_dir) if ns.history_dir else Path("experiments/.history")
        )
        manager = PromotionManager(ledger=ledger)
        patch_path = manager.promote(suite_id=ns.suite_id, variant_name=ns.variant, force=ns.force)
        print(f"Promoted. Patch YAML: {patch_path}")
        return 0
    except Exception as exc:
        if getattr(ns, "debug", False):
            raise
        print(f"Error: {exc}", file=sys.stderr)
        return 1


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "experiment",
        help="Run declarative experiment suites (baseline + variants).",
    )
    p.add_argument(
        "--history-dir",
        default=None,
        help="Override the ledger root (default: experiments/.history).",
    )
    p.add_argument("--debug", action="store_true", help="Re-raise with full traceback.")

    subs = p.add_subparsers(dest="subcommand", required=True)

    run_p = subs.add_parser("run", help="Execute a suite YAML and record results.")
    run_p.add_argument("--config", required=True, help="Path to suite YAML.")
    run_p.add_argument(
        "--provider",
        choices=["binance", "coinbase", "mock", "fixture"],
        default=None,
        help="Override the suite's provider.",
    )
    run_p.add_argument(
        "--days",
        type=_positive_int,
        default=None,
        help="Override the backtest window (positive integer).",
    )
    run_p.add_argument("--no-cache", action="store_true", help="Disable provider caching.")
    run_p.set_defaults(func=_handle_run)

    list_p = subs.add_parser("list", help="List recorded suite runs.")
    list_p.add_argument("--limit", type=int, default=20)
    list_p.set_defaults(func=_handle_list)

    show_p = subs.add_parser("show", help="Show the most recent report for a suite.")
    show_p.add_argument("suite_id")
    show_p.set_defaults(func=_handle_show)

    promote_p = subs.add_parser("promote", help="Promote a variant to the lineage store.")
    promote_p.add_argument("suite_id")
    promote_p.add_argument("variant")
    promote_p.add_argument(
        "--force",
        action="store_true",
        help="Promote even when the verdict is not PROMOTE.",
    )
    promote_p.set_defaults(func=_handle_promote)
