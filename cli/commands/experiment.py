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

        header = (
            f"{'Timestamp':<26} {'Suite':<30} {'Winner':<20} "
            f"{'Metric':<12} {'BaseSharpe':>10} {'BaseReturn%':>12}"
        )
        print(header)
        print("-" * len(header))
        for e in entries:
            baseline_sharpe = e.metrics.get("baseline_sharpe")
            baseline_return = e.metrics.get("baseline_return")
            sharpe_str = (
                f"{baseline_sharpe:>10.3f}" if baseline_sharpe is not None else f"{'—':>10}"
            )
            return_str = (
                f"{baseline_return:>12.2f}" if baseline_return is not None else f"{'—':>12}"
            )
            print(
                f"{e.timestamp:<26} {e.suite_id:<30} "
                f"{(e.winner or '—'):<20} {e.target_metric:<12} "
                f"{sharpe_str} {return_str}"
            )
        return 0
    except Exception as exc:
        if getattr(ns, "debug", False):
            raise
        logging.exception("Experiment list failed")
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
        logging.exception("Experiment show failed for suite_id=%s", ns.suite_id)
        print(f"Error: {exc}", file=sys.stderr)
        return 1


def _handle_diagnose(ns: argparse.Namespace) -> int:
    try:
        from src.experiments.diagnostics import SignalDiagnostic

        start = datetime.fromisoformat(ns.start.replace("Z", "+00:00"))
        end = datetime.fromisoformat(ns.end.replace("Z", "+00:00"))
        if start.tzinfo is None:
            start = start.replace(tzinfo=UTC)
        if end.tzinfo is None:
            end = end.replace(tzinfo=UTC)

        factory_kwargs: dict[str, object] = {}
        for kv in ns.factory_kwarg or []:
            if "=" not in kv:
                raise ValueError(f"--factory-kwarg must be KEY=VALUE, got {kv!r}")
            key, raw = kv.split("=", 1)
            factory_kwargs[key.strip()] = _coerce_cli_scalar(raw)

        diag = SignalDiagnostic()
        report = diag.run(
            strategy_name=ns.strategy,
            symbol=ns.symbol,
            timeframe=ns.timeframe,
            start=start,
            end=end,
            provider=ns.provider,
            use_cache=not ns.no_cache,
            random_seed=ns.random_seed,
            factory_kwargs=factory_kwargs or None,
        )
        if ns.format == "json":
            print(json.dumps(report.to_dict(), indent=2))
        else:
            print(report.render_text())
        return 0 if report.constant_signal_warning is None else 2
    except Exception as exc:
        if getattr(ns, "debug", False):
            raise
        logging.exception("Experiment diagnose failed")
        print(f"Error: {exc}", file=sys.stderr)
        return 1


def _coerce_cli_scalar(raw: str) -> object:
    """Best-effort coercion of CLI string values to int/float/bool/None/str."""
    stripped = raw.strip()
    if stripped.lower() in {"true", "false"}:
        return stripped.lower() == "true"
    if stripped.lower() in {"none", "null", ""}:
        return None
    try:
        return int(stripped)
    except ValueError:
        pass
    try:
        return float(stripped)
    except ValueError:
        pass
    return stripped


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
        logging.exception(
            "Experiment promote failed for suite_id=%s variant=%s",
            ns.suite_id,
            ns.variant,
        )
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
    list_p.add_argument("--limit", type=_positive_int, default=20)
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

    diag_p = subs.add_parser(
        "diagnose",
        help=(
            "Walk the strategy's signal generator bar-by-bar and report "
            "decision mix, predicted-return distribution, confidence "
            "distribution, and direction-conditional hit rate. Useful for "
            "catching broken signals that nonetheless produce non-zero P&L."
        ),
    )
    diag_p.add_argument("--strategy", required=True, help="Strategy name (e.g. ml_basic).")
    diag_p.add_argument("--symbol", default="BTCUSDT")
    diag_p.add_argument("--timeframe", default="1h")
    diag_p.add_argument("--start", required=True, help="ISO-8601 start (e.g. 2024-01-01).")
    diag_p.add_argument("--end", required=True, help="ISO-8601 end (e.g. 2024-12-31).")
    diag_p.add_argument(
        "--provider",
        choices=["binance", "coinbase", "mock", "fixture"],
        default="fixture",
    )
    diag_p.add_argument("--no-cache", action="store_true")
    diag_p.add_argument("--random-seed", type=int, default=None)
    diag_p.add_argument(
        "--factory-kwarg",
        action="append",
        metavar="KEY=VALUE",
        help=(
            "Keyword argument to pass to the strategy factory at "
            "construction time. Repeatable. Use for knobs like "
            "model_type=basic or max_leverage=2.0."
        ),
    )
    diag_p.add_argument("--format", choices=["text", "json"], default="text")
    diag_p.set_defaults(func=_handle_diagnose)
