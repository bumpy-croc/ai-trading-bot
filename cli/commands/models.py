from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config.paths import get_project_root
from src.ml.model_retention import (
    DEFAULT_KEEP_NEWEST,
    DEFAULT_REFERENCE_PATHS,
    RegistryStateError,
    apply_plan,
    plan_cleanup,
)
from src.prediction.config import PredictionConfig
from src.prediction.models.registry import PredictionModelRegistry


def _open_registry(cfg: PredictionConfig) -> PredictionModelRegistry | None:
    """Build the registry, or print why not (a missing directory raises)."""
    try:
        return PredictionModelRegistry(cfg)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None


def _handle_list(ns: argparse.Namespace) -> int:
    cfg = PredictionConfig.from_config_manager()
    base = Path(cfg.model_registry_path)
    if not base.exists():
        print("No models directory found")
        return 0
    reg = _open_registry(cfg)
    if reg is None:
        return 1
    print("Strategy models:")
    for b in reg.list_bundles():
        print(f"- {b.symbol} {b.timeframe} {b.model_type} -> {b.version_id}")
    return 0


def _handle_compare(ns: argparse.Namespace) -> int:
    cfg = PredictionConfig.from_config_manager()
    reg = _open_registry(cfg)
    if reg is None:
        return 1

    # Basic comparison: print metrics.json if present
    symbol = ns.symbol
    timeframe = ns.timeframe
    model_type = ns.model_type
    try:
        bundle = reg.select_bundle(symbol=symbol, model_type=model_type, timeframe=timeframe)
    except Exception as e:
        print(f"Error: {e}")
        return 1
    metrics = bundle.metrics or {}
    print(json.dumps(metrics, indent=2))
    return 0


def _handle_validate(ns: argparse.Namespace) -> int:
    cfg = PredictionConfig.from_config_manager()
    reg = _open_registry(cfg)
    if reg is None:
        return 1
    # Attempt reload to surface errors
    try:
        reg.reload_models()
        print("Validation OK")
        return 0
    except Exception as e:
        print(f"Validation failed: {e}")
        return 1


def _handle_promote(ns: argparse.Namespace) -> int:
    # For now, implement latest symlink creation only
    cfg = PredictionConfig.from_config_manager()
    base = Path(cfg.model_registry_path)
    symbol = ns.symbol
    model_type = ns.model_type
    version = ns.version
    mdir = base / symbol / model_type
    target = mdir / version
    latest = mdir / "latest"
    if not target.exists():
        print(f"Version not found: {target}")
        return 1
    # Remove existing symlink/dir named latest and recreate symlink
    if latest.exists() or latest.is_symlink():
        try:
            if latest.is_dir() and not latest.is_symlink():
                # Avoid deleting real directory named latest; require it to be symlink
                print("Refusing to overwrite non-symlink 'latest' directory")
                return 1
            latest.unlink()
        except Exception:
            pass
    latest.symlink_to(target.name)
    print(f"Promoted {symbol}/{model_type} -> {version}")
    return 0


def _handle_gate(ns: argparse.Namespace) -> int:
    from src.ml.promotion_gate import main as gate_main

    return gate_main([ns.comparison_json])


def _format_bytes(n: int) -> str:
    units = ("B", "KB", "MB", "GB")
    size, idx = float(n), 0
    while size >= 1024 and idx < len(units) - 1:
        size /= 1024
        idx += 1
    return f"{n} B" if idx == 0 else f"{size:.1f} {units[idx]}"


def _handle_cleanup(ns: argparse.Namespace) -> int:
    registry = Path(ns.registry or PredictionConfig.from_config_manager().model_registry_path)
    root = get_project_root()
    reference_paths = [root / p for p in DEFAULT_REFERENCE_PATHS]
    reference_paths += [Path(p) for p in ns.reference_path or []]

    try:
        plan = plan_cleanup(registry, keep_newest=ns.keep, reference_paths=reference_paths)
    except (RegistryStateError, ValueError) as e:
        print(f"Error: {e}")
        return 1

    print(f"Registry: {registry}")
    print(f"Protected versions: {len(plan.protected)}")
    if ns.verbose:
        for key, reasons in sorted(plan.protected.items()):
            print(f"  keep {key}: {'; '.join(reasons)}")
    if not plan.candidates:
        print("Nothing to remove.")
        return 0

    verb = "Removing" if ns.apply else "Would remove"
    print(f"{verb} {len(plan.candidates)} version(s):")
    for mv in plan.candidates:
        print(f"  {mv.key}  ({_format_bytes(mv.size_bytes)})")
    freed = "Space freed" if ns.apply else "Space that would be freed"
    print(f"{freed}: {_format_bytes(plan.bytes_freed)}")

    if not ns.apply:
        print("Dry run. Re-run with --apply to delete.")
        return 0
    try:
        apply_plan(registry, plan)
    except RegistryStateError as e:
        print(f"Error: {e}")
        return 1
    print("Done.")
    return 0


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("models", help="Manage ML models")
    sp = p.add_subparsers(dest="models_cmd", required=True)

    p_list = sp.add_parser("list", help="List available models and bundles")
    p_list.set_defaults(func=_handle_list)

    p_compare = sp.add_parser("compare", help="Show metrics for a selected bundle")
    p_compare.add_argument("symbol")
    p_compare.add_argument("timeframe")
    p_compare.add_argument("model_type")
    p_compare.set_defaults(func=_handle_compare)

    p_validate = sp.add_parser("validate", help="Validate registry and bundles")
    p_validate.set_defaults(func=_handle_validate)

    p_promote = sp.add_parser("promote", help="Promote a version as latest (symlink)")
    p_promote.add_argument("symbol")
    p_promote.add_argument("model_type")
    p_promote.add_argument("version")
    p_promote.set_defaults(func=_handle_promote)

    p_cleanup = sp.add_parser(
        "cleanup",
        help="Remove old model versions per the retention policy (dry run unless --apply)",
    )
    p_cleanup.add_argument(
        "--keep",
        type=int,
        default=DEFAULT_KEEP_NEWEST,
        help="Newest versions to keep per SYMBOL/TYPE (default: %(default)s)",
    )
    p_cleanup.add_argument("--registry", help="Registry directory (default: configured registry)")
    p_cleanup.add_argument(
        "--reference-path",
        action="append",
        help="Extra file or directory whose text pins versions (repeatable)",
    )
    p_cleanup.add_argument("--verbose", action="store_true", help="Also list protected versions")
    p_cleanup.add_argument(
        "--apply", action="store_true", help="Delete the listed versions (default: dry run)"
    )
    p_cleanup.set_defaults(func=_handle_cleanup)

    p_gate = sp.add_parser(
        "gate",
        help="Score a challenger vs incumbent (exit 0 only on PASS); see src/ml/promotion_gate.py",
    )
    p_gate.add_argument("comparison_json")
    p_gate.set_defaults(func=_handle_gate)
