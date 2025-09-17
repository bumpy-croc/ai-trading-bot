from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.prediction.config import PredictionConfig
from src.prediction.models.registry import PredictionModelRegistry


def _handle_list(ns: argparse.Namespace) -> int:
    cfg = PredictionConfig.from_config_manager()
    reg = PredictionModelRegistry(cfg)
    base = Path(cfg.model_registry_path)
    if not base.exists():
        print("No models directory found")
        return 0
    print("Strategy models:")
    for b in reg.list_bundles():
        print(f"- {b.symbol} {b.timeframe} {b.model_type} -> {b.version_id}")
    return 0


def _handle_compare(ns: argparse.Namespace) -> int:
    cfg = PredictionConfig.from_config_manager()
    reg = PredictionModelRegistry(cfg)

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
    reg = PredictionModelRegistry(cfg)
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


