from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from src.infrastructure.runtime.paths import get_project_root

PROJECT_ROOT = get_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_PATH = PROJECT_ROOT / "src"
if SRC_PATH.exists() and str(SRC_PATH) not in sys.path:
    sys.path.insert(1, str(SRC_PATH))

STRATEGIES_DIR = PROJECT_ROOT / "src" / "strategies" / "store"
STRATEGIES_DIR.mkdir(exist_ok=True, parents=True)


def _lazy_import_strategies():
    from src.strategies import (
        create_ensemble_weighted_strategy,
        create_ml_adaptive_strategy,
        create_ml_basic_strategy,
        create_ml_sentiment_strategy,
    )
    from src.strategies.components import Strategy, StrategyRegistry

    return {
        "StrategyRegistry": StrategyRegistry,
        "Strategy": Strategy,
        "MlBasic": create_ml_basic_strategy,
        "MlAdaptive": create_ml_adaptive_strategy,
        "MlSentiment": create_ml_sentiment_strategy,
        "EnsembleWeighted": create_ensemble_weighted_strategy,
    }


def _get_modified_strategy_files(specified: list[str] | None = None) -> list[Path]:
    if specified:
        return [Path(s).resolve() for s in specified]
    try:
        result = subprocess.run(
            ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []
    files = [f.strip() for f in result.stdout.splitlines() if f.strip()]
    return [
        (PROJECT_ROOT / f).resolve()
        for f in files
        if f.startswith("src/strategies/")
        and f.endswith(".py")
        and not f.endswith("__init__.py")
        and "components/" not in f
        and "adapters/" not in f
        and "migration/" not in f
    ]


def _calculate_strategy_hash(strategy_instance) -> str:
    try:
        params = strategy_instance.get_parameters()
        params_str = json.dumps(params, sort_keys=True)
        return hashlib.sha256(params_str.encode()).hexdigest()
    except Exception as exc:  # noqa: BLE001
        print(f"Warning: Could not calculate hash: {exc}")
        return ""


def _load_existing_version(strategy_name: str) -> dict[str, Any] | None:
    config_file = STRATEGIES_DIR / f"{strategy_name}.json"
    if config_file.exists():
        return json.loads(config_file.read_text())
    return None


def _prompt_for_changes(
    default_message: str = "Updated strategy implementation",
) -> tuple[list[str], bool]:
    print("\nDescribe the changes made (one per line, empty line to finish):")
    changes: list[str] = []
    while True:
        line = input("  - ").strip()
        if not line:
            break
        changes.append(line)
    if not changes:
        changes = [default_message]
    is_major = input("\nIs this a major version change? (y/N): ").lower() == "y"
    return changes, is_major


def _process_strategy_file(strategy_file: Path, auto_confirm: bool = False) -> bool:
    try:
        imports = _lazy_import_strategies()
        StrategyRegistry = imports["StrategyRegistry"]
        Strategy = imports["Strategy"]
        strategy_classes = {
            "MlBasic": imports["MlBasic"],
            "MlAdaptive": imports["MlAdaptive"],
            "MlSentiment": imports["MlSentiment"],
            "EnsembleWeighted": imports["EnsembleWeighted"],
        }
    except Exception as exc:  # noqa: BLE001
        print(f"✗ Failed to import strategies: {exc}")
        print("Skipping strategy versioning (imports failed)")
        return False
    strategy_name = strategy_file.stem
    class_name = "".join(word.capitalize() for word in strategy_name.split("_"))
    if class_name not in strategy_classes:
        print(f"⊘ Skipping {strategy_file} (not in registry)")
        return False
    print(f"\n{'=' * 60}")
    print(f"Processing: {strategy_file.relative_to(PROJECT_ROOT)}")
    print(f"{'=' * 60}")
    try:
        strategy_builder = strategy_classes[class_name]
        strategy_instance = strategy_builder()
    except Exception as exc:  # noqa: BLE001
        print(f"✗ Failed to instantiate {class_name}: {exc}")
        return False
    try:
        component_strategy = strategy_instance
        if not isinstance(component_strategy, Strategy):
            raise TypeError("Strategy builder did not return a Strategy instance")
    except Exception as exc:  # noqa: BLE001
        print("✗ Failed to initialize component strategy: " f"{exc}")
        return False
    current_hash = _calculate_strategy_hash(strategy_instance)
    existing_data = _load_existing_version(strategy_name)
    registry = StrategyRegistry()
    if existing_data:
        latest_version = (
            existing_data.get("versions", [])[-1] if existing_data.get("versions") else None
        )
        existing_hash = latest_version.get("component_hash", "") if latest_version else ""
        if current_hash == existing_hash:
            print("✓ No configuration changes detected, skipping version update")
            return False
        print("⚠ Configuration change detected!")
        print(f"  Old hash: {existing_hash[:16]}...")
        print(f"  New hash: {current_hash[:16]}...")
        strategy_id = registry.deserialize_strategy(existing_data)
        changes, is_major = (
            _prompt_for_changes() if not auto_confirm else (["Automated update"], False)
        )
        new_version = registry.update_strategy(
            strategy_id=strategy_id,
            strategy=component_strategy,
            changes=changes,
            is_major=is_major,
        )
        print(f"✓ Updated {strategy_name} to version {new_version}")
    else:
        strategy_id = registry.register_strategy(
            strategy=component_strategy,
            metadata={
                "created_by": "developer",
                "description": f"{class_name} trading strategy",
                "tags": ["ml"] if "Ml" in class_name else [],
                "status": "EXPERIMENTAL",
            },
        )
        print(f"✓ Registered {strategy_name} as version 1.0.0")
    config_file = STRATEGIES_DIR / f"{strategy_name}.json"
    config_file.write_text(
        json.dumps(registry.serialize_strategy(strategy_id), indent=2, sort_keys=True)
    )
    print(f"✓ Saved {strategy_name} version to {config_file.relative_to(PROJECT_ROOT)}")
    try:
        subprocess.run(["git", "add", str(config_file)], check=False)
    except FileNotFoundError:
        pass
    return True


def _handle_version(ns: argparse.Namespace) -> int:
    files = _get_modified_strategy_files(ns.strategy)
    if not files:
        print("✓ No strategy files detected for version update")
        return 0
    print(f"Found {len(files)} strategy file(s) to process:")
    for file in files:
        try:
            rel = file.relative_to(PROJECT_ROOT)
        except ValueError:
            rel = file
        print(f"  - {rel}")
    updated = 0
    for file in files:
        if _process_strategy_file(file, auto_confirm=ns.yes):
            updated += 1
    if updated:
        print(f"\n{'=' * 60}")
        print(f"✓ Updated {updated} strategy version(s)")
        print(f"  Config files staged for commit in {STRATEGIES_DIR.relative_to(PROJECT_ROOT)}")
        print(f"{'=' * 60}")
    return 0


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("strategies", help="Strategy management helpers")
    sub = parser.add_subparsers(dest="strategies_cmd", required=True)
    p_version = sub.add_parser(
        "version",
        help="Update strategy version manifests for modified strategy implementations",
    )
    p_version.add_argument(
        "--strategy",
        action="append",
        help="Specific strategy file(s) to process (default: staged git changes)",
    )
    p_version.add_argument(
        "--yes",
        action="store_true",
        help="Auto-confirm prompts using default change message",
    )
    p_version.set_defaults(func=_handle_version)
