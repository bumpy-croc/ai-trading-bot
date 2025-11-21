from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

from cli.core.forward import forward_to_module_main
from src.infrastructure.runtime.paths import get_project_root

# Conditionally import training commands only in non-Railway environments
try:
    from src.config.providers.railway_provider import RailwayProvider

    railway_provider = RailwayProvider()
    _IS_RAILWAY = railway_provider.is_available()
except ImportError:
    # Fallback to environment variables if Railway provider not available
    railway_indicators = [
        "RAILWAY_DEPLOYMENT_ID",
        "RAILWAY_PROJECT_ID",
        "RAILWAY_SERVICE_ID",
        "RAILWAY_ENVIRONMENT_ID",
    ]
    _IS_RAILWAY = any(key in os.environ for key in railway_indicators)

if not _IS_RAILWAY:
    try:
        from cli.commands.train_commands import train_model_main, train_price_model_main

        _TRAINING_AVAILABLE = True
    except ImportError:
        _TRAINING_AVAILABLE = False
        train_model_main = None  # type: ignore
        train_price_model_main = None  # type: ignore
else:
    _TRAINING_AVAILABLE = False
    train_model_main = None  # type: ignore
    train_price_model_main = None  # type: ignore

logger = logging.getLogger(__name__)

MODEL_REGISTRY = get_project_root() / "src" / "ml" / "models"


def _handle(ns: argparse.Namespace) -> int:
    tail = ns.args or []
    return forward_to_module_main("src.live.runner", tail)


def _date_range(days: int) -> tuple[str, str]:
    end = datetime.utcnow().date()
    start = end - timedelta(days=days)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def _latest_metadata(symbol: str, model_type: str) -> Path:
    return MODEL_REGISTRY / symbol.upper() / model_type / "latest" / "metadata.json"


def _list_registry() -> dict[str, dict[str, dict[str, list[str] | str | None]]]:
    catalog: dict[str, dict[str, dict[str, list[str] | str | None]]] = {}
    if not MODEL_REGISTRY.exists():
        return catalog

    for symbol_dir in sorted(p for p in MODEL_REGISTRY.iterdir() if p.is_dir()):
        symbol_entry: dict[str, dict[str, list[str] | str | None]] = {}
        for model_type_dir in sorted(p for p in symbol_dir.iterdir() if p.is_dir()):
            versions = sorted(
                d.name for d in model_type_dir.iterdir() if d.is_dir() and d.name != "latest"
            )
            latest_target = None
            latest_path = model_type_dir / "latest"
            if latest_path.exists() or latest_path.is_symlink():
                try:
                    latest_target = Path(os.readlink(latest_path)).name
                except OSError:
                    latest_target = None
            symbol_entry[model_type_dir.name] = {
                "latest": latest_target,
                "versions": versions,
            }
        if symbol_entry:
            catalog[symbol_dir.name] = symbol_entry
    return catalog


def _resolve_version_path(path_str: str) -> Path:
    """
    Resolve and validate a model version path.

    Security: Always resolves paths (including absolute ones) before validation
    to prevent symlink-based path traversal attacks. A crafted symlink inside
    the registry could otherwise redirect to directories outside the boundary.
    """
    candidate = Path(path_str)
    if not candidate.is_absolute():
        candidate = MODEL_REGISTRY / candidate

    # Resolve ALL paths (including absolute) to follow symlinks and normalize
    candidate = candidate.resolve()

    if not candidate.exists():
        raise FileNotFoundError(f"Model path does not exist: {candidate}")
    if MODEL_REGISTRY not in candidate.parents:
        raise ValueError("Model path must be inside the registry")
    return candidate


def _repoint_latest(version_dir: Path) -> None:
    """Update 'latest' symlink to point to the specified version directory.

    Uses atomic temp+replace pattern to avoid race conditions (consistent with artifacts.py).

    Args:
        version_dir: Path to version directory (e.g., .../BTCUSDT/basic/2025-10-27_14h_v1)

    Raises:
        OSError: If symlink operations fail due to permissions or platform limitations
        FileNotFoundError: If parent directory doesn't exist
    """
    model_type_dir = version_dir.parent
    latest_link = model_type_dir / "latest"
    temp_link = model_type_dir / f".latest.{version_dir.name}.tmp"

    try:
        # Clean up any stale temp symlink
        if temp_link.exists() or temp_link.is_symlink():
            temp_link.unlink()

        # Create new symlink with temporary name
        temp_link.symlink_to(version_dir.name)

        # Atomically replace old symlink (rename is atomic on POSIX systems)
        temp_link.replace(latest_link)

        logger.info(f"Updated 'latest' symlink to point to {version_dir.name}")

    except PermissionError as e:
        # Clean up temp symlink on failure
        if temp_link.exists() or temp_link.is_symlink():
            try:
                temp_link.unlink()
            except OSError:
                pass
        logger.error(f"Permission denied when updating symlink at {latest_link}: {e}")
        raise OSError("Failed to update 'latest' symlink: insufficient permissions") from e
    except OSError as e:
        # Clean up temp symlink on failure
        if temp_link.exists() or temp_link.is_symlink():
            try:
                temp_link.unlink()
            except OSError:
                pass
        # Catch platform-specific errors (e.g., Windows without symlink privileges)
        logger.error(f"Failed to create symlink at {latest_link}: {e}")
        raise OSError(
            f"Failed to update 'latest' symlink at {latest_link}. "
            f"On Windows, ensure Developer Mode is enabled or run with admin privileges."
        ) from e
    except FileNotFoundError as e:
        logger.error(f"Parent directory not found: {model_type_dir}")
        raise FileNotFoundError(f"Model type directory does not exist: {model_type_dir}") from e


def _control(ns: argparse.Namespace) -> int:
    if ns.control_cmd == "train":
        if not _TRAINING_AVAILABLE:
            print("❌ Model training is not available in Railway environments.")
            print(
                "Training requires heavy dependencies (tensorflow) that are excluded from Railway builds."
            )
            print("Use a local development environment for model training.")
            return 1
        start_date, end_date = _date_range(ns.days)
        if ns.sentiment:
            args = SimpleNamespace(
                symbol=ns.symbol,
                start_date=start_date,
                end_date=end_date,
                timeframe="1d",
                force_sentiment=True,
                force_price_only=False,
                epochs=ns.epochs,
                batch_size=32,
                sequence_length=120,
                skip_plots=False,
                skip_robustness=False,
                skip_onnx=False,
                disable_mixed_precision=False,
            )
            code = train_model_main(args)
            model_type = "sentiment"
        else:
            args = SimpleNamespace(
                symbol=ns.symbol,
                start_date=start_date,
                end_date=end_date,
                timeframe="1d",
                epochs=ns.epochs,
                batch_size=256,
                sequence_length=120,
            )
            code = train_price_model_main(args)
            model_type = "basic"

        if code != 0:
            print("❌ Model training failed")
            return code

        meta_path = _latest_metadata(ns.symbol, model_type)
        if meta_path.exists():
            with open(meta_path, encoding="utf-8") as fh:
                metadata = json.load(fh)
            print("✅ Model training complete")
            print(
                json.dumps({"latest": metadata.get("version_id"), "metadata": metadata}, indent=2)
            )
        else:
            print("✅ Model training complete (metadata unavailable)")

        if ns.auto_deploy:
            print("✅ Auto-deploy: latest symlink already updated; live components will pick it up")
        return 0

    if ns.control_cmd == "deploy-model":
        try:
            version_dir = _resolve_version_path(ns.model_path)
        except (FileNotFoundError, ValueError) as exc:
            print(f"❌ {exc}")
            return 1
        _repoint_latest(version_dir)
        print(
            f"✅ Latest model for {version_dir.parent.parent.name}/{version_dir.parent.name} set to {version_dir.name}"
        )
        if ns.close_positions:
            print("⚠️ close-positions requested (no-op in CLI helper)")
        return 0

    if ns.control_cmd == "list-models":
        print(json.dumps(_list_registry(), indent=2))
        return 0

    if ns.control_cmd == "status":
        status = {
            "connected": False,
            "running": False,
            "current_strategy": "Unknown",
            "active_positions": 0,
        }
        print(json.dumps(status, indent=2))
        return 0

    if ns.control_cmd == "emergency-stop":
        print("🚨 EMERGENCY STOP INITIATED (simulated)")
        return 0

    if ns.control_cmd == "swap-strategy":
        print(
            f"🔄 Strategy swap requested to {ns.strategy} (close_positions={ns.close_positions}) (simulated)"
        )
        return 0

    return 1


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("live", help="Run live trading")
    p.add_argument("args", nargs=argparse.REMAINDER, help="Arguments passed through to runner")
    p.set_defaults(func=_handle)

    # Control group
    pc = subparsers.add_parser(
        "live-control", help="Control live trading (train/deploy/swap/status)"
    )
    sub = pc.add_subparsers(dest="control_cmd", required=True)

    p_train = sub.add_parser("train", help="Train new model")
    p_train.add_argument("--symbol", default="BTCUSDT")
    p_train.add_argument("--sentiment", action="store_true")
    p_train.add_argument("--days", type=int, default=365)
    p_train.add_argument("--epochs", type=int, default=50)
    p_train.add_argument("--auto-deploy", action="store_true")
    p_train.set_defaults(func=_control)

    p_deploy = sub.add_parser("deploy-model", help="Deploy a staged model")
    p_deploy.add_argument("--model-path", required=True)
    p_deploy.add_argument("--close-positions", action="store_true")
    p_deploy.set_defaults(func=_control)

    p_models = sub.add_parser("list-models", help="List available models")
    p_models.set_defaults(func=_control)

    p_status = sub.add_parser("status", help="Show engine status")
    p_status.set_defaults(func=_control)

    p_stop = sub.add_parser("emergency-stop", help="Emergency stop trading")
    p_stop.set_defaults(func=_control)

    p_swap = sub.add_parser("swap-strategy", help="Hot-swap strategy (simulated)")
    p_swap.add_argument("--strategy", required=True)
    p_swap.add_argument("--close-positions", action="store_true")
    p_swap.set_defaults(func=_control)
