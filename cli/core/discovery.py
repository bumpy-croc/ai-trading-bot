from __future__ import annotations

import importlib
import inspect
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

# Ensure project root and src are in sys.path for absolute imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_PATH = PROJECT_ROOT / "src"
if SRC_PATH.exists() and str(SRC_PATH) not in sys.path:
    sys.path.insert(1, str(SRC_PATH))


@dataclass
class DiscoveredDashboard:
    name: str
    module_name: str
    object_name: str | None  # Class name ending with 'Dashboard', or None if only main()
    summary: str | None


def _import_module(module_name: str) -> ModuleType:
    try:
        return importlib.import_module(module_name)
    except Exception as exc:
        raise RuntimeError(f"Failed to import module '{module_name}': {exc}") from exc


def discover_dashboards() -> dict[str, DiscoveredDashboard]:
    """Discover dashboards under src/dashboards/*/dashboard.py.

    Returns a mapping of dashboard name -> DiscoveredDashboard.
    """
    dashboards: dict[str, DiscoveredDashboard] = {}
    base = SRC_PATH / "dashboards"
    if not base.exists():
        return dashboards

    for item in sorted(base.iterdir()):
        if not item.is_dir():
            continue
        module_path = item / "dashboard.py"
        if not module_path.exists():
            continue
        name = item.name
        module_name = f"src.dashboards.{name}.dashboard"
        try:
            mod = _import_module(module_name)
        except Exception:
            # Skip broken dashboards but keep CLI operational
            continue

        # Prefer a class named *Dashboard with a .run() method
        object_name: str | None = None
        for cls_name, cls_obj in inspect.getmembers(mod, inspect.isclass):
            if cls_obj.__module__ != mod.__name__:
                continue
            if cls_name.endswith("Dashboard") and hasattr(cls_obj, "run"):
                object_name = cls_name
                break

        # Fallback: module-level main()
        if object_name is None and not hasattr(mod, "main"):
            # If neither class nor main() exists, skip
            continue

        # Build a summary from docstrings if available
        summary = None
        if object_name is not None:
            summary = getattr(getattr(mod, object_name), "__doc__", None)
        if not summary:
            summary = getattr(mod, "__doc__", None)

        dashboards[name] = DiscoveredDashboard(
            name=name,
            module_name=module_name,
            object_name=object_name,
            summary=(summary or "").strip() or None,
        )

    return dashboards


def call_with_supported_params(func, maybe_kwargs: dict[str, Any]) -> Any:
    """Call a function with only the kwargs that it accepts (by name)."""
    import inspect as _inspect

    sig = _inspect.signature(func)
    accepted = {k: v for k, v in maybe_kwargs.items() if k in sig.parameters and v is not None}
    return func(**accepted)
