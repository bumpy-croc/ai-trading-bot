import types as _types
from importlib import import_module as _import_module
from typing import Union

from src.dashboards.monitoring import MonitoringDashboard  # noqa: F401

__all__ = ["MonitoringDashboard"]

# Back-compat shim for tests expecting src.monitoring.dashboard.*
dashboard: Union[_types.ModuleType, _types.SimpleNamespace]
try:
    _dash_mod = _import_module("src.dashboards.monitoring.dashboard")
    dashboard = _dash_mod
except Exception:  # pragma: no cover - best effort shim
    dashboard = _types.SimpleNamespace(MonitoringDashboard=MonitoringDashboard)
