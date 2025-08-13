import types as _types
from importlib import import_module as _import_module

from dashboards.monitoring import MonitoringDashboard  # noqa: F401

__all__ = ["MonitoringDashboard"]

# Back-compat shim for tests expecting src.monitoring.dashboard.*
try:
    _dash_mod = _import_module("dashboards.monitoring.dashboard")
    dashboard = _dash_mod
except Exception:  # pragma: no cover - best effort shim
    dashboard = _types.SimpleNamespace(MonitoringDashboard=MonitoringDashboard)
