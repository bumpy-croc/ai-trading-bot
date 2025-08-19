from .engine import Backtester as Backtester
from .models import Trade as Trade

# Optional import of dashboard for discoverability; safe if missing at runtime
try:  # noqa: SIM105 - limited scope, logs not critical in library init
    from dashboards.backtesting import BacktestDashboard  # type: ignore[import-untyped]
except Exception:  # pragma: no cover - dashboard optional
    BacktestDashboard = None  # type: ignore

__all__ = ["Backtester", "Trade", "BacktestDashboard"]
