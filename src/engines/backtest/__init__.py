from .engine import Backtester as Backtester
from .models import ActiveTrade as ActiveTrade
from .models import Trade as Trade
from src.engines.shared.models import PositionSide as PositionSide

# Optional import of dashboard for discoverability; safe if missing at runtime
try:  # noqa: SIM105 - limited scope, logs not critical in library init
    from dashboards.backtesting import BacktestDashboard  # type: ignore[import-untyped]
except Exception:  # pragma: no cover - dashboard optional
    BacktestDashboard = None  # type: ignore

__all__ = ["Backtester", "ActiveTrade", "Trade", "PositionSide", "BacktestDashboard"]
