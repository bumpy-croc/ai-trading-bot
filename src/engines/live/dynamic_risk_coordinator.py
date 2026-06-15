"""Live dynamic-risk adjustment (#486).

Applies the shared ``DynamicRiskHandler`` to per-entry position sizing using
live engine state (balance, peak/drawdown, session) and logs the resulting
adjustments for observability/audit. Moved verbatim out of
``LiveTradingEngine`` (a mechanical ``self.`` -> ``state.`` rewrite against an
engine backref); the engine keeps thin delegating wrappers (still called by the
trading loop and by ``LiveEntryCoordinator`` via ``state._apply_dynamic_risk_adjustment``).

Runs on the trading-loop thread; the coordinator holds no state of its own.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Protocol

from src.infrastructure.logging.events import log_risk_event

if TYPE_CHECKING:
    from src.database.manager import DatabaseManager
    from src.engines.shared.dynamic_risk_handler import DynamicRiskHandler
    from src.position_management.dynamic_risk import DynamicRiskManager
    from src.trading.performance import PerformanceTracker

logger = logging.getLogger(__name__)


class LiveDynamicRiskEngineState(Protocol):
    """Engine state the dynamic-risk coordinator reads at call time."""

    current_balance: float
    initial_balance: float
    trading_session_id: int | None
    dynamic_risk_manager: DynamicRiskManager | None
    performance_tracker: PerformanceTracker
    db_manager: DatabaseManager
    _dynamic_risk_handler: DynamicRiskHandler


class LiveDynamicRiskCoordinator:
    """Owns per-entry dynamic-risk sizing adjustment + its audit logging."""

    def __init__(self, engine_state: LiveDynamicRiskEngineState) -> None:
        """Bind to the engine's live state (see protocol for the surface)."""
        self._state = engine_state

    def apply_dynamic_risk_adjustment(
        self,
        original_size: float,
        current_time: datetime,
    ) -> float:
        """Apply dynamic risk adjustments to position size.

        Reduces position size during drawdown or adverse market conditions
        to preserve capital and prevent excessive losses.
        """
        state = self._state
        if state.dynamic_risk_manager is None:
            return original_size

        try:
            perf_metrics = state.performance_tracker.get_metrics()

            # Guard against zero/None balances to prevent division by zero in drawdown calc
            balance = (
                float(state.current_balance)
                if state.current_balance is not None and state.current_balance > 0
                else float(state.initial_balance)
            )
            peak = (
                float(perf_metrics.peak_balance)
                if perf_metrics.peak_balance is not None and perf_metrics.peak_balance > 0
                else balance
            )
            # Peak should never be less than current balance
            peak_balance = max(peak, balance)

            adjusted_size = state._dynamic_risk_handler.apply_dynamic_risk(
                original_size=original_size,
                current_time=current_time,
                balance=balance,
                peak_balance=peak_balance,
                trading_session_id=state.trading_session_id,
            )
            self.log_dynamic_risk_adjustments()
            return adjusted_size

        except Exception as e:
            logger.warning("Failed to apply dynamic risk adjustment: %s", e)
            return original_size

    def log_dynamic_risk_adjustments(self) -> None:
        """Log dynamic risk adjustments for observability and audit."""
        state = self._state
        adjustments = state._dynamic_risk_handler.get_adjustment_objects(clear=True)
        for adjustment in adjustments:
            logger.info(
                "🎛️ Dynamic risk adjustment applied: size factor=%.2f, reason=%s",
                adjustment.position_size_factor,
                adjustment.primary_reason,
            )
            # Log both factor values (for analysis) and sizes (for debugging)
            log_risk_event(
                "dynamic_risk_adjustment",
                position_size_factor=adjustment.position_size_factor,
                reason=adjustment.primary_reason,
                original_value=1.0,
                adjusted_value=adjustment.position_size_factor,
                original_size=adjustment.original_size,
                adjusted_size=adjustment.adjusted_size,
                current_drawdown=adjustment.current_drawdown,
            )

            if state.db_manager and state.trading_session_id:
                try:
                    # Extract adjustment type from primary_reason (e.g., "drawdown_reduction" -> "drawdown")
                    # Use safe extraction to handle edge cases where reason doesn't contain "_"
                    reason = adjustment.primary_reason or "unknown"
                    adjustment_type = reason.split("_")[0] if "_" in reason else reason

                    # Log factor values (not position sizes) for backward compatibility
                    state.db_manager.log_risk_adjustment(
                        session_id=state.trading_session_id,
                        adjustment_type=adjustment_type,
                        trigger_reason=adjustment.primary_reason,
                        parameter_name="position_size_factor",
                        original_value=1.0,
                        adjusted_value=adjustment.position_size_factor,
                        adjustment_factor=adjustment.position_size_factor,
                        current_drawdown=adjustment.current_drawdown,
                        performance_score=None,
                        volatility_level=None,
                    )
                except Exception as log_e:
                    logger.warning("Failed to log risk adjustment to database: %s", log_e)
