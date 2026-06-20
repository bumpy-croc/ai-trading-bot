"""Entry-handler behavior shared verbatim by the backtest and live engines.

These methods were byte-identical copies in ``EntryHandler`` (backtest) and
``LiveEntryHandler`` (live). Hosting them here makes backtest-live parity for
entry-plan extraction and dynamic-risk sizing hold by construction instead of
by code review (#486, CODE.md Backtest-Live Parity).
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

from src.engines.shared.entry_utils import extract_entry_plan
from src.engines.shared.models import PositionSide

if TYPE_CHECKING:
    from src.engines.shared.dynamic_risk_handler import DynamicRiskHandler
    from src.position_management.dynamic_risk import DynamicRiskManager


class SharedEntryHandlerMixin:
    """Entry-plan extraction and dynamic-risk sizing common to both engines.

    The inheriting handler must set ``dynamic_risk_manager`` and
    ``_dynamic_risk_handler`` in its ``__init__``.
    """

    dynamic_risk_manager: DynamicRiskManager | None
    _dynamic_risk_handler: DynamicRiskHandler

    def _extract_entry_plan(
        self,
        decision: Any,
        balance: float,
    ) -> tuple[PositionSide | None, float]:
        """Extract entry side and size from runtime decision.

        Args:
            decision: Runtime decision from strategy.
            balance: Current account balance.

        Returns:
            Tuple of (side, size_fraction).
        """
        plan = extract_entry_plan(decision, balance)
        if plan is None:
            return None, 0.0
        return plan.side, plan.size_fraction

    def _apply_dynamic_risk(
        self,
        original_size: float,
        current_time: datetime,
        balance: float,
        peak_balance: float,
        trading_session_id: int | None,
    ) -> float:
        """Apply dynamic risk adjustments to position size.

        Delegates to shared DynamicRiskHandler for consistent logic
        between backtest and live engines.

        Args:
            original_size: Original position size fraction.
            current_time: Current timestamp.
            balance: Current account balance.
            peak_balance: Peak account balance.
            trading_session_id: Session ID for logging.

        Returns:
            Adjusted position size fraction.
        """
        # Update handler's manager in case it changed
        self._dynamic_risk_handler.set_manager(self.dynamic_risk_manager)
        return self._dynamic_risk_handler.apply_dynamic_risk(
            original_size=original_size,
            current_time=current_time,
            balance=balance,
            peak_balance=peak_balance,
            trading_session_id=trading_session_id,
        )

    def get_dynamic_risk_adjustments(self) -> list[dict]:
        """Get and clear dynamic risk adjustments tracked by this handler.

        Returns:
            List of dynamic risk adjustment records.
        """
        return self._dynamic_risk_handler.get_adjustments(clear=True)
