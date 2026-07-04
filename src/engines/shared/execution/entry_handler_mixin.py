"""Entry-handler behavior shared verbatim by the backtest and live engines.

These methods were byte-identical copies in ``EntryHandler`` (backtest) and
``LiveEntryHandler`` (live). Hosting them here makes backtest-live parity for
entry-plan extraction and dynamic-risk sizing hold by construction instead of
by code review (#486, CODE.md Backtest-Live Parity).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from datetime import datetime
from typing import TYPE_CHECKING, Any

from src.engines.shared.entry_utils import extract_entry_plan
from src.engines.shared.exposure import gross_exposure_fraction
from src.engines.shared.models import PositionSide

if TYPE_CHECKING:
    from src.engines.shared.dynamic_risk_handler import DynamicRiskHandler
    from src.position_management.dynamic_risk import DynamicRiskManager
    from src.strategies.components.exposure_governor import ExposureGovernor

logger = logging.getLogger(__name__)


class SharedEntryHandlerMixin:
    """Entry-plan extraction and dynamic-risk sizing common to both engines.

    The inheriting handler must set ``dynamic_risk_manager`` and
    ``_dynamic_risk_handler`` in its ``__init__``. It may optionally set
    ``_exposure_governor`` and ``_positions_source`` to enable the shared
    pre-order exposure gate (#802).
    """

    dynamic_risk_manager: DynamicRiskManager | None
    _dynamic_risk_handler: DynamicRiskHandler
    # Optional pre-order gate wiring (#802). Absent/None => gate is inert.
    _exposure_governor: ExposureGovernor | None = None
    _positions_source: Callable[[], Iterable[Any]] | None = None

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

    def configure_exposure_gate(
        self,
        exposure_governor: ExposureGovernor | None,
        positions_source: Callable[[], Iterable[Any]] | None,
    ) -> None:
        """Wire the #802 pre-order exposure gate (governor + open-position source).

        Called by each engine after constructing the handler. With either arg
        ``None`` the gate stays inert, so parity and default behaviour are
        unchanged until an engine opts in.
        """
        self._exposure_governor = exposure_governor
        self._positions_source = positions_source

    def current_gross_exposure_fraction(self, equity: float) -> float:
        """Current gross open exposure as a fraction of ``equity`` (0.0 if unknown).

        Reads open positions from the injected ``_positions_source`` and delegates
        the arithmetic to the shared :func:`gross_exposure_fraction`, so backtest
        and live compute exposure identically. Never raises — an exposure-calc
        failure must not break entries; it degrades to 0.0 (uncapped) and logs.
        """
        source = self._positions_source
        if source is None:
            return 0.0
        try:
            positions = source()
            return gross_exposure_fraction(positions, equity)
        except Exception:  # noqa: BLE001 - exposure calc must never break entries
            logger.warning("current_gross_exposure_fraction failed; treating as 0", exc_info=True)
            return 0.0

    def apply_pre_order_gates(
        self,
        size_fraction: float,
        *,
        regime: Any,
        equity: float,
        now: datetime | None = None,
        extra_factor: float = 1.0,
    ) -> tuple[float, str | None]:
        """Apply the shared pre-order risk gates to a sized position fraction.

        Currently the regime-gated exposure governor (#802); #806/#807 extend
        this with event-window and circuit-breaker caps. Returns
        ``(allowed_fraction, reason_or_None)`` where ``reason`` is set only when
        the size was reduced (for entry-decision logging). Inert (returns the
        input unchanged) when no governor is wired or it is disabled.
        """
        governor = self._exposure_governor
        if governor is None or size_fraction <= 0 or not governor.enabled:
            return size_fraction, None
        gross = self.current_gross_exposure_fraction(equity)
        return governor.cap_fraction(
            size_fraction,
            regime=regime,
            gross_exposure_fraction=gross,
            extra_factor=extra_factor,
        )
