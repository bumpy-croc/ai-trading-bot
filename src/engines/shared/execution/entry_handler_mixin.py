"""Entry-handler behavior shared verbatim by the backtest and live engines.

These methods were byte-identical copies in ``EntryHandler`` (backtest) and
``LiveEntryHandler`` (live). Hosting them here makes backtest-live parity for
entry-plan extraction and dynamic-risk sizing hold by construction instead of
by code review (#486, CODE.md Backtest-Live Parity).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Iterable
from datetime import datetime
from typing import TYPE_CHECKING, Any

from src.engines.shared.entry_utils import extract_entry_plan
from src.engines.shared.exposure import gross_exposure_fraction
from src.engines.shared.models import PositionSide

if TYPE_CHECKING:
    from src.engines.shared.dynamic_risk_handler import DynamicRiskHandler
    from src.position_management.dynamic_risk import DynamicRiskManager
    from src.position_management.macro_events import MacroEventGuard
    from src.risk.circuit_breaker import AccountCircuitBreaker
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
    # Optional macro-event de-risking guard (#806). Absent/None => inert.
    _macro_event_guard: MacroEventGuard | None = None
    # Optional account-level circuit breaker (#807). Absent/None => inert.
    _circuit_breaker: AccountCircuitBreaker | None = None
    # Optional unrealized-P&L feed for the breaker's true-equity read (#986).
    # Absent/None => the breaker evaluates the caller-supplied balance as-is.
    _breaker_unrealized_provider: Callable[[], float] | None = None
    _breaker_equity_degraded: bool = False

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

    def configure_macro_guard(self, macro_guard: MacroEventGuard | None) -> None:
        """Wire the #806 macro-event de-risking guard (independent of the governor)."""
        self._macro_event_guard = macro_guard

    def configure_circuit_breaker(
        self,
        circuit_breaker: AccountCircuitBreaker | None,
        *,
        unrealized_pnl_provider: Callable[[], float] | None = None,
    ) -> None:
        """Wire the #807 account-level circuit breaker (None => inert).

        ``unrealized_pnl_provider`` supplies the mark-to-market unrealized P&L
        of open positions so the breaker evaluates TRUE EQUITY (#986 gap A):
        the live engine's cash balance never reflects an open position's loss,
        which is the exact move the breaker exists to halt. Without a provider
        the breaker evaluates the caller-supplied balance unchanged (backtest
        evaluates entries only while flat, where equity == cash by identity).
        """
        self._circuit_breaker = circuit_breaker
        self._breaker_unrealized_provider = unrealized_pnl_provider
        self._breaker_equity_degraded = False

    def _breaker_equity(self, cash_balance: float) -> float:
        """True equity for the breaker: cash + unrealized P&L of open positions.

        Fault-isolated: an unavailable or non-finite unrealized read degrades
        explicitly to balance-only with a WARNING (logged on transition, not
        every call) — it never raises into the entry path and never feeds the
        breaker a garbage equity that could spuriously halt or silently no-op.
        """
        provider = self._breaker_unrealized_provider
        if provider is None:
            return cash_balance
        try:
            unrealized = float(provider())
            if not math.isfinite(unrealized):
                raise ValueError(f"non-finite unrealized P&L: {unrealized}")
            equity = cash_balance + unrealized
            if equity <= 0 < cash_balance:
                # evaluate() ignores non-positive equity entirely — surface the
                # degenerate read instead of letting the breaker silently no-op.
                raise ValueError(f"non-positive equity {equity} from unrealized {unrealized}")
        except Exception as e:  # noqa: BLE001 - equity read must never break entries
            if not self._breaker_equity_degraded:
                self._breaker_equity_degraded = True
                logger.warning(
                    "Circuit-breaker equity read unavailable — degrading to "
                    "balance-only evaluation (balance $%.2f): %s",
                    cash_balance,
                    e,
                )
            return cash_balance
        if self._breaker_equity_degraded:
            self._breaker_equity_degraded = False
            logger.info("Circuit-breaker equity read recovered — resuming equity evaluation")
        return equity

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

        Order: the #807 account circuit breaker (hard halt → block new entries),
        then the #806 macro-event guard (blocks entries and halves the exposure
        cap inside a FOMC/CPI window), then the #802 regime-gated exposure
        governor. Returns ``(allowed_fraction, reason_or_None)`` where ``reason``
        is set only when the size was reduced/blocked (for entry-decision
        logging). Each gate is independently inert until wired/enabled, so
        default behaviour and backtest-live parity are unchanged.
        """
        if size_fraction <= 0:
            return size_fraction, None

        # #807: account-level circuit breaker. A hard halt blocks new entries;
        # in dry_run it logs but does not block (entries_blocked stays False).
        # Evaluated on TRUE equity (cash + unrealized, #986): the `equity`
        # argument callers pass here is the cash balance, blind to open-position
        # losses. Scoped to the breaker — the macro guard and exposure governor
        # below keep their original balance basis unchanged.
        breaker = self._circuit_breaker
        if breaker is not None and now is not None:
            decision = breaker.evaluate(self._breaker_equity(equity), now)
            if decision.entries_blocked:
                return 0.0, f"circuit_breaker_{decision.reason}"

        # #806: macro-event de-risking window (independent of the governor).
        guard = self._macro_event_guard
        if guard is not None and guard.enabled and now is not None:
            if not guard.entry_allowed(now):
                return 0.0, f"macro_event_block_{guard.active_event_name(now)}"
            extra_factor *= guard.exposure_factor(now)

        # #802: regime-gated gross exposure governor.
        governor = self._exposure_governor
        if governor is None or not governor.enabled:
            return size_fraction, None
        gross = self.current_gross_exposure_fraction(equity)
        return governor.cap_fraction(
            size_fraction,
            regime=regime,
            gross_exposure_fraction=gross,
            extra_factor=extra_factor,
        )
