"""Shared partial exit execution logic for both backtesting and live trading engines.

This module provides unified P&L calculation and fee/slippage handling for partial
exits, ensuring financial consistency between backtest and live results.

ARCHITECTURE:
- Single source of truth for partial exit calculations
- Used by both PositionTracker (backtest) and LivePositionTracker (live)
- Prevents divergence in financial calculations
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from src.config.constants import DEFAULT_FEE_RATE, DEFAULT_SLIPPAGE_RATE
from src.engines.shared.models import PositionSide
from src.performance.metrics import Side, cash_pnl, pnl_percent

logger = logging.getLogger(__name__)

# Epsilon for floating-point comparisons in financial calculations
EPSILON = 1e-9


@dataclass
class PartialExitExecutionResult:
    """Result of executing a partial exit.

    Attributes:
        realized_pnl: Net cash P&L after fees and slippage.
        gross_pnl: Gross cash P&L before costs.
        exit_fee: Fee charged on the exit.
        slippage_cost: Slippage cost on the exit.
        exit_notional: Notional value at exit (price-adjusted).
        pnl_percent: P&L as percentage (sized by exit fraction).
    """

    realized_pnl: float
    gross_pnl: float
    exit_fee: float
    slippage_cost: float
    exit_notional: float
    pnl_percent: float


class PartialExitExecutor:
    """Unified partial exit execution logic.

    This class provides consistent P&L calculation and fee/slippage handling
    for partial exits. Both backtesting and live trading engines delegate to
    this executor to ensure identical financial calculations.

    Attributes:
        fee_rate: Fee rate as decimal (e.g., 0.001 = 0.1%).
        slippage_rate: Slippage rate as decimal (e.g., 0.0005 = 0.05%).
    """

    def __init__(
        self,
        fee_rate: float = DEFAULT_FEE_RATE,
        slippage_rate: float = DEFAULT_SLIPPAGE_RATE,
    ) -> None:
        """Initialize the partial exit executor.

        Args:
            fee_rate: Fee rate per trade (default 0.1%).
            slippage_rate: Slippage rate per trade (default 0.05%).

        Raises:
            ValueError: If fee_rate or slippage_rate are negative.
        """
        if fee_rate < 0:
            raise ValueError(f"fee_rate must be non-negative, got {fee_rate}")
        if slippage_rate < 0:
            raise ValueError(f"slippage_rate must be non-negative, got {slippage_rate}")

        self.fee_rate = fee_rate
        self.slippage_rate = slippage_rate

    def execute_partial_exit(
        self,
        entry_price: float,
        exit_price: float,
        position_side: PositionSide | str,
        exit_fraction: float,
        basis_balance: float,
    ) -> PartialExitExecutionResult:
        """Execute a partial exit with fees and slippage.

        This is the single source of truth for partial exit calculations.
        Both engines must use this method to ensure identical results.

        Args:
            entry_price: Original entry price of the position.
            exit_price: Current exit price.
            position_side: Position side (LONG or SHORT).
            exit_fraction: Fraction of position being exited (0-1).
            basis_balance: Balance basis for P&L calculation.

        Returns:
            PartialExitExecutionResult with P&L and cost breakdown.

        Raises:
            ValueError: If inputs are invalid.
        """
        # Validate inputs
        if entry_price <= 0:
            raise ValueError(f"entry_price must be positive, got {entry_price}")
        if exit_price <= 0:
            raise ValueError(f"exit_price must be positive, got {exit_price}")
        if exit_fraction < 0 or exit_fraction > 1.0 + EPSILON:
            raise ValueError(f"exit_fraction must be in [0, 1], got {exit_fraction}")
        if basis_balance < 0:
            raise ValueError(f"basis_balance must be non-negative, got {basis_balance}")

        # Normalize position side
        side = self._normalize_side(position_side)

        # Calculate P&L percentage
        if side == Side.LONG:
            pnl_pct = pnl_percent(entry_price, exit_price, Side.LONG, exit_fraction)
        else:
            pnl_pct = pnl_percent(entry_price, exit_price, Side.SHORT, exit_fraction)

        # Calculate gross P&L (before fees and slippage)
        gross_pnl = cash_pnl(pnl_pct, basis_balance)

        # Calculate exit notional accounting for price change
        # CRITICAL: Use exit notional (price-adjusted) for accurate fee calculation
        # This matches real exchange behavior where fees are charged on actual value at exit time:
        # - Winning positions: selling more valuable assets → higher fee (correct)
        # - Losing positions: selling less valuable assets → lower fee (correct)
        entry_notional = basis_balance * exit_fraction
        price_adjustment = exit_price / entry_price if entry_price > 0 else 1.0
        exit_notional = entry_notional * price_adjustment

        # Calculate fees and slippage on exit notional
        exit_fee = abs(exit_notional * self.fee_rate)
        slippage_cost = abs(exit_notional * self.slippage_rate)

        # Calculate net P&L (after fees and slippage)
        realized_pnl = gross_pnl - exit_fee - slippage_cost

        logger.debug(
            "Partial exit: fraction=%.4f, gross_pnl=%.2f, fee=%.2f, slippage=%.2f, net_pnl=%.2f",
            exit_fraction,
            gross_pnl,
            exit_fee,
            slippage_cost,
            realized_pnl,
        )

        return PartialExitExecutionResult(
            realized_pnl=realized_pnl,
            gross_pnl=gross_pnl,
            exit_fee=exit_fee,
            slippage_cost=slippage_cost,
            exit_notional=exit_notional,
            pnl_percent=pnl_pct,
        )

    def _normalize_side(self, side: PositionSide | str) -> Side:
        """Normalize position side to Side enum for metrics calculations.

        Args:
            side: Position side as PositionSide enum or string.

        Returns:
            Side enum (LONG or SHORT).

        Raises:
            ValueError: If side is invalid.
        """
        if isinstance(side, PositionSide):
            return Side.LONG if side == PositionSide.LONG else Side.SHORT

        side_str = str(side).lower()
        if side_str == "long":
            return Side.LONG
        elif side_str == "short":
            return Side.SHORT
        else:
            raise ValueError(f"Invalid position side: {side}")


__all__ = [
    "PartialExitExecutor",
    "PartialExitExecutionResult",
]
