"""ExecutionEngine handles realistic trade execution mechanics.

Centralizes fee calculation, slippage modeling, and next-bar execution logic
to ensure consistent execution simulation across entry and exit paths.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from src.backtesting.models import ActiveTrade

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """Result of a trade execution operation."""

    trade: ActiveTrade | None = None
    entry_fee: float = 0.0
    slippage_cost: float = 0.0
    executed: bool = False
    pending: bool = False


class ExecutionEngine:
    """Handles realistic trade execution with fees, slippage, and timing.

    This class centralizes all execution mechanics to ensure consistent
    behavior across entry and exit paths. Supports both immediate execution
    and next-bar execution modes.
    """

    def __init__(
        self,
        fee_rate: float = 0.001,
        slippage_rate: float = 0.0005,
        use_next_bar_execution: bool = False,
    ) -> None:
        """Initialize execution engine with trading costs.

        Args:
            fee_rate: Fee percentage per trade (default 0.1%).
            slippage_rate: Slippage percentage per trade (default 0.05%).
            use_next_bar_execution: If True, queue entries for next bar's open.
        """
        self.fee_rate = fee_rate
        self.slippage_rate = slippage_rate
        self.use_next_bar_execution = use_next_bar_execution
        self._pending_entry: dict[str, Any] | None = None
        self.total_fees_paid = 0.0
        self.total_slippage_cost = 0.0

    @property
    def has_pending_entry(self) -> bool:
        """Check if there is a pending entry waiting for execution."""
        return self._pending_entry is not None

    @property
    def pending_entry(self) -> dict[str, Any] | None:
        """Get the current pending entry details."""
        return self._pending_entry

    def reset(self) -> None:
        """Reset execution state for a new backtest run."""
        self._pending_entry = None
        self.total_fees_paid = 0.0
        self.total_slippage_cost = 0.0

    def queue_entry(
        self,
        side: str,
        size_fraction: float,
        sl_pct: float,
        tp_pct: float,
        signal_price: float,
        signal_time: datetime,
        component_notional: float | None = None,
    ) -> None:
        """Queue an entry for execution on the next bar's open.

        Args:
            side: 'long' or 'short'.
            size_fraction: Position size as fraction of balance.
            sl_pct: Stop loss percentage from entry.
            tp_pct: Take profit percentage from entry.
            signal_price: Price at which signal was generated.
            signal_time: Time at which signal was generated.
            component_notional: Notional value for component tracking.
        """
        self._pending_entry = {
            "side": side,
            "size_fraction": size_fraction,
            "sl_pct": sl_pct,
            "tp_pct": tp_pct,
            "signal_time": signal_time,
            "signal_price": signal_price,
            "component_notional": component_notional,
        }
        logger.info(
            "Pending %s entry at signal price %.2f, will execute on next bar open",
            side,
            signal_price,
        )

    def execute_pending_entry(
        self,
        symbol: str,
        open_price: float,
        current_time: datetime,
        balance: float,
    ) -> ExecutionResult:
        """Execute a pending entry on the current bar's open.

        Args:
            symbol: Trading symbol.
            open_price: Opening price of the current bar.
            current_time: Current timestamp.
            balance: Current account balance.

        Returns:
            ExecutionResult with trade details and costs.
        """
        if self._pending_entry is None:
            return ExecutionResult(executed=False)

        pending = self._pending_entry
        self._pending_entry = None

        # Calculate entry price with slippage
        if pending["side"] == "long":
            # Buying: slippage works against us (higher price)
            entry_price = open_price * (1 + self.slippage_rate)
        else:
            # Shorting: slippage works against us (lower price)
            entry_price = open_price * (1 - self.slippage_rate)

        # Calculate costs
        position_notional = balance * pending["size_fraction"]
        entry_fee = abs(position_notional * self.fee_rate)
        slippage_cost = abs(position_notional * self.slippage_rate)

        # Track costs
        self.total_fees_paid += entry_fee
        self.total_slippage_cost += slippage_cost

        # Calculate SL/TP based on actual entry price
        if pending["side"] == "long":
            stop_loss = entry_price * (1 - pending["sl_pct"])
            take_profit = entry_price * (1 + pending["tp_pct"])
        else:
            stop_loss = entry_price * (1 + pending["sl_pct"])
            take_profit = entry_price * (1 - pending["tp_pct"])

        # Create trade
        trade = ActiveTrade(
            symbol=symbol,
            side=pending["side"],
            entry_price=entry_price,
            entry_time=current_time,
            size=pending["size_fraction"],
            stop_loss=stop_loss,
            take_profit=take_profit,
            entry_balance=balance - entry_fee,
        )
        trade.component_notional = pending.get(
            "component_notional", pending["size_fraction"] * balance
        )

        logger.info(
            "Entered %s at %.2f (open: %.2f) via next-bar execution",
            pending["side"],
            entry_price,
            open_price,
        )

        return ExecutionResult(
            trade=trade,
            entry_fee=entry_fee,
            slippage_cost=slippage_cost,
            executed=True,
        )

    def execute_immediate_entry(
        self,
        symbol: str,
        side: str,
        size_fraction: float,
        current_price: float,
        current_time: datetime,
        balance: float,
        stop_loss: float,
        take_profit: float,
        component_notional: float | None = None,
    ) -> ExecutionResult:
        """Execute an entry immediately with slippage applied.

        Args:
            symbol: Trading symbol.
            side: 'long' or 'short'.
            size_fraction: Position size as fraction of balance.
            current_price: Current market price.
            current_time: Current timestamp.
            balance: Current account balance.
            stop_loss: Stop loss price level.
            take_profit: Take profit price level.
            component_notional: Notional value for component tracking.

        Returns:
            ExecutionResult with trade details and costs.
        """
        # Calculate entry price with slippage
        if side == "long":
            entry_price = current_price * (1 + self.slippage_rate)
        else:
            entry_price = current_price * (1 - self.slippage_rate)

        # Calculate costs
        position_notional = balance * size_fraction
        entry_fee = abs(position_notional * self.fee_rate)
        slippage_cost = abs(position_notional * self.slippage_rate)

        # Track costs
        self.total_fees_paid += entry_fee
        self.total_slippage_cost += slippage_cost

        # Create trade
        trade = ActiveTrade(
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            entry_time=current_time,
            size=size_fraction,
            stop_loss=stop_loss,
            take_profit=take_profit,
            entry_balance=balance - entry_fee,
        )
        trade.component_notional = (
            component_notional if component_notional else size_fraction * balance
        )

        logger.info(
            "Entered %s at %.2f via immediate execution",
            side,
            entry_price,
        )

        return ExecutionResult(
            trade=trade,
            entry_fee=entry_fee,
            slippage_cost=slippage_cost,
            executed=True,
        )

    def calculate_exit_costs(
        self,
        base_price: float,
        side: str,
        position_notional: float,
    ) -> tuple[float, float, float]:
        """Calculate exit price and costs with slippage applied.

        Args:
            base_price: Base exit price (SL, TP, or close).
            side: 'long' or 'short' (the position side).
            position_notional: Notional value of the position.

        Returns:
            Tuple of (exit_price, exit_fee, slippage_cost).
        """
        # Apply slippage adversely
        if side == "long":
            # Selling long: slippage works against us (lower price)
            exit_price = base_price * (1 - self.slippage_rate)
        else:
            # Covering short: slippage works against us (higher price)
            exit_price = base_price * (1 + self.slippage_rate)

        exit_fee = abs(position_notional * self.fee_rate)
        slippage_cost = abs(position_notional * self.slippage_rate)

        # Track costs
        self.total_fees_paid += exit_fee
        self.total_slippage_cost += slippage_cost

        return exit_price, exit_fee, slippage_cost

    def clear_pending_entry(self) -> dict[str, Any] | None:
        """Clear and return any pending entry (for backtest end warning).

        Returns:
            The pending entry that was cleared, or None.
        """
        pending = self._pending_entry
        self._pending_entry = None
        return pending

    def get_execution_settings(self) -> dict[str, Any]:
        """Get current execution settings for result reporting.

        Returns:
            Dictionary of execution configuration.
        """
        return {
            "fee_rate": self.fee_rate,
            "slippage_rate": self.slippage_rate,
            "use_next_bar_execution": self.use_next_bar_execution,
            "use_high_low_for_stops": True,  # Always true in new design
        }

    def get_cost_summary(self) -> dict[str, float]:
        """Get summary of execution costs.

        Returns:
            Dictionary with total fees and slippage costs.
        """
        return {
            "total_fees": self.total_fees_paid,
            "total_slippage_cost": self.total_slippage_cost,
        }
