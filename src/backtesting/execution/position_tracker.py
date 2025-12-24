"""PositionTracker manages active trade state and MFE/MAE tracking.

Centralizes position lifecycle management including partial exits,
scale-ins, and performance metric tracking.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from src.backtesting.models import ActiveTrade, Trade
from src.config.constants import DEFAULT_MFE_MAE_PRECISION_DECIMALS
from src.performance.metrics import cash_pnl
from src.position_management.mfe_mae_tracker import MFEMAETracker, MFEMetrics

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class PositionCloseResult:
    """Result of closing a position."""

    trade: Trade
    pnl_cash: float
    mfe_mae_metrics: MFEMetrics | None


class PositionTracker:
    """Tracks active trade state, partial operations, and MFE/MAE metrics.

    This class manages the lifecycle of an active position including:
    - Position state (entry, current size, trailing stop status)
    - Partial exit and scale-in tracking
    - Maximum Favorable/Adverse Excursion (MFE/MAE) metrics
    """

    POSITION_KEY = "active"

    def __init__(
        self,
        mfe_mae_precision: int = DEFAULT_MFE_MAE_PRECISION_DECIMALS,
    ) -> None:
        """Initialize position tracker.

        Args:
            mfe_mae_precision: Decimal precision for MFE/MAE calculations.
        """
        self.current_trade: ActiveTrade | None = None
        self.mfe_mae_tracker = MFEMAETracker(precision_decimals=mfe_mae_precision)

    @property
    def has_position(self) -> bool:
        """Check if there is an active position."""
        return self.current_trade is not None

    @property
    def position_side(self) -> str | None:
        """Get the side of the current position."""
        if self.current_trade is None:
            return None
        return self.current_trade.side

    def reset(self) -> None:
        """Reset tracker state for a new backtest run."""
        self.current_trade = None
        self.mfe_mae_tracker.clear(self.POSITION_KEY)

    def open_position(self, trade: ActiveTrade) -> None:
        """Start tracking a new position.

        Args:
            trade: The active trade to track.
        """
        self.current_trade = trade
        self.mfe_mae_tracker.clear(self.POSITION_KEY)
        logger.debug(
            "Opened %s position at %.2f, size=%.4f",
            trade.side,
            trade.entry_price,
            trade.size,
        )

    def update_metrics(
        self,
        current_price: float,
        current_time: datetime,
    ) -> None:
        """Update MFE/MAE for active position.

        Args:
            current_price: Current market price.
            current_time: Current timestamp.
        """
        if self.current_trade is None:
            return

        try:
            self.mfe_mae_tracker.update_position_metrics(
                position_key=self.POSITION_KEY,
                entry_price=float(self.current_trade.entry_price),
                current_price=float(current_price),
                side=self.current_trade.side,
                position_fraction=float(self.current_trade.size),
                current_time=current_time,
            )
        except Exception as e:
            logger.debug("Failed to update MFE/MAE metrics: %s", e)

    def apply_partial_exit(
        self,
        exit_fraction: float,
        current_price: float,
        basis_balance: float,
    ) -> float:
        """Reduce position size via partial exit.

        Args:
            exit_fraction: Fraction of current position to exit.
            current_price: Current market price for PnL calculation.
            basis_balance: Balance basis for PnL calculation.

        Returns:
            Realized PnL in cash terms.
        """
        if self.current_trade is None:
            return 0.0

        # Calculate PnL for the exited portion
        if self.current_trade.side == "long":
            move = (current_price - self.current_trade.entry_price) / self.current_trade.entry_price
        else:
            move = (self.current_trade.entry_price - current_price) / self.current_trade.entry_price

        pnl_pct = move * exit_fraction
        pnl_cash = cash_pnl(pnl_pct, basis_balance)

        # Update position state
        self.current_trade.current_size = max(0.0, self.current_trade.current_size - exit_fraction)
        self.current_trade.partial_exits_taken += 1

        logger.debug(
            "Partial exit: %.4f of position, realized PnL=%.2f",
            exit_fraction,
            pnl_cash,
        )

        return pnl_cash

    def apply_scale_in(self, additional_size: float) -> None:
        """Increase position size via scale-in.

        Args:
            additional_size: Additional size fraction to add.
        """
        if self.current_trade is None:
            return

        new_current_size = self.current_trade.current_size + additional_size
        self.current_trade.current_size = min(1.0, new_current_size)
        self.current_trade.size = min(1.0, self.current_trade.size + additional_size)
        self.current_trade.scale_ins_taken += 1

        logger.debug(
            "Scale-in: +%.4f, new size=%.4f",
            additional_size,
            self.current_trade.current_size,
        )

    def update_trailing_stop(
        self,
        new_stop_loss: float,
        activated: bool,
        breakeven_triggered: bool,
    ) -> bool:
        """Update trailing stop state for active position.

        Args:
            new_stop_loss: New stop loss price.
            activated: Whether trailing stop is now activated.
            breakeven_triggered: Whether breakeven has been triggered.

        Returns:
            True if stop loss was updated.
        """
        if self.current_trade is None:
            return False

        changed = False

        # Only update if new stop is better
        current_sl = self.current_trade.stop_loss
        if self.current_trade.side == "long":
            should_update = current_sl is None or new_stop_loss > float(current_sl)
        else:
            should_update = current_sl is None or new_stop_loss < float(current_sl)

        if should_update:
            self.current_trade.stop_loss = new_stop_loss
            self.current_trade.trailing_stop_price = new_stop_loss
            changed = True

        if activated != self.current_trade.trailing_stop_activated:
            self.current_trade.trailing_stop_activated = activated
            changed = True

        if breakeven_triggered != self.current_trade.breakeven_triggered:
            self.current_trade.breakeven_triggered = breakeven_triggered
            changed = True

        return changed

    def close_position(
        self,
        exit_price: float,
        exit_time: datetime,
        exit_reason: str,
        basis_balance: float,
    ) -> PositionCloseResult:
        """Close the current position and compute final trade record.

        Args:
            exit_price: Exit price (after slippage).
            exit_time: Exit timestamp.
            exit_reason: Reason for exit.
            basis_balance: Balance basis for PnL calculation.

        Returns:
            PositionCloseResult with completed trade and metrics.

        Raises:
            ValueError: If no active position to close.
        """
        if self.current_trade is None:
            raise ValueError("No active position to close")

        trade = self.current_trade
        fraction = float(getattr(trade, "current_size", trade.size))

        # Calculate PnL
        if trade.side == "long":
            trade_pnl_pct = (
                (exit_price - trade.entry_price) / trade.entry_price
            ) * fraction
        else:
            trade_pnl_pct = (
                (trade.entry_price - exit_price) / trade.entry_price
            ) * fraction

        entry_balance = getattr(trade, "entry_balance", None)
        if entry_balance is not None and entry_balance > 0:
            actual_basis = float(entry_balance)
        else:
            actual_basis = basis_balance

        trade_pnl_cash = cash_pnl(trade_pnl_pct, actual_basis)

        # Get MFE/MAE metrics before clearing
        metrics = self.mfe_mae_tracker.get_position_metrics(self.POSITION_KEY)

        # Create completed trade record
        completed_trade = Trade(
            symbol=trade.symbol,
            side=trade.side,
            entry_price=trade.entry_price,
            exit_price=exit_price,
            entry_time=trade.entry_time,
            exit_time=exit_time,
            size=fraction,
            pnl=trade_pnl_cash,
            pnl_percent=trade_pnl_pct,
            exit_reason=exit_reason,
            stop_loss=trade.stop_loss,
            take_profit=trade.take_profit,
            mfe=metrics.mfe if metrics else 0.0,
            mae=metrics.mae if metrics else 0.0,
            mfe_price=metrics.mfe_price if metrics else None,
            mae_price=metrics.mae_price if metrics else None,
            mfe_time=metrics.mfe_time if metrics else None,
            mae_time=metrics.mae_time if metrics else None,
        )

        # Clear tracker
        self.mfe_mae_tracker.clear(self.POSITION_KEY)
        self.current_trade = None

        logger.info(
            "Closed %s at %.2f, PnL=%.2f (%.2f%%)",
            trade.side,
            exit_price,
            trade_pnl_cash,
            trade_pnl_pct * 100,
        )

        return PositionCloseResult(
            trade=completed_trade,
            pnl_cash=trade_pnl_cash,
            mfe_mae_metrics=metrics,
        )

    def get_position_state(self) -> dict | None:
        """Get current position state for external use.

        Returns:
            Dictionary with position details, or None if no position.
        """
        if self.current_trade is None:
            return None

        return {
            "symbol": self.current_trade.symbol,
            "side": self.current_trade.side,
            "entry_price": self.current_trade.entry_price,
            "entry_time": self.current_trade.entry_time,
            "size": self.current_trade.size,
            "current_size": self.current_trade.current_size,
            "original_size": self.current_trade.original_size,
            "stop_loss": self.current_trade.stop_loss,
            "take_profit": self.current_trade.take_profit,
            "trailing_stop_activated": self.current_trade.trailing_stop_activated,
            "breakeven_triggered": self.current_trade.breakeven_triggered,
            "partial_exits_taken": self.current_trade.partial_exits_taken,
            "scale_ins_taken": self.current_trade.scale_ins_taken,
        }
