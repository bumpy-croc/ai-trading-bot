"""Unified performance tracking for trading engines.

This module provides consistent performance metrics tracking for both
backtesting and live trading engines.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.engines.shared.models import BaseTrade

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Container for performance metrics.

    Attributes:
        total_trades: Total number of completed trades.
        winning_trades: Number of profitable trades.
        losing_trades: Number of unprofitable trades.
        total_pnl: Total realized profit/loss.
        total_fees_paid: Total fees paid.
        total_slippage_cost: Total slippage cost.
        max_drawdown: Maximum drawdown percentage.
        peak_balance: Highest balance achieved.
        current_balance: Current balance.
        win_rate: Percentage of winning trades.
        profit_factor: Ratio of gross profit to gross loss.
        avg_win: Average winning trade PnL.
        avg_loss: Average losing trade PnL (negative).
        largest_win: Largest winning trade PnL.
        largest_loss: Largest losing trade PnL (negative).
        avg_trade_duration_hours: Average trade duration.
    """

    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    total_fees_paid: float = 0.0
    total_slippage_cost: float = 0.0
    max_drawdown: float = 0.0
    peak_balance: float = 0.0
    current_balance: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    avg_trade_duration_hours: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "total_pnl": self.total_pnl,
            "total_fees_paid": self.total_fees_paid,
            "total_slippage_cost": self.total_slippage_cost,
            "max_drawdown": self.max_drawdown,
            "peak_balance": self.peak_balance,
            "current_balance": self.current_balance,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "largest_win": self.largest_win,
            "largest_loss": self.largest_loss,
            "avg_trade_duration_hours": self.avg_trade_duration_hours,
        }


class PerformanceTracker:
    """Unified performance metrics tracking.

    This class provides consistent performance tracking that is used
    by both backtesting and live trading engines.

    Attributes:
        initial_balance: Starting balance.
        current_balance: Current balance.
        peak_balance: Highest balance achieved.
        max_drawdown: Maximum drawdown percentage seen.
    """

    def __init__(self, initial_balance: float) -> None:
        """Initialize the performance tracker.

        Args:
            initial_balance: Starting account balance.
        """
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.peak_balance = initial_balance
        self.max_drawdown = 0.0

        # Trade statistics
        self._trades: list[dict] = []
        self._total_fees_paid = 0.0
        self._total_slippage_cost = 0.0
        self._total_pnl = 0.0
        self._gross_profit = 0.0
        self._gross_loss = 0.0
        self._winning_trades = 0
        self._losing_trades = 0
        self._total_duration_seconds = 0.0

        # Balance history for drawdown calculation
        self._balance_history: list[tuple[datetime, float]] = [
            (datetime.now(), initial_balance)
        ]

    def record_trade(
        self,
        trade: BaseTrade | Any,
        fee: float = 0.0,
        slippage: float = 0.0,
    ) -> None:
        """Record a completed trade.

        Args:
            trade: Completed trade object.
            fee: Total fee for the trade.
            slippage: Total slippage cost for the trade.
        """
        pnl = getattr(trade, "pnl", 0.0) or 0.0
        entry_time = getattr(trade, "entry_time", None)
        exit_time = getattr(trade, "exit_time", None)

        # Update trade counts
        if pnl > 0:
            self._winning_trades += 1
            self._gross_profit += pnl
        elif pnl < 0:
            self._losing_trades += 1
            self._gross_loss += abs(pnl)

        # Update totals
        self._total_pnl += pnl
        self._total_fees_paid += fee
        self._total_slippage_cost += slippage

        # Calculate duration
        if entry_time and exit_time:
            duration = (exit_time - entry_time).total_seconds()
            self._total_duration_seconds += duration

        # Store trade record
        self._trades.append({
            "pnl": pnl,
            "fee": fee,
            "slippage": slippage,
            "entry_time": entry_time,
            "exit_time": exit_time,
            "symbol": getattr(trade, "symbol", None),
            "side": str(getattr(trade, "side", None)),
        })

    def update_balance(
        self,
        balance: float,
        timestamp: datetime | None = None,
    ) -> None:
        """Update current balance and recalculate drawdown.

        Args:
            balance: New account balance.
            timestamp: Optional timestamp for the update.
        """
        self.current_balance = balance

        # Update peak balance
        if balance > self.peak_balance:
            self.peak_balance = balance

        # Calculate current drawdown
        if self.peak_balance > 0:
            current_drawdown = (self.peak_balance - balance) / self.peak_balance
            if current_drawdown > self.max_drawdown:
                self.max_drawdown = current_drawdown

        # Record in history
        ts = timestamp or datetime.now()
        self._balance_history.append((ts, balance))

    def get_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics.

        Returns:
            PerformanceMetrics with calculated values.
        """
        total_trades = self._winning_trades + self._losing_trades

        # Calculate win rate
        win_rate = 0.0
        if total_trades > 0:
            win_rate = self._winning_trades / total_trades

        # Calculate profit factor
        profit_factor = 0.0
        if self._gross_loss > 0:
            profit_factor = self._gross_profit / self._gross_loss

        # Calculate averages
        avg_win = 0.0
        if self._winning_trades > 0:
            avg_win = self._gross_profit / self._winning_trades

        avg_loss = 0.0
        if self._losing_trades > 0:
            avg_loss = -self._gross_loss / self._losing_trades

        # Calculate largest win/loss
        largest_win = 0.0
        largest_loss = 0.0
        for trade in self._trades:
            pnl = trade.get("pnl", 0)
            if pnl > largest_win:
                largest_win = pnl
            if pnl < largest_loss:
                largest_loss = pnl

        # Calculate average duration
        avg_duration_hours = 0.0
        if total_trades > 0:
            avg_duration_hours = (self._total_duration_seconds / total_trades) / 3600

        return PerformanceMetrics(
            total_trades=total_trades,
            winning_trades=self._winning_trades,
            losing_trades=self._losing_trades,
            total_pnl=self._total_pnl,
            total_fees_paid=self._total_fees_paid,
            total_slippage_cost=self._total_slippage_cost,
            max_drawdown=self.max_drawdown,
            peak_balance=self.peak_balance,
            current_balance=self.current_balance,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            largest_win=largest_win,
            largest_loss=largest_loss,
            avg_trade_duration_hours=avg_duration_hours,
        )

    def get_trade_history(self) -> list[dict]:
        """Get list of recorded trades.

        Returns:
            List of trade dictionaries.
        """
        return self._trades.copy()

    def get_balance_history(self) -> list[tuple[datetime, float]]:
        """Get balance history.

        Returns:
            List of (timestamp, balance) tuples.
        """
        return self._balance_history.copy()

    def get_total_return(self) -> float:
        """Get total return as a decimal.

        Returns:
            Total return (e.g., 0.15 for 15% return).
        """
        if self.initial_balance <= 0:
            return 0.0
        return (self.current_balance - self.initial_balance) / self.initial_balance

    def get_total_return_pct(self) -> float:
        """Get total return as a percentage.

        Returns:
            Total return percentage (e.g., 15.0 for 15%).
        """
        return self.get_total_return() * 100

    def reset(self, initial_balance: float | None = None) -> None:
        """Reset all tracking.

        Args:
            initial_balance: New initial balance, or use existing.
        """
        if initial_balance is not None:
            self.initial_balance = initial_balance
        self.current_balance = self.initial_balance
        self.peak_balance = self.initial_balance
        self.max_drawdown = 0.0
        self._trades.clear()
        self._total_fees_paid = 0.0
        self._total_slippage_cost = 0.0
        self._total_pnl = 0.0
        self._gross_profit = 0.0
        self._gross_loss = 0.0
        self._winning_trades = 0
        self._losing_trades = 0
        self._total_duration_seconds = 0.0
        self._balance_history = [(datetime.now(), self.initial_balance)]


__all__ = [
    "PerformanceTracker",
    "PerformanceMetrics",
]
