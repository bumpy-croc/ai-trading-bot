"""Unified performance tracking for trading engines.

This module provides consistent performance metrics tracking for both
backtesting and live trading engines.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, Protocol

import pandas as pd

from src.performance import metrics as perf_metrics

if TYPE_CHECKING:
    from src.engines.shared.models import BaseTrade


class TradeProtocol(Protocol):
    """Protocol for trade objects that can be recorded.

    This protocol defines the minimum interface required for trade tracking.
    """

    pnl: float | None
    entry_time: datetime | None
    exit_time: datetime | None
    symbol: str | None
    side: str | None

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics container.

    Attributes:
        # Trade statistics
        total_trades: Total number of completed trades.
        winning_trades: Number of profitable trades.
        losing_trades: Number of unprofitable trades.
        win_rate: Percentage of winning trades (decimal).

        # Returns
        total_pnl: Total realized profit/loss.
        total_return_pct: Total return as percentage.
        annualized_return: Annualized return (CAGR) as percentage.

        # Risk metrics
        max_drawdown: Maximum drawdown percentage.
        current_drawdown: Current drawdown percentage.
        sharpe_ratio: Annualized Sharpe ratio.
        sortino_ratio: Annualized Sortino ratio.
        calmar_ratio: Calmar ratio (return/max drawdown).
        var_95: Value at Risk at 95% confidence (decimal).

        # Trade quality
        profit_factor: Ratio of gross profit to gross loss.
        expectancy: Expected value per trade.
        avg_win: Average winning trade PnL.
        avg_loss: Average losing trade PnL (negative).
        largest_win: Largest winning trade PnL.
        largest_loss: Largest losing trade PnL (negative).

        # Efficiency
        avg_trade_duration_hours: Average trade duration in hours.
        consecutive_wins: Current consecutive winning trades.
        consecutive_losses: Current consecutive losing trades.

        # Costs
        total_fees_paid: Total fees paid.
        total_slippage_cost: Total slippage cost.

        # Balance tracking
        initial_balance: Starting balance.
        current_balance: Current balance.
        peak_balance: Highest balance achieved.
    """

    # Trade statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0

    # Returns
    total_pnl: float = 0.0
    total_return_pct: float = 0.0
    annualized_return: float = 0.0

    # Risk metrics
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    var_95: float = 0.0

    # Trade quality
    profit_factor: float = 0.0
    expectancy: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0

    # Efficiency
    avg_trade_duration_hours: float = 0.0
    consecutive_wins: int = 0
    consecutive_losses: int = 0

    # Costs
    total_fees_paid: float = 0.0
    total_slippage_cost: float = 0.0

    # Balance tracking
    initial_balance: float = 0.0
    current_balance: float = 0.0
    peak_balance: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            # Trade statistics
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": self.win_rate,
            # Returns
            "total_pnl": self.total_pnl,
            "total_return_pct": self.total_return_pct,
            "annualized_return": self.annualized_return,
            # Risk metrics
            "max_drawdown": self.max_drawdown,
            "current_drawdown": self.current_drawdown,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "calmar_ratio": self.calmar_ratio,
            "var_95": self.var_95,
            # Trade quality
            "profit_factor": self.profit_factor,
            "expectancy": self.expectancy,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "largest_win": self.largest_win,
            "largest_loss": self.largest_loss,
            # Efficiency
            "avg_trade_duration_hours": self.avg_trade_duration_hours,
            "consecutive_wins": self.consecutive_wins,
            "consecutive_losses": self.consecutive_losses,
            # Costs
            "total_fees_paid": self.total_fees_paid,
            "total_slippage_cost": self.total_slippage_cost,
            # Balance tracking
            "initial_balance": self.initial_balance,
            "current_balance": self.current_balance,
            "peak_balance": self.peak_balance,
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
        self._zero_pnl_trades = 0
        self._total_duration_seconds = 0.0
        self._max_trade_history = 10000  # Limit memory usage

        # Streak tracking
        self._current_streak_type: str | None = None  # 'win' or 'loss'
        self._current_win_streak = 0
        self._current_loss_streak = 0
        self._max_win_streak = 0
        self._max_loss_streak = 0

        # Balance history for drawdown calculation
        self._balance_history: list[tuple[datetime, float]] = [
            (datetime.now(), initial_balance)
        ]

        # Thread safety lock for mutable state (reentrant to allow nested calls)
        self._lock = threading.RLock()

    def record_trade(
        self,
        trade: TradeProtocol,
        fee: float = 0.0,
        slippage: float = 0.0,
    ) -> None:
        """Record a completed trade.

        Args:
            trade: Completed trade object.
            fee: Total fee for the trade.
            slippage: Total slippage cost for the trade.
        """
        # Extract trade attributes with explicit None handling
        pnl_attr = getattr(trade, "pnl", None)
        pnl = float(pnl_attr) if pnl_attr is not None else 0.0

        if pnl == 0.0 and pnl_attr is None:
            logger.warning(f"Trade {getattr(trade, 'symbol', 'unknown')} has None PnL")

        entry_time = getattr(trade, "entry_time", None)
        exit_time = getattr(trade, "exit_time", None)

        with self._lock:
            # Update trade counts (explicitly handle zero-PnL case)
            if pnl > 0:
                self._winning_trades += 1
                self._gross_profit += pnl
                # Update win streak
                if self._current_streak_type == "win":
                    self._current_win_streak += 1
                else:
                    self._current_streak_type = "win"
                    self._current_win_streak = 1
                    self._current_loss_streak = 0
                self._max_win_streak = max(self._max_win_streak, self._current_win_streak)
            elif pnl < 0:
                self._losing_trades += 1
                self._gross_loss += abs(pnl)
                # Update loss streak
                if self._current_streak_type == "loss":
                    self._current_loss_streak += 1
                else:
                    self._current_streak_type = "loss"
                    self._current_loss_streak = 1
                    self._current_win_streak = 0
                self._max_loss_streak = max(self._max_loss_streak, self._current_loss_streak)
            else:
                # Zero PnL trade (breakeven) - count but don't update P&L or streaks
                self._zero_pnl_trades += 1
                logger.debug(f"Recorded zero-PnL trade for {getattr(trade, 'symbol', 'unknown')}")

            # Update totals
            self._total_pnl += pnl
            self._total_fees_paid += fee
            self._total_slippage_cost += slippage

            # Calculate duration
            if entry_time and exit_time:
                duration = (exit_time - entry_time).total_seconds()
                self._total_duration_seconds += duration

            # Store trade record
            self._trades.append(
                {
                    "pnl": pnl,
                    "fee": fee,
                    "slippage": slippage,
                    "entry_time": entry_time,
                    "exit_time": exit_time,
                    "symbol": getattr(trade, "symbol", None),
                    "side": str(getattr(trade, "side", None)),
                }
            )

            # Limit memory usage by keeping only most recent trades
            if len(self._trades) > self._max_trade_history:
                self._trades = self._trades[-self._max_trade_history:]

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
        with self._lock:
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

        Calculates comprehensive metrics using pure functions from
        src.performance.metrics module.

        Returns:
            PerformanceMetrics with calculated values.
        """
        with self._lock:
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

            # Calculate expectancy
            expectancy_val = perf_metrics.expectancy(win_rate, avg_win, avg_loss)

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

            # Calculate total return percentage
            total_return_pct = perf_metrics.total_return(
                self.initial_balance, self.current_balance
            )

            # Calculate annualized return (CAGR)
            annualized_return = 0.0
            if len(self._balance_history) >= 2:
                start_time = self._balance_history[0][0]
                end_time = self._balance_history[-1][0]
                days = (end_time - start_time).days or 1
                annualized_return = perf_metrics.cagr(
                    self.initial_balance, self.current_balance, days
                )

            # Calculate current drawdown
            current_drawdown = 0.0
            if self.peak_balance > 0:
                current_drawdown = (
                    (self.peak_balance - self.current_balance) / self.peak_balance
                )

            # Get balance series for advanced metrics
            balance_series = self.get_balance_series()

            # Calculate Sharpe ratio
            sharpe_ratio = 0.0
            if len(balance_series) >= 2:
                sharpe_ratio = perf_metrics.sharpe(balance_series)

            # Calculate Sortino ratio
            sortino_ratio = 0.0
            if len(balance_series) >= 2:
                sortino_ratio = perf_metrics.sortino_ratio(balance_series)

            # Calculate Calmar ratio
            max_dd_pct = self.max_drawdown * 100.0
            calmar_ratio = perf_metrics.calmar_ratio(annualized_return, max_dd_pct)

            # Calculate VaR (95%) - requires minimum sample size for statistical validity
            var_95 = 0.0
            if len(balance_series) >= 30:
                returns = balance_series.pct_change().dropna()
                if len(returns) >= 20:
                    var_95 = perf_metrics.value_at_risk(returns, confidence=0.95)

            return PerformanceMetrics(
                # Trade statistics
                total_trades=total_trades,
                winning_trades=self._winning_trades,
                losing_trades=self._losing_trades,
                win_rate=win_rate,
                # Returns
                total_pnl=self._total_pnl,
                total_return_pct=total_return_pct,
                annualized_return=annualized_return,
                # Risk metrics
                max_drawdown=self.max_drawdown,
                current_drawdown=current_drawdown,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                calmar_ratio=calmar_ratio,
                var_95=var_95,
                # Trade quality
                profit_factor=profit_factor,
                expectancy=expectancy_val,
                avg_win=avg_win,
                avg_loss=avg_loss,
                largest_win=largest_win,
                largest_loss=largest_loss,
                # Efficiency
                avg_trade_duration_hours=avg_duration_hours,
                consecutive_wins=self._current_win_streak,
                consecutive_losses=self._current_loss_streak,
                # Costs
                total_fees_paid=self._total_fees_paid,
                total_slippage_cost=self._total_slippage_cost,
                # Balance tracking
                initial_balance=self.initial_balance,
                current_balance=self.current_balance,
                peak_balance=self.peak_balance,
            )

    def get_trade_history(self) -> list[dict]:
        """Get list of recorded trades.

        Returns:
            List of trade dictionaries.
        """
        with self._lock:
            return self._trades.copy()

    def get_balance_history(self) -> list[tuple[datetime, float]]:
        """Get balance history as list of tuples.

        Returns:
            List of (timestamp, balance) tuples.
        """
        with self._lock:
            return self._balance_history.copy()

    def get_balance_series(self) -> pd.Series:
        """Get balance history as pandas Series for metric calculations.

        Returns:
            pandas Series with datetime index and balance values.

        Note:
            Balance history is resampled to daily frequency for Sharpe/Sortino
            calculations. This uses end-of-day values only (.last()) with forward
            fill for missing days. Intraday volatility is not captured in the
            risk-adjusted metrics, which is acceptable for most trading strategies
            that hold positions for days or longer.
        """
        with self._lock:
            if not self._balance_history:
                return pd.Series(dtype=float)

            timestamps = [ts for ts, _ in self._balance_history]
            balances = [bal for _, bal in self._balance_history]

            series = pd.Series(balances, index=pd.DatetimeIndex(timestamps))

            # Resample to daily frequency for Sharpe/Sortino calculations
            # Note: Uses end-of-day values only. Intraday volatility is not captured.
            if len(series) > 1:
                series = series.resample("1D").last().ffill()

            return series

    def get_total_return(self) -> float:
        """Get total return as a decimal.

        Returns:
            Total return (e.g., 0.15 for 15% return).
        """
        with self._lock:
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
        with self._lock:
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
            self._zero_pnl_trades = 0
            self._total_duration_seconds = 0.0
            self._current_streak_type = None
            self._current_win_streak = 0
            self._current_loss_streak = 0
            self._max_win_streak = 0
            self._max_loss_streak = 0
            self._balance_history = [(datetime.now(), self.initial_balance)]


__all__ = [
    "PerformanceTracker",
    "PerformanceMetrics",
]
