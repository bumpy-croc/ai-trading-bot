"""Account-level monitoring for the live trading engine.

Balance/equity snapshots, inline status logging, performance-metric updates,
and the end-of-session summary. Extracted from ``LiveTradingEngine`` so
observability glue lives outside the orchestration loop (#486).

Thread-safety / lock ownership: the monitor holds no locks and owns no
mutable state. It reads balances/session ids off the engine at call time and
position state through ``LivePositionTracker``'s thread-safe ``positions``
snapshot property.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Protocol

from src.engines.live.execution.position_tracker import LivePositionTracker

if TYPE_CHECKING:
    from src.database.manager import DatabaseManager
    from src.engines.shared.models import BaseTrade
    from src.performance.tracker import PerformanceTracker

logger = logging.getLogger(__name__)


class MonitoringEngineState(Protocol):
    """Live engine state the monitor reads at call time.

    Attributes are read dynamically (not captured at construction) because
    balance, session id, and run state mutate throughout the engine lifecycle.
    """

    initial_balance: float
    current_balance: float
    trading_session_id: int | None
    completed_trades: list[BaseTrade]
    last_data_update: datetime | None
    is_running: bool
    live_position_tracker: LivePositionTracker
    performance_tracker: PerformanceTracker
    db_manager: DatabaseManager


class LiveAccountMonitor:
    """Snapshots, status lines, and performance summaries for a live session."""

    def __init__(self, engine_state: MonitoringEngineState) -> None:
        """Bind to the engine's live state (see protocol for what is read)."""
        self._state = engine_state

    def update_performance_metrics(self) -> None:
        """Update performance tracking metrics"""
        # Update performance tracker on every metric update cycle
        # Note: Less frequent than backtest (every candle vs every update cycle)
        # This trade-off reduces overhead while maintaining statistical validity for risk metrics
        state = self._state
        state.performance_tracker.update_balance(state.current_balance, timestamp=datetime.now(UTC))

    def log_account_snapshot(self) -> None:
        """Log current account state to database"""
        state = self._state
        try:
            # Calculate total exposure using the active fraction per position
            positions_snapshot = state.live_position_tracker.positions
            total_exposure = sum(
                float(pos.current_size if pos.current_size is not None else pos.size)
                * (
                    float(pos.entry_balance)
                    if pos.entry_balance is not None and pos.entry_balance > 0
                    else float(state.current_balance)
                )
                for pos in positions_snapshot.values()
            )

            # Calculate equity (balance + unrealized P&L)
            unrealized_pnl = sum(float(pos.unrealized_pnl) for pos in positions_snapshot.values())
            equity = float(state.current_balance) + unrealized_pnl

            # Calculate current drawdown percentage
            current_drawdown: float = 0
            perf_metrics = state.performance_tracker.get_metrics()
            if perf_metrics.peak_balance > 0:
                current_drawdown = (
                    (perf_metrics.peak_balance - state.current_balance)
                    / perf_metrics.peak_balance
                    * 100
                )

            # TODO: Calculate daily P&L (requires tracking of day start balance)
            daily_pnl = 0  # Placeholder

            # Log snapshot to database
            if state.trading_session_id is not None:
                state.db_manager.log_account_snapshot(
                    balance=state.current_balance,
                    equity=equity,
                    total_pnl=perf_metrics.total_pnl,
                    open_positions=state.live_position_tracker.position_count,
                    total_exposure=total_exposure,
                    drawdown=current_drawdown,
                    daily_pnl=daily_pnl,
                    session_id=state.trading_session_id,
                )
            else:
                logger.warning(
                    "⚠️ Cannot log account snapshot to database - no trading session ID available"
                )

        except Exception as e:
            logger.error("Failed to log account snapshot: %s", e)

    def log_status(self, symbol: str, current_price: float) -> None:
        """Log current trading status"""
        state = self._state
        total_unrealized = sum(
            float(pos.unrealized_pnl) for pos in state.live_position_tracker.positions.values()
        )
        perf_metrics = state.performance_tracker.get_metrics()
        win_rate = perf_metrics.win_rate * 100

        logger.info(
            f"📊 Status: {symbol} @ ${current_price:.2f} | "
            f"Balance: ${state.current_balance:.2f} | "
            f"Positions: {state.live_position_tracker.position_count} | "
            f"Unrealized: ${total_unrealized:.2f} | "
            f"Trades: {perf_metrics.total_trades} ({win_rate:.1f}% win)"
        )

    def print_final_stats(self) -> None:
        """Print final trading statistics"""
        state = self._state
        # Validate initial_balance before division to prevent crashes
        if state.initial_balance <= 0:
            logger.error(
                "Cannot calculate total return - invalid initial_balance: %.8f. "
                "Skipping final statistics.",
                state.initial_balance,
            )
            return

        total_return = (
            (state.current_balance - state.initial_balance) / state.initial_balance
        ) * 100
        perf_metrics = state.performance_tracker.get_metrics()
        win_rate = perf_metrics.win_rate * 100

        print("\n" + "=" * 60)
        print("🏁 FINAL TRADING STATISTICS")
        print("=" * 60)
        print(f"Initial Balance: ${state.initial_balance:,.2f}")
        print(f"Final Balance: ${state.current_balance:,.2f}")
        print(f"Total Return: {total_return:+.2f}%")
        print(f"Total PnL: ${perf_metrics.total_pnl:+,.2f}")
        print(f"Max Drawdown: {perf_metrics.max_drawdown * 100:.2f}%")
        print(f"Total Trades: {perf_metrics.total_trades}")
        print(f"Winning Trades: {perf_metrics.winning_trades}")
        print(f"Win Rate: {win_rate:.1f}%")
        print(f"Active Positions: {state.live_position_tracker.position_count}")

        if state.completed_trades:
            avg_trade = sum(trade.pnl for trade in state.completed_trades) / len(
                state.completed_trades
            )
            print(f"Average Trade: ${avg_trade:.2f}")

        print("=" * 60)

    def performance_summary(self) -> dict[str, Any]:
        """Get current performance summary"""
        state = self._state
        # Get comprehensive metrics from performance tracker
        perf_metrics = state.performance_tracker.get_metrics()

        # Convert to percentages for backward compatibility
        win_rate = perf_metrics.win_rate * 100
        current_drawdown = perf_metrics.current_drawdown * 100
        max_drawdown_pct = perf_metrics.max_drawdown * 100

        return {
            # Core metrics from tracker
            "initial_balance": state.initial_balance,
            "current_balance": state.current_balance,
            "total_return": perf_metrics.total_return_pct,
            "total_return_pct": perf_metrics.total_return_pct,
            "total_pnl": perf_metrics.total_pnl,
            "current_drawdown": current_drawdown,
            "max_drawdown_pct": max_drawdown_pct,
            "total_trades": perf_metrics.total_trades,
            "winning_trades": perf_metrics.winning_trades,
            "win_rate": win_rate,
            "win_rate_pct": win_rate,
            # New metrics from tracker
            "sharpe_ratio": perf_metrics.sharpe_ratio,
            "sortino_ratio": perf_metrics.sortino_ratio,
            "calmar_ratio": perf_metrics.calmar_ratio,
            "var_95": perf_metrics.var_95,
            "expectancy": perf_metrics.expectancy,
            "profit_factor": perf_metrics.profit_factor,
            "avg_win": perf_metrics.avg_win,
            "avg_loss": perf_metrics.avg_loss,
            "largest_win": perf_metrics.largest_win,
            "largest_loss": perf_metrics.largest_loss,
            "avg_trade_duration_hours": perf_metrics.avg_trade_duration_hours,
            "consecutive_wins": perf_metrics.consecutive_wins,
            "consecutive_losses": perf_metrics.consecutive_losses,
            "total_fees_paid": perf_metrics.total_fees_paid,
            "total_slippage_cost": perf_metrics.total_slippage_cost,
            # Live-specific metrics
            "active_positions": state.live_position_tracker.position_count,
            "last_update": (state.last_data_update.isoformat() if state.last_data_update else None),
            "is_running": state.is_running,
        }
