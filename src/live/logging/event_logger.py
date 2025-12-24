"""LiveEventLogger coordinates database and file logging for live trading events.

Centralizes all logging operations to reduce noise in the main trading loop
and make logging behavior easy to configure for different environments.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.database.manager import DatabaseManager
    from src.database.models import TradeSource
    from src.live.execution.position_tracker import LivePosition

logger = logging.getLogger(__name__)


class LiveEventLogger:
    """Coordinates logging for live trading events.

    This class centralizes all logging operations including:
    - Account snapshot logging to database
    - Strategy execution logging
    - Trade logging to file and database
    - Status logging to console
    - Final statistics printing

    Can be easily disabled for tests by setting log_to_database=False.
    """

    def __init__(
        self,
        db_manager: DatabaseManager | None,
        log_to_database: bool = True,
        log_trades_to_file: bool = True,
        session_id: int | None = None,
        trade_log_dir: str = "logs/trades",
    ) -> None:
        """Initialize event logger.

        Args:
            db_manager: Database manager for logging.
            log_to_database: Whether to log to database.
            log_trades_to_file: Whether to log trades to JSON file.
            session_id: Trading session ID.
            trade_log_dir: Directory for trade log files.
        """
        self.db_manager = db_manager
        self.log_to_database = log_to_database
        self.log_trades_to_file = log_trades_to_file
        self.session_id = session_id
        self.trade_log_dir = trade_log_dir

    @property
    def enabled(self) -> bool:
        """Check if database logging is enabled and available."""
        return self.log_to_database and self.db_manager is not None

    def set_session_id(self, session_id: int | None) -> None:
        """Update the session ID.

        Args:
            session_id: New session ID.
        """
        self.session_id = session_id

    def log_account_snapshot(
        self,
        balance: float,
        positions: dict[str, LivePosition],
        total_pnl: float,
        peak_balance: float,
    ) -> None:
        """Log current account state to database.

        Args:
            balance: Current account balance.
            positions: Dictionary of active positions.
            total_pnl: Total realized P&L.
            peak_balance: Peak account balance for drawdown calculation.
        """
        if not self.enabled or self.session_id is None:
            if self.log_to_database and self.session_id is None:
                logger.warning(
                    "Cannot log account snapshot - no trading session ID available"
                )
            return

        try:
            # Calculate total exposure using the active fraction per position
            total_exposure = sum(
                float(pos.current_size if pos.current_size is not None else pos.size)
                * (
                    float(pos.entry_balance)
                    if pos.entry_balance is not None and pos.entry_balance > 0
                    else float(balance)
                )
                for pos in positions.values()
            )

            # Calculate equity (balance + unrealized P&L)
            unrealized_pnl = sum(
                float(pos.unrealized_pnl) for pos in positions.values()
            )
            equity = float(balance) + unrealized_pnl

            # Calculate current drawdown percentage
            current_drawdown = 0.0
            if peak_balance > 0:
                current_drawdown = (peak_balance - balance) / peak_balance * 100

            # Daily P&L placeholder - would require day_start_balance tracking
            daily_pnl = 0.0

            self.db_manager.log_account_snapshot(
                balance=balance,
                equity=equity,
                total_pnl=total_pnl,
                open_positions=len(positions),
                total_exposure=total_exposure,
                drawdown=current_drawdown,
                daily_pnl=daily_pnl,
                session_id=self.session_id,
            )

        except Exception as e:
            logger.error("Failed to log account snapshot: %s", e)

    def log_status(
        self,
        symbol: str,
        current_price: float,
        balance: float,
        positions: dict[str, LivePosition],
        total_trades: int,
        winning_trades: int,
    ) -> None:
        """Log current trading status to console.

        Args:
            symbol: Trading symbol.
            current_price: Current market price.
            balance: Current account balance.
            positions: Dictionary of active positions.
            total_trades: Total number of completed trades.
            winning_trades: Number of winning trades.
        """
        total_unrealized = sum(
            float(pos.unrealized_pnl) for pos in positions.values()
        )
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

        logger.info(
            "Status: %s @ $%.2f | Balance: $%.2f | Positions: %d | "
            "Unrealized: $%.2f | Trades: %d (%.1f%% win)",
            symbol,
            current_price,
            balance,
            len(positions),
            total_unrealized,
            total_trades,
            win_rate,
        )

    def log_trade_to_file(
        self,
        symbol: str,
        side: str,
        size: float,
        entry_price: float,
        exit_price: float,
        entry_time: datetime,
        exit_time: datetime,
        pnl: float,
        pnl_percent: float,
        exit_reason: str,
    ) -> None:
        """Log completed trade to JSON file.

        Args:
            symbol: Trading symbol.
            side: Trade side (long/short).
            size: Position size.
            entry_price: Entry price.
            exit_price: Exit price.
            entry_time: Entry timestamp.
            exit_time: Exit timestamp.
            pnl: Profit/loss amount.
            pnl_percent: Profit/loss percentage.
            exit_reason: Reason for exit.
        """
        if not self.log_trades_to_file:
            return

        try:
            os.makedirs(self.trade_log_dir, exist_ok=True)

            log_file = os.path.join(
                self.trade_log_dir,
                f"trades_{datetime.now().strftime('%Y%m')}.json",
            )
            trade_data = {
                "timestamp": exit_time.isoformat(),
                "symbol": symbol,
                "side": side,
                "size": size,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "pnl": pnl,
                "pnl_percent": pnl_percent,
                "exit_reason": exit_reason,
                "duration_minutes": (exit_time - entry_time).total_seconds() / 60,
            }

            with open(log_file, "a") as f:
                f.write(json.dumps(trade_data) + "\n")

        except Exception as e:
            logger.error("Failed to log trade to file: %s", e, exc_info=True)

    def log_trade_to_database(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        exit_price: float,
        size: float,
        entry_time: datetime,
        exit_time: datetime,
        pnl: float,
        exit_reason: str,
        strategy_name: str,
        source: TradeSource,
        stop_loss: float | None = None,
        take_profit: float | None = None,
        mfe: float | None = None,
        mae: float | None = None,
        mfe_price: float | None = None,
        mae_price: float | None = None,
        mfe_time: datetime | None = None,
        mae_time: datetime | None = None,
    ) -> None:
        """Log completed trade to database.

        Args:
            symbol: Trading symbol.
            side: Trade side.
            entry_price: Entry price.
            exit_price: Exit price.
            size: Position size.
            entry_time: Entry timestamp.
            exit_time: Exit timestamp.
            pnl: Profit/loss amount.
            exit_reason: Reason for exit.
            strategy_name: Name of the strategy.
            source: Trade source (LIVE, PAPER).
            stop_loss: Stop loss price.
            take_profit: Take profit price.
            mfe: Maximum favorable excursion.
            mae: Maximum adverse excursion.
            mfe_price: Price at MFE.
            mae_price: Price at MAE.
            mfe_time: Time of MFE.
            mae_time: Time of MAE.
        """
        if not self.enabled:
            return

        try:
            self.db_manager.log_trade(
                symbol=symbol,
                side=side,
                entry_price=entry_price,
                exit_price=exit_price,
                size=size,
                entry_time=entry_time,
                exit_time=exit_time,
                pnl=pnl,
                exit_reason=exit_reason,
                strategy_name=strategy_name,
                source=source,
                stop_loss=stop_loss,
                take_profit=take_profit,
                session_id=self.session_id,
                mfe=mfe,
                mae=mae,
                mfe_price=mfe_price,
                mae_price=mae_price,
                mfe_time=mfe_time,
                mae_time=mae_time,
            )
        except Exception as e:
            logger.debug("Failed to log trade to database: %s", e)

    def log_entry_decision(
        self,
        strategy_name: str,
        symbol: str,
        current_price: float,
        timeframe: str,
        action_taken: str,
        signal_strength: float = 0.0,
        confidence_score: float = 0.5,
        position_size: float | None = None,
        indicators: dict | None = None,
        sentiment_data: dict | None = None,
        ml_predictions: dict | None = None,
        reasons: list[str] | None = None,
    ) -> None:
        """Log an entry decision to the database.

        Args:
            strategy_name: Name of the strategy.
            symbol: Trading symbol.
            current_price: Current market price.
            timeframe: Candle timeframe.
            action_taken: Action taken (e.g., 'opened_long', 'no_action').
            signal_strength: Strength of the signal (0-1).
            confidence_score: Confidence of prediction (0-1).
            position_size: Position size fraction.
            indicators: Extracted indicator values.
            sentiment_data: Extracted sentiment data.
            ml_predictions: ML prediction values.
            reasons: List of reasons for the decision.
        """
        if not self.enabled:
            return

        try:
            self.db_manager.log_strategy_execution(
                strategy_name=strategy_name,
                symbol=symbol,
                signal_type="entry",
                action_taken=action_taken,
                price=current_price,
                timeframe=timeframe,
                signal_strength=signal_strength,
                confidence_score=confidence_score,
                indicators=indicators,
                sentiment_data=sentiment_data,
                ml_predictions=ml_predictions,
                position_size=position_size,
                reasons=reasons or [],
                volume=indicators.get("volume") if indicators else None,
                volatility=indicators.get("volatility") if indicators else None,
                session_id=self.session_id,
            )
        except Exception as e:
            logger.debug("Failed to log entry decision: %s", e)

    def log_exit_decision(
        self,
        strategy_name: str,
        symbol: str,
        current_price: float,
        timeframe: str,
        action_taken: str,
        signal_strength: float = 0.0,
        confidence_score: float = 0.5,
        position_size: float | None = None,
        indicators: dict | None = None,
        sentiment_data: dict | None = None,
        ml_predictions: dict | None = None,
        reasons: list[str] | None = None,
    ) -> None:
        """Log an exit decision to the database.

        Args:
            strategy_name: Name of the strategy.
            symbol: Trading symbol.
            current_price: Current market price.
            timeframe: Candle timeframe.
            action_taken: Action taken (e.g., 'closed_position', 'hold_position').
            signal_strength: Strength of the signal (0-1).
            confidence_score: Confidence of prediction (0-1).
            position_size: Position size fraction.
            indicators: Extracted indicator values.
            sentiment_data: Extracted sentiment data.
            ml_predictions: ML prediction values.
            reasons: List of reasons for the decision.
        """
        if not self.enabled:
            return

        try:
            self.db_manager.log_strategy_execution(
                strategy_name=strategy_name,
                symbol=symbol,
                signal_type="exit",
                action_taken=action_taken,
                price=current_price,
                timeframe=timeframe,
                signal_strength=signal_strength,
                confidence_score=confidence_score,
                indicators=indicators,
                sentiment_data=sentiment_data,
                ml_predictions=ml_predictions,
                position_size=position_size,
                reasons=reasons or [],
                volume=indicators.get("volume") if indicators else None,
                volatility=indicators.get("volatility") if indicators else None,
                session_id=self.session_id,
            )
        except Exception as e:
            logger.debug("Failed to log exit decision: %s", e)

    def create_trading_session(
        self,
        strategy_name: str,
        symbol: str,
        timeframe: str,
        source: TradeSource,
        initial_balance: float,
        strategy_config: dict | None = None,
    ) -> int | None:
        """Create a new trading session.

        Args:
            strategy_name: Name of the strategy.
            symbol: Trading symbol.
            timeframe: Candle timeframe.
            source: Trade source (LIVE, PAPER).
            initial_balance: Initial account balance.
            strategy_config: Strategy configuration.

        Returns:
            Session ID, or None if creation failed.
        """
        if not self.enabled:
            return None

        try:
            date_str = datetime.now().strftime("%Y%m%d_%H%M")
            session_name = f"Live_{symbol}_{date_str}"
            session_id = self.db_manager.create_trading_session(
                strategy_name=strategy_name,
                symbol=symbol,
                timeframe=timeframe,
                mode=source,
                initial_balance=initial_balance,
                strategy_config=strategy_config or {},
                session_name=session_name,
            )
            self.session_id = session_id
            return session_id
        except Exception as e:
            logger.warning("Failed to create trading session: %s", e)
            return None

    def end_trading_session(self, final_balance: float) -> None:
        """End the current trading session.

        Args:
            final_balance: Final account balance.
        """
        if not self.enabled or self.session_id is None:
            return

        try:
            self.db_manager.end_trading_session(
                session_id=self.session_id,
                final_balance=final_balance,
            )
        except Exception as e:
            logger.debug("Failed to end trading session: %s", e)

    def print_final_stats(
        self,
        initial_balance: float,
        current_balance: float,
        total_pnl: float,
        max_drawdown: float,
        total_trades: int,
        winning_trades: int,
        positions: dict[str, LivePosition],
        completed_trades: list[Any] | None = None,
    ) -> None:
        """Print final trading statistics.

        Args:
            initial_balance: Starting balance.
            current_balance: Current balance.
            total_pnl: Total realized P&L.
            max_drawdown: Maximum drawdown (as decimal, e.g., 0.05 for 5%).
            total_trades: Total number of completed trades.
            winning_trades: Number of winning trades.
            positions: Dictionary of active positions.
            completed_trades: List of completed trade objects.
        """
        total_return = (
            (current_balance - initial_balance) / initial_balance * 100
            if initial_balance > 0
            else 0
        )
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

        print("\n" + "=" * 60)
        print("FINAL TRADING STATISTICS")
        print("=" * 60)
        print(f"Initial Balance: ${initial_balance:,.2f}")
        print(f"Final Balance: ${current_balance:,.2f}")
        print(f"Total Return: {total_return:+.2f}%")
        print(f"Total PnL: ${total_pnl:+,.2f}")
        print(f"Max Drawdown: {max_drawdown * 100:.2f}%")
        print(f"Total Trades: {total_trades}")
        print(f"Winning Trades: {winning_trades}")
        print(f"Win Rate: {win_rate:.1f}%")
        print(f"Active Positions: {len(positions)}")

        if completed_trades:
            avg_trade = sum(trade.pnl for trade in completed_trades) / len(
                completed_trades
            )
            print(f"Average Trade: ${avg_trade:.2f}")

        print("=" * 60)

    def get_stats_summary(
        self,
        initial_balance: float,
        current_balance: float,
        total_pnl: float,
        peak_balance: float,
        max_drawdown: float,
        total_trades: int,
        winning_trades: int,
        positions: dict[str, LivePosition],
        last_data_update: datetime | None,
        is_running: bool,
    ) -> dict[str, Any]:
        """Get current performance summary as dictionary.

        Args:
            initial_balance: Starting balance.
            current_balance: Current balance.
            total_pnl: Total realized P&L.
            peak_balance: Peak balance.
            max_drawdown: Maximum drawdown (as decimal).
            total_trades: Total number of completed trades.
            winning_trades: Number of winning trades.
            positions: Dictionary of active positions.
            last_data_update: Timestamp of last data update.
            is_running: Whether engine is running.

        Returns:
            Dictionary with performance metrics.
        """
        total_return = (
            (current_balance - initial_balance) / initial_balance * 100
            if initial_balance > 0
            else 0
        )
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        current_drawdown = (
            (peak_balance - current_balance) / peak_balance * 100
            if peak_balance > 0
            else 0
        )

        return {
            "initial_balance": initial_balance,
            "current_balance": current_balance,
            "total_return": total_return,
            "total_return_pct": total_return,
            "total_pnl": total_pnl,
            "current_drawdown": current_drawdown,
            "max_drawdown_pct": max_drawdown * 100,
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "win_rate": win_rate,
            "win_rate_pct": win_rate,
            "active_positions": len(positions),
            "last_update": last_data_update.isoformat() if last_data_update else None,
            "is_running": is_running,
        }
