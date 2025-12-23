"""EventLogger coordinates database logging for backtest events.

Centralizes all database logging operations to reduce noise in the main
backtest loop and make logging behavior easy to disable for tests.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.backtesting.models import Trade
    from src.database.manager import DatabaseManager
    from src.database.models import TradeSource

logger = logging.getLogger(__name__)


class EventLogger:
    """Coordinates database logging for backtest events.

    This class centralizes all database logging operations including:
    - Strategy execution logging
    - Trade logging
    - Risk adjustment logging
    - Session management

    Can be easily disabled for tests by setting log_to_database=False.
    """

    def __init__(
        self,
        db_manager: DatabaseManager | None,
        log_to_database: bool,
        session_id: int | None = None,
    ) -> None:
        """Initialize event logger.

        Args:
            db_manager: Database manager for logging.
            log_to_database: Whether to log to database.
            session_id: Trading session ID.
        """
        self.db_manager = db_manager
        self.log_to_database = log_to_database
        self.session_id = session_id

    @property
    def enabled(self) -> bool:
        """Check if logging is enabled and available."""
        return self.log_to_database and self.db_manager is not None

    def set_session_id(self, session_id: int | None) -> None:
        """Update the session ID.

        Args:
            session_id: New session ID.
        """
        self.session_id = session_id

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

    def log_completed_trade(
        self,
        trade: Trade,
        symbol: str,
        strategy_name: str,
        source: TradeSource,
    ) -> None:
        """Log a completed trade to the database.

        Args:
            trade: Completed trade record.
            symbol: Trading symbol.
            strategy_name: Name of the strategy.
            source: Trade source (BACKTEST).
        """
        if not self.enabled:
            return

        try:
            self.db_manager.log_trade(
                symbol=symbol,
                side=trade.side,
                entry_price=trade.entry_price,
                exit_price=trade.exit_price,
                size=trade.size,
                entry_time=trade.entry_time,
                exit_time=trade.exit_time,
                pnl=trade.pnl,
                exit_reason=trade.exit_reason,
                strategy_name=strategy_name,
                source=source,
                stop_loss=trade.stop_loss,
                take_profit=trade.take_profit,
                session_id=self.session_id,
                mfe=trade.mfe,
                mae=trade.mae,
                mfe_price=trade.mfe_price,
                mae_price=trade.mae_price,
                mfe_time=trade.mfe_time,
                mae_time=trade.mae_time,
            )
        except Exception as e:
            logger.debug("Failed to log completed trade: %s", e)

    def log_risk_adjustment(
        self,
        strategy_name: str,
        symbol: str,
        adjustment_type: str,
        current_price: float,
        timeframe: str,
        details: list[str],
    ) -> None:
        """Log a risk adjustment event.

        Args:
            strategy_name: Name of the strategy.
            symbol: Trading symbol.
            adjustment_type: Type of adjustment (e.g., 'trailing_stop_update').
            current_price: Current market price.
            timeframe: Candle timeframe.
            details: List of adjustment details.
        """
        if not self.enabled:
            return

        try:
            self.db_manager.log_strategy_execution(
                strategy_name=strategy_name,
                symbol=symbol,
                signal_type="risk_adjustment",
                action_taken=adjustment_type,
                price=current_price,
                timeframe=timeframe,
                reasons=details,
                session_id=self.session_id,
            )
        except Exception as e:
            logger.debug("Failed to log risk adjustment: %s", e)

    def create_trading_session(
        self,
        strategy_name: str,
        symbol: str,
        timeframe: str,
        source: TradeSource,
        initial_balance: float,
        strategy_config: dict | None = None,
        start_time: datetime | None = None,
    ) -> int | None:
        """Create a new trading session.

        Args:
            strategy_name: Name of the strategy.
            symbol: Trading symbol.
            timeframe: Candle timeframe.
            source: Trade source (BACKTEST).
            initial_balance: Initial account balance.
            strategy_config: Strategy configuration.
            start_time: Session start time.

        Returns:
            Session ID, or None if creation failed.
        """
        if not self.enabled:
            return None

        try:
            date_str = start_time.strftime("%Y%m%d") if start_time else "unknown"
            session_name = f"Backtest_{symbol}_{date_str}"
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

    def should_log_candle(self, candle_index: int, log_frequency: int = 10) -> bool:
        """Determine if this candle should be logged.

        Limits logging frequency to avoid spam for no-action candles.

        Args:
            candle_index: Current candle index.
            log_frequency: Log every N candles.

        Returns:
            True if this candle should be logged.
        """
        return candle_index % log_frequency == 0
