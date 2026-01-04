from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.database import DatabaseManager

logger = logging.getLogger(__name__)


def log_strategy_execution(
    db_manager: DatabaseManager | None,
    *,
    strategy_name: str,
    symbol: str,
    signal_type: str,
    action_taken: str,
    price: float,
    timeframe: str,
    signal_strength: float,
    confidence_score: float,
    indicators: dict[str, Any] | None = None,
    sentiment_data: dict[str, Any] | None = None,
    ml_predictions: dict[str, Any] | None = None,
    position_size: float | None = None,
    reasons: list[str] | None = None,
    volume: float | None = None,
    volatility: float | None = None,
    session_id: int | None = None,
) -> None:
    """Log a strategy execution event to the database.

    Performs best-effort logging - failures are logged at DEBUG level
    to prevent logging issues from affecting trading operations.

    Args:
        db_manager: Database manager instance (None to skip logging).
        strategy_name: Name of the executing strategy.
        symbol: Trading symbol (e.g., "BTCUSDT").
        signal_type: Type of signal generated.
        action_taken: Action taken based on the signal.
        price: Current price at execution time.
        timeframe: Trading timeframe (e.g., "1h", "4h").
        signal_strength: Strength of the signal (0.0 to 1.0).
        confidence_score: Confidence in the prediction (0.0 to 1.0).
        indicators: Technical indicator values at execution time.
        sentiment_data: Sentiment analysis data.
        ml_predictions: ML model prediction details.
        position_size: Size of position to take.
        reasons: List of reasons for the decision.
        volume: Trading volume.
        volatility: Market volatility measure.
        session_id: Trading session identifier.
    """
    if db_manager is None:
        return
    try:
        db_manager.log_strategy_execution(
            strategy_name=strategy_name,
            symbol=symbol,
            signal_type=signal_type,
            action_taken=action_taken,
            price=price,
            timeframe=timeframe,
            signal_strength=signal_strength,
            confidence_score=confidence_score,
            indicators=indicators,
            sentiment_data=sentiment_data if sentiment_data else None,
            ml_predictions=ml_predictions if ml_predictions else None,
            position_size=position_size,
            reasons=reasons or [],
            volume=volume,
            volatility=volatility,
            session_id=session_id,
        )
    except Exception as e:
        # Best-effort logging - log at DEBUG to aid troubleshooting
        # without failing the trade operation
        logger.debug("Failed to log strategy execution: %s", e, exc_info=True)
        return
