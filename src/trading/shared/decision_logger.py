from __future__ import annotations

from typing import Dict, Optional


def log_strategy_execution(
    db_manager,
    *,
    strategy_name: str,
    symbol: str,
    signal_type: str,
    action_taken: str,
    price: float,
    timeframe: str,
    signal_strength: float,
    confidence_score: float,
    indicators: Optional[Dict] = None,
    sentiment_data: Optional[Dict] = None,
    ml_predictions: Optional[Dict] = None,
    position_size: Optional[float] = None,
    reasons: Optional[list] = None,
    volume: Optional[float] = None,
    volatility: Optional[float] = None,
    session_id: Optional[int] = None,
) -> None:
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
    except Exception:
        # best-effort
        return