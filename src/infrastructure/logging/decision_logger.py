from __future__ import annotations


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
    indicators: dict | None = None,
    sentiment_data: dict | None = None,
    ml_predictions: dict | None = None,
    position_size: float | None = None,
    reasons: list | None = None,
    volume: float | None = None,
    volatility: float | None = None,
    session_id: int | None = None,
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
