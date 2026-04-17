"""
ML Sentiment Strategy - Component-Based Implementation

This strategy uses machine learning models trained with both price data and sentiment analysis.
It leverages the Fear & Greed Index to enhance prediction accuracy and trading decisions.

Key Features:
- Price + sentiment predictions using LSTM neural network
- 120-day sequence length for pattern recognition
- Fear & Greed Index sentiment integration
- Adaptive position sizing based on sentiment confidence
- 2% stop loss, 4% take profit risk management
- Robust fallback when sentiment data is unavailable
- Component-based architecture for better maintainability

Ideal for:
- Enhanced prediction accuracy with market sentiment
- Bull/bear market regime detection
- Trading during high-volatility periods
- Environments with reliable sentiment data access
"""

from __future__ import annotations

from src.config.constants import (
    DEFAULT_BASE_RISK_PER_TRADE,
    DEFAULT_MAX_POSITION_SIZE,
    DEFAULT_STRATEGY_BASE_FRACTION,
    DEFAULT_STRATEGY_MIN_CONFIDENCE,
    DEFAULT_TAKE_PROFIT_PCT,
)
from src.risk.risk_manager import RiskManager as EngineRiskManager
from src.risk.risk_manager import RiskParameters
from src.strategies.components import (
    ConfidenceWeightedSizer,
    CoreRiskAdapter,
    EnhancedRegimeDetector,
    MLSignalGenerator,
    Strategy,
)


def create_ml_sentiment_strategy(
    name: str = "MlSentiment",
    sequence_length: int = 120,
    model_name: str | None = None,
    model_type: str | None = None,
    timeframe: str | None = None,
    *,
    long_entry_threshold: float | None = None,
    short_entry_threshold: float | None = None,
    confidence_multiplier: float | None = None,
    base_fraction: float | None = None,
    min_confidence: float | None = None,
    min_confidence_floor: float | None = None,
    stop_loss_pct: float | None = None,
    take_profit_pct: float | None = None,
) -> Strategy:
    """
    Create ML Sentiment strategy using component composition.

    This strategy uses sentiment-aware signal generation to enhance
    prediction accuracy with market sentiment data.

    Args:
        name: Strategy name
        sequence_length: Number of candles for sequence prediction
        model_name: Model name for prediction engine
        model_type: Model type (e.g., "sentiment")
        timeframe: Model timeframe (e.g., "1h")
        long_entry_threshold: Minimum predicted return for long entry.
        short_entry_threshold: Maximum predicted return for short entry.
        confidence_multiplier: Scales |predicted_return| → confidence.
        base_fraction: Base fraction of balance for ConfidenceWeightedSizer.
        min_confidence: Minimum signal confidence before any position is opened.
        min_confidence_floor: Lower bound on the confidence factor once the
            min_confidence gate has passed (0.0 disables).
        stop_loss_pct: Override for the stop-loss percentage.
        take_profit_pct: Override for the take-profit percentage.

    Returns:
        Configured Strategy instance
    """
    # Create signal generator with ML Sentiment parameters (sentiment model)
    # Note: model_type and timeframe are passed but MLSignalGenerator currently
    # uses only model_name for registry selection. These parameters are retained
    # for future sentiment-specific model selection.
    signal_generator = MLSignalGenerator(
        name=f"{name}_signals",
        sequence_length=sequence_length,
        model_name=model_name,
        long_entry_threshold=long_entry_threshold,
        short_entry_threshold=short_entry_threshold,
        confidence_multiplier=confidence_multiplier,
    )
    # Store model_type/timeframe for potential future use in sentiment model selection
    if model_type:
        signal_generator.model_type = model_type
    if timeframe:
        signal_generator.model_timeframe = timeframe

    resolved_stop_loss = stop_loss_pct if stop_loss_pct is not None else DEFAULT_BASE_RISK_PER_TRADE
    resolved_take_profit = (
        take_profit_pct if take_profit_pct is not None else DEFAULT_TAKE_PROFIT_PCT
    )
    risk_parameters = RiskParameters(
        base_risk_per_trade=DEFAULT_BASE_RISK_PER_TRADE,
        default_take_profit_pct=resolved_take_profit,
        max_position_size=DEFAULT_MAX_POSITION_SIZE,
    )
    core_risk_manager = EngineRiskManager(risk_parameters)
    risk_overrides = {
        "position_sizer": "fixed_fraction",
        "base_fraction": DEFAULT_BASE_RISK_PER_TRADE,
        "max_fraction": DEFAULT_MAX_POSITION_SIZE,
        "stop_loss_pct": resolved_stop_loss,
        "take_profit_pct": resolved_take_profit,
    }
    risk_manager = CoreRiskAdapter(core_risk_manager)
    risk_manager.set_strategy_overrides(risk_overrides)

    # Create position sizer with confidence weighting
    position_sizer = ConfidenceWeightedSizer(
        base_fraction=(
            base_fraction if base_fraction is not None else DEFAULT_STRATEGY_BASE_FRACTION
        ),
        min_confidence=(
            min_confidence if min_confidence is not None else DEFAULT_STRATEGY_MIN_CONFIDENCE
        ),
        min_confidence_floor=(min_confidence_floor if min_confidence_floor is not None else 0.0),
    )

    # Create regime detector
    regime_detector = EnhancedRegimeDetector()

    strategy = Strategy(
        name=name,
        signal_generator=signal_generator,
        risk_manager=risk_manager,
        position_sizer=position_sizer,
        regime_detector=regime_detector,
    )
    strategy.stop_loss_pct = resolved_stop_loss
    strategy.take_profit_pct = resolved_take_profit
    strategy.risk_per_trade = DEFAULT_BASE_RISK_PER_TRADE
    strategy.set_risk_overrides(risk_overrides)
    return strategy
