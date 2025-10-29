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

    Returns:
        Configured Strategy instance
    """
    # Create signal generator with ML Sentiment parameters (sentiment model)
    signal_generator = MLSignalGenerator(
        name=f"{name}_signals",
        sequence_length=sequence_length,
        model_name=model_name,
    )

    risk_parameters = RiskParameters(
        base_risk_per_trade=0.02,
        default_take_profit_pct=0.04,
        max_position_size=0.1,
    )
    core_risk_manager = EngineRiskManager(risk_parameters)
    risk_overrides = {
        "position_sizer": "fixed_fraction",
        "base_fraction": 0.02,
        "max_fraction": 0.1,
        "stop_loss_pct": 0.02,
        "take_profit_pct": 0.04,
    }
    risk_manager = CoreRiskAdapter(core_risk_manager)
    risk_manager.set_strategy_overrides(risk_overrides)

    # Create position sizer with confidence weighting (20% base)
    position_sizer = ConfidenceWeightedSizer(
        base_fraction=0.2,
        min_confidence=0.3,
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
    strategy.stop_loss_pct = 0.02
    strategy.take_profit_pct = 0.04
    strategy.risk_per_trade = 0.02
    strategy._risk_overrides = risk_overrides
    return strategy
