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

from typing import Optional

from src.strategies.components import (
    Strategy,
    MLSignalGenerator,
    FixedRiskManager,
    ConfidenceWeightedSizer,
    EnhancedRegimeDetector,
)


def create_ml_sentiment_strategy(
    name: str = "MlSentiment",
    model_path: str = "src/ml/btcusdt_sentiment.onnx",
    sequence_length: int = 120,
    use_prediction_engine: Optional[bool] = None,
    model_name: Optional[str] = None,
    model_type: Optional[str] = None,
    timeframe: Optional[str] = None,
) -> Strategy:
    """
    Create ML Sentiment strategy using component composition.
    
    This strategy uses sentiment-aware signal generation to enhance
    prediction accuracy with market sentiment data.
    
    Args:
        name: Strategy name
        model_path: Path to ONNX model file (sentiment model)
        sequence_length: Number of candles for sequence prediction
        use_prediction_engine: Whether to use centralized prediction engine
        model_name: Model name for prediction engine
        model_type: Model type (e.g., "sentiment")
        timeframe: Model timeframe (e.g., "1h")
    
    Returns:
        Configured Strategy instance
    """
    # Create signal generator with ML Sentiment parameters (sentiment model)
    signal_generator = MLSignalGenerator(
        name=f"{name}_signals",
        model_path=model_path,
        sequence_length=sequence_length,
        use_prediction_engine=use_prediction_engine,
        model_name=model_name,
    )
    
    # Create fixed risk manager (2% risk per trade)
    risk_manager = FixedRiskManager(
        risk_per_trade=0.02,
        stop_loss_pct=0.02,
    )
    
    # Create position sizer with confidence weighting (20% base)
    position_sizer = ConfidenceWeightedSizer(
        base_fraction=0.2,
        min_confidence=0.3,
    )
    
    # Create regime detector
    regime_detector = EnhancedRegimeDetector()
    
    return Strategy(
        name=name,
        signal_generator=signal_generator,
        risk_manager=risk_manager,
        position_sizer=position_sizer,
        regime_detector=regime_detector,
    )
