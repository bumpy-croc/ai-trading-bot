"""
ML Adaptive Strategy - Component-Based Implementation

This strategy uses a machine learning model trained exclusively on price data (OHLCV).
It provides simple, reliable predictions without external dependencies like sentiment data.

Key Features:
- Price-only predictions using LSTM neural network
- 120-day sequence length for pattern recognition
- Normalized price inputs for better model performance
- 2% stop loss, 4% take profit risk management
- No external API dependencies
- Component-based architecture for better maintainability
- Dynamic regime-aware thresholds

Ideal for:
- Consistent, reliable trading signals
- Backtesting historical periods
- Environments where sentiment data is unavailable
- Simple deployment scenarios
"""

from typing import Optional

from src.strategies.components import (
    Strategy,
    MLSignalGenerator,
    RegimeAdaptiveRiskManager,
    ConfidenceWeightedSizer,
    EnhancedRegimeDetector,
)


def create_ml_adaptive_strategy(
    name: str = "MlAdaptive",
    model_path: str = "src/ml/btcusdt_price.onnx",
    sequence_length: int = 120,
    use_prediction_engine: Optional[bool] = None,
    model_name: Optional[str] = None,
) -> Strategy:
    """
    Create ML Adaptive strategy using component composition.
    
    This strategy uses regime-aware thresholds and adaptive risk management
    to adjust to changing market conditions.
    
    Args:
        name: Strategy name
        model_path: Path to ONNX model file
        sequence_length: Number of candles for sequence prediction
        use_prediction_engine: Whether to use centralized prediction engine
        model_name: Model name for prediction engine
    
    Returns:
        Configured Strategy instance
    """
    # Create signal generator with ML Adaptive parameters (regime-aware thresholds)
    signal_generator = MLSignalGenerator(
        name=f"{name}_signals",
        model_path=model_path,
        sequence_length=sequence_length,
        use_prediction_engine=use_prediction_engine,
        model_name=model_name,
    )
    
    # Create regime-adaptive risk manager
    risk_manager = RegimeAdaptiveRiskManager(
        base_risk=0.02,  # 2% base risk
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
