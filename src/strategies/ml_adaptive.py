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

from src.strategies.components import (
    ConfidenceWeightedSizer,
    EnhancedRegimeDetector,
    MLSignalGenerator,
    RegimeAdaptiveRiskManager,
    Strategy,
)


def create_ml_adaptive_strategy(
    name: str = "MlAdaptive",
    sequence_length: int = 120,
    model_name: str | None = None,
) -> Strategy:
    """
    Create ML Adaptive strategy using component composition.

    This strategy uses regime-aware thresholds and adaptive risk management
    to adjust to changing market conditions.

    Args:
        name: Strategy name
        sequence_length: Number of candles for sequence prediction
        model_name: Model name for prediction engine

    Returns:
        Configured Strategy instance
    """
    # Create signal generator with ML Adaptive parameters (regime-aware thresholds)
    signal_generator = MLSignalGenerator(
        name=f"{name}_signals",
        sequence_length=sequence_length,
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

    strategy = Strategy(
        name=name,
        signal_generator=signal_generator,
        risk_manager=risk_manager,
        position_sizer=position_sizer,
        regime_detector=regime_detector,
    )

    strategy._risk_overrides = {
        "partial_operations": {
            "exit_targets": [0.03, 0.06, 0.10],
            "exit_sizes": [0.25, 0.25, 0.50],
            "scale_in_thresholds": [0.02, 0.05],
            "scale_in_sizes": [0.25, 0.25],
            "max_scale_ins": 2,
        }
    }

    return strategy
