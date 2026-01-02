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

from src.config.constants import (
    DEFAULT_BASE_RISK_PER_TRADE,
    DEFAULT_MAX_SCALE_INS,
    DEFAULT_PARTIAL_EXIT_SIZES,
    DEFAULT_PARTIAL_EXIT_TARGETS,
    DEFAULT_SCALE_IN_SIZES,
    DEFAULT_SCALE_IN_THRESHOLDS,
    DEFAULT_STRATEGY_BASE_FRACTION,
    DEFAULT_STRATEGY_MIN_CONFIDENCE,
)
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
        base_risk=DEFAULT_BASE_RISK_PER_TRADE,
    )

    # Create position sizer with confidence weighting
    position_sizer = ConfidenceWeightedSizer(
        base_fraction=DEFAULT_STRATEGY_BASE_FRACTION,
        min_confidence=DEFAULT_STRATEGY_MIN_CONFIDENCE,
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

    strategy.set_risk_overrides({
        "partial_operations": {
            "exit_targets": list(DEFAULT_PARTIAL_EXIT_TARGETS),
            "exit_sizes": list(DEFAULT_PARTIAL_EXIT_SIZES),
            "scale_in_thresholds": list(DEFAULT_SCALE_IN_THRESHOLDS),
            "scale_in_sizes": list(DEFAULT_SCALE_IN_SIZES),
            "max_scale_ins": DEFAULT_MAX_SCALE_INS,
        }
    })

    return strategy
