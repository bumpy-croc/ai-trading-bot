"""
ML Basic Strategy - Component-Based Implementation

This strategy uses a machine learning model trained exclusively on price data (OHLCV).
It provides simple, reliable predictions without external dependencies like sentiment data.

Key Features:
- Price-only predictions using LSTM neural network
- 120-day sequence length for pattern recognition
- Normalized price inputs for better model performance
- 2% stop loss, 4% take profit risk management
- No external API dependencies
- Component-based architecture for better maintainability

Ideal for:
- Consistent, reliable trading signals
- Backtesting historical periods
- Environments where sentiment data is unavailable
- Simple deployment scenarios
"""

from typing import Any, Optional

from src.strategies.components import (
    Strategy,
    MLBasicSignalGenerator,
    FixedRiskManager,
    ConfidenceWeightedSizer,
    EnhancedRegimeDetector,
    HoldSignalGenerator,
    FixedFractionSizer,
    RegimeContext,
    TrendLabel,
    VolLabel,
)


def create_ml_basic_strategy(
    name: str = "MlBasic",
    model_path: str = "src/ml/btcusdt_price.onnx",
    sequence_length: int = 120,
    use_prediction_engine: Optional[bool] = None,
    model_name: Optional[str] = None,
    model_type: Optional[str] = None,
    timeframe: Optional[str] = None,
    fast_mode: bool = False,
) -> Strategy:
    """
    Create ML Basic strategy using component composition.

    Args:
        name: Strategy name
        model_path: Path to ONNX model file
        sequence_length: Number of candles for sequence prediction
        use_prediction_engine: Whether to use centralized prediction engine
        model_name: Model name for prediction engine
        model_type: Model type (e.g., "basic")
        timeframe: Model timeframe (e.g., "1h")

    Returns:
        Configured Strategy instance
    """
    if fast_mode:

        class _FastRegimeDetector:
            """Lightweight regime detector for fast test execution."""

            name = "fast_regime_detector"
            warmup_period = 0

            def detect_regime(self, df, index):
                return RegimeContext(
                    trend=TrendLabel.RANGE,
                    volatility=VolLabel.LOW,
                    confidence=1.0,
                    duration=index + 1,
                    strength=0.0,
                )

            def get_feature_generators(self):
                return []

        signal_generator = HoldSignalGenerator()
        risk_manager = FixedRiskManager(risk_per_trade=0.02, stop_loss_pct=0.02)
        position_sizer = FixedFractionSizer(fraction=0.001)
        regime_detector = _FastRegimeDetector()
    else:
        # Create signal generator with ML Basic parameters
        signal_generator = MLBasicSignalGenerator(
            name=f"{name}_signals",
            model_path=model_path,
            sequence_length=sequence_length,
            use_prediction_engine=use_prediction_engine,
            model_name=model_name,
            model_type=model_type,
            timeframe=timeframe,
        )

        # Create risk manager with fixed stop loss (2%)
        risk_manager = FixedRiskManager(
            risk_per_trade=0.02,
            stop_loss_pct=0.02,
        )

        # Create position sizer with confidence weighting (20% base)
        position_sizer = ConfidenceWeightedSizer(
            base_fraction=0.2,
            min_confidence=0.3,
        )
        regime_detector = EnhancedRegimeDetector()

    strategy = Strategy(
        name=name,
        signal_generator=signal_generator,
        risk_manager=risk_manager,
        position_sizer=position_sizer,
        regime_detector=regime_detector,
        enable_logging=not fast_mode,
    )

    return strategy
