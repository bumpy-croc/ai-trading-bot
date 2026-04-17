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
    FixedFractionSizer,
    HoldSignalGenerator,
    MLBasicSignalGenerator,
    RegimeContext,
    Strategy,
    TrendLabel,
    VolLabel,
)


def create_ml_basic_strategy(
    name: str = "MlBasic",
    sequence_length: int = 120,
    model_name: str | None = None,
    model_type: str | None = None,
    timeframe: str | None = None,
    fast_mode: bool = False,
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
    Create ML Basic strategy using component composition.

    Args:
        name: Strategy name
        sequence_length: Number of candles for sequence prediction
        model_name: Model name for prediction engine
        model_type: Model type (e.g., "basic")
        timeframe: Model timeframe (e.g., "1h")
        fast_mode: Enable fast mode for testing (disables ML)
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
    risk_parameters = RiskParameters(
        base_risk_per_trade=DEFAULT_BASE_RISK_PER_TRADE,
        default_take_profit_pct=(
            take_profit_pct if take_profit_pct is not None else DEFAULT_TAKE_PROFIT_PCT
        ),
        max_position_size=DEFAULT_MAX_POSITION_SIZE,
    )
    core_risk_manager = EngineRiskManager(risk_parameters)
    risk_overrides = {
        "position_sizer": "fixed_fraction",
        "base_fraction": DEFAULT_BASE_RISK_PER_TRADE,
        "max_fraction": DEFAULT_MAX_POSITION_SIZE,
        "stop_loss_pct": (
            stop_loss_pct if stop_loss_pct is not None else DEFAULT_BASE_RISK_PER_TRADE
        ),
        "take_profit_pct": (
            take_profit_pct if take_profit_pct is not None else DEFAULT_TAKE_PROFIT_PCT
        ),
    }

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
        risk_manager = CoreRiskAdapter(core_risk_manager)
        risk_manager.set_strategy_overrides(risk_overrides)
        position_sizer = FixedFractionSizer(fraction=0.001)
        regime_detector = _FastRegimeDetector()
    else:
        # Create signal generator with ML Basic parameters
        signal_generator = MLBasicSignalGenerator(
            name=f"{name}_signals",
            sequence_length=sequence_length,
            model_name=model_name,
            model_type=model_type,
            timeframe=timeframe,
            long_entry_threshold=long_entry_threshold,
            short_entry_threshold=short_entry_threshold,
            confidence_multiplier=confidence_multiplier,
        )

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
            min_confidence_floor=(
                min_confidence_floor if min_confidence_floor is not None else 0.0
            ),
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
