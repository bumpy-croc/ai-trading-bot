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

    Returns:
        Configured Strategy instance
    """
    risk_parameters = RiskParameters(
        base_risk_per_trade=DEFAULT_BASE_RISK_PER_TRADE,
        default_take_profit_pct=DEFAULT_TAKE_PROFIT_PCT,
        max_position_size=DEFAULT_MAX_POSITION_SIZE,
    )
    core_risk_manager = EngineRiskManager(risk_parameters)
    risk_overrides = {
        "position_sizer": "fixed_fraction",
        "base_fraction": DEFAULT_BASE_RISK_PER_TRADE,
        "max_fraction": DEFAULT_MAX_POSITION_SIZE,
        "stop_loss_pct": DEFAULT_BASE_RISK_PER_TRADE,  # Tight stop matching risk
        "take_profit_pct": DEFAULT_TAKE_PROFIT_PCT,
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
        )

        risk_manager = CoreRiskAdapter(core_risk_manager)
        risk_manager.set_strategy_overrides(risk_overrides)

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
