"""
ML Basic Strategy - Low Confidence Threshold Variant

This is an experimental variant of ml_basic with lower confidence threshold (0.1 vs 0.3).

Hypothesis: The ML model may produce directionally correct predictions even at low
confidence levels. By lowering the threshold, we can test if the model has signal
quality that's being filtered out by the conservative threshold.

Experiment: Lower min_confidence from 0.3 to 0.1
Expected: More trades executed, potentially better risk-adjusted returns if model
          predictions are directionally accurate even at low confidence.
"""

from __future__ import annotations

from src.risk.risk_manager import RiskManager as EngineRiskManager
from src.risk.risk_manager import RiskParameters
from src.strategies.components import (
    ConfidenceWeightedSizer,
    CoreRiskAdapter,
    EnhancedRegimeDetector,
    MLBasicSignalGenerator,
    Strategy,
)


def create_ml_basic_low_conf_strategy(
    name: str = "MlBasicLowConf",
    sequence_length: int = 120,
    model_name: str | None = None,
    model_type: str | None = None,
    timeframe: str | None = None,
) -> Strategy:
    """
    Create ML Basic strategy with lower confidence threshold (0.1 vs 0.3).

    Args:
        name: Strategy name
        sequence_length: Number of candles for sequence prediction
        model_name: Model name for prediction engine
        model_type: Model type (e.g., "basic")
        timeframe: Model timeframe (e.g., "1h")

    Returns:
        Configured Strategy instance with low confidence threshold
    """
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

    # EXPERIMENT: Lower confidence threshold to 0.1 (vs default 0.3)
    position_sizer = ConfidenceWeightedSizer(
        base_fraction=0.2,
        min_confidence=0.1,  # <-- Key change: lowered from 0.3
    )
    regime_detector = EnhancedRegimeDetector()

    strategy = Strategy(
        name=name,
        signal_generator=signal_generator,
        risk_manager=risk_manager,
        position_sizer=position_sizer,
        regime_detector=regime_detector,
        enable_logging=True,
    )

    return strategy
