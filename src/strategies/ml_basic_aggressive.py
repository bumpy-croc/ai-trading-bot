"""
ML Basic Strategy - Aggressive Risk Management Variant

This is an experimental variant of ml_basic with more aggressive take profit targets.

Hypothesis: Current strategy has 72% win rate but only 0.11% returns over 6 months.
The issue may be taking profits too early. By increasing TP from 4% to 8% while
keeping the same stop loss, we can potentially capture larger moves.

Experiment: Increase take_profit from 4% to 8%, keep stop_loss at 2%
Expected: Higher returns per trade, potentially better risk-adjusted returns even
          if win rate drops slightly.
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


def create_ml_basic_aggressive_strategy(
    name: str = "MlBasicAggressive",
    sequence_length: int = 120,
    model_name: str | None = None,
    model_type: str | None = None,
    timeframe: str | None = None,
) -> Strategy:
    """
    Create ML Basic strategy with aggressive take profit (8% vs 4%).

    Args:
        name: Strategy name
        sequence_length: Number of candles for sequence prediction
        model_name: Model name for prediction engine
        model_type: Model type (e.g., "basic")
        timeframe: Model timeframe (e.g., "1h")

    Returns:
        Configured Strategy instance with aggressive risk params
    """
    risk_parameters = RiskParameters(
        base_risk_per_trade=0.02,
        default_take_profit_pct=0.08,  # <-- Increased from 0.04
        max_position_size=0.1,
    )
    core_risk_manager = EngineRiskManager(risk_parameters)
    risk_overrides = {
        "position_sizer": "fixed_fraction",
        "base_fraction": 0.02,
        "max_fraction": 0.1,
        "stop_loss_pct": 0.02,  # Keep same
        "take_profit_pct": 0.08,  # <-- Doubled from 0.04
    }

    signal_generator = MLBasicSignalGenerator(
        name=f"{name}_signals",
        sequence_length=sequence_length,
        model_name=model_name,
        model_type=model_type,
        timeframe=timeframe,
    )

    risk_manager = CoreRiskAdapter(core_risk_manager)
    risk_manager.set_strategy_overrides(risk_overrides)

    # Keep same confidence threshold (0.3) and position sizing
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
        enable_logging=True,
    )

    return strategy
