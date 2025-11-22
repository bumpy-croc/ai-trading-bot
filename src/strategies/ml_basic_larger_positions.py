"""
ML Basic Strategy - Larger Position Size Variant

This experimental variant tests increased position sizing to capture meaningful returns.

Hypothesis: Current 0.11% returns over 6 months are due to position sizing that's too
conservative. With 72% win rate and 0.10% max drawdown, there's significant room to
increase position sizes while staying within acceptable risk bounds.

Experiment: Increase base_fraction from 0.02 (2%) to 0.05 (5%)
Expected: 2.5x higher returns (0.11% → 0.27% for same trades), still well under
          reasonable drawdown limits (5-10% is acceptable for crypto)
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


def create_ml_basic_larger_positions_strategy(
    name: str = "MlBasicLarger",
    sequence_length: int = 120,
    model_name: str | None = None,
    model_type: str | None = None,
    timeframe: str | None = None,
) -> Strategy:
    """
    Create ML Basic strategy with larger position sizes (5% vs 2%).

    Args:
        name: Strategy name
        sequence_length: Number of candles for sequence prediction
        model_name: Model name for prediction engine
        model_type: Model type (e.g., "basic")
        timeframe: Model timeframe (e.g., "1h")

    Returns:
        Configured Strategy instance with larger positions
    """
    risk_parameters = RiskParameters(
        base_risk_per_trade=0.05,  # <-- Increased from 0.02
        default_take_profit_pct=0.04,
        max_position_size=0.15,  # <-- Increased from 0.1
    )
    core_risk_manager = EngineRiskManager(risk_parameters)
    risk_overrides = {
        "position_sizer": "fixed_fraction",
        "base_fraction": 0.05,  # <-- Increased from 0.02
        "max_fraction": 0.15,  # <-- Increased from 0.1
        "stop_loss_pct": 0.02,
        "take_profit_pct": 0.04,
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

    # EXPERIMENT: Still use confidence weighting but with higher base
    # 5% base * 0.3 min confidence = 1.5% min position size (vs 0.6% baseline)
    position_sizer = ConfidenceWeightedSizer(
        base_fraction=0.2,  # This multiplies the base_fraction from risk_overrides
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
