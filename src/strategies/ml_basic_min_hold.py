"""
ML Basic Strategy - Minimum Hold Time Variant

This experimental variant addresses the premature exit issue identified in the
optimization analysis.

Root Cause: ML model signal reversals trigger exits before profit targets are
reached, causing economically useless returns (0.11% over 6 months despite 72% WR).

Hypothesis: Adding a minimum hold time prevents the ML model from flipping
predictions too quickly, allowing trades to reach profit targets.

Experiment: Add 4-hour minimum hold time (4 candles on 1h timeframe)
Expected: 5-10x improvement in returns (0.28% → 1.4-2.8% for same trades)

Code Change: Override should_exit_position() to reject signal reversals within
             minimum hold period.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Optional

from src.risk.risk_manager import RiskManager as EngineRiskManager
from src.risk.risk_manager import RiskParameters
from src.strategies.components import (
    ConfidenceWeightedSizer,
    CoreRiskAdapter,
    EnhancedRegimeDetector,
    MLBasicSignalGenerator,
    Strategy,
)

if TYPE_CHECKING:
    from src.strategies.components.risk_manager import MarketData, Position
    from src.strategies.components.regime_context import RegimeContext


class MinHoldTimeStrategy(Strategy):
    """
    Strategy variant with minimum hold time enforcement.

    Overrides should_exit_position to prevent premature exits due to
    signal reversals within the minimum hold period.
    """

    def __init__(
        self,
        min_hold_hours: float = 4.0,
        **kwargs,
    ):
        """
        Initialize strategy with minimum hold time.

        Args:
            min_hold_hours: Minimum hours to hold position before allowing exit
            **kwargs: Arguments for Strategy base class
        """
        super().__init__(**kwargs)
        self.min_hold_hours = min_hold_hours
        self.min_hold_timedelta = timedelta(hours=min_hold_hours)

    def should_exit_position(
        self,
        position: Position,
        current_data: MarketData,
        regime: Optional[RegimeContext] = None,
    ) -> bool:
        """
        Determine if position should be exited, respecting minimum hold time.

        Within minimum hold period: only allows stop-loss based exits
        After minimum hold period: normal exit logic (including signal reversals)

        Args:
            position: Current position to evaluate
            current_data: Current market data
            regime: Optional regime context

        Returns:
            True if position should be exited, False otherwise
        """
        # Calculate hold duration
        if not hasattr(current_data, "timestamp") or current_data.timestamp is None:
            # If no timestamp available, use parent behavior
            return super().should_exit_position(position, current_data, regime)

        if not hasattr(position, "entry_time") or position.entry_time is None:
            # If no entry time available, use parent behavior
            return super().should_exit_position(position, current_data, regime)

        # Parse timestamps if they're strings
        if isinstance(current_data.timestamp, str):
            current_time = datetime.fromisoformat(current_data.timestamp)
        else:
            current_time = current_data.timestamp

        if isinstance(position.entry_time, str):
            entry_time = datetime.fromisoformat(position.entry_time)
        else:
            entry_time = position.entry_time

        hold_duration = current_time - entry_time

        # Within minimum hold period: only check risk-based exits (stop loss)
        if hold_duration < self.min_hold_timedelta:
            # Only allow exit if risk manager says so (stop loss hit)
            try:
                return self.risk_manager.should_exit(position, current_data, regime)
            except Exception as e:
                self.logger.error(f"Error in min hold exit check: {e}")
                return False

        # After minimum hold period: normal exit logic
        return super().should_exit_position(position, current_data, regime)


def create_ml_basic_min_hold_strategy(
    name: str = "MlBasicMinHold",
    sequence_length: int = 120,
    model_name: str | None = None,
    model_type: str | None = None,
    timeframe: str | None = None,
    min_hold_hours: float = 4.0,
) -> Strategy:
    """
    Create ML Basic strategy with minimum hold time enforcement.

    Args:
        name: Strategy name
        sequence_length: Number of candles for sequence prediction
        model_name: Model name for prediction engine
        model_type: Model type (e.g., "basic")
        timeframe: Model timeframe (e.g., "1h")
        min_hold_hours: Minimum hours to hold positions (default: 4)

    Returns:
        Configured Strategy instance with minimum hold time
    """
    # Use same risk parameters as ml_basic_larger_positions (validated)
    risk_parameters = RiskParameters(
        base_risk_per_trade=0.05,
        max_risk_per_trade=0.15,
        default_take_profit_pct=0.04,
        max_position_size=0.15,
    )
    core_risk_manager = EngineRiskManager(risk_parameters)
    risk_overrides = {
        "position_sizer": "fixed_fraction",
        "base_fraction": 0.05,
        "max_fraction": 0.15,
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

    # Use standard CoreRiskAdapter
    risk_manager = CoreRiskAdapter(core_risk_manager)
    risk_manager.set_strategy_overrides(risk_overrides)

    # Use validated position sizing from Experiment 3
    position_sizer = ConfidenceWeightedSizer(
        base_fraction=0.5,  # Validated 2.5x improvement
        min_confidence=0.3,
    )
    regime_detector = EnhancedRegimeDetector()

    # Use MinHoldTimeStrategy instead of base Strategy class
    strategy = MinHoldTimeStrategy(
        min_hold_hours=min_hold_hours,
        name=name,
        signal_generator=signal_generator,
        risk_manager=risk_manager,
        position_sizer=position_sizer,
        regime_detector=regime_detector,
        enable_logging=True,
    )

    return strategy
