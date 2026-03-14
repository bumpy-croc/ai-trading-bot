"""
Hyper Growth Strategy - Aggressive Component-Based Implementation

Targets 500% annual returns by combining three key mechanisms from research:
1. ML-driven signal generation for directional alpha (66%+ win rate)
2. High base position sizing with regime-based leverage (up to 3x in bulls)
3. Aggressive risk overrides: wider stops, deeper drawdown tolerance

Key architectural decision: ML models produce very low raw confidence values
(0.01-0.05 per bar) even when their directional accuracy is high (66%+).
Standard risk managers and sizers multiply by confidence, crushing positions
to $10. This strategy uses a FlatRiskManager that returns a fixed fraction
of balance regardless of confidence — the ML direction filter IS the edge,
not the per-bar confidence score.

Reference: docs/research/500_percent_annual_returns.md
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from src.strategies.components import (
    EnhancedRegimeDetector,
    FixedFractionSizer,
    MLBasicSignalGenerator,
    MomentumSignalGenerator,
    Strategy,
)
from src.strategies.components.leverage_manager import LeverageManager
from src.strategies.components.position_sizer import LeveragedPositionSizer
from src.strategies.components.risk_manager import RiskManager

if TYPE_CHECKING:
    from src.strategies.components.regime_context import RegimeContext
    from src.strategies.components.signal_generator import Signal


class FlatRiskManager(RiskManager):
    """Risk manager that returns a fixed fraction of balance without
    scaling by signal confidence or strength.

    Standard risk managers multiply by confidence (0.01-0.05 for ML signals),
    reducing $1000 risk to $10. This manager trusts the ML direction filter
    and returns the full risk amount, letting the position sizer and leverage
    manager control sizing.
    """

    def __init__(
        self,
        risk_fraction: float = 0.10,
        stop_loss_pct: float = 0.10,
        min_confidence: float = 0.03,
    ):
        """Initialize flat risk manager.

        Args:
            risk_fraction: Fraction of balance to risk per trade (0.10 = 10%).
            stop_loss_pct: Stop loss percentage for exit decisions.
            min_confidence: Minimum signal confidence to allow a trade.
                ML signals below this are filtered out as noise.
        """
        super().__init__("flat_risk_manager")
        if not 0.01 <= risk_fraction <= 0.50:
            raise ValueError(
                f"risk_fraction must be between 0.01 and 0.50, got {risk_fraction}"
            )
        self.risk_fraction = risk_fraction
        self.stop_loss_pct = stop_loss_pct
        self.min_confidence = min_confidence

    def calculate_position_size(
        self,
        signal: "Signal",
        balance: float,
        regime: Optional["RegimeContext"] = None,
        **context: Any,
    ) -> float:
        """Return balance * risk_fraction without confidence/strength scaling.

        Uses a minimum confidence threshold to filter out low-quality signals.
        ML models produce 0.01-0.05 confidence on most bars but 0.10+ on
        higher-conviction signals that historically have 60%+ accuracy.
        """
        self.validate_inputs(balance)
        if signal.direction.value == "hold":
            return 0.0
        # Filter out very low confidence signals — below this threshold,
        # the ML model is essentially guessing
        if signal.confidence < self.min_confidence:
            return 0.0
        # Flat risk — no further confidence/strength scaling
        return balance * self.risk_fraction

    def should_exit(
        self,
        position: Any,
        current_data: Any,
        regime: Optional["RegimeContext"] = None,
        **context: Any,
    ) -> bool:
        """Exit when unrealized loss exceeds stop_loss_pct."""
        if not hasattr(position, "unrealized_pnl_pct"):
            return False
        return position.unrealized_pnl_pct <= -self.stop_loss_pct

    def get_stop_loss(
        self,
        entry_price: float,
        signal: "Signal",
        regime: Optional["RegimeContext"] = None,
        **context: Any,
    ) -> float:
        """Calculate stop loss based on fixed percentage."""
        if entry_price <= 0:
            raise ValueError(f"entry_price must be positive, got {entry_price}")
        if signal.direction.value == "buy":
            return entry_price * (1 - self.stop_loss_pct)
        if signal.direction.value == "sell":
            return entry_price * (1 + self.stop_loss_pct)
        return entry_price

    def get_parameters(self) -> dict[str, float]:
        """Return risk manager parameters."""
        return {
            "risk_fraction": self.risk_fraction,
            "stop_loss_pct": self.stop_loss_pct,
        }


# Aggressive leverage map: amplify in bulls, protect in bears
_HYPER_LEVERAGE_MAP = {
    "bull_high_conf": 3.0,   # Full leverage in confirmed bull trends
    "bull_low_conf": 2.0,    # Moderate leverage in early/uncertain bulls
    "range_high_conf": 1.5,  # Slight leverage in confirmed ranges
    "range_low_conf": 1.0,   # No leverage in uncertain ranges
    "bear_low_conf": 0.5,    # Minimal exposure in uncertain bears
    "bear_high_conf": 0.0,   # Cash in confirmed bears
}


def create_hyper_growth_strategy(
    name: str = "HyperGrowth",
    signal_source: str = "ml",
    risk_fraction: float = 0.25,
    base_fraction: float = 0.25,
    min_confidence: float = 0.10,
    max_leverage: float = 3.0,
    leverage_decay_rate: float = 0.20,
    min_regime_bars: int = 3,
    take_profit_pct: float = 0.30,
    stop_loss_pct: float = 0.10,
) -> Strategy:
    """Create hyper-growth strategy targeting 500% annual returns.

    Combines ML signal generation with aggressive position sizing and
    regime-based leverage to maximize compounding during favorable
    conditions while preserving capital during drawdowns.

    Args:
        name: Strategy name.
        signal_source: "ml" for ML predictions, "momentum" for breakouts.
        risk_fraction: Fraction of balance the risk manager allocates per trade.
        base_fraction: Base position size as fraction of balance.
        max_leverage: Maximum leverage multiplier safety cap.
        leverage_decay_rate: Exponential decay for smooth transitions.
        min_regime_bars: Bars before conviction scaling begins.
        take_profit_pct: Target profit percentage for scaling out.
        stop_loss_pct: Stop loss percentage.

    Returns:
        Configured Strategy instance with leverage.
    """
    # Signal generator
    if signal_source == "momentum":
        signal_generator = MomentumSignalGenerator(
            name=f"{name}_signals",
            momentum_entry_threshold=0.005,
            strong_momentum_threshold=0.015,
        )
    else:
        signal_generator = MLBasicSignalGenerator(name=f"{name}_signals")

    # Flat risk manager: returns full risk_fraction without confidence scaling
    # min_confidence filters out noise — only trade when ML has some conviction
    risk_manager = FlatRiskManager(
        risk_fraction=risk_fraction,
        stop_loss_pct=stop_loss_pct,
        min_confidence=min_confidence,
    )

    # Fixed fraction sizer WITHOUT confidence/strength scaling
    base_sizer = FixedFractionSizer(
        fraction=base_fraction,
        adjust_for_confidence=False,
        adjust_for_strength=False,
    )

    # Regime detector
    regime_detector = EnhancedRegimeDetector()

    # Leverage manager: amplify in bulls, cash in bears
    leverage_manager = LeverageManager(
        max_leverage=max_leverage,
        decay_rate=leverage_decay_rate,
        min_regime_bars=min_regime_bars,
        leverage_map=_HYPER_LEVERAGE_MAP,
    )

    # Wrap base sizer with leverage multiplier
    position_sizer = LeveragedPositionSizer(
        base_sizer=base_sizer,
        leverage_manager=leverage_manager,
        max_leveraged_fraction=0.50,
    )

    # Compose strategy
    strategy = Strategy(
        name=name,
        signal_generator=signal_generator,
        risk_manager=risk_manager,
        position_sizer=position_sizer,
        regime_detector=regime_detector,
    )

    # Expose components for introspection
    strategy.leverage_manager = leverage_manager
    strategy.base_position_size = base_fraction
    strategy.take_profit_pct = take_profit_pct

    # Engine-level risk overrides
    strategy.set_risk_overrides(
        {
            "position_sizer": "leveraged_fixed_fraction",
            "base_fraction": base_fraction,
            "min_fraction": 0.01,
            "max_fraction": min(base_fraction * max_leverage, 0.50),
            "stop_loss_pct": stop_loss_pct,
            "take_profit_pct": take_profit_pct,
            "leverage": {
                "enabled": True,
                "max_leverage": max_leverage,
                "decay_rate": leverage_decay_rate,
                "min_regime_bars": min_regime_bars,
            },
            "dynamic_risk": {
                "enabled": True,
                # Wider drawdown tolerance for hyper-growth target
                "drawdown_thresholds": [0.15, 0.30, 0.45],
                "risk_reduction_factors": [0.8, 0.5, 0.2],
                "recovery_thresholds": [0.08, 0.15],
            },
            "partial_operations": {
                "exit_targets": [0.08, 0.15, 0.30],
                "exit_sizes": [0.20, 0.30, 0.50],
                "scale_in_thresholds": [0.015, 0.03],
                "scale_in_sizes": [0.40, 0.25],
                "max_scale_ins": 2,
            },
            "trailing_stop": {
                "activation_threshold": 0.03,
                "trailing_distance_pct": 0.015,
                "breakeven_threshold": 0.05,
                "breakeven_buffer": 0.008,
            },
        }
    )

    return strategy
