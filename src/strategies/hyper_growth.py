"""
Hyper Growth Strategy - Aggressive Component-Based Implementation

Targets 500% annual returns by combining three key mechanisms from research:
1. ML-driven signal generation for directional alpha (66%+ win rate)
2. High base position sizing (20% of balance per trade)
3. Aggressive risk overrides: wider stops, deeper drawdown tolerance

NOTE: Leverage is DISABLED by default (max_leverage=1.0). Backtesting showed
that enabling leverage (e.g. 3x) hurt overall returns by -32% due to amplified
losses during volatile regime transitions. The LeverageManager infrastructure
and _HYPER_LEVERAGE_MAP remain in place for future experimentation — callers
can pass max_leverage > 1.0 to re-enable it.

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
from src.strategies.components.regime_context import TrendLabel, VolLabel
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
        min_confidence: float = 0.05,
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


# Aggressive leverage map: amplify in bulls, protect in bears.
# Keys are (TrendLabel, VolLabel) tuples matching LeverageManager.get_leverage_multiplier() lookups.
# NOTE: These raw multipliers are clamped by LeverageManager's max_leverage cap.
# With the default max_leverage=1.0 none of these values take effect — they exist
# so that callers who pass max_leverage > 1.0 get regime-aware scaling automatically.
_HYPER_LEVERAGE_MAP: dict[tuple[TrendLabel, VolLabel], float] = {
    (TrendLabel.TREND_UP, VolLabel.LOW): 3.0,      # Full leverage in confirmed bull + low vol
    (TrendLabel.TREND_UP, VolLabel.HIGH): 2.0,      # Moderate leverage in bull + high vol
    (TrendLabel.RANGE, VolLabel.LOW): 1.5,           # Slight leverage in confirmed range + low vol
    (TrendLabel.RANGE, VolLabel.HIGH): 1.0,          # No leverage in range + high vol
    (TrendLabel.TREND_DOWN, VolLabel.LOW): 0.5,      # Minimal exposure in bear + low vol
    (TrendLabel.TREND_DOWN, VolLabel.HIGH): 0.0,     # Cash in confirmed bear + high vol
}


def create_hyper_growth_strategy(
    name: str = "HyperGrowth",
    signal_source: str = "ml",
    risk_fraction: float = 0.20,
    base_fraction: float = 0.20,
    min_confidence: float = 0.05,
    max_leverage: float = 1.0,
    leverage_decay_rate: float = 0.20,
    min_regime_bars: int = 3,
    take_profit_pct: float = 0.30,
    stop_loss_pct: float = 0.20,
) -> Strategy:
    """Create hyper-growth strategy targeting 500% annual returns.

    Combines ML signal generation with aggressive position sizing to
    maximize compounding during favorable conditions while preserving
    capital during drawdowns.

    Leverage is DISABLED by default (max_leverage=1.0) because testing
    showed it reduces returns by -32%. The LeverageManager infrastructure
    is wired up so callers can pass max_leverage > 1.0 to experiment,
    but the default configuration runs without leverage amplification.

    Args:
        name: Strategy name.
        signal_source: "ml" for ML predictions, "momentum" for breakouts.
        risk_fraction: Fraction of balance the risk manager allocates per trade.
        base_fraction: Base position size as fraction of balance.
        min_confidence: Minimum signal confidence to allow a trade.
        max_leverage: Maximum leverage multiplier safety cap (default 1.0 =
            leverage disabled). Set > 1.0 to re-enable regime-based leverage.
        leverage_decay_rate: Exponential decay for smooth transitions.
        min_regime_bars: Bars before conviction scaling begins.
        take_profit_pct: Target profit percentage for scaling out.
        stop_loss_pct: Stop loss percentage.

    Returns:
        Configured Strategy instance.
    """
    # Signal generator
    if signal_source == "momentum":
        signal_generator = MomentumSignalGenerator(
            name=f"{name}_signals",
            momentum_entry_threshold=0.001,   # 0.1% — very sensitive
            strong_momentum_threshold=0.005,  # 0.5%
        )
    else:
        # ML signals — the model predicts direction with ~65% accuracy
        # but per-bar confidence is very low (0.01-0.10)
        signal_generator = MLBasicSignalGenerator(name=f"{name}_signals")

    risk_manager = FlatRiskManager(
        risk_fraction=risk_fraction,
        stop_loss_pct=stop_loss_pct,
        min_confidence=min_confidence,
    )

    base_sizer = FixedFractionSizer(
        fraction=base_fraction,
        adjust_for_confidence=False,
        adjust_for_strength=False,
    )

    regime_detector = EnhancedRegimeDetector()

    leverage_manager = LeverageManager(
        max_leverage=max_leverage,
        decay_rate=leverage_decay_rate,
        min_regime_bars=min_regime_bars,
        leverage_map=_HYPER_LEVERAGE_MAP,
    )

    position_sizer = LeveragedPositionSizer(
        base_sizer=base_sizer,
        leverage_manager=leverage_manager,
        max_leveraged_fraction=0.50,
    )

    strategy = Strategy(
        name=name,
        signal_generator=signal_generator,
        risk_manager=risk_manager,
        position_sizer=position_sizer,
        regime_detector=regime_detector,
    )

    strategy.leverage_manager = leverage_manager
    strategy.base_position_size = base_fraction
    strategy.take_profit_pct = take_profit_pct
    # Override default 25% cap to match max_leveraged_fraction (50%),
    # otherwise _validate_position_size() clamps leveraged positions to 25%
    strategy._max_position_pct = 0.50

    # Hold positions through signal flips — only exit on SL/TP/trailing stop.
    # ML signals flip direction every bar, but the trend persists for days.
    # Without this, positions are exited after 1 bar with 0.03% P&L.
    strategy._extra_metadata = {"ignore_signal_reversal": True}

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
