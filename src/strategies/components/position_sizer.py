"""
Position Sizer Components

This module defines the abstract PositionSizer interface and implementations
for calculating position sizes based on various factors in the component-based
strategy architecture.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

if TYPE_CHECKING:
    from .regime_context import RegimeContext
    from .runtime import FeatureGeneratorSpec
    from .signal_generator import Signal


class PositionSizer(ABC):
    """
    Abstract base class for position sizers

    Position sizers are responsible for calculating the optimal position size
    based on signal strength, confidence, available balance, risk parameters,
    and market regime conditions.
    """

    def __init__(self, name: str):
        """
        Initialize the position sizer

        Args:
            name: Unique name for this position sizer
        """
        self.name = name

    @abstractmethod
    def calculate_size(
        self,
        signal: "Signal",
        balance: float,
        risk_amount: float,
        regime: Optional["RegimeContext"] = None,
    ) -> float:
        """
        Calculate optimal position size

        Args:
            signal: Trading signal with strength and confidence
            balance: Available account balance
            risk_amount: Maximum amount to risk on this trade
            regime: Optional regime context for regime-aware sizing

        Returns:
            Position size in base currency

        Raises:
            ValueError: If input parameters are invalid
        """
        pass

    def validate_inputs(self, balance: float, risk_amount: float) -> None:
        """
        Validate common input parameters

        Args:
            balance: Account balance to validate
            risk_amount: Risk amount to validate

        Raises:
            ValueError: If parameters are invalid
        """
        if balance <= 0:
            raise ValueError(f"balance must be positive, got {balance}")

        if risk_amount < 0:
            raise ValueError(f"risk_amount must be non-negative, got {risk_amount}")

        if risk_amount > balance:
            raise ValueError(f"risk_amount ({risk_amount}) cannot exceed balance ({balance})")

    def apply_bounds_checking(
        self, size: float, balance: float, min_fraction: float = 0.001, max_fraction: float = 0.2
    ) -> float:
        """
        Apply bounds checking to position size

        Args:
            size: Calculated position size
            balance: Available balance
            min_fraction: Minimum position size as fraction of balance
            max_fraction: Maximum position size as fraction of balance

        Returns:
            Position size within bounds
        """
        min_size = balance * min_fraction
        max_size = balance * max_fraction

        return max(min_size, min(max_size, size))

    def get_parameters(self) -> dict[str, Any]:
        """
        Get position sizer parameters for logging and serialization

        Returns:
            Dictionary of parameter names and values
        """
        return {"name": self.name, "type": self.__class__.__name__}

    @property
    def warmup_period(self) -> int:
        """Return the minimum history required for the position sizer."""

        return 0

    def get_feature_generators(self) -> Sequence["FeatureGeneratorSpec"]:
        """Return feature generators required by the position sizer."""

        return []


class FixedFractionSizer(PositionSizer):
    """
    Fixed fraction position sizer

    Allocates a fixed percentage of the account balance to each position,
    with optional adjustments for signal strength and confidence.
    """

    def __init__(
        self,
        fraction: float = 0.02,
        adjust_for_confidence: bool = True,
        adjust_for_strength: bool = True,
    ):
        """
        Initialize fixed fraction sizer

        Args:
            fraction: Fixed fraction of balance to allocate (0.02 = 2%)
            adjust_for_confidence: Whether to adjust size based on signal confidence
            adjust_for_strength: Whether to adjust size based on signal strength
        """
        super().__init__("fixed_fraction_sizer")

        if not 0.001 <= fraction <= 0.5:  # 0.1% to 50%
            raise ValueError(f"fraction must be between 0.001 and 0.5, got {fraction}")

        self.fraction = fraction
        self.adjust_for_confidence = adjust_for_confidence
        self.adjust_for_strength = adjust_for_strength

    def calculate_size(
        self,
        signal: "Signal",
        balance: float,
        risk_amount: float,
        regime: Optional["RegimeContext"] = None,
    ) -> float:
        """Calculate position size as fixed fraction of balance"""
        self.validate_inputs(balance, risk_amount)

        if signal.direction.value == "hold":
            return 0.0

        # Respect zero risk limit from RiskManager (veto)
        if risk_amount <= 0:
            return 0.0

        # Base position size
        base_size = balance * self.fraction

        # Apply signal-based adjustments
        multiplier = 1.0

        if self.adjust_for_confidence:
            # Scale by confidence (minimum 20% of base size)
            confidence_mult = max(0.2, signal.confidence)
            multiplier *= confidence_mult

        if self.adjust_for_strength:
            # Scale by strength (minimum 20% of base size)
            strength_mult = max(0.2, signal.strength)
            multiplier *= strength_mult

        # Apply regime-based adjustments if available
        if regime is not None:
            regime_mult = self._get_regime_multiplier(regime)
            multiplier *= regime_mult

        # Calculate final size
        final_size = base_size * multiplier

        # Respect risk amount limit
        if risk_amount > 0:
            final_size = min(final_size, risk_amount)

        # Apply bounds checking
        return self.apply_bounds_checking(final_size, balance)

    def _get_regime_multiplier(self, regime: "RegimeContext") -> float:
        """Get position size multiplier based on regime"""
        multiplier = 1.0

        # Reduce size in high volatility
        if hasattr(regime, "volatility") and regime.volatility.value == "high_vol":
            multiplier *= 0.8

        # Reduce size in bear markets
        if hasattr(regime, "trend") and regime.trend.value == "trend_down":
            multiplier *= 0.7

        # Reduce size in range markets
        if hasattr(regime, "trend") and regime.trend.value == "range":
            multiplier *= 0.9

        # Reduce size when regime confidence is low
        if hasattr(regime, "confidence") and regime.confidence < 0.5:
            multiplier *= 0.8

        return max(0.1, multiplier)  # Minimum 10% of base size

    def get_parameters(self) -> dict[str, Any]:
        """Get fixed fraction sizer parameters"""
        params = super().get_parameters()
        params.update(
            {
                "fraction": self.fraction,
                "adjust_for_confidence": self.adjust_for_confidence,
                "adjust_for_strength": self.adjust_for_strength,
            }
        )
        return params


class ConfidenceWeightedSizer(PositionSizer):
    """
    Confidence-weighted position sizer

    Calculates position size primarily based on signal confidence,
    with optional adjustments for signal strength and regime conditions.
    """

    def __init__(self, base_fraction: float = 0.05, min_confidence: float = 0.3):
        """
        Initialize confidence-weighted sizer

        Args:
            base_fraction: Base fraction of balance when confidence is 1.0
            min_confidence: Minimum confidence required for non-zero position
        """
        super().__init__("confidence_weighted_sizer")

        if not 0.001 <= base_fraction <= 0.5:
            raise ValueError(f"base_fraction must be between 0.001 and 0.5, got {base_fraction}")

        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError(f"min_confidence must be between 0.0 and 1.0, got {min_confidence}")

        self.base_fraction = base_fraction
        self.min_confidence = min_confidence

    def calculate_size(
        self,
        signal: "Signal",
        balance: float,
        risk_amount: float,
        regime: Optional["RegimeContext"] = None,
    ) -> float:
        """Calculate position size weighted by signal confidence"""
        self.validate_inputs(balance, risk_amount)

        if signal.direction.value == "hold":
            return 0.0

        # Respect zero risk limit from RiskManager (veto)
        if risk_amount <= 0:
            return 0.0

        # Check minimum confidence threshold
        if signal.confidence < self.min_confidence:
            return 0.0

        # Base size scaled by confidence
        confidence_factor = signal.confidence
        base_size = balance * self.base_fraction * confidence_factor

        # Apply signal strength adjustment
        strength_factor = max(0.3, signal.strength)  # Minimum 30%
        adjusted_size = base_size * strength_factor

        # Apply regime-based adjustments if available
        if regime is not None:
            regime_mult = self._get_regime_multiplier(regime)
            adjusted_size *= regime_mult

        # Respect risk amount limit
        if risk_amount > 0:
            adjusted_size = min(adjusted_size, risk_amount)

        # Apply bounds checking
        return self.apply_bounds_checking(adjusted_size, balance)

    def _get_regime_multiplier(self, regime: "RegimeContext") -> float:
        """Get position size multiplier based on regime"""
        multiplier = 1.0

        # Increase size in bull markets with high confidence
        if (
            hasattr(regime, "trend")
            and regime.trend.value == "trend_up"
            and hasattr(regime, "confidence")
            and regime.confidence > 0.7
        ):
            multiplier *= 1.2

        # Reduce size in bear markets
        if hasattr(regime, "trend") and regime.trend.value == "trend_down":
            multiplier *= 0.6

        # Reduce size in high volatility
        if hasattr(regime, "volatility") and regime.volatility.value == "high_vol":
            multiplier *= 0.8

        # Scale by regime confidence
        if hasattr(regime, "confidence"):
            regime_conf_mult = max(0.5, regime.confidence)
            multiplier *= regime_conf_mult

        return max(0.1, multiplier)

    def get_parameters(self) -> dict[str, Any]:
        """Get confidence-weighted sizer parameters"""
        params = super().get_parameters()
        params.update({"base_fraction": self.base_fraction, "min_confidence": self.min_confidence})
        return params


class KellySizer(PositionSizer):
    """
    Kelly Criterion position sizer

    Calculates optimal position size using the Kelly Criterion based on
    estimated win probability and average win/loss ratios.
    """

    def __init__(
        self,
        win_rate: float = 0.55,
        avg_win: float = 0.02,
        avg_loss: float = 0.015,
        kelly_fraction: float = 0.25,
        lookback_period: int = 100,
    ):
        """
        Initialize Kelly Criterion sizer

        Args:
            win_rate: Estimated win rate (0.55 = 55%)
            avg_win: Average win as fraction (0.02 = 2%)
            avg_loss: Average loss as fraction (0.015 = 1.5%)
            kelly_fraction: Fraction of Kelly to use (0.25 = 25% Kelly)
            lookback_period: Period for updating win/loss statistics
        """
        super().__init__("kelly_sizer")

        if not 0.1 <= win_rate <= 0.9:
            raise ValueError(f"win_rate must be between 0.1 and 0.9, got {win_rate}")

        if avg_win <= 0:
            raise ValueError(f"avg_win must be positive, got {avg_win}")

        if avg_loss <= 0:
            raise ValueError(f"avg_loss must be positive, got {avg_loss}")

        if not 0.01 <= kelly_fraction <= 1.0:
            raise ValueError(f"kelly_fraction must be between 0.01 and 1.0, got {kelly_fraction}")

        self.win_rate = win_rate
        self.avg_win = avg_win
        self.avg_loss = avg_loss
        self.kelly_fraction = kelly_fraction
        self.lookback_period = lookback_period

        # Trade history for updating statistics
        self.trade_history = []

    def calculate_size(
        self,
        signal: "Signal",
        balance: float,
        risk_amount: float,
        regime: Optional["RegimeContext"] = None,
    ) -> float:
        """Calculate position size using Kelly Criterion"""
        self.validate_inputs(balance, risk_amount)

        if signal.direction.value == "hold":
            return 0.0

        # Respect zero risk limit from RiskManager (veto)
        if risk_amount <= 0:
            return 0.0

        # Calculate Kelly percentage
        kelly_pct = self._calculate_kelly_percentage()

        # Apply fractional Kelly
        fractional_kelly = kelly_pct * self.kelly_fraction

        # Adjust for signal confidence and strength
        confidence_adj = max(0.3, signal.confidence)
        strength_adj = max(0.3, signal.strength)

        adjusted_kelly = fractional_kelly * confidence_adj * strength_adj

        # Apply regime adjustments
        if regime is not None:
            regime_mult = self._get_regime_multiplier(regime)
            adjusted_kelly *= regime_mult

        # Calculate position size
        position_size = balance * adjusted_kelly

        # Respect risk amount limit
        if risk_amount > 0:
            position_size = min(position_size, risk_amount)

        # Apply bounds checking (Kelly can suggest large positions)
        return self.apply_bounds_checking(position_size, balance, max_fraction=0.15)

    def _calculate_kelly_percentage(self) -> float:
        """Calculate Kelly percentage using current win/loss statistics"""
        # Kelly formula: f = (bp - q) / b
        # where:
        # f = fraction of capital to wager
        # b = odds received on the wager (avg_win / avg_loss)
        # p = probability of winning (win_rate)
        # q = probability of losing (1 - win_rate)

        p = self.win_rate
        q = 1 - p
        b = self.avg_win / self.avg_loss

        kelly_f = (b * p - q) / b

        # Ensure Kelly percentage is reasonable
        return max(0.0, min(0.5, kelly_f))  # Cap at 50%

    def update_trade_result(self, win: bool, pnl_pct: float) -> None:
        """
        Update trade history with new result

        Args:
            win: Whether the trade was profitable
            pnl_pct: PnL as percentage of position size
        """
        self.trade_history.append({"win": win, "pnl_pct": abs(pnl_pct)})

        # Keep only recent history
        if len(self.trade_history) > self.lookback_period:
            self.trade_history = self.trade_history[-self.lookback_period :]

        # Update statistics if we have enough data
        if len(self.trade_history) >= 20:  # Minimum 20 trades
            self._update_statistics()

    def _update_statistics(self) -> None:
        """Update win rate and average win/loss from trade history"""
        if not self.trade_history:
            return

        wins = [trade for trade in self.trade_history if trade["win"]]
        losses = [trade for trade in self.trade_history if not trade["win"]]

        # Update win rate
        self.win_rate = len(wins) / len(self.trade_history)

        # Update average win/loss
        if wins:
            self.avg_win = np.mean([trade["pnl_pct"] for trade in wins])

        if losses:
            self.avg_loss = np.mean([trade["pnl_pct"] for trade in losses])

    def _get_regime_multiplier(self, regime: "RegimeContext") -> float:
        """Get position size multiplier based on regime"""
        multiplier = 1.0

        # Increase size in strong bull markets
        if (
            hasattr(regime, "trend")
            and regime.trend.value == "trend_up"
            and hasattr(regime, "confidence")
            and regime.confidence > 0.8
        ):
            multiplier *= 1.1

        # Reduce size in bear markets
        if hasattr(regime, "trend") and regime.trend.value == "trend_down":
            multiplier *= 0.7

        # Reduce size in high volatility (Kelly assumes constant volatility)
        if hasattr(regime, "volatility") and regime.volatility.value == "high_vol":
            multiplier *= 0.8

        return max(0.2, multiplier)

    def get_parameters(self) -> dict[str, Any]:
        """Get Kelly sizer parameters"""
        params = super().get_parameters()
        params.update(
            {
                "win_rate": self.win_rate,
                "avg_win": self.avg_win,
                "avg_loss": self.avg_loss,
                "kelly_fraction": self.kelly_fraction,
                "lookback_period": self.lookback_period,
                "trade_count": len(self.trade_history),
            }
        )
        return params


class RegimeAdaptiveSizer(PositionSizer):
    """
    Regime-adaptive position sizer

    Adjusts position sizing based on detected market regimes,
    with different sizing strategies for different market conditions.
    """

    def __init__(
        self,
        base_fraction: float = 0.03,
        regime_multipliers: dict[str, float] | None = None,
        volatility_adjustment: bool = True,
    ):
        """
        Initialize regime-adaptive sizer

        Args:
            base_fraction: Base fraction of balance to allocate (0.03 = 3%)
            regime_multipliers: Custom multipliers for different regimes
            volatility_adjustment: Whether to adjust for volatility within regimes
        """
        super().__init__("regime_adaptive_sizer")

        if not 0.001 <= base_fraction <= 0.2:
            raise ValueError(f"base_fraction must be between 0.001 and 0.2, got {base_fraction}")

        self.base_fraction = base_fraction
        self.volatility_adjustment = volatility_adjustment

        # Default regime multipliers
        default_multipliers = {
            "bull_low_vol": 1.8,  # Aggressive in favorable conditions
            "bull_high_vol": 1.2,  # Moderate in volatile bull market
            "bear_low_vol": 0.4,  # Conservative in bear market
            "bear_high_vol": 0.2,  # Very conservative in volatile bear
            "range_low_vol": 0.8,  # Reduced in sideways markets
            "range_high_vol": 0.3,  # Very reduced in volatile sideways
            "unknown": 0.5,  # Conservative when regime unclear
        }

        # Merge custom multipliers with defaults
        if regime_multipliers:
            self.regime_multipliers = {**default_multipliers, **regime_multipliers}
        else:
            self.regime_multipliers = default_multipliers
        # Validate multipliers
        for regime, multiplier in self.regime_multipliers.items():
            if not 0.1 <= multiplier <= 3.0:
                raise ValueError(
                    f"regime multiplier for {regime} must be between 0.1 and 3.0, got {multiplier}"
                )

    def calculate_size(
        self,
        signal: "Signal",
        balance: float,
        risk_amount: float,
        regime: Optional["RegimeContext"] = None,
    ) -> float:
        """Calculate position size based on regime-specific parameters"""
        self.validate_inputs(balance, risk_amount)

        if signal.direction.value == "hold":
            return 0.0

        # Get regime-specific multiplier
        regime_multiplier = self._get_regime_multiplier(regime)

        # Base position size
        base_size = balance * self.base_fraction * regime_multiplier

        # Apply signal-based adjustments
        confidence_adj = max(0.2, signal.confidence)
        strength_adj = max(0.2, signal.strength)

        adjusted_size = base_size * confidence_adj * strength_adj

        # Apply regime confidence scaling
        if regime is not None and hasattr(regime, "confidence"):
            regime_confidence_adj = max(0.3, regime.confidence)
            adjusted_size *= regime_confidence_adj

        # Apply volatility adjustment within regime
        if self.volatility_adjustment and regime is not None:
            volatility_adj = self._get_volatility_adjustment(regime)
            adjusted_size *= volatility_adj

        # Respect risk amount limit
        if risk_amount > 0:
            adjusted_size = min(adjusted_size, risk_amount)

        # Apply bounds checking with regime-specific limits
        max_fraction = self._get_max_fraction_for_regime(regime)
        return self.apply_bounds_checking(adjusted_size, balance, max_fraction=max_fraction)

    def _get_regime_multiplier(self, regime: Optional["RegimeContext"]) -> float:
        """Get position size multiplier based on current regime"""
        if regime is None:
            return self.regime_multipliers["unknown"]

        # Determine regime key based on trend and volatility
        trend_key = "unknown"
        if hasattr(regime, "trend"):
            if regime.trend.value == "trend_up":
                trend_key = "bull"
            elif regime.trend.value == "trend_down":
                trend_key = "bear"
            else:
                trend_key = "range"

        vol_key = "high_vol"
        if hasattr(regime, "volatility"):
            vol_key = "low_vol" if regime.volatility.value == "low_vol" else "high_vol"

        regime_key = f"{trend_key}_{vol_key}"

        return self.regime_multipliers.get(regime_key, self.regime_multipliers["unknown"])

    def _get_volatility_adjustment(self, regime: "RegimeContext") -> float:
        """Get additional volatility adjustment within regime"""
        if not hasattr(regime, "strength"):
            return 1.0

        # Use regime strength as proxy for volatility stability
        # Higher strength = more stable regime = larger positions
        strength = regime.strength

        if strength >= 0.8:
            return 1.1  # Increase size in very stable regimes
        elif strength >= 0.6:
            return 1.0  # Normal size in stable regimes
        elif strength >= 0.4:
            return 0.9  # Slight reduction in moderately stable regimes
        else:
            return 0.7  # Significant reduction in unstable regimes

    def _get_max_fraction_for_regime(self, regime: Optional["RegimeContext"]) -> float:
        """Get maximum position fraction based on regime"""
        if regime is None:
            return 0.1  # 10% max for unknown regime

        # More aggressive limits in favorable regimes
        if hasattr(regime, "trend") and hasattr(regime, "volatility"):
            if regime.trend.value == "trend_up" and regime.volatility.value == "low_vol":
                return 0.25  # 25% max in bull low vol
            elif regime.trend.value == "trend_down":
                return 0.08  # 8% max in bear market
            elif regime.volatility.value == "high_vol":
                return 0.12  # 12% max in high volatility

        return 0.15  # 15% default max

    def update_regime_multipliers(self, new_multipliers: dict[str, float]) -> None:
        """
        Update regime multipliers (useful for optimization)

        Args:
            new_multipliers: New multiplier values
        """
        # Validate new multipliers
        for regime, multiplier in new_multipliers.items():
            if not 0.1 <= multiplier <= 3.0:
                raise ValueError(
                    f"regime multiplier for {regime} must be between 0.1 and 3.0, got {multiplier}"
                )

        self.regime_multipliers.update(new_multipliers)

    def get_regime_allocation(self, regime: Optional["RegimeContext"]) -> dict[str, float]:
        """
        Get detailed allocation breakdown for a regime

        Args:
            regime: Regime context

        Returns:
            Dictionary with allocation details
        """
        if regime is None:
            return {
                "regime_key": "unknown",
                "regime_multiplier": self.regime_multipliers["unknown"],
                "volatility_adjustment": 1.0,
                "max_fraction": 0.1,
            }

        # Get regime key
        trend_key = "unknown"
        if hasattr(regime, "trend"):
            if regime.trend.value == "trend_up":
                trend_key = "bull"
            elif regime.trend.value == "trend_down":
                trend_key = "bear"
            else:
                trend_key = "range"

        vol_key = "high_vol"
        if hasattr(regime, "volatility"):
            vol_key = "low_vol" if regime.volatility.value == "low_vol" else "high_vol"

        regime_key = f"{trend_key}_{vol_key}"

        return {
            "regime_key": regime_key,
            "regime_multiplier": self._get_regime_multiplier(regime),
            "volatility_adjustment": self._get_volatility_adjustment(regime),
            "max_fraction": self._get_max_fraction_for_regime(regime),
        }

    def get_parameters(self) -> dict[str, Any]:
        """Get regime-adaptive sizer parameters"""
        params = super().get_parameters()
        params.update(
            {
                "base_fraction": self.base_fraction,
                "volatility_adjustment": self.volatility_adjustment,
                "regime_multipliers": self.regime_multipliers,
            }
        )
        return params


# Utility functions for position sizing
def calculate_position_from_risk(
    risk_amount: float, entry_price: float, stop_loss_price: float
) -> float:
    """
    Calculate position size based on risk amount and stop loss distance

    Args:
        risk_amount: Maximum amount to risk
        entry_price: Entry price for the position
        stop_loss_price: Stop loss price

    Returns:
        Position size that risks the specified amount
    """
    if entry_price <= 0 or stop_loss_price <= 0:
        raise ValueError("Prices must be positive")

    if risk_amount <= 0:
        return 0.0

    # Calculate risk per unit
    risk_per_unit = abs(entry_price - stop_loss_price)

    if risk_per_unit == 0:
        return 0.0

    # Calculate position size
    return risk_amount / risk_per_unit


def calculate_risk_from_position(
    position_size: float, entry_price: float, stop_loss_price: float
) -> float:
    """
    Calculate risk amount from position size and stop loss distance

    Args:
        position_size: Size of the position
        entry_price: Entry price for the position
        stop_loss_price: Stop loss price

    Returns:
        Total risk amount for the position
    """
    if position_size <= 0:
        return 0.0

    if entry_price <= 0 or stop_loss_price <= 0:
        raise ValueError("Prices must be positive")

    # Calculate risk per unit
    risk_per_unit = abs(entry_price - stop_loss_price)

    # Calculate total risk
    return position_size * risk_per_unit


def validate_position_size(
    position_size: float, balance: float, min_fraction: float = 0.001, max_fraction: float = 0.2
) -> bool:
    """
    Validate that position size is within acceptable bounds

    Args:
        position_size: Position size to validate
        balance: Available balance
        min_fraction: Minimum position size as fraction of balance
        max_fraction: Maximum position size as fraction of balance

    Returns:
        True if position size is valid, False otherwise
    """
    if position_size < 0:
        return False

    if balance <= 0:
        return False

    position_fraction = position_size / balance

    return min_fraction <= position_fraction <= max_fraction
