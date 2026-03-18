"""
Leverage Manager - Regime-Based Dynamic Leverage

Determines optimal leverage multiplier based on market regime state.
Amplifies returns during confirmed trends and reduces exposure during
uncertain or bearish markets.

Key features:
- Regime-to-leverage mapping (bull/bear/range x vol)
- Smooth exponential transitions between leverage levels
- Conviction scaling based on regime duration
- Configurable caps and thresholds
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

from src.config.constants import (
    DEFAULT_LEVERAGE_DECAY_RATE,
    DEFAULT_MAX_LEVERAGE,
    DEFAULT_MIN_REGIME_BARS,
)
from src.strategies.components.regime_context import RegimeContext, TrendLabel, VolLabel

logger = logging.getLogger(__name__)


@dataclass
class LeverageState:
    """Tracks current leverage state for smooth transitions."""

    current_leverage: float = 1.0
    target_leverage: float = 1.0
    last_trend: TrendLabel | None = None
    last_volatility: VolLabel | None = None


# Default leverage targets for each regime combination
DEFAULT_LEVERAGE_MAP: dict[tuple[TrendLabel, VolLabel], float] = {
    (TrendLabel.TREND_UP, VolLabel.LOW): 2.5,  # Bull: strong leverage
    (TrendLabel.TREND_UP, VolLabel.HIGH): 1.75,  # Mild bull: moderate leverage
    (TrendLabel.RANGE, VolLabel.LOW): 1.0,  # Range: no leverage
    (TrendLabel.RANGE, VolLabel.HIGH): 1.0,  # Range: no leverage
    (TrendLabel.TREND_DOWN, VolLabel.LOW): 0.5,  # Mild bear: reduced exposure
    (TrendLabel.TREND_DOWN, VolLabel.HIGH): 0.0,  # Bear: cash/defensive
}


class LeverageManager:
    """Determines optimal leverage based on regime state.

    Uses exponential smoothing to transition between leverage levels and
    increases conviction (and leverage) the longer a regime persists.

    Args:
        max_leverage: Maximum allowed leverage multiplier (safety cap).
        decay_rate: Exponential decay rate for smooth transitions (0-1).
            Higher values produce faster transitions.
        min_regime_bars: Minimum bars in a regime before conviction scaling
            begins increasing leverage toward target.
        leverage_map: Override default regime-to-leverage mapping.
            Keys are (TrendLabel, VolLabel) tuples, values are target leverage.

    Thread-Safety:
        This class is NOT thread-safe. It maintains mutable state that is
        modified on every call to get_leverage_multiplier(). Designed for
        single-threaded use per instance. In the standard architecture,
        each strategy owns a separate LeverageManager and runs in a single
        trading thread. If shared across threads, callers must synchronize
        with external locking.
    """

    # Maximum excess bars for conviction scaling plateau - prevents
    # over-leveraging in extended regimes
    MAX_REGIME_EXCESS_BARS = 100

    def __init__(
        self,
        max_leverage: float = DEFAULT_MAX_LEVERAGE,
        decay_rate: float = DEFAULT_LEVERAGE_DECAY_RATE,
        min_regime_bars: int = DEFAULT_MIN_REGIME_BARS,
        leverage_map: dict[tuple[TrendLabel, VolLabel], float] | None = None,
    ) -> None:
        if not math.isfinite(max_leverage) or max_leverage <= 0:
            raise ValueError(f"max_leverage must be finite and positive, got {max_leverage}")
        if not math.isfinite(decay_rate) or not 0.0 < decay_rate <= 1.0:
            raise ValueError(f"decay_rate must be finite and in (0, 1], got {decay_rate}")
        if min_regime_bars < 0:
            raise ValueError(f"min_regime_bars must be non-negative, got {min_regime_bars}")

        self.max_leverage = max_leverage
        self.decay_rate = decay_rate
        self.min_regime_bars = min_regime_bars
        self.leverage_map = dict(leverage_map or DEFAULT_LEVERAGE_MAP)
        self._state = LeverageState()

        # Validate and clamp leverage map values
        for key, value in self.leverage_map.items():
            if value < 0.0:
                raise ValueError(f"leverage_map values must be non-negative, got {value} for {key}")
            if not math.isfinite(value):
                raise ValueError(f"leverage_map values must be finite, got {value} for {key}")
            if value > self.max_leverage:
                self.leverage_map[key] = self.max_leverage

    def get_leverage_multiplier(self, regime: RegimeContext) -> float:
        """Calculate the current leverage multiplier for a given regime.

        Applies three layers:
        1. Look up base target from regime map
        2. Scale by conviction (regime duration and confidence)
        3. Smooth transition from current level via exponential decay

        Args:
            regime: Current market regime context.

        Returns:
            Leverage multiplier in [0.0, max_leverage].
        """
        # Look up base target for this regime combination
        raw_target = self.leverage_map.get((regime.trend, regime.volatility), 1.0)

        # Apply conviction scaling based on regime duration
        conviction = self._compute_conviction(regime)
        scaled_target = self._apply_conviction(raw_target, conviction)

        # Clamp to bounds
        scaled_target = max(0.0, min(scaled_target, self.max_leverage))

        # Track last regime for transition detection
        self._state.last_trend = regime.trend
        self._state.last_volatility = regime.volatility

        # Update target
        self._state.target_leverage = scaled_target

        # Smooth transition via exponential decay
        self._state.current_leverage = self._smooth_transition(
            self._state.current_leverage,
            self._state.target_leverage,
        )

        return self._state.current_leverage

    def reset(self) -> None:
        """Reset internal state to defaults."""
        self._state = LeverageState()

    @property
    def current_leverage(self) -> float:
        """Return the current smoothed leverage value."""
        return self._state.current_leverage

    @property
    def target_leverage(self) -> float:
        """Return the current target leverage value."""
        return self._state.target_leverage

    def get_parameters(self) -> dict[str, object]:
        """Return configuration parameters for logging."""
        return {
            "max_leverage": self.max_leverage,
            "decay_rate": self.decay_rate,
            "min_regime_bars": self.min_regime_bars,
            "leverage_map": {f"{k[0].value}_{k[1].value}": v for k, v in self.leverage_map.items()},
        }

    def _compute_conviction(self, regime: RegimeContext) -> float:
        """Compute conviction score from regime duration and confidence.

        Conviction ramps from 0.0 to 1.0 as the regime persists beyond
        min_regime_bars. Uses a logarithmic curve so conviction builds
        quickly at first then plateaus.

        Args:
            regime: Current regime context with duration and confidence.

        Returns:
            Conviction in [0.0, 1.0].
        """
        duration = regime.duration
        if duration < self.min_regime_bars:
            # Below minimum duration: scale linearly from 0 to base
            base_conviction = duration / max(self.min_regime_bars, 1)
        else:
            # Beyond minimum: logarithmic ramp toward 1.0
            excess = duration - self.min_regime_bars
            # log2(x+2)/log2(max+2) gives a nice curve from ~0.5 to 1.0
            max_excess = self.MAX_REGIME_EXCESS_BARS
            base_conviction = 0.5 + 0.5 * math.log2(excess + 2) / math.log2(max_excess + 2)
            base_conviction = min(base_conviction, 1.0)

        # Weight by regime confidence
        confidence_weight = max(0.0, min(1.0, regime.confidence))
        return base_conviction * confidence_weight

    def _apply_conviction(self, raw_target: float, conviction: float) -> float:
        """Scale raw leverage target toward neutral (1.0) based on conviction.

        Low conviction pulls leverage toward 1.0 (neutral).
        High conviction lets leverage reach its full target.

        Args:
            raw_target: Base leverage from regime map.
            conviction: Conviction score in [0.0, 1.0].

        Returns:
            Conviction-adjusted leverage target.
        """
        neutral = 1.0
        return neutral + (raw_target - neutral) * conviction

    def _smooth_transition(self, current: float, target: float) -> float:
        """Apply exponential smoothing between current and target leverage.

        Args:
            current: Current leverage level.
            target: Target leverage level.

        Returns:
            Smoothed leverage value.
        """
        return current + self.decay_rate * (target - current)
