from __future__ import annotations

from dataclasses import dataclass

from src.config.constants import (
    DEFAULT_BREAKEVEN_BUFFER,
    DEFAULT_TRAILING_ACTIVATION_THRESHOLD,
    DEFAULT_TRAILING_DISTANCE_PCT,
)


@dataclass
class TrailingStopPolicy:
    """Configurable trailing stop and breakeven policy.

    Percent inputs are decimals (e.g., 0.015 = 1.5%).
    Exactly one of trailing_distance_pct or atr_multiplier may be provided; if both
    are set, atr-based distance takes precedence when ATR is available.
    """

    activation_threshold: float = DEFAULT_TRAILING_ACTIVATION_THRESHOLD  # 1.5% position gain to start trailing
    trailing_distance_pct: float | None = DEFAULT_TRAILING_DISTANCE_PCT  # 0.5% trail distance
    atr_multiplier: float | None = None  # e.g., 1.5 * ATR
    breakeven_threshold: float | None = None  # if None, breakeven is disabled
    breakeven_buffer: float = DEFAULT_BREAKEVEN_BUFFER  # 0.1% above breakeven for long (below for short)

    def compute_distance(self, price: float, atr: float | None) -> float | None:
        if (
            atr is not None
            and atr > 0
            and self.atr_multiplier is not None
            and self.atr_multiplier > 0
        ):
            return float(atr) * float(self.atr_multiplier)
        if self.trailing_distance_pct is not None and self.trailing_distance_pct > 0:
            return float(price) * float(self.trailing_distance_pct)
        return None

    def _pnl_fraction(
        self, entry_price: float, current_price: float, side: str, position_fraction: float
    ) -> float:
        """Calculate position-level PnL percentage.

        We use position-level PnL instead of portfolio-level (sized) PnL for
        consistent risk management regardless of position size.

        Args:
            entry_price: Entry price for the position
            current_price: Current market price
            side: "long" or "short"
            position_fraction: Position size as fraction of balance (0.0-1.0).
                              Returns 0.0 if <= 0 (no position = no stops needed)

        Returns:
            Position-level PnL as a decimal (e.g., 0.10 for 10% gain)
        """
        # No position = no PnL to protect
        if position_fraction <= 0:
            return 0.0

        if entry_price <= 0:
            return 0.0

        raw = (
            (current_price - entry_price) / entry_price
            if side == "long"
            else (entry_price - current_price) / entry_price
        )
        return raw

    def update_trailing_stop(
        self,
        *,
        side: str,
        entry_price: float,
        current_price: float,
        existing_stop: float | None,
        position_fraction: float,
        atr: float | None = None,
        trailing_activated: bool = False,
        breakeven_triggered: bool = False,
    ) -> tuple[float | None, bool, bool]:
        """Return (new_stop, trailing_activated, breakeven_triggered).

        - Never loosens the stop against the trade.
        - Breakeven move (with buffer) has priority once threshold is met (if enabled).
        - Works for long and short.
        """
        # Compute sized PnL fraction (decimal, e.g., 0.015 for +1.5%)
        pnl_frac = self._pnl_fraction(entry_price, current_price, side, max(0.0, position_fraction))

        # Determine activation
        if not trailing_activated and pnl_frac >= self.activation_threshold:
            trailing_activated = True

        # If not activated, nothing to do
        if not trailing_activated:
            return existing_stop, trailing_activated, breakeven_triggered

        # Breakeven trigger: once met, keep as triggered (only if enabled)
        if (
            not breakeven_triggered
            and self.breakeven_threshold is not None
            and self.breakeven_threshold > 0
            and pnl_frac >= self.breakeven_threshold
        ):
            breakeven_triggered = True

        new_stop: float | None = existing_stop

        # If breakeven triggered, compute breakeven stop with buffer
        if (
            breakeven_triggered
            and self.breakeven_threshold is not None
            and self.breakeven_threshold > 0
        ):
            if side == "long":
                be = entry_price * (1.0 + max(0.0, self.breakeven_buffer))
                new_stop = be if new_stop is None else max(float(new_stop), float(be))
            else:
                be = entry_price * (1.0 - max(0.0, self.breakeven_buffer))
                new_stop = be if new_stop is None else min(float(new_stop), float(be))
            return new_stop, trailing_activated, breakeven_triggered

        # Otherwise, compute trailing distance
        distance = self.compute_distance(current_price, atr)
        if distance is None or distance <= 0:
            return new_stop, trailing_activated, breakeven_triggered

        if side == "long":
            candidate = float(current_price) - float(distance)
            new_stop = candidate if new_stop is None else max(float(new_stop), candidate)
        else:
            candidate = float(current_price) + float(distance)
            new_stop = candidate if new_stop is None else min(float(new_stop), candidate)

        return new_stop, trailing_activated, breakeven_triggered
