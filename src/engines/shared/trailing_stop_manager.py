"""Unified trailing stop management for trading engines.

This module provides consistent trailing stop logic for both
backtesting and live trading engines.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pandas as pd

from src.config.constants import DEFAULT_ATR_PERIOD, DEFAULT_BREAKEVEN_BUFFER
from src.engines.shared.models import normalize_side
from src.utils.price_targets import PriceTargetCalculator

if TYPE_CHECKING:
    from src.engines.shared.models import BasePosition
    from src.position_management.trailing_stops import TrailingStopPolicy

logger = logging.getLogger(__name__)


@dataclass
class TrailingStopUpdate:
    """Result of a trailing stop update check.

    Attributes:
        updated: Whether the trailing stop was updated.
        new_stop_price: New stop price if updated.
        log_message: Optional log message describing the update.
        breakeven_triggered: Whether breakeven was just triggered.
        trailing_activated: Whether trailing stop was just activated.
    """

    updated: bool
    new_stop_price: float | None = None
    log_message: str | None = None
    breakeven_triggered: bool = False
    trailing_activated: bool = False


class TrailingStopManager:
    """Unified trailing stop management.

    This class provides consistent trailing stop logic for both
    backtesting and live trading engines.

    The trailing stop works in two phases:
    1. Breakeven phase: Once profit reaches breakeven_threshold, move stop to entry + buffer
    2. Trailing phase: Once profit reaches activation_threshold, trail the stop

    Attributes:
        policy: The trailing stop policy to apply.
    """

    def __init__(self, policy: TrailingStopPolicy | None) -> None:
        """Initialize the trailing stop manager.

        Args:
            policy: Trailing stop policy to apply, or None to disable.
        """
        self.policy = policy

    def update(
        self,
        position: BasePosition | Any,
        current_price: float,
        df: pd.DataFrame | None = None,
        index: int | None = None,
    ) -> TrailingStopUpdate:
        """Update trailing stop based on current price.

        Args:
            position: The current position.
            current_price: Current market price.
            df: DataFrame with market data (for ATR calculation).
            index: Current candle index (for ATR calculation).

        Returns:
            TrailingStopUpdate with the result of the check.
        """
        if self.policy is None:
            return TrailingStopUpdate(updated=False)

        if position is None:
            return TrailingStopUpdate(updated=False)

        # Get position details
        entry_price = position.entry_price
        side = normalize_side(getattr(position, "side", None))
        current_stop = getattr(position, "stop_loss", None) or getattr(
            position, "trailing_stop_price", None
        )

        # Validate entry_price to prevent division by zero
        if entry_price <= 0 or not math.isfinite(entry_price):
            logger.error(
                "Invalid entry_price %.8f for trailing stop calculation - cannot update",
                entry_price,
            )
            return TrailingStopUpdate(updated=False)

        # Validate current_price to prevent corrupt calculations
        if current_price <= 0 or not math.isfinite(current_price):
            logger.error(
                "Invalid current_price %.8f for trailing stop calculation - cannot update",
                current_price,
            )
            return TrailingStopUpdate(updated=False)

        # Calculate position-level PnL percentage (not sized by position fraction)
        # This provides consistent risk management regardless of position size
        if side == "long":
            pnl_pct = (current_price - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - current_price) / entry_price

        # Check breakeven first
        breakeven_triggered = False
        if not getattr(position, "breakeven_triggered", False):
            breakeven_result = self._check_breakeven(
                position, current_price, pnl_pct, side, entry_price
            )
            if breakeven_result.updated:
                return breakeven_result

        # Check trailing stop activation
        trailing_result = self._check_trailing_activation(
            position, current_price, pnl_pct, side, entry_price, current_stop, df, index
        )

        return trailing_result

    def _check_breakeven(
        self,
        position: Any,
        current_price: float,
        pnl_pct: float,
        side: str,
        entry_price: float,
    ) -> TrailingStopUpdate:
        """Check and apply breakeven trigger.

        Args:
            position: The current position.
            current_price: Current market price.
            pnl_pct: Current PnL percentage.
            side: Position side ('long' or 'short').
            entry_price: Position entry price.

        Returns:
            TrailingStopUpdate with result (does NOT modify position).
        """
        breakeven_threshold = getattr(self.policy, "breakeven_threshold", None)
        breakeven_buffer = getattr(self.policy, "breakeven_buffer", DEFAULT_BREAKEVEN_BUFFER)

        if breakeven_threshold is None or breakeven_threshold <= 0:
            return TrailingStopUpdate(updated=False)

        if pnl_pct >= breakeven_threshold:
            # Calculate breakeven stop price using shared calculator
            new_stop = PriceTargetCalculator.breakeven(
                entry_price=entry_price,
                buffer_pct=breakeven_buffer,
                side=side,
            )

            # Return result without modifying position - caller is responsible
            # for applying the update via position_tracker
            return TrailingStopUpdate(
                updated=True,
                new_stop_price=new_stop,
                log_message=f"Breakeven triggered at {pnl_pct:.2%}, stop moved to {new_stop:.2f}",
                breakeven_triggered=True,
            )

        return TrailingStopUpdate(updated=False)

    def _check_trailing_activation(
        self,
        position: Any,
        current_price: float,
        pnl_pct: float,
        side: str,
        entry_price: float,
        current_stop: float | None,
        df: pd.DataFrame | None,
        index: int | None,
    ) -> TrailingStopUpdate:
        """Check and apply trailing stop activation and updates.

        Args:
            position: The current position.
            current_price: Current market price.
            pnl_pct: Current PnL percentage.
            side: Position side ('long' or 'short').
            entry_price: Position entry price.
            current_stop: Current stop loss price.
            df: DataFrame with market data (for ATR calculation).
            index: Current candle index (for ATR calculation).

        Returns:
            TrailingStopUpdate with result (does NOT modify position).
        """
        activation_threshold = getattr(self.policy, "activation_threshold", None)
        trailing_distance_pct = getattr(self.policy, "trailing_distance_pct", None)
        atr_multiplier = getattr(self.policy, "atr_multiplier", None)

        if activation_threshold is None:
            return TrailingStopUpdate(updated=False)

        # Check if trailing should be activated
        is_activated = getattr(position, "trailing_stop_activated", False)
        just_activated = False

        if not is_activated and pnl_pct >= activation_threshold:
            # Don't modify position directly - just track that we need to activate
            is_activated = True
            just_activated = True

        if not is_activated:
            return TrailingStopUpdate(updated=False)

        # Calculate trailing distance
        trailing_distance = self._calculate_trailing_distance(
            current_price, trailing_distance_pct, atr_multiplier, df, index
        )

        if trailing_distance is None:
            return TrailingStopUpdate(
                updated=just_activated,
                trailing_activated=just_activated,
            )

        # Calculate new stop price
        if side == "long":
            new_stop = current_price - trailing_distance
            should_update = current_stop is None or new_stop > current_stop
        else:
            new_stop = current_price + trailing_distance
            should_update = current_stop is None or new_stop < current_stop

        if should_update:
            # Return result without modifying position - caller is responsible
            # for applying the update via position_tracker
            return TrailingStopUpdate(
                updated=True,
                new_stop_price=new_stop,
                log_message=f"Trailing stop {'activated and ' if just_activated else ''}moved to {new_stop:.2f}",
                trailing_activated=just_activated,
            )

        return TrailingStopUpdate(
            updated=just_activated,
            trailing_activated=just_activated,
        )

    def _calculate_trailing_distance(
        self,
        current_price: float,
        trailing_distance_pct: float | None,
        atr_multiplier: float | None,
        df: pd.DataFrame | None,
        index: int | None,
    ) -> float | None:
        """Calculate the trailing stop distance.

        Args:
            current_price: Current market price.
            trailing_distance_pct: Distance as percentage of price.
            atr_multiplier: ATR multiplier for ATR-based trailing.
            df: DataFrame with market data.
            index: Current candle index.

        Returns:
            Trailing distance in price units, or None if cannot be calculated.
        """
        # Try percentage-based trailing first
        if trailing_distance_pct is not None:
            return current_price * trailing_distance_pct

        # Try ATR-based trailing
        if atr_multiplier is not None and df is not None and index is not None:
            atr = self._get_atr(df, index)
            if atr is not None and atr > 0:
                return atr * atr_multiplier

        return None

    def _get_atr(
        self, df: pd.DataFrame, index: int, period: int = DEFAULT_ATR_PERIOD
    ) -> float | None:
        """Get ATR value from DataFrame.

        Args:
            df: DataFrame with OHLCV data.
            index: Current candle index.
            period: ATR period.

        Returns:
            ATR value or None if not available.
        """
        # Check if ATR is pre-calculated
        if "atr" in df.columns:
            try:
                return float(df["atr"].iloc[index])
            except (IndexError, ValueError, KeyError):
                pass

        if "ATR" in df.columns:
            try:
                return float(df["ATR"].iloc[index])
            except (IndexError, ValueError, KeyError):
                pass

        # Calculate ATR if not available
        try:
            if index < period:
                return None

            high = df["high"].iloc[max(0, index - period) : index + 1]
            low = df["low"].iloc[max(0, index - period) : index + 1]
            close = df["close"].iloc[max(0, index - period) : index + 1]

            tr = pd.concat(
                [
                    high - low,
                    (high - close.shift(1)).abs(),
                    (low - close.shift(1)).abs(),
                ],
                axis=1,
            ).max(axis=1)

            return float(tr.mean())
        except Exception:
            return None


__all__ = [
    "TrailingStopManager",
    "TrailingStopUpdate",
]
