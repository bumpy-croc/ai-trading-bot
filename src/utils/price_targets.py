"""Price target calculation utilities for stop loss and take profit levels.

This module provides consistent SL/TP price calculations used throughout
the trading system to prevent divergence between different components.

USAGE:
    from src.utils.price_targets import PriceTargetCalculator

    # Simple percentage-based calculation
    sl = PriceTargetCalculator.stop_loss(entry_price=100.0, pct=0.05, side="long")
    tp = PriceTargetCalculator.take_profit(entry_price=100.0, pct=0.04, side="long")

    # Calculate both at once
    sl, tp = PriceTargetCalculator.sl_tp(
        entry_price=100.0, sl_pct=0.05, tp_pct=0.04, side="long"
    )

    # ATR-based stop loss
    sl = PriceTargetCalculator.stop_loss_atr(
        entry_price=100.0, atr=2.5, multiplier=1.5, side="long"
    )

    # Breakeven price with buffer
    be = PriceTargetCalculator.breakeven(
        entry_price=100.0, buffer_pct=0.002, side="long"
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

SideType = Literal["long", "short"]


@dataclass(frozen=True)
class PriceTargets:
    """Container for stop loss and take profit prices."""

    stop_loss: float
    take_profit: float


class PriceTargetCalculator:
    """Utility for calculating price targets (stop loss, take profit, breakeven).

    Provides consistent calculations used by both backtest and live engines.
    All methods are stateless class methods for easy use without instantiation.

    Side Convention:
        - "long": Buying at entry, stop loss below entry, take profit above entry
        - "short": Selling at entry, stop loss above entry, take profit below entry
    """

    @staticmethod
    def _normalize_side(side: str) -> SideType:
        """Normalize side string to lowercase 'long' or 'short'.

        Args:
            side: Position side (case-insensitive).

        Returns:
            Normalized side string.

        Raises:
            ValueError: If side is not 'long' or 'short'.
        """
        normalized = side.lower().strip()
        if normalized not in ("long", "short"):
            raise ValueError(f"Side must be 'long' or 'short', got: {side}")
        return normalized  # type: ignore[return-value]

    @classmethod
    def stop_loss(
        cls,
        entry_price: float,
        pct: float,
        side: str,
    ) -> float:
        """Calculate stop loss price from percentage.

        Args:
            entry_price: Entry price of the position.
            pct: Stop loss percentage as decimal (e.g., 0.05 for 5%).
            side: Position side ('long' or 'short').

        Returns:
            Stop loss price level.

        Examples:
            >>> PriceTargetCalculator.stop_loss(100.0, 0.05, "long")
            95.0
            >>> PriceTargetCalculator.stop_loss(100.0, 0.05, "short")
            105.0
        """
        normalized_side = cls._normalize_side(side)
        if normalized_side == "long":
            return entry_price * (1 - pct)
        else:  # short
            return entry_price * (1 + pct)

    @classmethod
    def take_profit(
        cls,
        entry_price: float,
        pct: float,
        side: str,
    ) -> float:
        """Calculate take profit price from percentage.

        Args:
            entry_price: Entry price of the position.
            pct: Take profit percentage as decimal (e.g., 0.04 for 4%).
            side: Position side ('long' or 'short').

        Returns:
            Take profit price level.

        Examples:
            >>> PriceTargetCalculator.take_profit(100.0, 0.04, "long")
            104.0
            >>> PriceTargetCalculator.take_profit(100.0, 0.04, "short")
            96.0
        """
        normalized_side = cls._normalize_side(side)
        if normalized_side == "long":
            return entry_price * (1 + pct)
        else:  # short
            return entry_price * (1 - pct)

    @classmethod
    def sl_tp(
        cls,
        entry_price: float,
        sl_pct: float,
        tp_pct: float,
        side: str,
    ) -> tuple[float, float]:
        """Calculate both stop loss and take profit prices.

        Args:
            entry_price: Entry price of the position.
            sl_pct: Stop loss percentage as decimal.
            tp_pct: Take profit percentage as decimal.
            side: Position side ('long' or 'short').

        Returns:
            Tuple of (stop_loss_price, take_profit_price).

        Examples:
            >>> PriceTargetCalculator.sl_tp(100.0, 0.05, 0.04, "long")
            (95.0, 104.0)
            >>> PriceTargetCalculator.sl_tp(100.0, 0.05, 0.04, "short")
            (105.0, 96.0)
        """
        return (
            cls.stop_loss(entry_price, sl_pct, side),
            cls.take_profit(entry_price, tp_pct, side),
        )

    @classmethod
    def sl_tp_targets(
        cls,
        entry_price: float,
        sl_pct: float,
        tp_pct: float,
        side: str,
    ) -> PriceTargets:
        """Calculate both stop loss and take profit as PriceTargets object.

        Args:
            entry_price: Entry price of the position.
            sl_pct: Stop loss percentage as decimal.
            tp_pct: Take profit percentage as decimal.
            side: Position side ('long' or 'short').

        Returns:
            PriceTargets with stop_loss and take_profit attributes.
        """
        sl, tp = cls.sl_tp(entry_price, sl_pct, tp_pct, side)
        return PriceTargets(stop_loss=sl, take_profit=tp)

    @classmethod
    def stop_loss_atr(
        cls,
        entry_price: float,
        atr: float,
        multiplier: float,
        side: str,
    ) -> float:
        """Calculate stop loss price using ATR (Average True Range).

        Args:
            entry_price: Entry price of the position.
            atr: Current ATR value.
            multiplier: ATR multiplier for stop distance.
            side: Position side ('long' or 'short').

        Returns:
            Stop loss price level.

        Examples:
            >>> PriceTargetCalculator.stop_loss_atr(100.0, 2.0, 1.5, "long")
            97.0  # 100 - (2.0 * 1.5)
            >>> PriceTargetCalculator.stop_loss_atr(100.0, 2.0, 1.5, "short")
            103.0  # 100 + (2.0 * 1.5)
        """
        normalized_side = cls._normalize_side(side)
        stop_distance = atr * multiplier

        if normalized_side == "long":
            return entry_price - stop_distance
        else:  # short
            return entry_price + stop_distance

    @classmethod
    def breakeven(
        cls,
        entry_price: float,
        buffer_pct: float,
        side: str,
    ) -> float:
        """Calculate breakeven price with optional buffer.

        Breakeven is the price at which a position can be exited without loss.
        The buffer adds a small profit margin beyond exact breakeven.

        Args:
            entry_price: Entry price of the position.
            buffer_pct: Buffer percentage as decimal (e.g., 0.002 for 0.2%).
            side: Position side ('long' or 'short').

        Returns:
            Breakeven price level.

        Examples:
            >>> PriceTargetCalculator.breakeven(100.0, 0.002, "long")
            100.2  # Just above entry for longs
            >>> PriceTargetCalculator.breakeven(100.0, 0.002, "short")
            99.8   # Just below entry for shorts
        """
        normalized_side = cls._normalize_side(side)
        buffer = max(0.0, buffer_pct)  # Ensure non-negative

        if normalized_side == "long":
            return entry_price * (1 + buffer)
        else:  # short
            return entry_price * (1 - buffer)

    @classmethod
    def pct_from_stop_loss(
        cls,
        entry_price: float,
        stop_loss_price: float,
        side: str,
    ) -> float:
        """Calculate stop loss percentage from absolute price.

        Useful for converting absolute stop prices to percentages.

        Args:
            entry_price: Entry price of the position.
            stop_loss_price: Absolute stop loss price.
            side: Position side ('long' or 'short').

        Returns:
            Stop loss percentage as decimal.

        Examples:
            >>> PriceTargetCalculator.pct_from_stop_loss(100.0, 95.0, "long")
            0.05
            >>> PriceTargetCalculator.pct_from_stop_loss(100.0, 105.0, "short")
            0.05
        """
        normalized_side = cls._normalize_side(side)
        if entry_price == 0:
            return 0.0

        if normalized_side == "long":
            return (entry_price - stop_loss_price) / entry_price
        else:  # short
            return (stop_loss_price - entry_price) / entry_price

    @classmethod
    def pct_from_take_profit(
        cls,
        entry_price: float,
        take_profit_price: float,
        side: str,
    ) -> float:
        """Calculate take profit percentage from absolute price.

        Useful for converting absolute TP prices to percentages.

        Args:
            entry_price: Entry price of the position.
            take_profit_price: Absolute take profit price.
            side: Position side ('long' or 'short').

        Returns:
            Take profit percentage as decimal.

        Examples:
            >>> PriceTargetCalculator.pct_from_take_profit(100.0, 104.0, "long")
            0.04
            >>> PriceTargetCalculator.pct_from_take_profit(100.0, 96.0, "short")
            0.04
        """
        normalized_side = cls._normalize_side(side)
        if entry_price == 0:
            return 0.0

        if normalized_side == "long":
            return (take_profit_price - entry_price) / entry_price
        else:  # short
            return (entry_price - take_profit_price) / entry_price


__all__ = [
    "PriceTargetCalculator",
    "PriceTargets",
]
