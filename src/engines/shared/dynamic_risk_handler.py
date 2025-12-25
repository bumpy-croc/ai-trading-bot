"""Unified dynamic risk adjustment logic for trading engines.

This module provides consistent dynamic risk adjustment logic used by
both backtesting and live trading engines to ensure parity in risk management.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.position_management.dynamic_risk import DynamicRiskManager

logger = logging.getLogger(__name__)


@dataclass
class DynamicRiskAdjustment:
    """Record of a dynamic risk adjustment.

    Attributes:
        timestamp: When the adjustment was made.
        position_size_factor: Factor applied to position size (1.0 = no change).
        stop_loss_tightening: Factor for tightening stop loss.
        daily_risk_factor: Daily risk factor applied.
        primary_reason: Main reason for the adjustment.
        current_drawdown: Current drawdown percentage.
        balance: Account balance at adjustment time.
        peak_balance: Peak balance at adjustment time.
        original_size: Original position size before adjustment.
        adjusted_size: Position size after adjustment.
    """

    timestamp: datetime
    position_size_factor: float
    stop_loss_tightening: float
    daily_risk_factor: float
    primary_reason: str
    current_drawdown: float | None
    balance: float
    peak_balance: float
    original_size: float
    adjusted_size: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "timestamp": self.timestamp,
            "position_size_factor": self.position_size_factor,
            "stop_loss_tightening": self.stop_loss_tightening,
            "daily_risk_factor": self.daily_risk_factor,
            "primary_reason": self.primary_reason,
            "current_drawdown": self.current_drawdown,
            "balance": self.balance,
            "peak_balance": self.peak_balance,
            "original_size": self.original_size,
            "adjusted_size": self.adjusted_size,
        }


class DynamicRiskHandler:
    """Unified dynamic risk adjustment handler.

    This class provides consistent dynamic risk adjustment logic that is
    used by both the backtesting and live trading engines.

    Attributes:
        dynamic_risk_manager: Optional manager for risk adjustments.
        adjustments: List of tracked adjustments.
        significance_threshold: Threshold for logging significant adjustments.
    """

    def __init__(
        self,
        dynamic_risk_manager: DynamicRiskManager | None = None,
        significance_threshold: float = 0.1,
    ) -> None:
        """Initialize the dynamic risk handler.

        Args:
            dynamic_risk_manager: Manager for calculating risk adjustments.
            significance_threshold: Threshold for tracking adjustments (default 10%).
        """
        self.dynamic_risk_manager = dynamic_risk_manager
        self.significance_threshold = significance_threshold
        self._adjustments: list[DynamicRiskAdjustment] = []

    def set_manager(self, manager: DynamicRiskManager | None) -> None:
        """Update the dynamic risk manager.

        Args:
            manager: New manager to use, or None to disable.
        """
        self.dynamic_risk_manager = manager

    def apply_dynamic_risk(
        self,
        original_size: float,
        current_time: datetime,
        balance: float,
        peak_balance: float,
        trading_session_id: int | None = None,
    ) -> float:
        """Apply dynamic risk adjustments to position size.

        Reduces position size during drawdown or adverse market conditions
        to preserve capital and prevent excessive losses.

        Args:
            original_size: Original position size fraction.
            current_time: Current timestamp.
            balance: Current account balance.
            peak_balance: Peak account balance.
            trading_session_id: Session ID for logging.

        Returns:
            Adjusted position size fraction.
        """
        if self.dynamic_risk_manager is None:
            return original_size

        try:
            adjustments = self.dynamic_risk_manager.calculate_dynamic_risk_adjustments(
                current_balance=balance,
                peak_balance=peak_balance,
                session_id=trading_session_id,
            )

            adjusted_size = original_size * adjustments.position_size_factor

            # Track significant adjustments for post-trade analysis
            if abs(adjustments.position_size_factor - 1.0) > self.significance_threshold:
                logger.debug(
                    "Dynamic risk adjustment at %s: size factor=%.2f, reason=%s",
                    current_time,
                    adjustments.position_size_factor,
                    adjustments.primary_reason,
                )

                adjustment = DynamicRiskAdjustment(
                    timestamp=current_time,
                    position_size_factor=adjustments.position_size_factor,
                    stop_loss_tightening=adjustments.stop_loss_tightening,
                    daily_risk_factor=adjustments.daily_risk_factor,
                    primary_reason=adjustments.primary_reason,
                    current_drawdown=adjustments.adjustment_details.get(
                        "current_drawdown"
                    ),
                    balance=balance,
                    peak_balance=peak_balance,
                    original_size=original_size,
                    adjusted_size=adjusted_size,
                )
                self._adjustments.append(adjustment)

            return adjusted_size

        except (AttributeError, ValueError, KeyError, TypeError) as e:
            logger.warning("Failed to apply dynamic risk adjustment: %s", e)
            return original_size

    def get_adjustments(self, clear: bool = True) -> list[dict[str, Any]]:
        """Get tracked dynamic risk adjustments.

        Args:
            clear: Whether to clear the adjustments after retrieval.

        Returns:
            List of adjustment records as dictionaries.
        """
        adjustments = [adj.to_dict() for adj in self._adjustments]
        if clear:
            self._adjustments.clear()
        return adjustments

    def get_adjustment_objects(self, clear: bool = True) -> list[DynamicRiskAdjustment]:
        """Get tracked adjustments as objects.

        Args:
            clear: Whether to clear after retrieval.

        Returns:
            List of DynamicRiskAdjustment objects.
        """
        adjustments = self._adjustments.copy()
        if clear:
            self._adjustments.clear()
        return adjustments

    def clear_adjustments(self) -> None:
        """Clear all tracked adjustments."""
        self._adjustments.clear()

    @property
    def has_adjustments(self) -> bool:
        """Check if there are any tracked adjustments."""
        return len(self._adjustments) > 0

    @property
    def adjustment_count(self) -> int:
        """Get the number of tracked adjustments."""
        return len(self._adjustments)


__all__ = [
    "DynamicRiskHandler",
    "DynamicRiskAdjustment",
]
