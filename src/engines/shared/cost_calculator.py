"""Unified fee and slippage calculation for trading engines.

This module provides consistent cost calculation logic for both
backtesting and live trading engines.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CostResult:
    """Result of a cost calculation.

    Attributes:
        executed_price: The price after applying slippage.
        fee: The fee charged for the trade.
        slippage_cost: The cost due to slippage.
        total_cost: Total cost (fee + slippage as monetary value).
    """

    executed_price: float
    fee: float
    slippage_cost: float

    @property
    def total_cost(self) -> float:
        """Calculate total cost."""
        return self.fee + self.slippage_cost


class CostCalculator:
    """Unified fee and slippage calculation.

    This class provides consistent cost modeling for both backtesting
    and live trading engines.

    Attributes:
        fee_rate: Fee rate as a decimal (e.g., 0.001 for 0.1%).
        slippage_rate: Slippage rate as a decimal (e.g., 0.0005 for 0.05%).
        total_fees_paid: Running total of fees paid.
        total_slippage_cost: Running total of slippage costs.
    """

    def __init__(
        self,
        fee_rate: float = 0.001,
        slippage_rate: float = 0.0005,
    ) -> None:
        """Initialize the cost calculator.

        Args:
            fee_rate: Fee rate as a decimal (default 0.1%).
            slippage_rate: Slippage rate as a decimal (default 0.05%).
        """
        self.fee_rate = fee_rate
        self.slippage_rate = slippage_rate
        self.total_fees_paid: float = 0.0
        self.total_slippage_cost: float = 0.0

    def calculate_entry_costs(
        self,
        price: float,
        notional: float,
        side: str,
    ) -> CostResult:
        """Calculate entry costs including slippage and fees.

        For entries:
        - Long positions get worse entry (higher price)
        - Short positions get worse entry (lower price)

        Args:
            price: The intended entry price.
            notional: The notional value of the trade.
            side: Trade side ('long' or 'short').

        Returns:
            CostResult with executed price, fee, and slippage cost.
        """
        side_lower = side.lower() if isinstance(side, str) else str(side).lower()

        # Apply slippage adversely for entry
        if side_lower == "long":
            executed_price = price * (1 + self.slippage_rate)
        else:  # short
            executed_price = price * (1 - self.slippage_rate)

        # Calculate costs
        slippage_cost = abs(executed_price - price) * (notional / price)
        fee = notional * self.fee_rate

        # Track totals
        self.total_fees_paid += fee
        self.total_slippage_cost += slippage_cost

        return CostResult(
            executed_price=executed_price,
            fee=fee,
            slippage_cost=slippage_cost,
        )

    def calculate_exit_costs(
        self,
        price: float,
        notional: float,
        side: str,
    ) -> CostResult:
        """Calculate exit costs including slippage and fees.

        For exits:
        - Long positions closing get worse exit (lower price)
        - Short positions closing get worse exit (higher price)

        Args:
            price: The intended exit price.
            notional: The notional value of the trade (at exit).
            side: Trade side ('long' or 'short').

        Returns:
            CostResult with executed price, fee, and slippage cost.
        """
        side_lower = side.lower() if isinstance(side, str) else str(side).lower()

        # Apply slippage adversely for exit (opposite of entry)
        if side_lower == "long":
            executed_price = price * (1 - self.slippage_rate)
        else:  # short
            executed_price = price * (1 + self.slippage_rate)

        # Calculate costs
        slippage_cost = abs(executed_price - price) * (notional / price)
        fee = notional * self.fee_rate

        # Track totals
        self.total_fees_paid += fee
        self.total_slippage_cost += slippage_cost

        return CostResult(
            executed_price=executed_price,
            fee=fee,
            slippage_cost=slippage_cost,
        )

    def calculate_fee(self, notional: float) -> float:
        """Calculate just the fee for a given notional value.

        Args:
            notional: The notional value of the trade.

        Returns:
            The fee amount.
        """
        return notional * self.fee_rate

    def reset_totals(self) -> None:
        """Reset the running totals of fees and slippage."""
        self.total_fees_paid = 0.0
        self.total_slippage_cost = 0.0

    def get_settings(self) -> dict:
        """Get the current cost calculator settings.

        Returns:
            Dictionary with fee_rate and slippage_rate.
        """
        return {
            "fee_rate": self.fee_rate,
            "slippage_rate": self.slippage_rate,
        }


__all__ = [
    "CostCalculator",
    "CostResult",
]
