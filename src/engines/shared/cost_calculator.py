"""Unified fee and slippage calculation for trading engines.

This module provides consistent cost calculation logic for both
backtesting and live trading engines.
"""

from __future__ import annotations

from dataclasses import dataclass

LIQUIDITY_MAKER = "maker"
LIQUIDITY_TAKER = "taker"
ZERO_VALUE = 0.0


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
        maker_fee_rate: Fee rate applied to maker liquidity (defaults to fee_rate).
        slippage_rate: Slippage rate as a decimal (e.g., 0.0005 for 0.05%).
        total_fees_paid: Running total of fees paid.
        total_slippage_cost: Running total of slippage costs.
    """

    def __init__(
        self,
        fee_rate: float = 0.001,
        slippage_rate: float = 0.0005,
        maker_fee_rate: float | None = None,
    ) -> None:
        """Initialize the cost calculator.

        Args:
            fee_rate: Fee rate as a decimal (default 0.1%).
            slippage_rate: Slippage rate as a decimal (default 0.05%).
            maker_fee_rate: Fee rate for maker liquidity (defaults to fee_rate).
        """
        self.fee_rate = fee_rate
        self.maker_fee_rate = fee_rate if maker_fee_rate is None else maker_fee_rate
        self.slippage_rate = slippage_rate
        self.total_fees_paid: float = ZERO_VALUE
        self.total_slippage_cost: float = ZERO_VALUE

    def _resolve_fee_rate(self, liquidity: str | None) -> float:
        """Resolve the fee rate based on liquidity type."""
        if liquidity == LIQUIDITY_MAKER:
            return self.maker_fee_rate
        if liquidity == LIQUIDITY_TAKER:
            return self.fee_rate
        return self.fee_rate

    def calculate_entry_costs(
        self,
        price: float,
        notional: float,
        side: str,
        liquidity: str | None = None,
    ) -> CostResult:
        """Calculate entry costs including slippage and fees.

        For entries:
        - Long positions get worse entry (higher price)
        - Short positions get worse entry (lower price)

        Args:
            price: The intended entry price.
            notional: The notional value of the trade.
            side: Trade side ('long' or 'short').
            liquidity: Liquidity type ('maker' or 'taker').

        Returns:
            CostResult with executed price, fee, and slippage cost.

        Raises:
            ValueError: If price <= 0, notional < 0, or side is invalid.
        """
        # Input validation
        if price <= 0:
            raise ValueError(f"Price must be positive, got {price}")
        if notional < 0:
            raise ValueError(f"Notional must be non-negative, got {notional}")

        side_lower = side.lower() if isinstance(side, str) else str(side).lower()
        if side_lower not in ("long", "short"):
            raise ValueError(f"Side must be 'long' or 'short', got '{side}'")

        # Apply slippage adversely for entry unless maker liquidity is specified
        if liquidity == LIQUIDITY_MAKER:
            executed_price = price
            slippage_cost = ZERO_VALUE
        else:
            if side_lower == "long":
                executed_price = price * (1 + self.slippage_rate)
            else:  # short
                executed_price = price * (1 - self.slippage_rate)
            slippage_cost = abs(executed_price - price) * (notional / price)

        # Calculate costs
        fee_rate = self._resolve_fee_rate(liquidity)
        fee = notional * fee_rate

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
        liquidity: str | None = None,
        apply_slippage: bool = True,
    ) -> CostResult:
        """Calculate exit costs including slippage and fees.

        For exits:
        - Long positions closing get worse exit (lower price)
        - Short positions closing get worse exit (higher price)

        Args:
            price: The intended exit price.
            notional: The notional value of the trade (at exit).
            side: Trade side ('long' or 'short').
            liquidity: Liquidity type ('maker' or 'taker').
            apply_slippage: When False, slippage is suppressed.

        Returns:
            CostResult with executed price, fee, and slippage cost.

        Raises:
            ValueError: If price <= 0, notional < 0, or side is invalid.
        """
        # Input validation
        if price <= 0:
            raise ValueError(f"Price must be positive, got {price}")
        if notional < 0:
            raise ValueError(f"Notional must be non-negative, got {notional}")

        side_lower = side.lower() if isinstance(side, str) else str(side).lower()
        if side_lower not in ("long", "short"):
            raise ValueError(f"Side must be 'long' or 'short', got '{side}'")

        # Apply slippage adversely for exit unless maker liquidity is specified
        if not apply_slippage or liquidity == LIQUIDITY_MAKER:
            executed_price = price
            slippage_cost = ZERO_VALUE
        else:
            if side_lower == "long":
                executed_price = price * (1 - self.slippage_rate)
            else:  # short
                executed_price = price * (1 + self.slippage_rate)
            slippage_cost = abs(executed_price - price) * (notional / price)

        # Calculate costs
        fee_rate = self._resolve_fee_rate(liquidity)
        fee = notional * fee_rate

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
        self.total_fees_paid = ZERO_VALUE
        self.total_slippage_cost = ZERO_VALUE

    def get_settings(self) -> dict:
        """Get the current cost calculator settings.

        Returns:
            Dictionary with fee_rate and slippage_rate.
        """
        return {
            "fee_rate": self.fee_rate,
            "maker_fee_rate": self.maker_fee_rate,
            "slippage_rate": self.slippage_rate,
        }


__all__ = [
    "CostCalculator",
    "CostResult",
]
