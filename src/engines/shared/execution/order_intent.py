"""Order intent definition for execution modeling."""

from __future__ import annotations

from dataclasses import dataclass

from src.data_providers.exchange_interface import OrderSide, OrderType


@dataclass(frozen=True)
class OrderIntent:
    """Represents the desired order and its execution constraints.

    This intent is used by execution models to decide if an order fills
    and at what price. In this system, TAKE_PROFIT intents behave like
    limit orders, while STOP_LOSS intents behave like stop-market orders
    unless a limit_price is explicitly provided.
    """

    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    limit_price: float | None = None
    stop_price: float | None = None
    time_in_force: str = "GTC"
    reduce_only: bool = False
    exit_reason: str | None = None

    def is_limit_order(self) -> bool:
        """Return True when the intent represents a limit-style order."""
        return self.order_type in (OrderType.LIMIT, OrderType.TAKE_PROFIT)

    def is_stop_order(self) -> bool:
        """Return True when the intent represents a stop-style order."""
        return self.order_type == OrderType.STOP_LOSS

    def is_market_order(self) -> bool:
        """Return True when the intent represents a market order."""
        return self.order_type == OrderType.MARKET
