"""Hand-written exchange stub for restart-adoption tests (#683).

A plain class, not a MagicMock: an unexpected lookup raises AttributeError instead of
returning a truthy auto-attribute that reads as "the exchange holds 1.0".
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from src.data_providers.exchange_interface import Order, OrderSide, OrderStatus, OrderType


def make_stop_order(
    order_id: str,
    symbol: str,
    status: OrderStatus,
    *,
    filled_quantity: float = 0.0,
    quantity: float = 1.0,
) -> Order:
    """A stop-loss order as the exchange reports it."""
    now = datetime.now(UTC)
    return Order(
        order_id=order_id,
        symbol=symbol,
        side=OrderSide.SELL,
        order_type=OrderType.STOP_LOSS,
        quantity=quantity,
        price=None,
        status=status,
        filled_quantity=filled_quantity,
        average_price=None,
        commission=0.0,
        commission_asset="USDT",
        create_time=now,
        update_time=now,
    )


class StubMarginExchange:
    """Cross-margin holdings per base asset, known orders, switchable lookup failure."""

    is_margin_mode = True

    def __init__(
        self,
        net_assets: dict[str, float] | None = None,
        symbol_info: dict[str, Any] | None = None,
        orders: dict[str, Order] | None = None,
    ) -> None:
        self.net_assets = dict(net_assets or {})
        self.symbol_info = symbol_info
        self.orders = dict(orders or {})
        self.fail_lookups = False
        self.cancelled_orders: list[tuple[str, str]] = []

    def get_margin_account_asset(self, asset: str) -> dict[str, str] | None:
        if self.fail_lookups:
            return None
        held = self.net_assets.get(asset, 0.0)
        return {"asset": asset, "netAsset": str(held), "borrowed": "0", "free": str(held)}

    def get_margin_borrowed(self, asset: str) -> float | None:
        return None if self.fail_lookups else 0.0

    def get_symbol_info(self, symbol: str) -> dict[str, Any] | None:
        return self.symbol_info

    def get_order(self, order_id: str, symbol: str) -> Order | None:
        return self.orders.get(order_id)

    def cancel_order(self, order_id: str, symbol: str) -> bool:
        self.cancelled_orders.append((order_id, symbol))
        return True
