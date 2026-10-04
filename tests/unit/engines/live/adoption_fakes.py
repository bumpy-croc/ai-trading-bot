"""Hand-written exchange stub for restart-adoption tests (#683).

A plain class, not a MagicMock: an unexpected lookup raises AttributeError instead of
returning a truthy auto-attribute that reads as "the exchange holds 1.0".
"""

from __future__ import annotations

from typing import Any


class StubMarginExchange:
    """Cross-margin holdings per base asset, with switchable lookup failure."""

    is_margin_mode = True

    def __init__(
        self,
        net_assets: dict[str, float] | None = None,
        symbol_info: dict[str, Any] | None = None,
    ) -> None:
        self.net_assets = dict(net_assets or {})
        self.symbol_info = symbol_info
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

    def cancel_order(self, order_id: str, symbol: str) -> bool:
        self.cancelled_orders.append((order_id, symbol))
        return True
