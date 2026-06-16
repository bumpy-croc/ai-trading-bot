"""Shared commission helpers for normalizing exchange fill fees to USD.

Single source of truth used by both ``LiveExecutionEngine`` (entry/exit fee
reconciliation) and ``PositionReconciler`` (offline-close trade logging) so the
base-asset → USD conversion is never duplicated (CODE.md: "never duplicate
financial logic — use shared modules").
"""

from __future__ import annotations

import math
from typing import Any

# Quote assets the bot trades against; longest-first so 4-char quotes match before "USD".
_QUOTE_ASSETS = ("USDT", "BUSD", "USDC", "USD")


def split_base_quote(symbol: str) -> tuple[str, str]:
    """Split a pair symbol into ``(base, quote)`` (e.g. ``ETHUSDT`` -> ``("ETH", "USDT")``).

    Returns ``(symbol, "")`` when no known quote suffix matches.
    """
    s = symbol.upper()
    for quote in _QUOTE_ASSETS:
        if s.endswith(quote) and len(s) > len(quote):
            return s[: -len(quote)], quote
    return s, ""


def order_commission_usd(order_details: Any, symbol: str, price: float) -> float | None:
    """Convert an exchange fill commission to quote/USD using its ``commission_asset``.

    Binance denominates commission in the *received* asset: the base asset on BUYs
    (e.g. ETH), the quote asset on SELLs (e.g. USDT). Booking a base-asset commission
    as if it were USD silently under-reports fees (and corrupts the USD balance/ledger).

    Returns the commission in quote/USD when it can be determined, or ``None`` when it
    cannot be reliably converted (commission paid in a discount asset like BNB, or an
    unknown/empty asset) — signalling the caller to fall back to the modelled USD fee
    rather than book a wrong-unit value.
    """
    raw = float(getattr(order_details, "commission", 0.0) or 0.0)
    if raw <= 0:
        return 0.0
    asset = str(getattr(order_details, "commission_asset", "") or "").upper()
    if not asset:
        return None
    base, quote = split_base_quote(symbol)
    if quote and asset == quote:
        return raw  # already quote/USD
    if base and asset == base and price > 0 and math.isfinite(price):
        return raw * price  # base asset -> quote via fill price
    return None  # discount/unknown asset -> not convertible here
