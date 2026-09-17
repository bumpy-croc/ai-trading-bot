"""Shared closing-SELL quantity guard for last-resort emergency closes.

Mirrors the free-base cap + floored lot snap that
``LiveExecutionEngine._close_live_order`` applies to its normal market-close
SELL path: Binance deducts the entry BUY's commission from the base fill, so
a closing SELL sized off the raw filled quantity can exceed real holdings and
get rejected with -2010. Three last-resort emergency-close sites
(``entry_coordinator.py``, ``reconciliation.py``) sent an uncapped closing
SELL and hit the exact same hazard -- worse there, since the reject leaves an
already-inconsistent position open and unprotected instead of just failing a
routine close (#989).

Short-cover BUYs are NOT covered here: they are funded from quote and must
repay the full base borrow, so callers keep sending those uncapped and
nearest-rounded, exactly as before this guard existed.
"""

from __future__ import annotations

import logging
import math
from typing import Any

from src.config.constants import HOLDINGS_CAP_MIN_RATIO
from src.trading.balance_retry import read_free_balance_with_retry
from src.trading.precision import quantize_to_step
from src.trading.symbols.factory import base_asset_from_symbol

logger = logging.getLogger(__name__)


def cap_closing_sell_quantity(
    exchange_interface: Any,
    *,
    symbol: str,
    quantity: float,
) -> float:
    """Cap a closing SELL's quantity to what is actually free and lot-sized.

    Returns the safe-to-submit quantity, floored to the symbol's LOT_SIZE step.
    Returns ``0.0`` when the position is not honestly sellable -- free balance
    covers less than ``HOLDINGS_CAP_MIN_RATIO`` of the intended amount (inventory
    locked by an untracked order, or the step-size floor rounds it away).
    Callers MUST treat ``0.0`` as "abort the close, escalate" -- never as
    "there is nothing to sell, so do nothing is fine."

    A lookup failure (no exchange interface, an unreadable balance, missing
    symbol info) degrades to skipping the affected guard rather than blocking
    an already-critical close on a transient error -- the same fail-open
    contract ``LiveExecutionEngine._free_base_for_close``/``_normalize_quantity``
    use for the primary close path.
    """
    if quantity <= 0 or exchange_interface is None:
        return 0.0

    intended_quantity = quantity
    base_asset = base_asset_from_symbol(symbol)

    def _read_free_base() -> float | None:
        try:
            balance = exchange_interface.get_balance(base_asset)
            return float(balance.free) if balance is not None else None
        except Exception as e:
            logger.warning(
                "Could not read free %s balance for emergency-close sizing: %s",
                base_asset,
                e,
            )
            return None

    # No stop was just cancelled ahead of these emergency closes, so a low
    # reading isn't a stale post-cancel snapshot -- min_required=None makes
    # exactly one read (see read_free_balance_with_retry).
    free_base = read_free_balance_with_retry(_read_free_base, min_required=None, context=symbol)
    if free_base is not None and free_base < quantity:
        logger.warning(
            "Emergency close sell qty %.8f for %s exceeds free base balance %.8f -- "
            "capping to holdings to avoid -2010.",
            quantity,
            symbol,
            free_base,
        )
        quantity = free_base

    quantity = _floor_to_lot_step(exchange_interface, symbol, quantity)

    if quantity < intended_quantity * HOLDINGS_CAP_MIN_RATIO:
        logger.critical(
            "Refusing emergency close of %s: only %.8f of the intended %.8f is sellable "
            "(free base %s) -- inventory may be locked by an untracked order or cannot be "
            "lot-sized honestly. Refusing to sell a fraction and book a full close.",
            symbol,
            quantity,
            intended_quantity,
            "unknown" if free_base is None else f"{free_base:.8f}",
        )
        return 0.0

    return quantity


def _floor_to_lot_step(exchange_interface: Any, symbol: str, quantity: float) -> float:
    """Round ``quantity`` DOWN to the symbol's LOT_SIZE step, or leave it as-is.

    Never rounds UP -- an emergency close must never sell more than it holds.
    Any failure to read usable symbol info degrades to the raw quantity,
    matching ``LiveExecutionEngine._normalize_quantity``'s fail-open contract.
    """
    if quantity <= 0:
        return 0.0
    try:
        symbol_info = exchange_interface.get_symbol_info(symbol)
    except Exception as e:
        logger.warning(
            "Failed to fetch symbol info for %s: %s -- using uncapped-balance quantity",
            symbol,
            e,
        )
        return quantity

    if not symbol_info or not isinstance(symbol_info, dict):
        return quantity

    step_size = symbol_info.get("step_size")
    if not isinstance(step_size, int | float) or step_size <= 0 or not math.isfinite(step_size):
        return quantity

    # +epsilon so a quantity that is mathematically an exact lot multiple but
    # stored a hair low (float noise) isn't truncated a whole step down
    # (mirrors _normalize_quantity's floor branch).
    floored = math.floor(quantity / step_size + 1e-9) * step_size
    if not math.isfinite(floored):
        return quantity
    return quantize_to_step(floored, step_size)
