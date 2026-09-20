"""Single source of truth for mapping Binance order-type strings to ``OrderType``.

Shared by the REST parsing path (``BinanceProvider``) and the WebSocket path
(``OrderTracker``) so a newly supported Binance type is a one-site edit.
"""

import logging

from src.data_providers.exchange_interface import OrderType

logger = logging.getLogger(__name__)

BINANCE_ORDER_TYPE_MAP: dict[str, OrderType] = {
    "MARKET": OrderType.MARKET,
    "LIMIT": OrderType.LIMIT,
    "STOP_LOSS": OrderType.STOP_LOSS,
    "STOP_LOSS_LIMIT": OrderType.STOP_LOSS,
    "TAKE_PROFIT": OrderType.TAKE_PROFIT,
    "TAKE_PROFIT_LIMIT": OrderType.TAKE_PROFIT,
}


def map_binance_order_type(binance_type: str) -> OrderType:
    """Map a Binance order type to ``OrderType``.

    Unknown types default to MARKET (the neutral value) but are logged so a new
    Binance type surfaces immediately instead of being silently misclassified.
    """
    mapped = BINANCE_ORDER_TYPE_MAP.get(binance_type)
    if mapped is None:
        logger.warning("Unknown Binance order type %r — defaulting to MARKET", binance_type)
        return OrderType.MARKET
    return mapped
