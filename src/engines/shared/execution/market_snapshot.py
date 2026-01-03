"""Market snapshot data used for execution modeling."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class MarketSnapshot:
    """Represents market data used to decide if and how orders fill."""

    symbol: str
    timestamp: datetime
    last_price: float
    high: float
    low: float
    close: float
    volume: float
    bid: float | None = None
    ask: float | None = None

    def has_quotes(self) -> bool:
        """Return True when bid and ask quotes are available."""
        return self.bid is not None and self.ask is not None
