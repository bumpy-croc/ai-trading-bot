"""Margin interest tracker for calculating borrow costs on short positions."""

from __future__ import annotations

import logging
import math
from datetime import datetime
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class _ExchangeWithInterestHistory(Protocol):
    """Protocol for exchanges that support margin interest history queries."""

    def get_margin_interest_history(
        self, *, asset: str, start_time: int
    ) -> list[dict[str, Any]]: ...


class MarginInterestTracker:
    """Query wrapper for margin borrow interest costs.

    Calculates total margin borrow interest accrued on a position
    between its entry time and the present by querying the exchange API.
    """

    def __init__(self, exchange: _ExchangeWithInterestHistory) -> None:
        self._exchange = exchange

    def get_position_interest_cost(
        self, asset: str, entry_time: datetime
    ) -> float:
        """Return total margin interest cost for an asset since entry_time.

        Sums interest from all exchange records since the position was opened.
        Returns 0.0 on error or when no records exist.
        """
        try:
            start_time_ms = int(entry_time.timestamp() * 1000)
            records = self._exchange.get_margin_interest_history(
                asset=asset, start_time=start_time_ms
            )
        except Exception:
            logger.warning(
                "Failed to fetch margin interest history for %s", asset
            )
            return 0.0

        if not records:
            return 0.0

        # Binance returns max 500 records per call; warn if truncated
        if len(records) >= 500:
            logger.warning(
                "Interest history for %s returned %d records (may be truncated)",
                asset,
                len(records),
            )

        total = 0.0
        for record in records:
            try:
                value = float(record["interest"])
            except (ValueError, KeyError, TypeError):
                logger.warning(
                    "Skipping record with invalid interest value: %s",
                    record.get("interest", "<missing>"),
                )
                continue

            if not math.isfinite(value):
                logger.warning(
                    "Skipping non-finite interest value: %s",
                    record["interest"],
                )
                continue

            total += value

        logger.debug(
            "Total margin interest for %s since %s: %.8f",
            asset,
            entry_time.isoformat(),
            total,
        )
        return total
