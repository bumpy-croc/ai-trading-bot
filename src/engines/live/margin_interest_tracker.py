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
        self, asset: str, start_time: int, end_time: int | None = None
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
        if entry_time.tzinfo is None:
            logger.warning(
                "entry_time for %s is naive (no timezone) — assuming UTC",
                asset,
            )

        try:
            start_time_ms = int(entry_time.timestamp() * 1000)
            records = self._fetch_all_records(asset, start_time_ms)
        except Exception as e:
            logger.warning(
                "Failed to fetch margin interest history for %s: %s",
                asset, e, exc_info=True,
            )
            return 0.0

        if not records:
            return 0.0

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

            if value <= 0:
                logger.warning(
                    "Skipping non-positive interest value: %s",
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

    _MAX_PAGES = 10  # Safety limit to prevent runaway pagination
    _PAGE_SIZE = 500  # Binance returns max 500 records per call

    def _fetch_all_records(
        self, asset: str, start_time_ms: int
    ) -> list[dict[str, Any]]:
        """Fetch all interest records, paginating if Binance returns a full page."""
        all_records: list[dict[str, Any]] = []
        current_start = start_time_ms

        for page in range(self._MAX_PAGES):
            records = self._exchange.get_margin_interest_history(
                asset=asset, start_time=current_start
            )
            if not records:
                break

            all_records.extend(records)

            if len(records) < self._PAGE_SIZE:
                break

            # Use last record's timestamp + 1ms as next page start
            last_time = records[-1].get("interestAccuredTime")
            if last_time is None:
                logger.warning(
                    "Cannot paginate interest history for %s — missing timestamp",
                    asset,
                )
                break
            current_start = int(last_time) + 1

            if page > 0:
                logger.info(
                    "Paginating interest history for %s (page %d, %d records so far)",
                    asset, page + 1, len(all_records),
                )

        return all_records
