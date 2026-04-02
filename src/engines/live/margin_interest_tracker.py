"""Margin interest tracker for calculating borrow costs on short positions."""

from __future__ import annotations

import logging
import math
import time
from datetime import UTC, datetime
from typing import Any, Protocol

logger = logging.getLogger(__name__)

_RETRY_DELAY_SECONDS = 1.0


class _ExchangeWithInterestHistory(Protocol):
    """Protocol for exchanges that support margin interest history queries."""

    def get_margin_interest_history(
        self, asset: str, start_time: int, end_time: int | None = None,
        page: int = 1,
    ) -> list[dict[str, Any]]: ...


class MarginInterestTracker:
    """Query wrapper for margin borrow interest costs.

    Calculates total margin borrow interest accrued on a position
    between its entry time and the present by querying the exchange API.
    """

    def __init__(self, exchange: _ExchangeWithInterestHistory) -> None:
        self._exchange = exchange

    def get_position_interest_cost(
        self, asset: str, entry_time: datetime, retries: int = 1
    ) -> float:
        """Return total margin interest cost for an asset since entry_time.

        Sums interest from all exchange records since the position was opened.
        Returns 0.0 on error or when no records exist.

        Args:
            asset: Base asset symbol (e.g. "BTC").
            entry_time: Position open time.
            retries: Number of retry attempts on API failure (default 1).
        """
        if entry_time.tzinfo is None:
            logger.warning(
                "entry_time for %s is naive (no timezone) — normalizing to UTC",
                asset,
            )
            entry_time = entry_time.replace(tzinfo=UTC)

        start_time_ms = int(entry_time.timestamp() * 1000)
        records: list[dict[str, Any]] = []
        last_error: Exception | None = None

        for attempt in range(1 + retries):
            try:
                records = self._fetch_all_records(asset, start_time_ms)
                last_error = None
                break
            except Exception as e:
                last_error = e
                if attempt < retries:
                    logger.info(
                        "Retrying margin interest fetch for %s (attempt %d/%d): %s",
                        asset, attempt + 1, 1 + retries, e,
                    )
                    time.sleep(_RETRY_DELAY_SECONDS)

        if last_error is not None:
            logger.warning(
                "Failed to fetch margin interest history for %s after %d attempts: %s",
                asset, 1 + retries, last_error, exc_info=True,
            )
            return 0.0

        if not records:
            return 0.0

        total = 0.0
        for record in records:
            if not isinstance(record, dict):
                logger.warning(
                    "Skipping non-dict interest record: %s", type(record).__name__
                )
                continue
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
    _PAGE_SIZE = 100  # Must match size param passed to Binance API

    def _fetch_all_records(
        self, asset: str, start_time_ms: int
    ) -> list[dict[str, Any]]:
        """Fetch all interest records using page-number pagination."""
        all_records: list[dict[str, Any]] = []

        for page_num in range(1, self._MAX_PAGES + 1):
            records = self._exchange.get_margin_interest_history(
                asset=asset, start_time=start_time_ms, page=page_num
            )
            if not records:
                break

            all_records.extend(records)

            if len(records) < self._PAGE_SIZE:
                break

            if page_num > 1:
                logger.info(
                    "Paginating interest history for %s (page %d, %d records so far)",
                    asset, page_num, len(all_records),
                )
        else:
            # Loop exhausted MAX_PAGES without breaking
            logger.warning(
                "Interest history for %s hit %d-page limit (%d records) — "
                "total may be understated for very long-held positions",
                asset, self._MAX_PAGES, len(all_records),
            )

        return all_records
