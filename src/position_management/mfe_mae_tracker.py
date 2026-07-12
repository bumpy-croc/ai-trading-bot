from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass
from datetime import datetime

from src.performance.metrics import Side, pnl_percent

logger = logging.getLogger(__name__)


@dataclass
class MFEMetrics:
    """Raw price excursion extremes relative to entry.

    ``mfe``/``mae`` are unsized price moves as decimal fractions
    (e.g. +0.05 = +5%), gross of fees and slippage, and always consistent
    with their ``mfe_price``/``mae_price`` companions:
    ``mfe == pnl_percent(entry_price, mfe_price, side)``.
    """

    mfe: float = 0.0
    mae: float = 0.0
    mfe_price: float | None = None
    mae_price: float | None = None
    mfe_time: datetime | None = None
    mae_time: datetime | None = None


class MFEMAETracker:
    """
    Tracks Maximum Favorable/Adverse Excursion for positions.

    Values are RAW price excursions relative to entry, stored as decimal
    fractions (e.g., +0.05 = +5%). No position sizing and no fee/slippage
    adjustment is applied — that keeps ``mfe``/``mae`` in the same units as
    ``trades.pnl_percent`` and derivable from the ``mfe_price``/``mae_price``
    companion columns. Consumers that need sized or cost-netted excursions
    apply ``trades.size`` and the shared cost calculator themselves.

    Thread Safety:
        This class is thread-safe for concurrent access to different position
        keys. The same position key should not be updated concurrently from
        multiple threads.
    """

    def __init__(self, precision_decimals: int = 8):
        self.precision_decimals = precision_decimals
        # In-memory cache keyed by position_id or order_id
        self._cache: dict[str | int, MFEMetrics] = {}
        # Lock protects _cache from concurrent write operations
        self._lock = threading.Lock()

    @staticmethod
    def calculate_mfe_mae(
        entry_price: float,
        current_price: float,
        side: str | Side,
    ) -> tuple[float, float]:
        """Return current excursion (mfe_candidate, mae_candidate).

        Both candidates are raw price moves from entry as decimal fractions:
        a favorable move yields (move, 0.0), an adverse move (0.0, move).
        """
        # Validate prices to prevent NaN/Infinity propagation in MFE/MAE calculations
        if not math.isfinite(entry_price) or entry_price <= 0:
            return 0.0, 0.0
        if not math.isfinite(current_price) or current_price <= 0:
            return 0.0, 0.0

        side_enum = side if isinstance(side, Side) else Side(side)

        move = pnl_percent(entry_price, current_price, side_enum)

        mfe_cand = move if move > 0 else 0.0
        mae_cand = move if move < 0 else 0.0

        return mfe_cand, mae_cand

    def update_position_metrics(
        self,
        position_key: str | int,
        entry_price: float,
        current_price: float,
        side: str | Side,
        current_time: datetime,
    ) -> MFEMetrics:
        """Update rolling MFE/MAE for a position and return the updated metrics."""
        # Validate position_key is hashable before using as dict key
        if not isinstance(position_key, str | int):
            raise TypeError(f"position_key must be str or int, got {type(position_key)}")

        mfe_cand, mae_cand = self.calculate_mfe_mae(
            entry_price=entry_price,
            current_price=current_price,
            side=side,
        )

        # Update cache with lock to prevent race conditions
        with self._lock:
            metrics = self._cache.get(position_key, MFEMetrics())
            # Update MFE
            if mfe_cand > (metrics.mfe or 0.0):
                metrics.mfe = round(float(mfe_cand), self.precision_decimals)
                metrics.mfe_price = current_price
                metrics.mfe_time = current_time
            # Update MAE (most negative)
            if mae_cand < (metrics.mae or 0.0):
                metrics.mae = round(float(mae_cand), self.precision_decimals)
                metrics.mae_price = current_price
                metrics.mae_time = current_time

            self._cache[position_key] = metrics
            return metrics

    def get_position_metrics(self, position_key: str | int) -> MFEMetrics | None:
        """Get metrics for a position without lock.

        Thread Safety:
            This method performs an unlocked read. For concurrent updates to the
            same position_key, callers must use external synchronization. Returns
            either a complete MFEMetrics object or None; never a partially updated
            object due to Python's GIL guaranteeing atomic reads.
        """
        return self._cache.get(position_key)

    def clear(self, position_key: str | int | None = None):
        with self._lock:
            if position_key is None:
                self._cache.clear()
            else:
                self._cache.pop(position_key, None)
