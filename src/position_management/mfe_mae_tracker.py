from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple

from src.performance.metrics import Side, pnl_percent


@dataclass
class MFEMetrics:
    mfe: float = 0.0  # decimal fraction relative to entry (sized if fraction provided)
    mae: float = 0.0
    mfe_price: Optional[float] = None
    mae_price: Optional[float] = None
    mfe_time: Optional[datetime] = None
    mae_time: Optional[datetime] = None


class MFEMAETracker:
    """
    Tracks Maximum Favorable/Adverse Excursion for positions.

    Values are stored as decimal fractions relative to entry (e.g., +0.05 = +5%).
    The position fraction can be applied for sized returns via `position_fraction`.
    """

    def __init__(self, precision_decimals: int = 8):
        self.precision_decimals = precision_decimals
        # In-memory cache keyed by position_id or order_id
        self._cache: dict[str | int, MFEMetrics] = {}

    @staticmethod
    def calculate_mfe_mae(
        entry_price: float,
        current_price: float,
        side: str | Side,
        position_fraction: float = 1.0,
        as_sized: bool = True,
    ) -> Tuple[float, float]:
        """Return current excursion (mfe_candidate, mae_candidate) as decimal fractions.

        If `as_sized` is True, returns sized PnL fractions using `position_fraction`.
        """
        side_enum = side if isinstance(side, Side) else Side(side)
        move = pnl_percent(entry_price, current_price, side_enum, position_fraction if as_sized else 1.0)
        # Positive move contributes to MFE candidate; negative to MAE candidate
        mfe_cand = max(0.0, move)
        mae_cand = min(0.0, move)
        return mfe_cand, mae_cand

    def update_position_metrics(
        self,
        position_key: str | int,
        entry_price: float,
        current_price: float,
        side: str | Side,
        position_fraction: float,
        current_time: datetime,
    ) -> MFEMetrics:
        """Update rolling MFE/MAE for a position and return the updated metrics."""
        metrics = self._cache.get(position_key, MFEMetrics())
        mfe_cand, mae_cand = self.calculate_mfe_mae(
            entry_price=entry_price,
            current_price=current_price,
            side=side,
            position_fraction=position_fraction,
        )

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
        return self._cache.get(position_key)

    def clear(self, position_key: str | int | None = None):
        if position_key is None:
            self._cache.clear()
        else:
            self._cache.pop(position_key, None)