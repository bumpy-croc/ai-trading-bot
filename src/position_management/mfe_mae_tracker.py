from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from src.performance.metrics import Side, pnl_percent


@dataclass
class MFEMetrics:
    mfe: float = 0.0  # decimal fraction relative to entry (sized if fraction provided)
    mae: float = 0.0
    mfe_price: float | None = None
    mae_price: float | None = None
    mfe_time: datetime | None = None
    mae_time: datetime | None = None


class MFEMAETracker:
    """
    Tracks Maximum Favorable/Adverse Excursion for positions.

    Values are stored as decimal fractions relative to entry (e.g., +0.05 = +5%).
    The position fraction can be applied for sized returns via `position_fraction`.
    MFE/MAE metrics account for exit fees and slippage to reflect achievable profit/loss.
    """

    def __init__(
        self,
        precision_decimals: int = 8,
        fee_rate: float = 0.001,
        slippage_rate: float = 0.0005,
    ):
        self.precision_decimals = precision_decimals
        self.fee_rate = fee_rate
        self.slippage_rate = slippage_rate
        # In-memory cache keyed by position_id or order_id
        self._cache: dict[str | int, MFEMetrics] = {}

    @staticmethod
    def calculate_mfe_mae(
        entry_price: float,
        current_price: float,
        side: str | Side,
        position_fraction: float = 1.0,
        as_sized: bool = True,
        fee_rate: float = 0.001,
        slippage_rate: float = 0.0005,
    ) -> tuple[float, float]:
        """Return current excursion (mfe_candidate, mae_candidate) as decimal fractions.

        If `as_sized` is True, returns sized PnL fractions using `position_fraction`.
        Net MFE/MAE accounts for exit fees and slippage to reflect achievable profit/loss.
        """
        side_enum = side if isinstance(side, Side) else Side(side)

        # Calculate gross price movement
        move = pnl_percent(
            entry_price, current_price, side_enum, position_fraction if as_sized else 1.0
        )

        # Calculate exit costs as percentage of position value
        # For MFE: subtract costs since they reduce achievable profit
        # For MAE: add costs since they worsen losses
        exit_cost_rate = fee_rate + slippage_rate

        # Adjust for costs to get net achievable excursion
        net_move = move - exit_cost_rate if as_sized else move

        # Positive move contributes to MFE candidate; negative to MAE candidate
        mfe_cand = max(0.0, net_move) if move > 0 else 0.0
        mae_cand = min(0.0, net_move) if move < 0 else 0.0

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
            fee_rate=self.fee_rate,
            slippage_rate=self.slippage_rate,
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
