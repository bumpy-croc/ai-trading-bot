from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime


@dataclass
class TradeMFERecord:
    """Record of MFE/MAE metrics for a single trade.

    Attributes:
        strategy_name: Name of the strategy that generated the trade
        mfe: Maximum Favorable Excursion as decimal fraction
        mae: Maximum Adverse Excursion as decimal fraction
        mfe_time: Timestamp when MFE occurred
        mae_time: Timestamp when MAE occurred
    """

    strategy_name: str
    mfe: float
    mae: float
    mfe_time: datetime | None
    mae_time: datetime | None


class MFEMAEAnalyzer:
    """Analyzes Maximum Favorable/Adverse Excursion metrics for trade performance evaluation.

    Provides statistical analysis of MFE/MAE ratios, exit timing efficiency,
    and optimal exit point identification across trade histories.

    MFE (Maximum Favorable Excursion): Largest unrealized profit reached during trade
    MAE (Maximum Adverse Excursion): Largest unrealized loss reached during trade

    These metrics help identify:
    - Whether exits are capturing available profit (MFE vs realized PnL)
    - Risk tolerance adequacy (MAE vs stop-loss settings)
    - Optimal profit-taking levels (common MFE patterns)
    """

    @staticmethod
    def _safe_float(value, default=0.0):
        """Convert to float, replacing NaN/Infinity with default.

        Handles None, NaN, Infinity, and type conversion errors gracefully.
        """
        try:
            result = float(value or default)
            return result if math.isfinite(result) else default
        except (TypeError, ValueError):
            return default
    def calculate_avg_mfe_mae_by_strategy(
        self, trades: Iterable[dict], strategy_name: str | None = None
    ) -> dict:
        """Calculate average MFE and MAE for trades, optionally filtered by strategy.

        Args:
            trades: Iterable of trade dictionaries with 'mfe', 'mae', 'strategy' keys
            strategy_name: Optional strategy name filter (None = all strategies)

        Returns:
            Dictionary with 'avg_mfe' and 'avg_mae' keys as decimal fractions
        """
        records = [
            t for t in trades if (strategy_name is None or t.get("strategy") == strategy_name)
        ]
        if not records:
            return {"avg_mfe": 0.0, "avg_mae": 0.0}
        # Use _safe_float to filter out NaN/Infinity values from corrupted data
        mfe_vals = [self._safe_float(t.get("mfe", 0.0)) for t in records]
        mae_vals = [self._safe_float(t.get("mae", 0.0)) for t in records]
        return {
            "avg_mfe": sum(mfe_vals) / len(mfe_vals) if mfe_vals else 0.0,
            "avg_mae": sum(mae_vals) / len(mae_vals) if mae_vals else 0.0,
        }

    def calculate_mfe_mae_ratios(self, trades: Iterable[dict]) -> dict:
        """Calculate average MFE/MAE ratio across trades.

        Higher ratios indicate trades that reached larger favorable excursions
        relative to adverse excursions, suggesting effective risk/reward setup.

        Args:
            trades: Iterable of trade dictionaries with 'mfe' and 'mae' keys

        Returns:
            Dictionary with 'avg_ratio' key
        """
        ratios = []
        for t in trades:
            # Use _safe_float to filter out NaN/Infinity values from corrupted data
            mae = abs(self._safe_float(t.get("mae", 0.0)))
            mfe = self._safe_float(t.get("mfe", 0.0))
            if mae > 0:
                ratios.append(mfe / mae)
        return {"avg_ratio": (sum(ratios) / len(ratios)) if ratios else 0.0}

    def analyze_exit_timing_efficiency(self, trades: Iterable[dict]) -> dict:
        """Approximate exit timing efficiency by comparing realized PnL to MFE.

        TODO: Requires intra-trade tick data to compute ideal vs actual exits accurately.
        Current implementation approximates efficiency as (realized PnL / MFE), clamped to [0, 1].

        High efficiency (>0.8) suggests exits capture most available profit.
        Low efficiency (<0.4) indicates premature exits leaving profit on table.

        Args:
            trades: Iterable of trade dictionaries with 'pnl_percent' and 'mfe' keys

        Returns:
            Dictionary with 'avg_exit_efficiency' key (0.0 to 1.0)
        """
        efficiencies = []
        for t in trades:
            # Use _safe_float to filter out NaN/Infinity values from corrupted data
            pnl = self._safe_float(t.get("pnl_percent", 0.0)) / 100.0
            mfe = self._safe_float(t.get("mfe", 0.0))
            if mfe > 0:
                eff = max(0.0, min(1.0, pnl / mfe))
                efficiencies.append(eff)
        return {
            "avg_exit_efficiency": (sum(efficiencies) / len(efficiencies)) if efficiencies else 0.0
        }

    def identify_optimal_exit_points(self, trades: Iterable[dict]) -> list[TradeMFERecord]:
        """Extract MFE/MAE records from trades for pattern analysis.

        Use these records to identify common MFE levels where exits should be placed
        or common MAE levels that indicate stop-loss placement is too tight/loose.

        Args:
            trades: Iterable of trade dictionaries with MFE/MAE data

        Returns:
            List of TradeMFERecord objects for further analysis
        """
        out: list[TradeMFERecord] = []
        for t in trades:
            out.append(
                TradeMFERecord(
                    strategy_name=str(t.get("strategy", "")),
                    mfe=self._safe_float(t.get("mfe", 0.0)),
                    mae=self._safe_float(t.get("mae", 0.0)),
                    mfe_time=t.get("mfe_time"),
                    mae_time=t.get("mae_time"),
                )
            )
        return out
