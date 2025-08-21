from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional
from datetime import datetime


@dataclass
class TradeMFERecord:
    strategy_name: str
    mfe: float
    mae: float
    mfe_time: Optional[datetime]
    mae_time: Optional[datetime]


class MFEMAEAnalyzer:
    def calculate_avg_mfe_mae_by_strategy(self, trades: Iterable[dict], strategy_name: Optional[str] = None) -> dict:
        records = [t for t in trades if (strategy_name is None or t.get("strategy") == strategy_name)]
        if not records:
            return {"avg_mfe": 0.0, "avg_mae": 0.0}
        mfe_vals = [float(t.get("mfe", 0.0) or 0.0) for t in records]
        mae_vals = [float(t.get("mae", 0.0) or 0.0) for t in records]
        return {
            "avg_mfe": sum(mfe_vals) / len(mfe_vals) if mfe_vals else 0.0,
            "avg_mae": sum(mae_vals) / len(mae_vals) if mae_vals else 0.0,
        }

    def calculate_mfe_mae_ratios(self, trades: Iterable[dict]) -> dict:
        ratios = []
        for t in trades:
            mae = abs(float(t.get("mae", 0.0) or 0.0))
            mfe = float(t.get("mfe", 0.0) or 0.0)
            if mae > 0:
                ratios.append(mfe / mae)
        return {"avg_ratio": (sum(ratios) / len(ratios)) if ratios else 0.0}

    def analyze_exit_timing_efficiency(self, trades: Iterable[dict]) -> dict:
        # Placeholder: requires intra-trade series to compute ideal vs actual exits
        # Here, we approximate efficiency as actual pnl divided by MFE (clamped 0..1)
        efficiencies = []
        for t in trades:
            pnl = float(t.get("pnl_percent", 0.0) or 0.0) / 100.0
            mfe = float(t.get("mfe", 0.0) or 0.0)
            if mfe > 0:
                eff = max(0.0, min(1.0, pnl / mfe))
                efficiencies.append(eff)
        return {"avg_exit_efficiency": (sum(efficiencies) / len(efficiencies)) if efficiencies else 0.0}

    def identify_optimal_exit_points(self, trades: Iterable[dict]) -> list[TradeMFERecord]:
        out: list[TradeMFERecord] = []
        for t in trades:
            out.append(
                TradeMFERecord(
                    strategy_name=str(t.get("strategy", "")),
                    mfe=float(t.get("mfe", 0.0) or 0.0),
                    mae=float(t.get("mae", 0.0) or 0.0),
                    mfe_time=t.get("mfe_time"),
                    mae_time=t.get("mae_time"),
                )
            )
        return out