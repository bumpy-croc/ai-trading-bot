"""Regression harness comparing legacy and runtime backtests."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable

import pandas as pd

from src.backtesting.engine import Backtester
from src.data_providers.data_provider import DataProvider
from src.strategies.base import BaseStrategy
from src.strategies.components import Strategy as ComponentStrategy, StrategyRuntime


class DataFrameProvider(DataProvider):
    """Simple data provider that serves a static DataFrame."""

    def __init__(self, frame: pd.DataFrame):
        self._frame = frame.copy()

    def get_historical_data(
        self,
        symbol: str,
        timeframe: str,
        start: datetime | None,
        end: datetime | None,
    ) -> pd.DataFrame:
        return self._frame.copy()

    def get_current_price(self, symbol: str) -> float:
        if self._frame.empty:
            return 0.0
        return float(self._frame["close"].iloc[-1])

    def get_live_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        return self._frame.copy()

    def update_live_data(self, symbol: str, timeframe: str) -> None:
        return None


@dataclass
class BacktestComparisonResult:
    """Structured comparison output for regression harness."""

    legacy_results: dict[str, Any]
    runtime_results: dict[str, Any]
    matching: bool
    differences: dict[str, Any]


def _default_metric_extractor(results: dict[str, Any]) -> dict[str, Any]:
    return {
        "final_balance": round(float(results.get("final_balance", 0.0)), 8),
        "total_trades": int(results.get("total_trades", 0)),
        "win_rate": round(float(results.get("win_rate", 0.0)), 8),
        "max_drawdown": round(float(results.get("max_drawdown", 0.0)), 8),
        "total_return": round(float(results.get("total_return", 0.0)), 8),
    }


def compare_backtests(
    legacy_strategy: BaseStrategy,
    runtime_strategy: BaseStrategy | ComponentStrategy | StrategyRuntime,
    market_data: pd.DataFrame,
    metric_extractor: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> BacktestComparisonResult:
    """Run the same dataset through both strategies and compare metrics."""

    if metric_extractor is None:
        metric_extractor = _default_metric_extractor

    provider = DataFrameProvider(market_data)
    start = market_data.index[0] if len(market_data.index) else datetime.utcnow()
    end = market_data.index[-1] if len(market_data.index) else None

    legacy_backtester = Backtester(
        strategy=legacy_strategy,
        data_provider=provider,
        log_to_database=False,
        enable_dynamic_risk=False,
    )
    runtime_backtester = Backtester(
        strategy=runtime_strategy,
        data_provider=provider,
        log_to_database=False,
        enable_dynamic_risk=False,
    )

    legacy_results = legacy_backtester.run(symbol="TEST", timeframe="1h", start=start, end=end)
    runtime_results = runtime_backtester.run(symbol="TEST", timeframe="1h", start=start, end=end)

    legacy_metrics = metric_extractor(legacy_results)
    runtime_metrics = metric_extractor(runtime_results)

    differences: dict[str, Any] = {}
    matching = True
    for key in sorted(set(legacy_metrics) | set(runtime_metrics)):
        if legacy_metrics.get(key) != runtime_metrics.get(key):
            matching = False
            differences[key] = {
                "legacy": legacy_metrics.get(key),
                "runtime": runtime_metrics.get(key),
            }

    return BacktestComparisonResult(
        legacy_results=legacy_results,
        runtime_results=runtime_results,
        matching=matching,
        differences=differences,
    )
