"""Tests for the runtime regression harness."""

import pandas as pd

from src.strategies.adapters.legacy_adapter import LegacyStrategyAdapter
from src.strategies.components import (
    FixedFractionSizer,
    FixedRiskManager,
    HoldSignalGenerator,
    Signal,
    SignalDirection,
    SignalGenerator,
    Strategy,
    StrategyRuntime,
)
from src.strategies.migration.runtime_regression import compare_backtests


def _build_market_data(rows: int = 24) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=rows, freq="h")
    prices = pd.Series(100.0 + pd.RangeIndex(len(index)), index=index)
    frame = pd.DataFrame(
        {
            "open": prices,
            "high": prices + 1.0,
            "low": prices - 1.0,
            "close": prices,
            "volume": 1000.0,
        },
        index=index,
    )
    frame.index.name = "timestamp"
    return frame


class BuySignalGenerator(SignalGenerator):
    def __init__(self):
        super().__init__("always_buy")

    def generate_signal(self, df: pd.DataFrame, index: int, regime=None) -> Signal:
        return Signal(SignalDirection.BUY, 1.0, 1.0, {})

    def get_confidence(self, df: pd.DataFrame, index: int) -> float:
        return 1.0


def _build_component_strategy(
    fraction: float = 0.02,
    generator: SignalGenerator | None = None,
) -> Strategy:
    signal = generator or HoldSignalGenerator()
    risk = FixedRiskManager(risk_per_trade=0.01, stop_loss_pct=0.05)
    sizer = FixedFractionSizer(fraction=fraction)
    return Strategy("hold_component", signal, risk, sizer)


def test_compare_backtests_matches_for_identical_strategies():
    data = _build_market_data()
    component_strategy = _build_component_strategy()
    legacy_adapter = LegacyStrategyAdapter(
        component_strategy.signal_generator,
        component_strategy.risk_manager,
        component_strategy.position_sizer,
    )

    result = compare_backtests(legacy_adapter, component_strategy, data)

    assert result.matching is True
    assert result.differences == {}
    assert result.legacy_results["total_trades"] == 0
    assert result.runtime_results["total_trades"] == 0


def test_compare_backtests_reports_differences_when_metrics_diverge():
    data = _build_market_data()
    component_strategy = _build_component_strategy(generator=BuySignalGenerator())
    legacy_adapter = LegacyStrategyAdapter(
        component_strategy.signal_generator,
        component_strategy.risk_manager,
        component_strategy.position_sizer,
    )

    # Modify component strategy to force different sizing for runtime
    component_strategy.position_sizer = FixedFractionSizer(fraction=0.05)

    result = compare_backtests(legacy_adapter, component_strategy, data)

    assert result.matching is False
    assert "final_balance" in result.differences


def test_compare_backtests_accepts_strategy_runtime_instance():
    data = _build_market_data()
    component_strategy = _build_component_strategy()
    legacy_adapter = LegacyStrategyAdapter(
        component_strategy.signal_generator,
        component_strategy.risk_manager,
        component_strategy.position_sizer,
    )

    runtime = StrategyRuntime(component_strategy)

    result = compare_backtests(legacy_adapter, runtime, data)

    assert result.matching is True
    assert result.differences == {}
