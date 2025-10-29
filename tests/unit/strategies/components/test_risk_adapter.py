"""Unit tests for the CoreRiskAdapter."""

from __future__ import annotations

import pandas as pd
import pytest

from src.risk.risk_manager import RiskManager as EngineRiskManager
from src.risk.risk_manager import RiskParameters
from src.strategies.components import CoreRiskAdapter, Signal, SignalDirection


@pytest.fixture()
def sample_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.5],
            "volume": [1_000],
        }
    )


@pytest.fixture()
def buy_signal() -> Signal:
    return Signal(direction=SignalDirection.BUY, strength=1.0, confidence=1.0, metadata={})


def test_calculate_position_size_uses_core_fraction(
    sample_dataframe: pd.DataFrame, buy_signal: Signal
) -> None:
    params = RiskParameters(base_risk_per_trade=0.02, max_position_size=0.2)
    core_manager = EngineRiskManager(params)
    adapter = CoreRiskAdapter(core_manager)
    adapter.set_strategy_overrides({"position_sizer": "fixed_fraction", "base_fraction": 0.05})

    size = adapter.calculate_position_size(
        buy_signal,
        balance=10_000,
        regime=None,
        df=sample_dataframe,
        index=0,
        price=float(sample_dataframe["close"].iloc[0]),
        indicators={},
    )

    assert pytest.approx(size, rel=1e-6) == 500.0


def test_get_stop_loss_uses_overrides_without_dataframe(buy_signal: Signal) -> None:
    params = RiskParameters()
    adapter = CoreRiskAdapter(EngineRiskManager(params))
    adapter.set_strategy_overrides({"stop_loss_pct": 0.02})

    stop = adapter.get_stop_loss(100.0, buy_signal, None)
    assert pytest.approx(stop, rel=1e-6) == 98.0


def test_policy_bundle_reflects_risk_parameters(
    sample_dataframe: pd.DataFrame, buy_signal: Signal
) -> None:
    params = RiskParameters(
        partial_exit_targets=[0.02, 0.04],
        partial_exit_sizes=[0.5, 0.5],
        scale_in_thresholds=[0.03],
        scale_in_sizes=[0.25],
        max_scale_ins=1,
        trailing_activation_threshold=0.01,
        trailing_distance_pct=0.005,
        default_take_profit_pct=0.04,
    )
    adapter = CoreRiskAdapter(EngineRiskManager(params))
    bundle = adapter.get_position_policies(
        buy_signal,
        balance=10_000,
        regime=None,
        df=sample_dataframe,
        index=0,
        price=float(sample_dataframe["close"].iloc[0]),
        indicators={},
    )

    assert bundle is not None
    assert bundle.partial_exit is not None
    assert bundle.partial_exit.exit_targets == [0.02, 0.04]
    assert bundle.trailing_stop is not None
    assert pytest.approx(bundle.trailing_stop.activation_threshold, rel=1e-6) == 0.01
