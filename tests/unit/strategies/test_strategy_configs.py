"""Regression tests for strategy component wiring and configuration."""

from __future__ import annotations

import pytest

from src.strategies.components.ml_signal_generator import (
    MLBasicSignalGenerator,
    MLSignalGenerator,
)
from src.strategies.components.momentum_signal_generator import MomentumSignalGenerator
from src.strategies.components.position_sizer import ConfidenceWeightedSizer
from src.strategies.components.risk_manager import (
    FixedRiskManager,
    RegimeAdaptiveRiskManager,
    VolatilityRiskManager,
)
from src.strategies.components.signal_generator import WeightedVotingSignalGenerator
from src.strategies.ensemble_weighted import EnsembleWeighted
from src.strategies.ml_adaptive import MlAdaptive
from src.strategies.ml_basic import MlBasic
from src.strategies.ml_sentiment import MlSentiment
from src.strategies.momentum_leverage import MomentumLeverage


@pytest.mark.parametrize(
    "factory, signal_type",
    [
        (MlBasic, MLBasicSignalGenerator),
        (MlAdaptive, MLSignalGenerator),
        (MlSentiment, MLSignalGenerator),
        (MomentumLeverage, MomentumSignalGenerator),
    ],
)
def test_primary_signal_generator_types(factory, signal_type) -> None:
    strategy = factory()
    assert isinstance(strategy.signal_generator, signal_type)
    assert isinstance(strategy.position_sizer, ConfidenceWeightedSizer)


@pytest.mark.parametrize(
    "factory, risk_type",
    [
        (MlBasic, FixedRiskManager),
        (MlSentiment, FixedRiskManager),
        (MlAdaptive, RegimeAdaptiveRiskManager),
        (MomentumLeverage, VolatilityRiskManager),
        (EnsembleWeighted, VolatilityRiskManager),
    ],
)
def test_risk_managers_are_configured(factory, risk_type) -> None:
    strategy = factory()
    assert isinstance(strategy.risk_manager, risk_type)
    overrides = strategy.get_risk_overrides()
    assert overrides is None or isinstance(overrides, dict)


def test_ensemble_weighted_uses_voting_signal_generator() -> None:
    strategy = EnsembleWeighted()
    assert isinstance(strategy.signal_generator, WeightedVotingSignalGenerator)
    assert strategy.get_risk_overrides()["partial_operations"]["max_scale_ins"] == 4


@pytest.mark.parametrize(
    "strategy, expected_keys",
    [
        (MlBasic(), {"model_path", "sequence_length", "stop_loss_pct", "take_profit_pct"}),
        (MlAdaptive(), {"model_path", "sequence_length", "stop_loss_pct", "take_profit_pct"}),
        (MlSentiment(), {"model_path", "sequence_length", "sentiment_boost_multiplier", "strategy_name"}),
        (MomentumLeverage(), {"base_position_size", "stop_loss_pct", "take_profit_pct"}),
        (EnsembleWeighted(), {"base_position_size", "stop_loss_pct", "take_profit_pct"}),
    ],
)
def test_get_parameters_include_expected_fields(strategy, expected_keys) -> None:
    params = strategy.get_parameters()
    assert expected_keys.issubset(params)
    name_key = "name" if "name" in params else "strategy_name"
    assert params[name_key] == strategy.name


def test_strategies_expose_component_info_snapshot() -> None:
    strategy = MlBasic()
    info = strategy.get_component_info()
    assert info["signal_generator"]["name"].endswith("signals")
    assert "risk_manager" in info
    assert "position_sizer" in info
