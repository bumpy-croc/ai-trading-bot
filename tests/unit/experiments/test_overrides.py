"""Tests ensuring every Phase 1 override key actually mutates the component."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from src.experiments.runner import ExperimentRunner
from src.experiments.schemas import ExperimentConfig, ParameterSet


@pytest.fixture
def runner() -> ExperimentRunner:
    return ExperimentRunner()


def _cfg(strategy_name: str, values: dict[str, object]) -> ExperimentConfig:
    end = datetime.now(UTC)
    start = end - timedelta(days=1)
    return ExperimentConfig(
        strategy_name=strategy_name,
        symbol="BTCUSDT",
        timeframe="1h",
        start=start,
        end=end,
        initial_balance=1000.0,
        parameters=ParameterSet(name="override", values=values),
    )


def test_load_strategy_supports_all_ml_variants(runner: ExperimentRunner) -> None:
    for name in ("ml_basic", "ml_adaptive", "ml_sentiment"):
        strategy = runner._load_strategy(name)
        assert strategy is not None, f"failed to load {name}"


def test_signal_generator_thresholds_override(runner: ExperimentRunner) -> None:
    strategy = runner._load_strategy("ml_basic")
    cfg = _cfg(
        "ml_basic",
        {
            "ml_basic.long_entry_threshold": 0.0003,
            "ml_basic.short_entry_threshold": -0.0003,
            "ml_basic.confidence_multiplier": 20,
        },
    )
    runner._apply_parameter_overrides(strategy, cfg)

    sg = strategy.signal_generator
    assert pytest.approx(sg.long_entry_threshold, rel=1e-9) == 0.0003
    assert pytest.approx(sg.short_entry_threshold, rel=1e-9) == -0.0003
    assert sg.confidence_multiplier == 20


def test_regime_thresholds_override_on_ml_adaptive(runner: ExperimentRunner) -> None:
    strategy = runner._load_strategy("ml_adaptive")
    cfg = _cfg(
        "ml_adaptive",
        {
            "ml_adaptive.short_threshold_trend_up": -0.0001,
            "ml_adaptive.short_threshold_trend_down": -0.0009,
            "ml_adaptive.short_threshold_range": -0.0004,
            "ml_adaptive.short_threshold_high_vol": -0.0003,
            "ml_adaptive.short_threshold_low_vol": -0.0008,
            "ml_adaptive.short_threshold_confidence_multiplier": 0.4,
        },
    )
    runner._apply_parameter_overrides(strategy, cfg)

    sg = strategy.signal_generator
    assert pytest.approx(sg.short_threshold_trend_up, rel=1e-9) == -0.0001
    assert pytest.approx(sg.short_threshold_trend_down, rel=1e-9) == -0.0009
    assert pytest.approx(sg.short_threshold_range, rel=1e-9) == -0.0004
    assert pytest.approx(sg.short_threshold_high_vol, rel=1e-9) == -0.0003
    assert pytest.approx(sg.short_threshold_low_vol, rel=1e-9) == -0.0008
    assert pytest.approx(sg.short_threshold_confidence_multiplier, rel=1e-9) == 0.4


def test_position_sizer_overrides(runner: ExperimentRunner) -> None:
    strategy = runner._load_strategy("ml_basic")
    cfg = _cfg(
        "ml_basic",
        {
            "ml_basic.base_fraction": 0.25,
            "ml_basic.min_confidence": 0.45,
            "ml_basic.min_confidence_floor": 0.5,
        },
    )
    runner._apply_parameter_overrides(strategy, cfg)

    ps = strategy.position_sizer
    assert pytest.approx(ps.base_fraction, rel=1e-9) == 0.25
    assert pytest.approx(ps.min_confidence, rel=1e-9) == 0.45
    assert pytest.approx(ps.min_confidence_floor, rel=1e-9) == 0.5


def test_stop_loss_and_take_profit_overrides(runner: ExperimentRunner) -> None:
    strategy = runner._load_strategy("ml_basic")
    cfg = _cfg(
        "ml_basic",
        {
            "ml_basic.stop_loss_pct": 0.04,
            "ml_basic.take_profit_pct": 0.08,
        },
    )
    runner._apply_parameter_overrides(strategy, cfg)

    # Both are cached in _risk_overrides — the risk manager consults this at
    # runtime regardless of whether the strategy exposes them as instance attrs.
    overrides = getattr(strategy, "_risk_overrides", {}) or {}
    assert pytest.approx(overrides.get("stop_loss_pct", 0.0), rel=1e-9) == 0.04
    assert pytest.approx(overrides.get("take_profit_pct", 0.0), rel=1e-9) == 0.08


def test_factory_kwargs_respected_for_ml_basic() -> None:
    from src.strategies.ml_basic import create_ml_basic_strategy

    strategy = create_ml_basic_strategy(
        long_entry_threshold=0.0007,
        short_entry_threshold=-0.0008,
        confidence_multiplier=15.0,
        base_fraction=0.12,
        min_confidence=0.4,
        min_confidence_floor=0.2,
        stop_loss_pct=0.03,
        take_profit_pct=0.05,
    )

    assert pytest.approx(strategy.signal_generator.long_entry_threshold, rel=1e-9) == 0.0007
    assert pytest.approx(strategy.signal_generator.short_entry_threshold, rel=1e-9) == -0.0008
    assert pytest.approx(strategy.signal_generator.confidence_multiplier, rel=1e-9) == 15.0
    assert pytest.approx(strategy.position_sizer.base_fraction, rel=1e-9) == 0.12
    assert pytest.approx(strategy.position_sizer.min_confidence, rel=1e-9) == 0.4
    assert pytest.approx(strategy.position_sizer.min_confidence_floor, rel=1e-9) == 0.2


def test_default_thresholds_unchanged_when_no_override() -> None:
    """Baseline guardrail: defaults preserved so runs without overrides are parity."""
    from src.strategies.components.ml_signal_generator import (
        MLBasicSignalGenerator,
        MLSignalGenerator,
    )

    basic_sg = MLBasicSignalGenerator()
    assert basic_sg.long_entry_threshold == MLBasicSignalGenerator.LONG_ENTRY_THRESHOLD
    assert basic_sg.short_entry_threshold == MLBasicSignalGenerator.SHORT_ENTRY_THRESHOLD
    assert basic_sg.confidence_multiplier == MLBasicSignalGenerator.CONFIDENCE_MULTIPLIER

    adaptive_sg = MLSignalGenerator()
    assert adaptive_sg.long_entry_threshold == MLSignalGenerator.LONG_ENTRY_THRESHOLD
    assert adaptive_sg.short_entry_threshold == MLSignalGenerator.SHORT_ENTRY_THRESHOLD
    assert adaptive_sg.confidence_multiplier == MLSignalGenerator.CONFIDENCE_MULTIPLIER
    assert adaptive_sg.short_threshold_trend_up == MLSignalGenerator.SHORT_THRESHOLD_TREND_UP
    assert adaptive_sg.short_threshold_trend_down == MLSignalGenerator.SHORT_THRESHOLD_TREND_DOWN
    assert adaptive_sg.short_threshold_range == MLSignalGenerator.SHORT_THRESHOLD_RANGE
    assert adaptive_sg.short_threshold_high_vol == MLSignalGenerator.SHORT_THRESHOLD_HIGH_VOL
    assert adaptive_sg.short_threshold_low_vol == MLSignalGenerator.SHORT_THRESHOLD_LOW_VOL
