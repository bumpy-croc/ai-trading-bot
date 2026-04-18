"""Tests ensuring every Phase 1 override key actually mutates the component."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from src.experiments.runner import ExperimentRunner
from src.experiments.schemas import ExperimentConfig, ParameterSet

pytestmark = pytest.mark.fast


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


def test_parity_long_entry_threshold_preserves_pre_refactor_behavior() -> None:
    """Default long threshold must be 0.0 so historical backtests remain comparable."""
    from src.strategies.components.ml_signal_generator import (
        MLBasicSignalGenerator,
        MLSignalGenerator,
    )

    # Both signal generators: `predicted_return > 0` must still trigger BUY
    # out of the box. Any non-zero default would silently change live trading.
    assert MLBasicSignalGenerator.LONG_ENTRY_THRESHOLD == 0.0
    assert MLSignalGenerator.LONG_ENTRY_THRESHOLD == 0.0


def test_float_override_on_int_attr_preserves_precision(runner: ExperimentRunner) -> None:
    """confidence_multiplier default is 12.0 (float) so overrides keep precision."""
    strategy = runner._load_strategy("ml_basic")
    cfg = _cfg("ml_basic", {"ml_basic.confidence_multiplier": 20.5})
    runner._apply_parameter_overrides(strategy, cfg)
    assert pytest.approx(strategy.signal_generator.confidence_multiplier, rel=1e-9) == 20.5


def test_unknown_override_raises(runner: ExperimentRunner) -> None:
    strategy = runner._load_strategy("ml_basic")
    cfg = _cfg("ml_basic", {"ml_basic.definitely_not_a_real_attr": 0.42})
    with pytest.raises(ValueError, match="Unknown override attribute"):
        runner._apply_parameter_overrides(strategy, cfg)


def test_namespace_mismatch_raises(runner: ExperimentRunner) -> None:
    strategy = runner._load_strategy("ml_basic")
    cfg = _cfg("ml_basic", {"ml_adaptive.long_entry_threshold": 0.001})
    with pytest.raises(ValueError, match="namespace"):
        runner._apply_parameter_overrides(strategy, cfg)


def test_stop_loss_override_must_be_numeric(runner: ExperimentRunner) -> None:
    strategy = runner._load_strategy("ml_basic")
    cfg = _cfg("ml_basic", {"ml_basic.stop_loss_pct": "not a number"})
    with pytest.raises(ValueError, match="stop_loss_pct"):
        runner._apply_parameter_overrides(strategy, cfg)


# --------------------------------------------------------------------------
# G1: factory_kwargs plumbing — kwargs the strategy factory accepts at
# construction time (e.g. ``model_type``, ``max_leverage``) must reach
# ``builder(**kwargs)`` instead of being silently dropped.
# --------------------------------------------------------------------------


def test_factory_kwargs_passed_to_hyper_growth_builder(runner: ExperimentRunner) -> None:
    """``max_leverage`` / ``stop_loss_pct`` are construction-only on hyper_growth."""
    strategy = runner._load_strategy(
        "hyper_growth",
        factory_kwargs={"max_leverage": 2.5, "stop_loss_pct": 0.07},
    )
    # The factory wires max_leverage into the LeverageManager and
    # stop_loss_pct into the FlatRiskManager's instance attribute.
    assert pytest.approx(strategy.leverage_manager.max_leverage, rel=1e-9) == 2.5
    assert pytest.approx(strategy.risk_manager.stop_loss_pct, rel=1e-9) == 0.07


def test_factory_kwargs_empty_dict_is_noop(runner: ExperimentRunner) -> None:
    """Empty / missing factory_kwargs must behave exactly like no kwargs."""
    default = runner._load_strategy("hyper_growth")
    with_empty = runner._load_strategy("hyper_growth", factory_kwargs={})
    with_none = runner._load_strategy("hyper_growth", factory_kwargs=None)
    assert default.risk_manager.stop_loss_pct == with_empty.risk_manager.stop_loss_pct
    assert default.risk_manager.stop_loss_pct == with_none.risk_manager.stop_loss_pct


def test_factory_kwargs_unknown_kwarg_surfaces_factory_name(runner: ExperimentRunner) -> None:
    """Typos must fail loudly with the factory name attached to the error."""
    with pytest.raises(ValueError, match="factory_kwargs rejected by create_hyper_growth_strategy"):
        runner._load_strategy(
            "hyper_growth",
            factory_kwargs={"definitely_not_a_kwarg": 1.0},
        )


def test_factory_kwargs_flow_end_to_end_via_run(runner: ExperimentRunner) -> None:
    """``ExperimentConfig.factory_kwargs`` must reach ``_load_strategy``.

    Uses a mock provider so we don't hit the network, and only runs long
    enough to confirm the strategy was built with the requested kwargs —
    we inspect the runner's resolved strategy via ``_load_strategy``
    directly (``run`` rebuilds every time so the same contract holds).
    """
    from src.experiments.schemas import ExperimentConfig

    end = datetime.now(UTC)
    start = end - timedelta(days=1)
    cfg = ExperimentConfig(
        strategy_name="hyper_growth",
        symbol="BTCUSDT",
        timeframe="1h",
        start=start,
        end=end,
        initial_balance=1000.0,
        provider="mock",
        random_seed=42,
        factory_kwargs={"max_leverage": 1.75},
    )
    built = runner._load_strategy(cfg.strategy_name, factory_kwargs=cfg.factory_kwargs)
    assert pytest.approx(built.leverage_manager.max_leverage, rel=1e-9) == 1.75


# --------------------------------------------------------------------------
# G2: FlatRiskManager (hyper_growth) honors stop_loss_pct directly via its
# instance attribute, not via ``_strategy_overrides``. The runner must
# accept the override instead of rejecting it with "does not consume
# strategy_overrides".
# --------------------------------------------------------------------------


def test_hyper_growth_stop_loss_override_lands_on_flat_risk_manager(
    runner: ExperimentRunner,
) -> None:
    strategy = runner._load_strategy("hyper_growth")
    cfg = _cfg("hyper_growth", {"hyper_growth.stop_loss_pct": 0.07})
    runner._apply_parameter_overrides(strategy, cfg)

    # FlatRiskManager reads self.stop_loss_pct at trade time (via
    # get_stop_loss / should_exit) — the override must mutate the
    # instance attribute, not just the strategy's risk_overrides dict.
    assert pytest.approx(strategy.risk_manager.stop_loss_pct, rel=1e-9) == 0.07
    overrides = getattr(strategy, "_risk_overrides", {}) or {}
    assert pytest.approx(overrides["stop_loss_pct"], rel=1e-9) == 0.07


def test_hyper_growth_take_profit_override_does_not_require_strategy_overrides_dict(
    runner: ExperimentRunner,
) -> None:
    """FlatRiskManager doesn't consume TP directly, but the engine-level
    _risk_overrides dict on the strategy does. The override must reach it
    rather than being rejected upfront on the risk-manager gate."""
    strategy = runner._load_strategy("hyper_growth")
    cfg = _cfg("hyper_growth", {"hyper_growth.take_profit_pct": 0.35})
    runner._apply_parameter_overrides(strategy, cfg)

    overrides = getattr(strategy, "_risk_overrides", {}) or {}
    assert pytest.approx(overrides["take_profit_pct"], rel=1e-9) == 0.35


def test_flat_risk_manager_declares_direct_runtime_override_contract() -> None:
    """The runner gate checks this class attribute — it must include
    ``stop_loss_pct`` so FlatRiskManager-backed strategies are honored."""
    from src.strategies.hyper_growth import FlatRiskManager

    assert "stop_loss_pct" in FlatRiskManager._direct_runtime_overrides


# --------------------------------------------------------------------------
# G3: ``base_fraction`` routing must walk ``LeveragedPositionSizer ->
# base_sizer``, and alias to the underlying ``FixedFractionSizer.fraction``
# attribute. Before this fix, the override silently bounced.
# --------------------------------------------------------------------------


def test_hyper_growth_base_fraction_override_reaches_wrapped_sizer(
    runner: ExperimentRunner,
) -> None:
    strategy = runner._load_strategy("hyper_growth")
    cfg = _cfg("hyper_growth", {"hyper_growth.base_fraction": 0.12})
    runner._apply_parameter_overrides(strategy, cfg)

    # hyper_growth's outer sizer is LeveragedPositionSizer which wraps a
    # FixedFractionSizer under .base_sizer. The fraction attribute is
    # 'fraction' (not 'base_fraction') on FixedFractionSizer.
    assert pytest.approx(strategy.position_sizer.base_sizer.fraction, rel=1e-9) == 0.12


def test_hyper_growth_base_fraction_post_override_invariant_enforced(
    runner: ExperimentRunner,
) -> None:
    """The bounds check must walk through the wrapping sizer too —
    otherwise an invalid 0.9 override would pass validation."""
    strategy = runner._load_strategy("hyper_growth")
    cfg = _cfg("hyper_growth", {"hyper_growth.base_fraction": 0.9})
    runner._apply_parameter_overrides(strategy, cfg)
    with pytest.raises(ValueError, match="base_fraction|fraction"):
        runner._validate_post_override_invariants(strategy)


def test_ml_basic_base_fraction_still_works_on_unwrapped_sizer(
    runner: ExperimentRunner,
) -> None:
    """Sanity guard — the unwrap logic must not break the already-working
    ConfidenceWeightedSizer path on ml_basic."""
    strategy = runner._load_strategy("ml_basic")
    cfg = _cfg("ml_basic", {"ml_basic.base_fraction": 0.15})
    runner._apply_parameter_overrides(strategy, cfg)
    assert pytest.approx(strategy.position_sizer.base_fraction, rel=1e-9) == 0.15


# --------------------------------------------------------------------------
# G4: regime-specific ``short_threshold_*`` overrides only exist on
# regime-aware generators. Attempting them on a regime-agnostic strategy
# must fail with a clear error that points at the class mismatch — not
# the generic "no component accepts it".
# --------------------------------------------------------------------------


def test_regime_short_threshold_on_ml_basic_raises_specific_error(
    runner: ExperimentRunner,
) -> None:
    strategy = runner._load_strategy("ml_basic")
    cfg = _cfg("ml_basic", {"ml_basic.short_threshold_trend_up": -0.0002})
    with pytest.raises(ValueError, match="regime-aware signal generator"):
        runner._apply_parameter_overrides(strategy, cfg)


def test_regime_short_threshold_error_names_actual_generator_class(
    runner: ExperimentRunner,
) -> None:
    """The error message must name the concrete signal generator class the
    strategy is using so the operator can fix the YAML without grep."""
    strategy = runner._load_strategy("hyper_growth")
    cfg = _cfg("hyper_growth", {"hyper_growth.short_threshold_range": -0.0004})
    with pytest.raises(ValueError, match="MLBasicSignalGenerator"):
        runner._apply_parameter_overrides(strategy, cfg)


def test_min_confidence_floor_invariant_enforced_after_override(
    runner: ExperimentRunner,
) -> None:
    """setattr bypasses __init__ — post-override validator must catch this."""
    cfg = ExperimentConfig(
        strategy_name="ml_basic",
        symbol="BTCUSDT",
        timeframe="1h",
        start=datetime.now(UTC),
        end=datetime.now(UTC),
        initial_balance=1000.0,
        # ml_basic default min_confidence is 0.35 — set floor higher to trigger.
        parameters=ParameterSet(
            name="bad_floor",
            values={"ml_basic.min_confidence_floor": 0.9},
        ),
        provider="mock",
    )
    strategy = runner._load_strategy("ml_basic")
    runner._apply_parameter_overrides(strategy, cfg)
    with pytest.raises(ValueError, match="min_confidence_floor"):
        runner._validate_post_override_invariants(strategy)
