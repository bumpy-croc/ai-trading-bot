"""Unit tests for the TARGET-REDESIGN tournament's exam-only strategy factory.

Preregistration §5: a new exam-only strategy factory, mirroring
src/strategies/ml_basic.py's wiring pattern exactly (CoreRiskAdapter(
EngineRiskManager(RiskParameters(...))) + ConfidenceWeightedSizer(
base_fraction=0.2, min_confidence=0.3)) -- NOT HyperGrowth's FlatRiskManager
+ FixedFractionSizer(adjust_for_confidence=False), which #938 proved cannot
express confidence/magnitude differences at all.
"""

from unittest.mock import patch

import pytest

from src.config.constants import (
    DEFAULT_MAX_HOLDING_HOURS,
    DEFAULT_STOP_LOSS_PCT,
    DEFAULT_STRATEGY_BASE_FRACTION,
    DEFAULT_STRATEGY_MIN_CONFIDENCE,
    DEFAULT_TAKE_PROFIT_PCT,
)
from src.strategies.components import (
    ConfidenceWeightedSizer,
    CoreRiskAdapter,
    EnhancedRegimeDetector,
    HoldSignalGenerator,
    Strategy,
)
from src.strategies.components.exam_signal_generator import (
    ClassificationExamSignalGenerator,
    MetaLabelExamSignalGenerator,
    SmoothedReturnExamSignalGenerator,
)
from src.strategies.exam_target_redesign import (
    _MODEL_VERSION_OVERRIDE_ENV_VAR,
    create_exam_binary_direction_strategy,
    create_exam_meta_label_strategy,
    create_exam_smoothed_return_strategy,
    create_exam_strategy,
    create_exam_triple_barrier_strategy,
)

pytestmark = pytest.mark.unit


class TestCreateExamStrategy:
    def test_returns_a_strategy(self):
        strategy = create_exam_strategy(signal_generator=HoldSignalGenerator())

        assert isinstance(strategy, Strategy)
        assert strategy.signal_generator is not None
        assert strategy.risk_manager is not None
        assert strategy.position_sizer is not None
        assert strategy.regime_detector is not None

    def test_risk_overrides_position_sizer_is_fixed_fraction_not_confidence_weighted(self):
        """Regression guard found via Phase 2b item 6's real end-to-end
        acceptance run: risk_overrides["position_sizer"] is a STRING key
        consumed by the CORE risk manager's OWN, unrelated
        "confidence_weighted" fraction calculator
        (src/risk/risk_manager.py::_calculate_raw_fraction), which reads a
        "prediction_confidence" indicator that NOTHING in this codebase ever
        populates (grep confirms risk_manager.py is the only reference) --
        it silently returns fraction=0.0 always, which
        ConfidenceWeightedSizer.calculate_size then treats as a hard veto
        (risk_amount <= 0 -> position size 0), producing PERMANENT ZERO
        TRADES regardless of signal confidence or direction. The REAL
        confidence-weighting happens one layer up, via the
        ConfidenceWeightedSizer OBJECT assigned to Strategy.position_sizer
        (unaffected by this string) -- ml_basic.py's create_ml_basic_strategy
        (which this function mirrors) correctly uses "fixed_fraction" here;
        this exam scaffolding had drifted from that pattern."""
        strategy = create_exam_strategy(signal_generator=HoldSignalGenerator())

        overrides = strategy.get_risk_overrides()
        assert overrides["position_sizer"] == "fixed_fraction"

    def test_risk_manager_actually_sizes_a_confident_signal_nonzero(self):
        """End-to-end guard for the same bug: calculate_position_size (the
        risk_amount Stage-1 cap ConfidenceWeightedSizer's veto depends on)
        must return a positive value for a real, confident signal -- not
        just check the override string in isolation."""
        import pandas as pd

        from src.strategies.components.signal_generator import Signal, SignalDirection

        strategy = create_exam_strategy(signal_generator=HoldSignalGenerator())
        df = pd.DataFrame(
            {
                "open": [100.0] * 150,
                "high": [101.0] * 150,
                "low": [99.0] * 150,
                "close": [100.0] * 150,
                "volume": [1000.0] * 150,
            }
        )
        signal = Signal(direction=SignalDirection.BUY, strength=0.6, confidence=0.6, metadata={})

        risk_amount = strategy.risk_manager.calculate_position_size(
            signal, 1000.0, None, df=df, index=100, price=100.0
        )

        assert risk_amount > 0.0

    def test_uses_the_supplied_signal_generator(self):
        signal_generator = HoldSignalGenerator()
        strategy = create_exam_strategy(signal_generator=signal_generator)

        assert strategy.signal_generator is signal_generator

    def test_uses_confidence_weighted_sizer_not_fixed_fraction(self):
        """#938: FixedFractionSizer(adjust_for_confidence=False) (HyperGrowth's
        wiring) cannot express confidence/magnitude differences at all --
        this is the load-bearing fix, verified directly on the wired object."""
        strategy = create_exam_strategy(signal_generator=HoldSignalGenerator())

        assert isinstance(strategy.position_sizer, ConfidenceWeightedSizer)

    def test_confidence_weighted_sizer_default_params(self):
        strategy = create_exam_strategy(signal_generator=HoldSignalGenerator())

        sizer = strategy.position_sizer
        assert sizer.base_fraction == pytest.approx(DEFAULT_STRATEGY_BASE_FRACTION)
        assert sizer.min_confidence == pytest.approx(DEFAULT_STRATEGY_MIN_CONFIDENCE)
        assert sizer.min_confidence_floor == pytest.approx(0.0)

    def test_confidence_weighted_sizer_params_are_overridable(self):
        strategy = create_exam_strategy(
            signal_generator=HoldSignalGenerator(),
            base_fraction=0.15,
            min_confidence=0.4,
            min_confidence_floor=0.1,
        )

        sizer = strategy.position_sizer
        assert sizer.base_fraction == pytest.approx(0.15)
        assert sizer.min_confidence == pytest.approx(0.4)
        assert sizer.min_confidence_floor == pytest.approx(0.1)

    def test_uses_core_risk_adapter_wrapping_engine_risk_manager(self):
        """NOT FlatRiskManager -- the ml_basic-pattern wiring."""
        strategy = create_exam_strategy(signal_generator=HoldSignalGenerator())

        assert isinstance(strategy.risk_manager, CoreRiskAdapter)
        assert type(strategy.risk_manager).__name__ != "FlatRiskManager"

    def test_uses_enhanced_regime_detector(self):
        strategy = create_exam_strategy(signal_generator=HoldSignalGenerator())

        assert isinstance(strategy.regime_detector, EnhancedRegimeDetector)

    def test_default_stops_are_ratified_risk_limits_defaults_not_hypergrowth(self):
        """Ratified risk-limits.json defaults (5%/4%), NOT prod HyperGrowth's
        10%/30% -- per the preregistration §5/Deviation 2."""
        strategy = create_exam_strategy(signal_generator=HoldSignalGenerator())

        assert strategy.risk_manager.stop_loss_pct == pytest.approx(DEFAULT_STOP_LOSS_PCT)
        assert strategy.risk_manager.take_profit_pct == pytest.approx(DEFAULT_TAKE_PROFIT_PCT)
        assert DEFAULT_STOP_LOSS_PCT == pytest.approx(0.05)
        assert DEFAULT_TAKE_PROFIT_PCT == pytest.approx(0.04)

    def test_stops_are_overridable(self):
        strategy = create_exam_strategy(
            signal_generator=HoldSignalGenerator(),
            stop_loss_pct=0.03,
            take_profit_pct=0.06,
        )

        assert strategy.risk_manager.stop_loss_pct == pytest.approx(0.03)
        assert strategy.risk_manager.take_profit_pct == pytest.approx(0.06)

    def test_max_holding_hours_defaults_to_336(self):
        """Set on BOTH RiskParameters.time_exits (so any caller reading the
        underlying PortfolioRiskManager.params sees it) and
        Strategy._risk_overrides via set_risk_overrides (the priority-order
        lookup build_time_exit_policy actually uses) -- verified end-to-end
        through build_time_exit_policy itself."""
        from src.engines.shared.risk_configuration import build_time_exit_policy

        strategy = create_exam_strategy(signal_generator=HoldSignalGenerator())

        overrides = strategy.get_risk_overrides()
        assert overrides["time_exits"] == {"max_holding_hours": DEFAULT_MAX_HOLDING_HOURS}
        assert DEFAULT_MAX_HOLDING_HOURS == 336

        policy = build_time_exit_policy(strategy, strategy.risk_manager)
        assert policy is not None
        assert policy.max_holding_hours == DEFAULT_MAX_HOLDING_HOURS

    def test_max_holding_hours_overridable(self):
        from src.engines.shared.risk_configuration import build_time_exit_policy

        strategy = create_exam_strategy(
            signal_generator=HoldSignalGenerator(), max_holding_hours=48
        )

        policy = build_time_exit_policy(strategy, strategy.risk_manager)
        assert policy is not None
        assert policy.max_holding_hours == 48

    def test_custom_name(self):
        strategy = create_exam_strategy(
            signal_generator=HoldSignalGenerator(), name="EntrantB_BinaryDirection"
        )
        assert strategy.name == "EntrantB_BinaryDirection"

    def test_default_name(self):
        strategy = create_exam_strategy(signal_generator=HoldSignalGenerator())
        assert strategy.name


class TestPerEntrantExamStrategyFactories:
    """Per-entrant factory functions -- CLI-selectable via backtest.py's
    exam-only strategy registration (Phase 2b item 4). Each wires the
    entrant's SignalGenerator into create_exam_strategy's shared harness
    wiring, and each supports pinning to a SPECIFIC (non-latest) model
    version per fold via ATB_MODEL_VERSION_OVERRIDE -- exam-only, read
    inside this module rather than the shared registry, so it can never
    affect live trading (PredictionModelRegistry.select_bundle always
    resolves latest regardless)."""

    def test_binary_direction_uses_classification_signal_generator(self):
        strategy = create_exam_binary_direction_strategy()

        assert isinstance(strategy, Strategy)
        assert isinstance(strategy.signal_generator, ClassificationExamSignalGenerator)
        assert strategy.signal_generator.model_name is None

    def test_triple_barrier_uses_classification_signal_generator(self):
        strategy = create_exam_triple_barrier_strategy()

        assert isinstance(strategy.signal_generator, ClassificationExamSignalGenerator)
        assert strategy.signal_generator.model_name is None

    def test_smoothed_return_uses_smoothed_return_signal_generator(self):
        strategy = create_exam_smoothed_return_strategy()

        assert isinstance(strategy.signal_generator, SmoothedReturnExamSignalGenerator)
        assert strategy.signal_generator.model_name is None

    def test_binary_direction_pins_model_version_via_env_var(self, monkeypatch):
        monkeypatch.setenv(_MODEL_VERSION_OVERRIDE_ENV_VAR, "2026-01-01_1h_v1")

        strategy = create_exam_binary_direction_strategy(symbol="ETHUSDT", timeframe="4h")

        assert strategy.signal_generator.model_name == "ETHUSDT:4h:price:2026-01-01_1h_v1"

    def test_triple_barrier_pins_model_version_via_env_var(self, monkeypatch):
        monkeypatch.setenv(_MODEL_VERSION_OVERRIDE_ENV_VAR, "2026-01-01_1h_v1")

        strategy = create_exam_triple_barrier_strategy(symbol="BTCUSDT", timeframe="1h")

        assert strategy.signal_generator.model_name == "BTCUSDT:1h:price:2026-01-01_1h_v1"

    def test_smoothed_return_pins_model_version_via_env_var(self, monkeypatch):
        monkeypatch.setenv(_MODEL_VERSION_OVERRIDE_ENV_VAR, "2026-01-01_1h_v1")

        strategy = create_exam_smoothed_return_strategy(symbol="BTCUSDT", timeframe="1h")

        assert strategy.signal_generator.model_name == "BTCUSDT:1h:price:2026-01-01_1h_v1"

    def test_env_var_unset_leaves_model_name_none(self, monkeypatch):
        monkeypatch.delenv(_MODEL_VERSION_OVERRIDE_ENV_VAR, raising=False)

        strategy = create_exam_binary_direction_strategy()

        assert strategy.signal_generator.model_name is None

    def test_each_entrant_gets_a_distinct_strategy_name(self):
        names = {
            create_exam_binary_direction_strategy().name,
            create_exam_triple_barrier_strategy().name,
            create_exam_smoothed_return_strategy().name,
        }
        assert len(names) == 3

    def test_registry_model_type_and_sequence_length_are_overridable(self):
        strategy = create_exam_binary_direction_strategy(
            registry_model_type="sentiment", sequence_length=60
        )

        assert strategy.signal_generator.sequence_length == 60

    def test_uses_confidence_weighted_sizer_not_fixed_fraction(self):
        """Same #938 requirement as create_exam_strategy -- verified end-to-
        end through the per-entrant factories, not just the shared helper."""
        for strategy in (
            create_exam_binary_direction_strategy(),
            create_exam_triple_barrier_strategy(),
            create_exam_smoothed_return_strategy(),
        ):
            assert isinstance(strategy.position_sizer, ConfidenceWeightedSizer)
            assert type(strategy.risk_manager).__name__ != "FlatRiskManager"


class TestCreateExamMetaLabelStrategy:
    """Entrant (a): wraps a live MLBasicSignalGenerator (the primary) with
    MetaLabelExamSignalGenerator (the profitability gate). Mocks
    MLBasicSignalGenerator's construction -- it raises hard when no bundle
    exists for the requested symbol/model_type/timeframe (unlike the other
    three entrants' generators, which degrade gracefully to a None
    prediction_engine), so these tests must not depend on the dev
    registry's live contents."""

    def test_wires_meta_label_signal_generator_wrapping_the_primary(self):
        with patch("src.strategies.exam_target_redesign.MLBasicSignalGenerator") as mock_primary:
            mock_primary_instance = mock_primary.return_value
            strategy = create_exam_meta_label_strategy()

        assert isinstance(strategy, Strategy)
        assert isinstance(strategy.signal_generator, MetaLabelExamSignalGenerator)
        assert strategy.signal_generator.primary_signal_generator is mock_primary_instance

    def test_primary_model_type_threaded_to_ml_basic_signal_generator(self):
        with patch("src.strategies.exam_target_redesign.MLBasicSignalGenerator") as mock_primary:
            create_exam_meta_label_strategy(
                symbol="ETHUSDT", timeframe="4h", primary_model_type="sentiment"
            )

        mock_primary.assert_called_once_with(
            model_type="sentiment", timeframe="4h", symbol="ETHUSDT"
        )

    def test_min_confidence_threaded_to_meta_label_gate(self):
        with patch("src.strategies.exam_target_redesign.MLBasicSignalGenerator"):
            strategy = create_exam_meta_label_strategy(min_confidence=0.7)

        assert strategy.signal_generator.min_confidence == pytest.approx(0.7)

    def test_pins_model_version_via_env_var(self, monkeypatch):
        monkeypatch.setenv(_MODEL_VERSION_OVERRIDE_ENV_VAR, "2026-01-01_1h_v1")
        with patch("src.strategies.exam_target_redesign.MLBasicSignalGenerator"):
            strategy = create_exam_meta_label_strategy(symbol="BTCUSDT", timeframe="1h")

        assert strategy.signal_generator.model_name == "BTCUSDT:1h:meta_label:2026-01-01_1h_v1"

    def test_strategy_name(self):
        with patch("src.strategies.exam_target_redesign.MLBasicSignalGenerator"):
            strategy = create_exam_meta_label_strategy()
        assert strategy.name == "EntrantA_MetaLabel"

    def test_uses_confidence_weighted_sizer_not_fixed_fraction(self):
        with patch("src.strategies.exam_target_redesign.MLBasicSignalGenerator"):
            strategy = create_exam_meta_label_strategy()

        assert isinstance(strategy.position_sizer, ConfidenceWeightedSizer)
        assert type(strategy.risk_manager).__name__ != "FlatRiskManager"
