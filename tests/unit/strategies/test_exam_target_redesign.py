"""Unit tests for the TARGET-REDESIGN tournament's exam-only strategy factory.

Preregistration §5: a new exam-only strategy factory, mirroring
src/strategies/ml_basic.py's wiring pattern exactly (CoreRiskAdapter(
EngineRiskManager(RiskParameters(...))) + ConfidenceWeightedSizer(
base_fraction=0.2, min_confidence=0.3)) -- NOT HyperGrowth's FlatRiskManager
+ FixedFractionSizer(adjust_for_confidence=False), which #938 proved cannot
express confidence/magnitude differences at all.
"""

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
from src.strategies.exam_target_redesign import create_exam_strategy

pytestmark = pytest.mark.unit


class TestCreateExamStrategy:
    def test_returns_a_strategy(self):
        strategy = create_exam_strategy(signal_generator=HoldSignalGenerator())

        assert isinstance(strategy, Strategy)
        assert strategy.signal_generator is not None
        assert strategy.risk_manager is not None
        assert strategy.position_sizer is not None
        assert strategy.regime_detector is not None

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
