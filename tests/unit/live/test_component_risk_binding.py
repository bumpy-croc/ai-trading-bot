import pytest

from src.live.trading_engine import LiveTradingEngine
from src.risk.risk_manager import RiskManager, RiskParameters
from src.strategies.components import Strategy as ComponentStrategy
from src.strategies.components.risk_adapter import CoreRiskAdapter


class DummyComponent:
    def __init__(self, name: str) -> None:
        self.name = name
        self.warmup_period = 0

    def get_feature_generators(self):  # pragma: no cover - interface compatibility
        return []


class DummySizer(DummyComponent):
    def size_position(self, *args, **kwargs):  # pragma: no cover - interface compatibility
        return 0.0


def _build_component_strategy(risk_params: RiskParameters) -> ComponentStrategy:
    core_manager = RiskManager(risk_params)
    adapter = CoreRiskAdapter(core_manager)
    return ComponentStrategy(
        name="test_component",
        signal_generator=DummyComponent("signal"),
        risk_manager=adapter,
        position_sizer=DummySizer("sizer"),
        regime_detector=None,
        enable_logging=False,
    )


def test_engine_uses_component_risk_parameters_when_none_provided():
    component_params = RiskParameters(
        base_risk_per_trade=0.015,
        max_position_size=0.1,
        default_take_profit_pct=0.04,
    )
    strategy = _build_component_strategy(component_params)

    engine = LiveTradingEngine(
        strategy=strategy,
        data_provider=object(),
        enable_live_trading=False,
        log_trades=False,
    )

    params = engine.risk_manager.params
    assert params.base_risk_per_trade == pytest.approx(0.015)
    assert params.max_position_size == pytest.approx(0.1)
    assert params.default_take_profit_pct == pytest.approx(0.04)


def test_engine_merges_engine_and_component_parameters():
    component_params = RiskParameters(
        base_risk_per_trade=0.015,
        max_position_size=0.1,
        default_take_profit_pct=0.04,
    )
    engine_params = RiskParameters(
        base_risk_per_trade=0.025,
        max_position_size=0.2,
    )

    strategy = _build_component_strategy(component_params)

    engine = LiveTradingEngine(
        strategy=strategy,
        data_provider=object(),
        risk_parameters=engine_params,
        enable_live_trading=False,
        log_trades=False,
    )

    params = engine.risk_manager.params
    assert params.base_risk_per_trade == pytest.approx(0.025)
    assert params.max_position_size == pytest.approx(0.2)
    assert params.default_take_profit_pct == pytest.approx(0.04)


def test_engine_preserves_component_overrides_for_default_engine_parameters():
    component_params = RiskParameters(
        trailing_activation_threshold=None,
        trailing_distance_pct=None,
        partial_exit_targets=[],
        partial_exit_sizes=[],
        scale_in_thresholds=[],
        scale_in_sizes=[],
    )

    strategy = _build_component_strategy(component_params)

    engine = LiveTradingEngine(
        strategy=strategy,
        data_provider=object(),
        risk_parameters=RiskParameters(),
        enable_live_trading=False,
        log_trades=False,
    )

    params = engine.risk_manager.params
    assert params.trailing_activation_threshold is None
    assert params.trailing_distance_pct is None
    assert params.partial_exit_targets == []
    assert params.partial_exit_sizes == []
    assert params.scale_in_thresholds == []
    assert params.scale_in_sizes == []
