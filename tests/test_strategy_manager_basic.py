from live.strategy_manager import StrategyManager


def test_strategy_manager_load_ml_basic():
    sm = StrategyManager()
    strategy = sm.load_strategy("ml_basic")
    assert strategy is not None
    assert strategy.name.lower() == "mlbasic"