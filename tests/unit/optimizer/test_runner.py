from datetime import datetime, timedelta

from src.optimizer.runner import ExperimentRunner
from src.optimizer.schemas import ExperimentConfig, ParameterSet


def test_runner_with_mock_provider_executes():
    runner = ExperimentRunner()
    end = datetime.now()
    start = end - timedelta(days=3)

    cfg = ExperimentConfig(
        strategy_name="ml_basic",
        symbol="BTCUSDT",
        timeframe="1h",
        start=start,
        end=end,
        initial_balance=1000.0,
        provider="mock",
        use_cache=False,
    )

    result = runner.run(cfg)

    # Basic shape assertions
    assert result.config.strategy_name == "ml_basic"
    assert result.final_balance >= 0
    assert isinstance(result.total_trades, int)
    assert isinstance(result.win_rate, float)
    assert isinstance(result.sharpe_ratio, float)


def test_runner_applies_parameter_overrides_to_components():
    runner = ExperimentRunner()
    strategy = runner._load_strategy("ml_basic")

    start = datetime.now() - timedelta(days=1)
    end = datetime.now()
    cfg = ExperimentConfig(
        strategy_name="ml_basic",
        symbol="BTCUSDT",
        timeframe="1h",
        start=start,
        end=end,
        initial_balance=1000.0,
        parameters=ParameterSet(
            name="override",
            values={
                "MlBasic.stop_loss_pct": 0.05,
                "MlBasic.risk_per_trade": 0.03,
                "MlBasic.base_fraction": 0.15,
                "MlBasic.sequence_length": 60,
            },
        ),
    )

    runner._apply_parameter_overrides(strategy, cfg)

    assert strategy.risk_manager.stop_loss_pct == 0.05
    assert strategy.risk_manager.risk_per_trade == 0.03
    assert strategy.position_sizer.base_fraction == 0.15
    assert strategy.signal_generator.sequence_length == 60
