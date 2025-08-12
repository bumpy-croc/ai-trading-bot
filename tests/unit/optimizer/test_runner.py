from datetime import datetime, timedelta

from src.optimizer.schemas import ExperimentConfig
from src.optimizer.runner import ExperimentRunner


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