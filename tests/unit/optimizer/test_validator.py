from datetime import UTC, datetime, timedelta

from src.optimizer.schemas import ExperimentConfig, ExperimentResult
from src.optimizer.validator import StatisticalValidator, ValidationConfig


def _mk_result(ann_ret: float, mdd: float) -> ExperimentResult:
    now = datetime.now(UTC)
    cfg = ExperimentConfig(
        strategy_name="ml_basic",
        symbol="BTCUSDT",
        timeframe="1h",
        start=now - timedelta(days=7),
        end=now,
        initial_balance=1000.0,
    )
    return ExperimentResult(
        config=cfg,
        total_trades=50,
        win_rate=50.0,
        total_return=5.0,
        annualized_return=ann_ret,
        max_drawdown=mdd,
        sharpe_ratio=1.0,
        final_balance=1050.0,
    )


def test_validator_reports_pass_when_candidate_better():
    baseline = [_mk_result(ann_ret=10.0, mdd=10.0) for _ in range(3)]
    candidate = [_mk_result(ann_ret=15.0, mdd=8.0) for _ in range(3)]

    validator = StatisticalValidator(
        ValidationConfig(bootstrap_samples=200, p_value_threshold=0.5, min_effect_size=0.1)
    )
    report = validator.validate(baseline, candidate)

    assert report.p_value >= 0.0 and report.p_value <= 1.0
    assert report.effect_size != 0.0
    assert report.passed is True
