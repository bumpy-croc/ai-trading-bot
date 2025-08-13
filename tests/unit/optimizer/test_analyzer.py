from datetime import datetime, timedelta

from src.optimizer.schemas import ExperimentConfig, ExperimentResult
from src.optimizer.analyzer import PerformanceAnalyzer, AnalyzerConfig


def _make_result(**overrides) -> ExperimentResult:
    start = datetime.now() - timedelta(days=30)
    end = datetime.now()
    cfg = ExperimentConfig(
        strategy_name="ml_basic",
        symbol="BTCUSDT",
        timeframe="1h",
        start=start,
        end=end,
        initial_balance=1000.0,
    )
    base = dict(
        config=cfg,
        total_trades=100,
        win_rate=50.0,  # percent
        total_return=10.0,
        annualized_return=15.0,
        max_drawdown=10.0,  # percent
        sharpe_ratio=1.0,
        final_balance=1100.0,
        session_id=None,
    )
    base.update(overrides)
    return ExperimentResult(**base)


def test_analyzer_suggests_reducing_risk_on_high_drawdown():
    # Max DD above threshold (default 0.15 -> 15%)
    result = _make_result(max_drawdown=25.0, sharpe_ratio=0.8, win_rate=55.0)
    analyzer = PerformanceAnalyzer(AnalyzerConfig())

    suggestions = analyzer.analyze([result])

    assert any(
        s.target == "risk" and s.change.get("risk.max_position_size", 0) < 0
        for s in suggestions
    ), "Expected risk reduction suggestion when drawdown exceeds threshold"


def test_analyzer_suggests_small_risk_increase_on_low_sharpe_with_ok_drawdown():
    # Sharpe below threshold, drawdown acceptable
    result = _make_result(max_drawdown=10.0, sharpe_ratio=0.2)
    analyzer = PerformanceAnalyzer(AnalyzerConfig())
    suggestions = analyzer.analyze([result])

    assert any(
        s.target == "risk" and s.change.get("risk.max_position_size", 0) > 0
        for s in suggestions
    ), "Expected small risk increase suggestion when Sharpe is low but drawdown ok"


def test_analyzer_suggests_strategy_sl_tp_adjustment_on_low_win_rate():
    # Win rate below threshold (default 0.45 -> 45%)
    result = _make_result(win_rate=40.0)
    analyzer = PerformanceAnalyzer(AnalyzerConfig())
    suggestions = analyzer.analyze([result])

    strat_sugg = [s for s in suggestions if s.target == "strategy:ml_basic"]
    assert strat_sugg, "Expected strategy-level suggestion for low win rate"
    change = strat_sugg[0].change
    assert "MlBasic.stop_loss_pct" in change and change["MlBasic.stop_loss_pct"] < 0
    assert "MlBasic.take_profit_pct" in change and change["MlBasic.take_profit_pct"] > 0