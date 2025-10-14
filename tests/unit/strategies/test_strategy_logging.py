from src.strategies.ml_basic import MlBasic


class Recorder:
    def __init__(self):
        self.calls = []

    def log_strategy_execution(self, **kwargs):
        self.calls.append(kwargs)


def test_strategy_log_execution_defaults_to_trading_pair():
    strategy = MlBasic()
    recorder = Recorder()
    strategy.set_database_manager(recorder)

    strategy.log_execution(
        signal_type="entry",
        action_taken="buy",
        price=101.0,
        signal_strength=0.8,
        confidence_score=0.9,
        position_size=0.5,
        reasons=["test"],
    )

    assert len(recorder.calls) == 1
    call = recorder.calls[0]
    assert call["symbol"] == strategy.trading_pair
    assert call["strategy_name"] == strategy.__class__.__name__


def test_strategy_log_execution_merges_additional_context():
    strategy = MlBasic()
    recorder = Recorder()
    strategy.set_database_manager(recorder)

    strategy.log_execution(
        signal_type="exit",
        action_taken="sell",
        price=99.0,
        additional_context={"sl": "tight", "note": "risk"},
    )

    recorded = recorder.calls[0]
    assert "sl=tight" in recorded["reasons"]
    assert "note=risk" in recorded["reasons"]
