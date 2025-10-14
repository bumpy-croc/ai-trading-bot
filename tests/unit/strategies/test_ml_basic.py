import pandas as pd
import pytest

from src.strategies.components.signal_generator import Signal, SignalDirection
from src.strategies.ml_basic import MlBasic


@pytest.fixture()
def price_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 102.0, 103.0],
            "low": [99.0, 100.0, 101.0],
            "close": [100.5, 101.5, 102.5],
            "volume": [1_000.0, 1_100.0, 1_200.0],
        }
    )


def test_ml_basic_generates_decision(monkeypatch, price_data):
    strategy = MlBasic()

    def fake_signal(df, index, regime):
        return Signal(
            direction=SignalDirection.BUY,
            strength=0.8,
            confidence=0.9,
            metadata={}
        )

    monkeypatch.setattr(strategy.signal_generator, "generate_signal", fake_signal)
    monkeypatch.setattr(
        strategy.risk_manager,
        "calculate_position_size",
        lambda signal, balance, regime: balance * 0.1,
    )
    monkeypatch.setattr(
        strategy.position_sizer,
        "calculate_size",
        lambda signal, balance, risk_amount, regime: risk_amount,
    )

    decision = strategy.process_candle(price_data, index=2, balance=1_000.0)

    assert decision.signal.direction == SignalDirection.BUY
    assert pytest.approx(decision.position_size, rel=1e-3) == 100.0
    assert decision.metadata["strategy_name"] == strategy.name
    assert decision.metadata["market_data"]["close"] == pytest.approx(102.5)


def test_ml_basic_parameters_exposed():
    strategy = MlBasic(model_path="custom.onnx", sequence_length=64)

    params = strategy.get_parameters()

    assert params["model_path"] == "custom.onnx"
    assert params["sequence_length"] == 64
    assert params["stop_loss_pct"] == pytest.approx(0.02)
