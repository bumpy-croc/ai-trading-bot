import json
from pathlib import Path

import pandas as pd
import pytest

from src.prediction.config import PredictionConfig
from src.strategies.ml_basic import create_ml_basic_strategy

pytestmark = pytest.mark.integration


def _make_price_df(n: int = 200) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": [100.0 + i * 0.1 for i in range(n)],
            "high": [101.0 + i * 0.1 for i in range(n)],
            "low": [99.0 + i * 0.1 for i in range(n)],
            "close": [100.5 + i * 0.1 for i in range(n)],
            "volume": [1000 + i for i in range(n)],
        }
    )


def _write_bundle(root: Path, symbol: str, model_type: str, timeframe: str, version: str):
    d = root / symbol / model_type / version
    d.mkdir(parents=True, exist_ok=True)
    (d / "model.onnx").write_bytes(b"dummy")
    (d / "metadata.json").write_text(
        json.dumps(
            {
                "symbol": symbol,
                "model_type": model_type,
                "timeframe": timeframe,
                "version_id": version,
            }
        )
    )


def test_strategy_uses_registry_predictions(tmp_path: Path, monkeypatch):
    # Only BTC is present, ETH missing (should still serve BTC predictions)
    root = tmp_path / "models"
    _write_bundle(root, "BTCUSDT", "basic", "1h", "2025-01-01_1h_v1")

    cfg = PredictionConfig.from_config_manager()
    monkeypatch.setattr(cfg, "model_registry_path", str(root))

    # Initialize strategy with registry-enabled engine
    strategy = create_ml_basic_strategy(
        use_prediction_engine=True,
        model_type="basic",
        timeframe="1h",
    )

    # Patch prediction lookup so the test stays deterministic
    generator = strategy.signal_generator
    invoked = {}

    def _fake_get_ml_prediction(self, df, index):
        invoked["index"] = index
        return 125.0

    monkeypatch.setattr(generator, "_get_ml_prediction", _fake_get_ml_prediction.__get__(generator))

    df = _make_price_df()
    decision = strategy.process_candle(df, index=150, balance=10_000.0)

    assert "index" in invoked  # ensure prediction was requested
    assert decision.signal.metadata.get("prediction") == 125.0
