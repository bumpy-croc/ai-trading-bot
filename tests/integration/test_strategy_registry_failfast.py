import json
from pathlib import Path

import pandas as pd
import pytest

from src.prediction.config import PredictionConfig
from src.strategies.ml_basic import MlBasic

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


def test_strategy_uses_registry_fail_fast(tmp_path: Path, monkeypatch):
    # Only BTC is present, ETH missing
    root = tmp_path / "models"
    _write_bundle(root, "BTCUSDT", "basic", "1h", "2025-01-01_1h_v1")

    cfg = PredictionConfig.from_config_manager()
    monkeypatch.setattr(cfg, "model_registry_path", str(root))

    # Initialize strategy with registry-enabled engine
    s = MlBasic(use_prediction_engine=True, model_type="basic", timeframe="1h")

    # Run calculate_indicators; with single symbol it should proceed fine
    df = _make_price_df()
    out = s.calculate_indicators(df)
    assert "ml_prediction" in out.columns


