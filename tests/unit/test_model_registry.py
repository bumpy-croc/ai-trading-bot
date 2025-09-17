from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.prediction.config import PredictionConfig
from src.prediction.models.exceptions import ModelLoadError
from src.prediction.models.registry import PredictionModelRegistry, StrategyModel


def _make_bundle(tmpdir: Path, symbol: str, model_type: str, timeframe: str, version: str) -> Path:
    base = tmpdir / symbol / model_type / version
    base.mkdir(parents=True, exist_ok=True)
    (base / "model.onnx").write_bytes(b"dummy")
    (base / "metadata.json").write_text(
        json.dumps(
            {
                "symbol": symbol,
                "model_type": model_type,
                "timeframe": timeframe,
                "version_id": version,
            }
        )
    )
    return base


def test_select_bundle_and_many(tmp_path: Path, monkeypatch):
    reg_root = tmp_path / "models"
    _make_bundle(reg_root, "BTCUSDT", "basic", "1h", "2025-01-01_1h_v1")
    _make_bundle(reg_root, "ETHUSDT", "basic", "1h", "2025-01-01_1h_v1")

    cfg = PredictionConfig.from_config_manager()
    monkeypatch.setattr(cfg, "model_registry_path", str(reg_root))

    reg = PredictionModelRegistry(cfg)
    b = reg.select_bundle(symbol="BTCUSDT", model_type="basic", timeframe="1h")
    assert isinstance(b, StrategyModel)
    sel = reg.select_many([
        ("BTCUSDT", "basic", "1h"),
        ("ETHUSDT", "basic", "1h"),
    ])
    assert len(sel) == 2


def test_select_many_fail_fast(tmp_path: Path, monkeypatch):
    reg_root = tmp_path / "models"
    _make_bundle(reg_root, "BTCUSDT", "basic", "1h", "2025-01-01_1h_v1")

    cfg = PredictionConfig.from_config_manager()
    monkeypatch.setattr(cfg, "model_registry_path", str(reg_root))
    reg = PredictionModelRegistry(cfg)

    with pytest.raises(ModelLoadError):
        reg.select_many([
            ("BTCUSDT", "basic", "1h"),
            ("ETHUSDT", "basic", "1h"),  # missing
        ])


