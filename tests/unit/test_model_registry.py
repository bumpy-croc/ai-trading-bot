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
    sel = reg.select_many(
        [
            ("BTCUSDT", "basic", "1h"),
            ("ETHUSDT", "basic", "1h"),
        ]
    )
    assert len(sel) == 2


def test_select_many_fail_fast(tmp_path: Path, monkeypatch):
    reg_root = tmp_path / "models"
    _make_bundle(reg_root, "BTCUSDT", "basic", "1h", "2025-01-01_1h_v1")

    cfg = PredictionConfig.from_config_manager()
    monkeypatch.setattr(cfg, "model_registry_path", str(reg_root))
    reg = PredictionModelRegistry(cfg)

    with pytest.raises(ModelLoadError):
        reg.select_many(
            [
                ("BTCUSDT", "basic", "1h"),
                ("ETHUSDT", "basic", "1h"),  # missing
            ]
        )


# ---- Cache invalidation tests ----


from unittest.mock import MagicMock, call  # noqa: E402, isort:skip


def test_invalidate_cache_maps_structured_name_to_runner(tmp_path: Path, monkeypatch):
    reg_root = tmp_path / "models"
    _make_bundle(reg_root, "BTCUSDT", "basic", "1h", "2025-01-01_1h_v1")

    cfg = PredictionConfig.from_config_manager()
    monkeypatch.setattr(cfg, "model_registry_path", str(reg_root))

    mock_cache_manager = MagicMock()
    reg = PredictionModelRegistry(cfg, mock_cache_manager)

    # First attempt returns 0 (not found), second returns 2 for runner name
    mock_cache_manager.invalidate_model.side_effect = [0, 2]

    invalidated = reg.invalidate_cache("BTCUSDT:1h:basic")

    assert invalidated == 2
    assert mock_cache_manager.invalidate_model.call_args_list == [
        call("BTCUSDT:1h:basic"),
        call("model.onnx"),
    ]


def test_invalidate_cache_clear_all(tmp_path: Path, monkeypatch):
    reg_root = tmp_path / "models"
    _make_bundle(reg_root, "BTCUSDT", "basic", "1h", "2025-01-01_1h_v1")

    cfg = PredictionConfig.from_config_manager()
    monkeypatch.setattr(cfg, "model_registry_path", str(reg_root))

    mock_cache_manager = MagicMock()
    mock_cache_manager.clear.return_value = 5
    reg = PredictionModelRegistry(cfg, mock_cache_manager)

    invalidated = reg.invalidate_cache()

    assert invalidated == 5
    mock_cache_manager.clear.assert_called_once_with()
