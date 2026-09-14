"""Regression tests for the bundle-metadata contract (#1049).

Cloud-trained bundles used to omit ``price_normalization``; both denormalization
sites treat its absence as "nothing to do", so a model emitting normalized
~[0,1] output was compared against real prices with no error raised.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from src.ml.cloud.exceptions import ModelPromotionError
from src.ml.cloud.promotion import promote_model_version
from src.ml.model_metadata import (
    ensure_bundle_metadata_complete,
    missing_prediction_keys,
    uses_rolling_minmax_features,
    validate_bundle_metadata,
)
from src.prediction.config import PredictionConfig
from src.prediction.models.exceptions import ModelLoadError, ModelNotAvailableError
from src.prediction.models.registry import PredictionModelRegistry

PRICE_ONLY_FEATURES = [
    "close_normalized",
    "volume_normalized",
    "high_normalized",
    "low_normalized",
    "open_normalized",
]

# The keys a locally-trained bundle carries; a cloud bundle must match on these.
LOCAL_BUNDLE_KEYS = ("price_normalization", "model_file", "framework")


def _cloud_style_metadata(**overrides) -> dict:
    """Metadata as the in-container trainer writes it: no prediction keys."""
    metadata = {
        "symbol": "ETHUSDT",
        "model_type": "price",
        "timeframe": "1h",
        "version_id": "2026-08-30_07h_v1",
        "task_type": "regression",
        "sequence_length": 120,
        "feature_names": list(PRICE_ONLY_FEATURES),
    }
    metadata.update(overrides)
    return metadata


def _write_bundle(root: Path, symbol: str, model_type: str, version: str, metadata: dict) -> Path:
    base = root / symbol / model_type / version
    base.mkdir(parents=True, exist_ok=True)
    (base / "model.onnx").write_bytes(b"dummy")
    (base / "metadata.json").write_text(json.dumps(metadata))
    return base


def test_rolling_minmax_detected_from_features():
    assert uses_rolling_minmax_features(_cloud_style_metadata()) is True


def test_non_normalized_bundle_is_not_flagged():
    metadata = _cloud_style_metadata(feature_names=["rsi", "macd", "atr"])
    assert uses_rolling_minmax_features(metadata) is False
    assert missing_prediction_keys(metadata) == []


def test_classification_bundle_is_not_flagged():
    metadata = _cloud_style_metadata(task_type="binary_classification")
    assert uses_rolling_minmax_features(metadata) is False


def test_cloud_bundle_reaches_local_schema_after_enrichment(tmp_path: Path):
    bundle = _write_bundle(tmp_path, "ETHUSDT", "price", "v1", _cloud_style_metadata())

    added = ensure_bundle_metadata_complete(bundle)

    assert sorted(added) == sorted(LOCAL_BUNDLE_KEYS)
    enriched = json.loads((bundle / "metadata.json").read_text())
    for key in LOCAL_BUNDLE_KEYS:
        assert enriched[key], f"{key} still missing after enrichment"
    assert enriched["price_normalization"] == {
        "method": "rolling_minmax",
        "window": 120,
        "target_feature": "close",
    }
    validate_bundle_metadata(enriched, bundle_id="ETHUSDT/price/v1")


def test_enrichment_writes_atomically(tmp_path: Path):
    """#1132: ensure_bundle_metadata_complete must write metadata.json via
    write-temp-then-rename (matching gate.py's validation_audit.json and
    meta_labels.py's checkpoint pattern), not a direct in-place json.dump, so
    a crash mid-write never leaves a torn metadata.json -- the exact file the
    registry's fail-loud check gates loading on.

    Real atomicity is a timing property that can't be forced from an
    interrupted process in a test; this pins the write-temp-then-rename
    shape (a distinct temp path is written first, then os.replace()'d onto
    the real path) rather than the previous single json.dump(..., handle).
    """
    bundle = _write_bundle(tmp_path, "ETHUSDT", "price", "v1", _cloud_style_metadata())
    metadata_path = bundle / "metadata.json"
    tmp_path_expected = metadata_path.with_suffix(".json.tmp")

    real_replace = os.replace
    calls: list[tuple[Path, Path]] = []

    def _spy_replace(src, dst):
        src, dst = Path(src), Path(dst)
        # The temp file must be fully written (and the real metadata.json
        # must still hold its pre-enrichment content) at the moment of
        # rename -- proves the write happens before, not during, the swap.
        assert src.exists()
        calls.append((src, dst))
        return real_replace(src, dst)

    with patch("src.ml.model_metadata.os.replace", side_effect=_spy_replace):
        added = ensure_bundle_metadata_complete(bundle)

    assert added
    assert calls == [(tmp_path_expected, metadata_path)]
    assert not tmp_path_expected.exists()
    enriched = json.loads(metadata_path.read_text())
    assert enriched["price_normalization"]["method"] == "rolling_minmax"


def test_enrichment_is_idempotent_and_never_overwrites(tmp_path: Path):
    existing = {"method": "rolling_minmax", "window": 60, "target_feature": "close"}
    bundle = _write_bundle(
        tmp_path,
        "ETHUSDT",
        "price",
        "v1",
        _cloud_style_metadata(price_normalization=existing),
    )

    ensure_bundle_metadata_complete(bundle)
    assert ensure_bundle_metadata_complete(bundle) == []
    assert json.loads((bundle / "metadata.json").read_text())["price_normalization"] == existing


def test_loading_bundle_without_price_normalization_raises(tmp_path: Path, monkeypatch):
    reg_root = tmp_path / "models"
    bundle = _write_bundle(
        reg_root, "ETHUSDT", "basic", "2026-08-30_07h_v1", _cloud_style_metadata()
    )

    cfg = PredictionConfig.from_config_manager()
    monkeypatch.setattr(cfg, "model_registry_path", str(reg_root))
    registry = PredictionModelRegistry(cfg)

    with pytest.raises(ModelLoadError, match="price_normalization"):
        registry._load_bundle("ETHUSDT", "basic", bundle)


def test_incomplete_bundle_is_never_served(tmp_path: Path, monkeypatch):
    """The registry scan must not index a bundle it cannot denormalize.

    Failing to "no model available" makes the strategy fail safe to HOLD; the
    hazard #1049 created was serving normalized output as a real price.
    """
    reg_root = tmp_path / "models"
    _write_bundle(reg_root, "ETHUSDT", "basic", "2026-08-30_07h_v1", _cloud_style_metadata())

    cfg = PredictionConfig.from_config_manager()
    monkeypatch.setattr(cfg, "model_registry_path", str(reg_root))
    registry = PredictionModelRegistry(cfg)

    with pytest.raises(ModelNotAvailableError):
        registry.select_bundle(symbol="ETHUSDT", model_type="basic", timeframe="1h")
    assert (
        registry.get_bundle_by_key("ETHUSDT:1h:basic:2026-08-30_07h_v1") is None
    ), "a pinned version that cannot be denormalized must not be returned"


def test_loading_succeeds_once_metadata_is_complete(tmp_path: Path, monkeypatch):
    reg_root = tmp_path / "models"
    bundle = _write_bundle(
        reg_root, "ETHUSDT", "basic", "2026-08-30_07h_v1", _cloud_style_metadata()
    )
    ensure_bundle_metadata_complete(bundle)

    cfg = PredictionConfig.from_config_manager()
    monkeypatch.setattr(cfg, "model_registry_path", str(reg_root))
    registry = PredictionModelRegistry(cfg)

    loaded = registry.select_bundle(symbol="ETHUSDT", model_type="basic", timeframe="1h")
    assert loaded.metadata["price_normalization"]["method"] == "rolling_minmax"


def test_promotion_refuses_incomplete_bundle(tmp_path: Path):
    reg_root = tmp_path / "models"
    _write_bundle(reg_root, "ETHUSDT", "price", "v1", _cloud_style_metadata())

    with pytest.raises(ModelPromotionError, match="price_normalization"):
        promote_model_version("ETHUSDT", "v1", registry_root=reg_root)

    assert not (reg_root / "ETHUSDT" / "basic" / "v1").exists()


def test_promotion_allows_complete_bundle(tmp_path: Path):
    reg_root = tmp_path / "models"
    bundle = _write_bundle(reg_root, "ETHUSDT", "price", "v1", _cloud_style_metadata())
    ensure_bundle_metadata_complete(bundle)

    promoted = promote_model_version("ETHUSDT", "v1", registry_root=reg_root)

    assert promoted.is_dir()
    assert (promoted / "metadata.json").exists()
