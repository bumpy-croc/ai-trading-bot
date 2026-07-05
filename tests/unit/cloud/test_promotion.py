"""Unit tests for cloud model promotion (price/ -> basic/)."""

import json
from pathlib import Path

import pytest

from src.ml.cloud.exceptions import ModelPromotionError
from src.ml.cloud.promotion import promote_model_version

VERSION = "2026-07-05_10h00m00s_v1"


@pytest.fixture
def registry(tmp_path: Path) -> Path:
    """Create a registry with a cloud-trained bundle under BTCUSDT/price/."""
    source_dir = tmp_path / "BTCUSDT" / "price" / VERSION
    source_dir.mkdir(parents=True)
    (source_dir / "model.onnx").write_text("onnx-bytes")
    (source_dir / "feature_schema.json").write_text("{}")
    metadata = {"symbol": "BTCUSDT", "training_date": "2026-07-05", "version_id": VERSION}
    (source_dir / "metadata.json").write_text(json.dumps(metadata))
    # price/latest points at the cloud-synced bundle
    (tmp_path / "BTCUSDT" / "price" / "latest").symlink_to(VERSION)
    return tmp_path


@pytest.mark.fast
class TestPromoteModelVersion:
    """Tests for promote_model_version."""

    def test_copies_bundle_to_target_type(self, registry: Path) -> None:
        target = promote_model_version("BTCUSDT", VERSION, registry_root=registry)

        expected = registry / "BTCUSDT" / "basic" / VERSION
        assert target == expected
        assert (expected / "model.onnx").read_text() == "onnx-bytes"
        assert (expected / "metadata.json").exists()
        assert (expected / "feature_schema.json").exists()

    def test_source_bundle_left_intact(self, registry: Path) -> None:
        promote_model_version("BTCUSDT", VERSION, registry_root=registry)

        source = registry / "BTCUSDT" / "price" / VERSION
        assert (source / "model.onnx").read_text() == "onnx-bytes"

    def test_does_not_create_latest_by_default(self, registry: Path) -> None:
        promote_model_version("BTCUSDT", VERSION, registry_root=registry)

        assert not (registry / "BTCUSDT" / "basic" / "latest").is_symlink()
        assert not (registry / "BTCUSDT" / "basic" / "latest").exists()

    def test_preserves_existing_latest_by_default(self, registry: Path) -> None:
        basic_dir = registry / "BTCUSDT" / "basic"
        basic_dir.mkdir(parents=True)
        live_version = basic_dir / "2025-10-30_12h_v1"
        live_version.mkdir()
        (basic_dir / "latest").symlink_to(live_version.name)

        promote_model_version("BTCUSDT", VERSION, registry_root=registry)

        assert (basic_dir / "latest").resolve() == live_version.resolve()

    def test_set_latest_updates_symlink(self, registry: Path) -> None:
        basic_dir = registry / "BTCUSDT" / "basic"
        basic_dir.mkdir(parents=True)
        live_version = basic_dir / "2025-10-30_12h_v1"
        live_version.mkdir()
        (basic_dir / "latest").symlink_to(live_version.name)

        promote_model_version("BTCUSDT", VERSION, set_latest=True, registry_root=registry)

        assert (basic_dir / "latest").resolve() == (basic_dir / VERSION).resolve()

    def test_refuses_to_overwrite_existing_target(self, registry: Path) -> None:
        existing = registry / "BTCUSDT" / "basic" / VERSION
        existing.mkdir(parents=True)
        (existing / "model.onnx").write_text("pre-existing")

        with pytest.raises(ModelPromotionError, match="already exists"):
            promote_model_version("BTCUSDT", VERSION, registry_root=registry)

        assert (existing / "model.onnx").read_text() == "pre-existing"

    def test_missing_source_rejected(self, registry: Path) -> None:
        with pytest.raises(ModelPromotionError, match="not found"):
            promote_model_version("BTCUSDT", "2099-01-01_00h00m00s_v1", registry_root=registry)

    def test_source_without_model_file_rejected(self, registry: Path) -> None:
        bad_version = "2026-07-05_11h00m00s_v1"
        bad_dir = registry / "BTCUSDT" / "price" / bad_version
        bad_dir.mkdir(parents=True)
        (bad_dir / "metadata.json").write_text("{}")

        with pytest.raises(ModelPromotionError, match="model file"):
            promote_model_version("BTCUSDT", bad_version, registry_root=registry)

    @pytest.mark.parametrize("bad_component", ["..", "a/b", "", "x/../y"])
    def test_unsafe_path_components_rejected(self, registry: Path, bad_component: str) -> None:
        with pytest.raises(ModelPromotionError):
            promote_model_version(bad_component, VERSION, registry_root=registry)

    def test_custom_source_and_target_types(self, registry: Path) -> None:
        target = promote_model_version(
            "BTCUSDT",
            VERSION,
            source_type="price",
            target_type="scratch",
            registry_root=registry,
        )

        assert target == registry / "BTCUSDT" / "scratch" / VERSION
        assert (target / "model.onnx").exists()
