"""Security regression tests for cloud orchestrator path handling.

Covers the validation branches that guard destructive rmtree/copytree/symlink
operations in the local model-registry sync against attacker-controlled
``metadata.json`` fields and registry-escaping paths.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.ml.cloud.exceptions import ArtifactSyncError
from src.ml.cloud.orchestrator import (
    CloudTrainingOrchestrator,
    _validate_registry_component,
)


class TestValidateRegistryComponent:
    @pytest.mark.fast
    @pytest.mark.parametrize("value", ["2025-10-27_14h_v1", "basic", "sentiment", "v1.2.3"])
    def test_accepts_simple_identifiers(self, value):
        assert _validate_registry_component(value, "version_id") == value

    @pytest.mark.fast
    @pytest.mark.parametrize(
        "value",
        ["..", ".", "../../etc", "a/b", "a\\b", "", "x/../y", "/abs"],
    )
    def test_rejects_unsafe_values(self, value):
        with pytest.raises(ArtifactSyncError):
            _validate_registry_component(value, "model_type")


class TestSyncLocalArtifactsContainment:
    @pytest.mark.fast
    def test_rejects_path_escaping_registry(self, tmp_path):
        """A model_type/version that resolves outside the registry must raise
        before any filesystem mutation."""
        # Bypass __init__ — the containment check does not use instance state.
        orch = object.__new__(CloudTrainingOrchestrator)
        registry = tmp_path / "registry"
        registry.mkdir()
        artifact = tmp_path / "artifact"
        artifact.mkdir()

        with pytest.raises(ArtifactSyncError, match="escapes registry root"):
            orch._sync_local_artifacts(
                artifact_path=artifact,
                local_registry=registry,
                symbol="BTCUSDT",
                model_type="../../../../tmp",
                version_id="evil",
            )

        # Nothing was created outside the registry.
        assert not (tmp_path / "tmp").exists()

    @pytest.mark.fast
    def test_allows_normal_path_into_registry(self, tmp_path):
        orch = object.__new__(CloudTrainingOrchestrator)
        registry = tmp_path / "registry"
        registry.mkdir()
        artifact = tmp_path / "artifact"
        artifact.mkdir()
        (artifact / "model.onnx").write_text("x")

        result = orch._sync_local_artifacts(
            artifact_path=artifact,
            local_registry=registry,
            symbol="BTCUSDT",
            model_type="basic",
            version_id="2025-10-27_14h_v1",
        )
        expected = registry / "BTCUSDT" / "basic" / "2025-10-27_14h_v1"
        assert Path(result) == expected
        assert (expected / "model.onnx").exists()
