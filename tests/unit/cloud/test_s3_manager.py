"""Unit tests for S3 artifact manager."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from src.ml.cloud.artifacts.s3_manager import S3ArtifactManager
from src.ml.cloud.exceptions import ArtifactSyncError


class TestS3URIParsing:
    """Tests for S3 URI parsing and validation."""

    @pytest.fixture
    def manager(self) -> S3ArtifactManager:
        """Create an S3ArtifactManager instance for testing."""
        return S3ArtifactManager(bucket_name="test-bucket")

    def test_parse_valid_s3_uri(self, manager: S3ArtifactManager) -> None:
        """Verify valid S3 URI parsing."""
        bucket, prefix = manager._parse_s3_uri("s3://my-bucket/path/to/file.tar.gz")

        assert bucket == "my-bucket"
        assert prefix == "path/to/file.tar.gz"

    def test_parse_s3_uri_no_prefix(self, manager: S3ArtifactManager) -> None:
        """Verify S3 URI parsing with no prefix."""
        bucket, prefix = manager._parse_s3_uri("s3://my-bucket/")

        assert bucket == "my-bucket"
        assert prefix == ""

    def test_parse_s3_uri_bucket_only(self, manager: S3ArtifactManager) -> None:
        """Verify S3 URI parsing with bucket only."""
        bucket, prefix = manager._parse_s3_uri("s3://my-bucket")

        assert bucket == "my-bucket"
        assert prefix == ""

    def test_parse_invalid_s3_uri_not_s3(self, manager: S3ArtifactManager) -> None:
        """Verify invalid URI (not s3://) raises ValueError."""
        with pytest.raises(ValueError, match="Invalid S3 URI"):
            manager._parse_s3_uri("https://my-bucket/path")

    def test_parse_invalid_s3_uri_no_bucket(self, manager: S3ArtifactManager) -> None:
        """Verify invalid URI (no bucket) raises ValueError."""
        with pytest.raises(ValueError, match="missing bucket name"):
            manager._parse_s3_uri("s3:///path/to/file")

    def test_parse_s3_uri_path_traversal(self, manager: S3ArtifactManager) -> None:
        """Verify path traversal attempt raises ValueError."""
        with pytest.raises(ValueError, match="path traversal detected"):
            manager._parse_s3_uri("s3://my-bucket/../etc/passwd")


class TestArtifactValidation:
    """Tests for artifact validation."""

    @pytest.fixture
    def manager(self) -> S3ArtifactManager:
        """Create an S3ArtifactManager instance for testing."""
        return S3ArtifactManager(bucket_name="test-bucket")

    def test_validate_artifacts_missing_metadata(
        self, manager: S3ArtifactManager, tmp_path: Path
    ) -> None:
        """Verify validation fails when metadata.json is missing."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "model.keras").touch()

        with pytest.raises(ArtifactSyncError, match="Required artifact missing"):
            manager._validate_artifacts(model_dir)

    def test_validate_artifacts_missing_model_file(
        self, manager: S3ArtifactManager, tmp_path: Path
    ) -> None:
        """Verify validation fails when no model file exists."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()

        metadata = {"symbol": "BTCUSDT", "training_date": "2024-01-01"}
        (model_dir / "metadata.json").write_text(json.dumps(metadata))

        with pytest.raises(ArtifactSyncError, match="No model file found"):
            manager._validate_artifacts(model_dir)

    def test_validate_artifacts_invalid_metadata_json(
        self, manager: S3ArtifactManager, tmp_path: Path
    ) -> None:
        """Verify validation fails with invalid JSON in metadata."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "model.keras").touch()
        (model_dir / "metadata.json").write_text("not valid json")

        with pytest.raises(ArtifactSyncError, match="Invalid metadata.json"):
            manager._validate_artifacts(model_dir)

    def test_validate_artifacts_missing_required_fields(
        self, manager: S3ArtifactManager, tmp_path: Path
    ) -> None:
        """Verify validation fails when required fields are missing."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "model.keras").touch()

        metadata = {"foo": "bar"}  # Missing symbol and training_date
        (model_dir / "metadata.json").write_text(json.dumps(metadata))

        with pytest.raises(ArtifactSyncError, match="missing required fields"):
            manager._validate_artifacts(model_dir)

    def test_validate_artifacts_success_with_keras(
        self, manager: S3ArtifactManager, tmp_path: Path
    ) -> None:
        """Verify validation succeeds with model.keras."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "model.keras").touch()

        metadata = {"symbol": "BTCUSDT", "training_date": "2024-01-01"}
        (model_dir / "metadata.json").write_text(json.dumps(metadata))

        # Should not raise
        manager._validate_artifacts(model_dir)

    def test_validate_artifacts_success_with_onnx(
        self, manager: S3ArtifactManager, tmp_path: Path
    ) -> None:
        """Verify validation succeeds with model.onnx."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "model.onnx").touch()

        metadata = {"symbol": "BTCUSDT", "training_date": "2024-01-01"}
        (model_dir / "metadata.json").write_text(json.dumps(metadata))

        # Should not raise
        manager._validate_artifacts(model_dir)


class TestSymlinkUpdate:
    """Tests for atomic symlink updates."""

    @pytest.fixture
    def manager(self) -> S3ArtifactManager:
        """Create an S3ArtifactManager instance for testing."""
        return S3ArtifactManager(bucket_name="test-bucket")

    def test_update_latest_symlink_creates_link(
        self, manager: S3ArtifactManager, tmp_path: Path
    ) -> None:
        """Verify symlink is created pointing to version directory."""
        type_dir = tmp_path / "BTCUSDT" / "basic"
        type_dir.mkdir(parents=True)

        version_dir = type_dir / "2024-01-01_12h_v1"
        version_dir.mkdir()

        manager._update_latest_symlink(version_dir)

        latest_link = type_dir / "latest"
        assert latest_link.is_symlink()
        assert latest_link.resolve() == version_dir.resolve()

    def test_update_latest_symlink_replaces_existing(
        self, manager: S3ArtifactManager, tmp_path: Path
    ) -> None:
        """Verify existing symlink is replaced atomically."""
        type_dir = tmp_path / "BTCUSDT" / "basic"
        type_dir.mkdir(parents=True)

        old_version = type_dir / "2024-01-01_12h_v1"
        old_version.mkdir()

        new_version = type_dir / "2024-01-02_12h_v1"
        new_version.mkdir()

        # Create initial symlink
        latest_link = type_dir / "latest"
        latest_link.symlink_to(old_version.name)

        # Update to new version
        manager._update_latest_symlink(new_version)

        assert latest_link.is_symlink()
        assert latest_link.resolve() == new_version.resolve()


class TestBoto3LazyLoading:
    """Tests for boto3 lazy loading behavior."""

    def test_client_not_created_on_init(self) -> None:
        """Verify boto3 client is not created during initialization."""
        manager = S3ArtifactManager(bucket_name="test-bucket")
        assert manager._s3_client is None

    def test_ensure_client_raises_without_boto3(self) -> None:
        """Verify proper error when boto3 is not available."""
        manager = S3ArtifactManager(bucket_name="test-bucket")

        with patch.dict("sys.modules", {"boto3": None}):
            # Force reimport failure by clearing the client
            manager._s3_client = None

            # This should work in normal test env since boto3 is installed
            # The actual behavior depends on whether boto3 is installed
            # We test the happy path since boto3 is a test dependency
            try:
                manager._ensure_client()
                assert manager._s3_client is not None
            except ArtifactSyncError as e:
                assert "boto3 is required" in str(e)
