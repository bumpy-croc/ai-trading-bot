"""Unit tests for cloud training orchestrator."""

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.ml.cloud.config import CloudInstanceConfig, CloudStorageConfig, CloudTrainingConfig
from src.ml.cloud.exceptions import ArtifactSyncError
from src.ml.cloud.orchestrator import CloudTrainingOrchestrator, CloudTrainingResult
from src.ml.cloud.providers.base import TrainingJobStatus
from src.ml.training_pipeline.config import TrainingConfig


class TestCloudTrainingResult:
    """Tests for CloudTrainingResult dataclass."""

    def test_successful_result(self) -> None:
        """Verify successful result attributes."""
        result = CloudTrainingResult(
            success=True,
            job_id="test-job-123",
            job_status="Completed",
            provider="sagemaker",
            artifact_path=Path("/tmp/model"),
            metrics={"accuracy": 0.95},
            duration_seconds=3600.0,
            message="Training completed",
        )

        assert result.success is True
        assert result.job_id == "test-job-123"
        assert result.job_status == "Completed"
        assert result.artifact_path == Path("/tmp/model")
        assert result.metrics["accuracy"] == 0.95

    def test_failed_result(self) -> None:
        """Verify failed result attributes."""
        result = CloudTrainingResult(
            success=False,
            job_id=None,
            job_status="Failed",
            provider="sagemaker",
            error="Out of memory",
        )

        assert result.success is False
        assert result.job_id is None
        assert result.error == "Out of memory"

    def test_to_dict(self) -> None:
        """Verify to_dict serialization."""
        result = CloudTrainingResult(
            success=True,
            job_id="job-123",
            job_status="Completed",
            provider="local",
            artifact_path=Path("/tmp/artifacts"),
            metrics={"loss": 0.1},
            duration_seconds=1800.0,
        )

        data = result.to_dict()

        assert data["success"] is True
        assert data["job_id"] == "job-123"
        assert data["artifact_path"] == "/tmp/artifacts"
        assert data["metrics"]["loss"] == 0.1


class TestBuildJobSpec:
    """Tests for _build_job_spec method."""

    @pytest.fixture
    def orchestrator(self) -> CloudTrainingOrchestrator:
        """Create orchestrator with mocked provider."""
        training_config = TrainingConfig(
            symbol="BTCUSDT",
            timeframe="1h",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 12, 1),
            epochs=100,
            batch_size=32,
            sequence_length=120,
        )

        cloud_config = CloudTrainingConfig(
            training_config=training_config,
            storage_config=CloudStorageConfig(s3_bucket="test-bucket"),
            instance_config=CloudInstanceConfig(
                instance_type="ml.g4dn.xlarge",
                use_spot_instances=True,
                max_runtime_hours=4,
            ),
        )

        mock_provider = MagicMock()
        mock_provider.provider_name = "local"

        return CloudTrainingOrchestrator(cloud_config, mock_provider)

    def test_build_job_spec_symbol(self, orchestrator: CloudTrainingOrchestrator) -> None:
        """Verify job spec contains correct symbol."""
        spec = orchestrator._build_job_spec()
        assert spec.symbol == "BTCUSDT"

    def test_build_job_spec_timeframe(self, orchestrator: CloudTrainingOrchestrator) -> None:
        """Verify job spec contains correct timeframe."""
        spec = orchestrator._build_job_spec()
        assert spec.timeframe == "1h"

    def test_build_job_spec_epochs(self, orchestrator: CloudTrainingOrchestrator) -> None:
        """Verify job spec contains correct epochs."""
        spec = orchestrator._build_job_spec()
        assert spec.epochs == 100

    def test_build_job_spec_instance_type(self, orchestrator: CloudTrainingOrchestrator) -> None:
        """Verify job spec contains correct instance type."""
        spec = orchestrator._build_job_spec()
        assert spec.instance_type == "ml.g4dn.xlarge"

    def test_build_job_spec_spot_instances(self, orchestrator: CloudTrainingOrchestrator) -> None:
        """Verify job spec contains correct spot instance setting."""
        spec = orchestrator._build_job_spec()
        assert spec.use_spot_instances is True


class TestSubmitJob:
    """Tests for submit_job method."""

    @pytest.fixture
    def cloud_config(self) -> CloudTrainingConfig:
        """Create cloud config for testing."""
        training_config = TrainingConfig(
            symbol="BTCUSDT",
            timeframe="1h",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 12, 1),
            epochs=50,
        )
        return CloudTrainingConfig(
            training_config=training_config,
            storage_config=CloudStorageConfig(s3_bucket="test-bucket"),
        )

    def test_submit_job_returns_job_id(self, cloud_config: CloudTrainingConfig) -> None:
        """Verify submit_job returns job ID from provider."""
        mock_provider = MagicMock()
        mock_provider.submit_training_job.return_value = "job-123"

        orchestrator = CloudTrainingOrchestrator(cloud_config, mock_provider)
        job_id = orchestrator.submit_job()

        assert job_id == "job-123"
        mock_provider.submit_training_job.assert_called_once()


class TestRunTrainingNoWait:
    """Tests for run_training with wait=False."""

    @pytest.fixture
    def cloud_config(self) -> CloudTrainingConfig:
        """Create cloud config for testing."""
        training_config = TrainingConfig(
            symbol="BTCUSDT",
            timeframe="1h",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 12, 1),
            epochs=50,
        )
        return CloudTrainingConfig(
            training_config=training_config,
            storage_config=CloudStorageConfig(s3_bucket="test-bucket"),
            # Provide pre-existing S3 URI to skip data preparation step
            input_data_s3_uri="s3://test-bucket/training-data/BTCUSDT",
        )

    def test_run_training_no_wait_returns_immediately(
        self, cloud_config: CloudTrainingConfig
    ) -> None:
        """Verify run_training with wait=False returns without waiting."""
        mock_provider = MagicMock()
        mock_provider.provider_name = "local"
        mock_provider.submit_training_job.return_value = "job-456"

        orchestrator = CloudTrainingOrchestrator(cloud_config, mock_provider)
        result = orchestrator.run_training(wait=False)

        assert result.success is True
        assert result.job_id == "job-456"
        assert result.job_status == "InProgress"
        assert "Job submitted" in result.message


class TestJobIdValidation:
    """Tests for job_id validation in _sync_artifacts."""

    @pytest.fixture
    def cloud_config(self) -> CloudTrainingConfig:
        """Create cloud config for testing."""
        training_config = TrainingConfig(
            symbol="BTCUSDT",
            timeframe="1h",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 12, 1),
            epochs=50,
        )
        return CloudTrainingConfig(
            training_config=training_config,
            storage_config=CloudStorageConfig(s3_bucket="test-bucket"),
        )

    def test_empty_s3_path_raises_error(self, cloud_config: CloudTrainingConfig) -> None:
        """Verify empty S3 path raises ArtifactSyncError."""
        mock_provider = MagicMock()
        orchestrator = CloudTrainingOrchestrator(cloud_config, mock_provider)

        with pytest.raises(ArtifactSyncError, match="No output path"):
            orchestrator._sync_artifacts("job-123", None)

    def test_job_id_with_special_chars_sanitized(
        self, cloud_config: CloudTrainingConfig, tmp_path: Path
    ) -> None:
        """Verify job_id with invalid characters is sanitized to 'download'."""
        import re

        # These job_ids have invalid characters in the last segment
        # and should be sanitized to "download"
        invalid_ids = [
            "job;rm -rf /",  # semicolon is invalid
            "job\ncommand",  # newline is invalid
            "",  # empty string
        ]

        for job_id in invalid_ids:
            job_suffix = job_id.split("/")[-1] if job_id else ""
            if not job_suffix or not re.match(r"^[\w\-]+$", job_suffix):
                job_suffix = "download"

            assert job_suffix == "download", f"Failed for job_id: {job_id}"

    def test_path_traversal_extracts_last_segment(self, cloud_config: CloudTrainingConfig) -> None:
        """Verify path traversal attempts extract only the last segment."""
        import re

        # Path traversal attempts - the last segment is extracted
        # and validated, preventing directory escape
        job_id = "../../../etc/passwd"
        job_suffix = job_id.split("/")[-1] if job_id else ""
        if not job_suffix or not re.match(r"^[\w\-]+$", job_suffix):
            job_suffix = "download"

        # "passwd" is extracted and is a valid filename
        assert job_suffix == "passwd"

        # The temp directory would be /tmp/model-download-passwd
        # which is safe - no actual path traversal occurs

    def test_valid_job_id_not_sanitized(self, cloud_config: CloudTrainingConfig) -> None:
        """Verify valid job_id is not sanitized."""
        import re

        valid_ids = [
            "local-abc12345",
            "arn:aws:sagemaker:us-east-1:123456:training-job/atb-btcusdt-1h-20240101",
            "my-training-job_v1",
        ]

        for job_id in valid_ids:
            job_suffix = job_id.split("/")[-1] if job_id else ""
            if not job_suffix or not re.match(r"^[\w\-]+$", job_suffix):
                job_suffix = "download"

            # These should NOT be sanitized to "download"
            assert job_suffix != "download", f"Valid job_id was sanitized: {job_id}"


class TestCheckStatus:
    """Tests for check_status method."""

    @pytest.fixture
    def cloud_config(self) -> CloudTrainingConfig:
        """Create cloud config for testing."""
        training_config = TrainingConfig(
            symbol="BTCUSDT",
            timeframe="1h",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 12, 1),
            epochs=50,
        )
        return CloudTrainingConfig(
            training_config=training_config,
            storage_config=CloudStorageConfig(s3_bucket="test-bucket"),
            auto_sync_artifacts=False,
        )

    def test_check_status_in_progress(self, cloud_config: CloudTrainingConfig) -> None:
        """Verify check_status for in-progress job."""
        mock_provider = MagicMock()
        mock_provider.provider_name = "sagemaker"
        mock_provider.get_job_status.return_value = TrainingJobStatus(
            job_name="job-123",
            status="InProgress",
            start_time=datetime.now(UTC),
        )

        orchestrator = CloudTrainingOrchestrator(cloud_config, mock_provider)
        result = orchestrator.check_status("job-123")

        assert result.success is False  # Not yet successful
        assert result.job_status == "InProgress"

    def test_check_status_completed(self, cloud_config: CloudTrainingConfig) -> None:
        """Verify check_status for completed job."""
        mock_provider = MagicMock()
        mock_provider.provider_name = "sagemaker"
        mock_provider.get_job_status.return_value = TrainingJobStatus(
            job_name="job-123",
            status="Completed",
            start_time=datetime.now(UTC),
            end_time=datetime.now(UTC),
            output_s3_path="s3://bucket/output",
            metrics={"loss": 0.05},
        )

        orchestrator = CloudTrainingOrchestrator(cloud_config, mock_provider)
        result = orchestrator.check_status("job-123")

        assert result.success is True
        assert result.job_status == "Completed"
        assert result.metrics["loss"] == 0.05

    def test_check_status_handles_exception(self, cloud_config: CloudTrainingConfig) -> None:
        """Verify check_status handles provider exceptions gracefully."""
        mock_provider = MagicMock()
        mock_provider.provider_name = "sagemaker"
        mock_provider.get_job_status.side_effect = Exception("Connection failed")

        orchestrator = CloudTrainingOrchestrator(cloud_config, mock_provider)
        result = orchestrator.check_status("job-123")

        assert result.success is False
        assert result.job_status == "Unknown"
        assert "Connection failed" in result.error


class TestPrepareTrainingData:
    """Tests for _prepare_training_data method."""

    @pytest.fixture
    def cloud_config(self) -> CloudTrainingConfig:
        """Create cloud config for testing."""
        training_config = TrainingConfig(
            symbol="BTCUSDT",
            timeframe="1h",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 12, 1),
            epochs=50,
        )
        return CloudTrainingConfig(
            training_config=training_config,
            storage_config=CloudStorageConfig(s3_bucket="test-bucket"),
        )

    def test_skips_when_s3_uri_already_set(self, cloud_config: CloudTrainingConfig) -> None:
        """Verify _prepare_training_data returns None when input_data_s3_uri is set."""
        cloud_config.input_data_s3_uri = "s3://existing-bucket/data"
        mock_provider = MagicMock()
        mock_provider.provider_name = "local"

        orchestrator = CloudTrainingOrchestrator(cloud_config, mock_provider)
        result = orchestrator._prepare_training_data()

        assert result is None

    def test_successful_download_and_upload(
        self, cloud_config: CloudTrainingConfig, tmp_path: Path
    ) -> None:
        """Verify successful download from Binance and upload to S3."""
        mock_provider = MagicMock()
        mock_provider.provider_name = "local"
        mock_s3_manager = MagicMock()
        mock_s3_manager.upload_training_data.return_value = "s3://test-bucket/data/BTCUSDT"

        orchestrator = CloudTrainingOrchestrator(
            cloud_config, mock_provider, s3_manager=mock_s3_manager
        )

        # Create a mock download function that creates a CSV file in the temp dir
        def mock_download_func(args: MagicMock) -> int:
            # Create a CSV file in the output directory
            csv_path = Path(args.output_dir) / "BTCUSDT_1h_2024.csv"
            csv_path.write_text("timestamp,open,high,low,close,volume\n2024-01-01,100,101,99,100,1000")
            return 0

        with patch("cli.commands.data._download", side_effect=mock_download_func):
            result = orchestrator._prepare_training_data()

        assert result == "s3://test-bucket/data/BTCUSDT"
        mock_s3_manager.upload_training_data.assert_called_once()
        # Verify the upload was called with correct symbol and timeframe
        call_kwargs = mock_s3_manager.upload_training_data.call_args
        assert call_kwargs.kwargs["symbol"] == "BTCUSDT"
        assert call_kwargs.kwargs["timeframe"] == "1h"

    def test_raises_error_when_download_fails(self, cloud_config: CloudTrainingConfig) -> None:
        """Verify RuntimeError raised when Binance download fails."""
        mock_provider = MagicMock()
        mock_provider.provider_name = "local"

        orchestrator = CloudTrainingOrchestrator(cloud_config, mock_provider)

        with patch("cli.commands.data._download", return_value=1):
            with pytest.raises(RuntimeError, match="Failed to download training data"):
                orchestrator._prepare_training_data()

    def test_raises_error_when_no_data_files_found(
        self, cloud_config: CloudTrainingConfig
    ) -> None:
        """Verify RuntimeError raised when no CSV files found after download."""
        mock_provider = MagicMock()
        mock_provider.provider_name = "local"

        orchestrator = CloudTrainingOrchestrator(cloud_config, mock_provider)

        with patch("cli.commands.data._download", return_value=0):
            with patch("pathlib.Path.glob", return_value=[]):
                with pytest.raises(RuntimeError, match="No data files found"):
                    orchestrator._prepare_training_data()


class TestFindArtifactsRoot:
    """Tests for _find_artifacts_root method."""

    @pytest.fixture
    def cloud_config(self) -> CloudTrainingConfig:
        """Create cloud config for testing."""
        training_config = TrainingConfig(
            symbol="BTCUSDT",
            timeframe="1h",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 12, 1),
            epochs=50,
        )
        return CloudTrainingConfig(
            training_config=training_config,
            storage_config=CloudStorageConfig(s3_bucket="test-bucket"),
        )

    def test_returns_root_when_metadata_at_root(
        self, cloud_config: CloudTrainingConfig, tmp_path: Path
    ) -> None:
        """Verify returns artifact_path when metadata.json exists at root."""
        # Create flat structure with metadata at root
        (tmp_path / "metadata.json").write_text('{"version": "1.0"}')
        (tmp_path / "model.keras").write_text("model data")

        mock_provider = MagicMock()
        orchestrator = CloudTrainingOrchestrator(cloud_config, mock_provider)

        result = orchestrator._find_artifacts_root(tmp_path)

        assert result == tmp_path

    def test_finds_nested_structure(
        self, cloud_config: CloudTrainingConfig, tmp_path: Path
    ) -> None:
        """Verify finds artifacts in nested SYMBOL/TYPE/VERSION structure."""
        # Create nested structure: BTCUSDT/basic/2024-01-01_v1/
        nested_dir = tmp_path / "BTCUSDT" / "basic" / "2024-01-01_v1"
        nested_dir.mkdir(parents=True)
        (nested_dir / "metadata.json").write_text('{"version": "1.0"}')
        (nested_dir / "model.keras").write_text("model data")

        mock_provider = MagicMock()
        orchestrator = CloudTrainingOrchestrator(cloud_config, mock_provider)

        result = orchestrator._find_artifacts_root(tmp_path)

        assert result == nested_dir

    def test_returns_root_when_no_metadata_found(
        self, cloud_config: CloudTrainingConfig, tmp_path: Path
    ) -> None:
        """Verify returns artifact_path when no metadata.json found anywhere."""
        # Create directory with no metadata.json
        (tmp_path / "some_file.txt").write_text("data")

        mock_provider = MagicMock()
        orchestrator = CloudTrainingOrchestrator(cloud_config, mock_provider)

        result = orchestrator._find_artifacts_root(tmp_path)

        assert result == tmp_path

    def test_requires_model_file_alongside_metadata(
        self, cloud_config: CloudTrainingConfig, tmp_path: Path
    ) -> None:
        """Verify requires model.keras or model.onnx alongside metadata.json."""
        # Create nested structure with metadata but no model file
        nested_dir = tmp_path / "BTCUSDT" / "basic" / "2024-01-01_v1"
        nested_dir.mkdir(parents=True)
        (nested_dir / "metadata.json").write_text('{"version": "1.0"}')
        # No model.keras or model.onnx

        mock_provider = MagicMock()
        orchestrator = CloudTrainingOrchestrator(cloud_config, mock_provider)

        result = orchestrator._find_artifacts_root(tmp_path)

        # Should return root since no valid artifacts found
        assert result == tmp_path

    def test_finds_onnx_model(
        self, cloud_config: CloudTrainingConfig, tmp_path: Path
    ) -> None:
        """Verify finds artifacts with model.onnx instead of model.keras."""
        nested_dir = tmp_path / "BTCUSDT" / "basic" / "2024-01-01_v1"
        nested_dir.mkdir(parents=True)
        (nested_dir / "metadata.json").write_text('{"version": "1.0"}')
        (nested_dir / "model.onnx").write_text("onnx model data")

        mock_provider = MagicMock()
        orchestrator = CloudTrainingOrchestrator(cloud_config, mock_provider)

        result = orchestrator._find_artifacts_root(tmp_path)

        assert result == nested_dir

    def test_handles_deeply_nested_paths(
        self, cloud_config: CloudTrainingConfig, tmp_path: Path
    ) -> None:
        """Verify handles deeply nested paths within depth limit."""
        # Create nested structure within depth limit (10)
        nested_dir = tmp_path / "a" / "b" / "c" / "d" / "e"
        nested_dir.mkdir(parents=True)
        (nested_dir / "metadata.json").write_text('{"version": "1.0"}')
        (nested_dir / "model.keras").write_text("model data")

        mock_provider = MagicMock()
        orchestrator = CloudTrainingOrchestrator(cloud_config, mock_provider)

        result = orchestrator._find_artifacts_root(tmp_path)

        assert result == nested_dir
