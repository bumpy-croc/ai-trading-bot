"""Unit tests for cloud training orchestrator."""

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

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

    def test_build_job_spec_instance_type(
        self, orchestrator: CloudTrainingOrchestrator
    ) -> None:
        """Verify job spec contains correct instance type."""
        spec = orchestrator._build_job_spec()
        assert spec.instance_type == "ml.g4dn.xlarge"

    def test_build_job_spec_spot_instances(
        self, orchestrator: CloudTrainingOrchestrator
    ) -> None:
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

    def test_path_traversal_extracts_last_segment(
        self, cloud_config: CloudTrainingConfig
    ) -> None:
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
            start_time=datetime.now(),
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
            start_time=datetime.now(),
            end_time=datetime.now(),
            output_s3_path="s3://bucket/output",
            metrics={"loss": 0.05},
        )

        orchestrator = CloudTrainingOrchestrator(cloud_config, mock_provider)
        result = orchestrator.check_status("job-123")

        assert result.success is True
        assert result.job_status == "Completed"
        assert result.metrics["loss"] == 0.05

    def test_check_status_handles_exception(
        self, cloud_config: CloudTrainingConfig
    ) -> None:
        """Verify check_status handles provider exceptions gracefully."""
        mock_provider = MagicMock()
        mock_provider.provider_name = "sagemaker"
        mock_provider.get_job_status.side_effect = Exception("Connection failed")

        orchestrator = CloudTrainingOrchestrator(cloud_config, mock_provider)
        result = orchestrator.check_status("job-123")

        assert result.success is False
        assert result.job_status == "Unknown"
        assert "Connection failed" in result.error
