"""Unit tests for cloud training providers."""

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from src.ml.cloud.exceptions import ArtifactSyncError
from src.ml.cloud.providers import get_provider
from src.ml.cloud.providers.base import TrainingJobSpec, TrainingJobStatus
from src.ml.cloud.providers.local import LocalProvider


class TestProviderFactory:
    """Tests for the provider factory function."""

    def test_get_local_provider(self) -> None:
        """Verify local provider is returned correctly."""
        provider = get_provider("local")

        assert provider.provider_name == "local"
        assert provider.is_available() is True

    def test_get_sagemaker_provider(self) -> None:
        """Verify sagemaker provider is returned correctly."""
        provider = get_provider("sagemaker")
        assert provider.provider_name == "sagemaker"

    def test_get_unknown_provider_raises_error(self) -> None:
        """Verify unknown provider raises ProviderNotAvailableError."""
        from src.ml.cloud.exceptions import ProviderNotAvailableError

        with pytest.raises(ProviderNotAvailableError, match="not found"):
            get_provider("unknown_provider")


class TestLocalProvider:
    """Tests for LocalProvider implementation."""

    @pytest.fixture
    def provider(self) -> LocalProvider:
        """Create a LocalProvider instance for testing."""
        return LocalProvider()

    @pytest.fixture
    def job_spec(self) -> TrainingJobSpec:
        """Create a sample TrainingJobSpec for testing."""
        return TrainingJobSpec(
            symbol="BTCUSDT",
            timeframe="1h",
            start_date="2024-01-01T00:00:00",
            end_date="2024-12-01T00:00:00",
            epochs=10,
            batch_size=32,
            sequence_length=120,
            instance_type="ml.g4dn.xlarge",
            use_spot_instances=True,
            max_runtime_seconds=3600,
            output_s3_path="s3://test-bucket/models/",
        )

    def test_is_available(self, provider: LocalProvider) -> None:
        """Verify local provider is always available."""
        assert provider.is_available() is True

    def test_provider_name(self, provider: LocalProvider) -> None:
        """Verify provider name is 'local'."""
        assert provider.provider_name == "local"

    def test_submit_training_job_returns_job_id(
        self, provider: LocalProvider, job_spec: TrainingJobSpec
    ) -> None:
        """Verify submit_training_job returns a valid job ID."""
        with patch.object(provider, "_run_local_training"):
            job_id = provider.submit_training_job(job_spec)

        assert job_id.startswith("local-")
        assert len(job_id) == 14  # "local-" + 8 hex chars

    def test_submit_training_job_creates_in_progress_status(
        self, provider: LocalProvider, job_spec: TrainingJobSpec
    ) -> None:
        """Verify job status is InProgress after submission."""
        with patch.object(provider, "_run_local_training"):
            job_id = provider.submit_training_job(job_spec)

        status = provider.get_job_status(job_id)
        assert status.status == "InProgress"
        assert status.job_name == job_id
        assert status.start_time is not None
        assert status.end_time is None

    def test_get_job_status_unknown_job(self, provider: LocalProvider) -> None:
        """Verify get_job_status raises for unknown job."""
        with pytest.raises(ValueError, match="Job not found"):
            provider.get_job_status("unknown-job-id")

    def test_cancel_job(self, provider: LocalProvider, job_spec: TrainingJobSpec) -> None:
        """Verify cancel_job marks job as stopped."""
        with patch.object(provider, "_run_local_training"):
            job_id = provider.submit_training_job(job_spec)

        provider.cancel_job(job_id)

        status = provider.get_job_status(job_id)
        assert status.status == "Stopped"
        assert status.end_time is not None

    def test_cancel_nonexistent_job_no_error(self, provider: LocalProvider) -> None:
        """Verify canceling nonexistent job does not raise."""
        provider.cancel_job("nonexistent-job")  # Should not raise

    def test_download_artifacts_requires_successful_job(
        self, provider: LocalProvider, job_spec: TrainingJobSpec
    ) -> None:
        """Verify download_artifacts fails for non-successful jobs."""
        with patch.object(provider, "_run_local_training"):
            job_id = provider.submit_training_job(job_spec)

        with pytest.raises(ArtifactSyncError, match="Cannot get artifacts"):
            provider.download_artifacts(job_id, Path("/tmp/test"))


class TestTrainingJobSpec:
    """Tests for TrainingJobSpec dataclass."""

    def test_to_hyperparameters(self) -> None:
        """Verify to_hyperparameters returns correct format."""
        spec = TrainingJobSpec(
            symbol="ETHUSDT",
            timeframe="4h",
            start_date="2024-01-01T00:00:00",
            end_date="2024-06-01T00:00:00",
            epochs=50,
            batch_size=64,
            sequence_length=60,
            instance_type="ml.g4dn.xlarge",
            use_spot_instances=False,
            max_runtime_seconds=7200,
            output_s3_path="s3://bucket/path",
            hyperparameters={"custom_param": "value"},
        )

        params = spec.to_hyperparameters()

        assert params["symbol"] == "ETHUSDT"
        assert params["timeframe"] == "4h"
        assert params["start_date"] == "2024-01-01T00:00:00"
        assert params["end_date"] == "2024-06-01T00:00:00"
        assert params["epochs"] == "50"
        assert params["batch_size"] == "64"
        assert params["sequence_length"] == "60"
        assert params["custom_param"] == "value"


class TestTrainingJobStatus:
    """Tests for TrainingJobStatus dataclass."""

    def test_is_terminal_completed(self) -> None:
        """Verify Completed status is terminal."""
        status = TrainingJobStatus(
            job_name="test-job",
            status="Completed",
            start_time=datetime.now(UTC),
            end_time=datetime.now(UTC),
        )
        assert status.is_terminal is True

    def test_is_terminal_failed(self) -> None:
        """Verify Failed status is terminal."""
        status = TrainingJobStatus(
            job_name="test-job",
            status="Failed",
            start_time=datetime.now(UTC),
            end_time=datetime.now(UTC),
            failure_reason="Out of memory",
        )
        assert status.is_terminal is True

    def test_is_terminal_stopped(self) -> None:
        """Verify Stopped status is terminal."""
        status = TrainingJobStatus(
            job_name="test-job",
            status="Stopped",
            start_time=datetime.now(UTC),
            end_time=datetime.now(UTC),
        )
        assert status.is_terminal is True

    def test_is_terminal_in_progress(self) -> None:
        """Verify InProgress status is not terminal."""
        status = TrainingJobStatus(
            job_name="test-job",
            status="InProgress",
            start_time=datetime.now(UTC),
        )
        assert status.is_terminal is False

    def test_is_successful_completed(self) -> None:
        """Verify Completed status is successful."""
        status = TrainingJobStatus(
            job_name="test-job",
            status="Completed",
            start_time=datetime.now(UTC),
            end_time=datetime.now(UTC),
        )
        assert status.is_successful is True

    def test_is_successful_failed(self) -> None:
        """Verify Failed status is not successful."""
        status = TrainingJobStatus(
            job_name="test-job",
            status="Failed",
            start_time=datetime.now(UTC),
            end_time=datetime.now(UTC),
        )
        assert status.is_successful is False

    def test_duration_seconds(self) -> None:
        """Verify duration_seconds calculation."""
        start = datetime(2024, 1, 1, 10, 0, 0, tzinfo=UTC)
        end = datetime(2024, 1, 1, 11, 30, 0, tzinfo=UTC)

        status = TrainingJobStatus(
            job_name="test-job",
            status="Completed",
            start_time=start,
            end_time=end,
        )

        assert status.duration_seconds == 5400.0  # 1.5 hours in seconds

    def test_duration_seconds_no_end_time(self) -> None:
        """Verify duration_seconds returns None when job is running."""
        status = TrainingJobStatus(
            job_name="test-job",
            status="InProgress",
            start_time=datetime.now(UTC),
        )
        assert status.duration_seconds is None
