"""Unit tests for cloud training orchestrator."""

import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
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

    def test_build_job_spec_default_architecture_hyperparameters(
        self, orchestrator: CloudTrainingOrchestrator
    ) -> None:
        """Verify default architecture selection reaches the job hyperparameters."""
        spec = orchestrator._build_job_spec()
        assert spec.hyperparameters["model_type"] == "cnn_lstm"
        assert spec.hyperparameters["model_variant"] == "default"

    def test_build_job_spec_custom_architecture_hyperparameters(self) -> None:
        """Verify --model-type/--model-variant selections reach the job hyperparameters."""
        training_config = TrainingConfig(
            symbol="BTCUSDT",
            timeframe="1h",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 12, 1),
            model_type="tcn_attention",
            model_variant="deep",
        )
        cloud_config = CloudTrainingConfig(
            training_config=training_config,
            storage_config=CloudStorageConfig(s3_bucket="test-bucket"),
        )
        orchestrator = CloudTrainingOrchestrator(cloud_config, MagicMock())

        spec = orchestrator._build_job_spec()

        assert spec.hyperparameters["model_type"] == "tcn_attention"
        assert spec.hyperparameters["model_variant"] == "deep"
        # SageMaker requires string hyperparameter values end-to-end
        assert spec.to_hyperparameters()["model_type"] == "tcn_attention"
        assert spec.to_hyperparameters()["model_variant"] == "deep"

    def test_build_job_spec_default_target_type_hyperparameters(
        self, orchestrator: CloudTrainingOrchestrator
    ) -> None:
        """Phase 2b item 1: default target_type/target_horizon must reach
        the job hyperparameters too (regression/1, preserving current
        behavior for every job submitted before --target-type existed)."""
        spec = orchestrator._build_job_spec()
        assert spec.hyperparameters["target_type"] == "regression"
        assert spec.hyperparameters["target_horizon"] == "1"

    def test_build_job_spec_custom_target_type_hyperparameters(self) -> None:
        training_config = TrainingConfig(
            symbol="BTCUSDT",
            timeframe="1h",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 12, 1),
            model_type="tft",
            target_type="binary_direction",
            target_horizon=6,
        )
        cloud_config = CloudTrainingConfig(
            training_config=training_config,
            storage_config=CloudStorageConfig(s3_bucket="test-bucket"),
        )
        orchestrator = CloudTrainingOrchestrator(cloud_config, MagicMock())

        spec = orchestrator._build_job_spec()

        assert spec.hyperparameters["target_type"] == "binary_direction"
        assert spec.hyperparameters["target_horizon"] == "6"
        # SageMaker requires string hyperparameter values end-to-end
        assert spec.to_hyperparameters()["target_type"] == "binary_direction"
        assert spec.to_hyperparameters()["target_horizon"] == "6"

    def test_build_job_spec_default_primary_model_type_hyperparameters(
        self, orchestrator: CloudTrainingOrchestrator
    ) -> None:
        """Phase 2b item 3: primary_model_type=None must round-trip through
        the string-only hyperparameters dict as "" (SageMaker requires
        string values; entrypoint.py::parse_hyperparameters treats "" as
        unset)."""
        spec = orchestrator._build_job_spec()
        assert spec.hyperparameters["primary_model_type"] == ""

    def test_build_job_spec_custom_primary_model_type_hyperparameters(self) -> None:
        training_config = TrainingConfig(
            symbol="BTCUSDT",
            timeframe="1h",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 12, 1),
            target_type="meta_label",
            primary_model_type="basic",
        )
        cloud_config = CloudTrainingConfig(
            training_config=training_config,
            storage_config=CloudStorageConfig(s3_bucket="test-bucket"),
        )
        orchestrator = CloudTrainingOrchestrator(cloud_config, MagicMock())

        spec = orchestrator._build_job_spec()

        assert spec.hyperparameters["target_type"] == "meta_label"
        assert spec.hyperparameters["primary_model_type"] == "basic"

    def test_build_job_spec_default_use_mock_data_hyperparameters(
        self, orchestrator: CloudTrainingOrchestrator
    ) -> None:
        spec = orchestrator._build_job_spec()
        assert spec.hyperparameters["use_mock_data"] == "false"

    def test_build_job_spec_custom_use_mock_data_hyperparameters(self) -> None:
        training_config = TrainingConfig(
            symbol="BTCUSDT",
            timeframe="1h",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 12, 1),
            use_mock_data=True,
        )
        cloud_config = CloudTrainingConfig(
            training_config=training_config,
            storage_config=CloudStorageConfig(s3_bucket="test-bucket"),
        )
        orchestrator = CloudTrainingOrchestrator(cloud_config, MagicMock())

        spec = orchestrator._build_job_spec()

        assert spec.hyperparameters["use_mock_data"] == "true"


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

    @pytest.mark.fast
    def test_submit_job_returns_job_id(self, cloud_config: CloudTrainingConfig) -> None:
        """Verify submit_job returns job ID from provider."""
        cloud_config.input_data_s3_uri = "s3://test-bucket/training-data/BTCUSDT"
        mock_provider = MagicMock()
        mock_provider.submit_training_job.return_value = "job-123"

        orchestrator = CloudTrainingOrchestrator(cloud_config, mock_provider)
        job_id = orchestrator.submit_job()

        assert job_id == "job-123"
        mock_provider.submit_training_job.assert_called_once()

    @pytest.mark.fast
    def test_submit_job_uploads_data_channel(self, cloud_config: CloudTrainingConfig) -> None:
        """submit_job must run the same data-upload step as run_training.

        Without an uploaded data channel the in-container Binance fetch fails
        (AWS IPs are blocked), which is why --no-wait used to be broken.
        """
        mock_provider = MagicMock()
        mock_provider.submit_training_job.return_value = "job-async"
        mock_s3_manager = MagicMock()
        mock_s3_manager.upload_training_data.return_value = "s3://test-bucket/data/BTCUSDT"

        orchestrator = CloudTrainingOrchestrator(
            cloud_config, mock_provider, s3_manager=mock_s3_manager
        )

        corpus = pd.DataFrame(
            {"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0},
            index=pd.date_range("2024-01-01", periods=24, freq="1h", tz="UTC"),
        )
        with patch("src.ml.training_pipeline.ingestion.load_training_corpus", return_value=corpus):
            job_id = orchestrator.submit_job()

        assert job_id == "job-async"
        mock_s3_manager.upload_training_data.assert_called_once()
        spec = mock_provider.submit_training_job.call_args.args[0]
        assert spec.input_data_s3_uri == "s3://test-bucket/data/BTCUSDT"

    @pytest.mark.fast
    def test_submit_job_skips_upload_when_uri_provided(
        self, cloud_config: CloudTrainingConfig
    ) -> None:
        """Verify submit_job skips data prep when input_data_s3_uri is preset."""
        cloud_config.input_data_s3_uri = "s3://existing-bucket/data"
        mock_provider = MagicMock()
        mock_provider.submit_training_job.return_value = "job-preset"
        mock_s3_manager = MagicMock()

        orchestrator = CloudTrainingOrchestrator(
            cloud_config, mock_provider, s3_manager=mock_s3_manager
        )
        job_id = orchestrator.submit_job()

        assert job_id == "job-preset"
        mock_s3_manager.upload_training_data.assert_not_called()
        spec = mock_provider.submit_training_job.call_args.args[0]
        assert spec.input_data_s3_uri == "s3://existing-bucket/data"


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

    @staticmethod
    def _corpus_frame() -> "pd.DataFrame":
        """Small OHLCV corpus as returned by load_training_corpus."""
        index = pd.date_range("2024-01-01", periods=24, freq="1h", tz="UTC")
        return pd.DataFrame(
            {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 1000.0},
            index=index,
        )

    def test_skips_when_s3_uri_already_set(self, cloud_config: CloudTrainingConfig) -> None:
        """Verify _prepare_training_data returns None when input_data_s3_uri is set."""
        cloud_config.input_data_s3_uri = "s3://existing-bucket/data"
        mock_provider = MagicMock()
        mock_provider.provider_name = "local"

        orchestrator = CloudTrainingOrchestrator(cloud_config, mock_provider)
        result = orchestrator._prepare_training_data()

        assert result is None

    def test_successful_corpus_load_and_upload(self, cloud_config: CloudTrainingConfig) -> None:
        """Verify the single-source corpus loader feeds the S3 upload."""
        mock_provider = MagicMock()
        mock_provider.provider_name = "local"
        mock_s3_manager = MagicMock()
        mock_s3_manager.upload_training_data.return_value = "s3://test-bucket/data/BTCUSDT"

        orchestrator = CloudTrainingOrchestrator(
            cloud_config, mock_provider, s3_manager=mock_s3_manager
        )

        with patch(
            "src.ml.training_pipeline.ingestion.load_training_corpus",
            return_value=self._corpus_frame(),
        ) as mock_corpus:
            result = orchestrator._prepare_training_data()

        assert result == "s3://test-bucket/data/BTCUSDT"
        mock_corpus.assert_called_once()
        ctx = mock_corpus.call_args.args[0]
        assert ctx.config is cloud_config.training_config
        mock_s3_manager.upload_training_data.assert_called_once()
        call_kwargs = mock_s3_manager.upload_training_data.call_args
        assert call_kwargs.kwargs["symbol"] == "BTCUSDT"
        assert call_kwargs.kwargs["timeframe"] == "1h"

    def test_uploaded_csv_has_timestamp_column(self, cloud_config: CloudTrainingConfig) -> None:
        """The container's loader requires a 'timestamp' column in the uploaded CSV."""
        mock_provider = MagicMock()
        mock_provider.provider_name = "local"
        mock_s3_manager = MagicMock()
        uploaded: dict[str, str] = {}

        def capture_upload(symbol: str, timeframe: str, data_files: list[Path]) -> str:
            # Read inside the call: the temp dir is deleted once _prepare returns
            uploaded["name"] = data_files[0].name
            uploaded["header"] = data_files[0].read_text().splitlines()[0]
            uploaded["rows"] = str(len(data_files[0].read_text().splitlines()) - 1)
            return "s3://test-bucket/data/BTCUSDT"

        mock_s3_manager.upload_training_data.side_effect = capture_upload

        orchestrator = CloudTrainingOrchestrator(
            cloud_config, mock_provider, s3_manager=mock_s3_manager
        )

        with patch(
            "src.ml.training_pipeline.ingestion.load_training_corpus",
            return_value=self._corpus_frame(),
        ):
            orchestrator._prepare_training_data()

        assert uploaded["header"].startswith("timestamp,")
        assert uploaded["rows"] == "24"
        assert uploaded["name"] == "BTCUSDT_1h_2024-01-01_2024-12-01.csv"

    def test_raises_error_when_corpus_load_fails(self, cloud_config: CloudTrainingConfig) -> None:
        """Corpus failures propagate loudly - there is no silent fallback source."""
        mock_provider = MagicMock()
        mock_provider.provider_name = "local"

        orchestrator = CloudTrainingOrchestrator(cloud_config, mock_provider)

        with patch(
            "src.ml.training_pipeline.ingestion.load_training_corpus",
            side_effect=RuntimeError("Training corpus for BTCUSDT 1h could not be loaded"),
        ):
            with pytest.raises(RuntimeError, match="could not be loaded"):
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

    def test_finds_onnx_model(self, cloud_config: CloudTrainingConfig, tmp_path: Path) -> None:
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


def _make_orchestrator(symbol: str = "BTCUSDT") -> CloudTrainingOrchestrator:
    """Build an orchestrator with a mocked provider for sync tests."""
    training_config = TrainingConfig(
        symbol=symbol,
        timeframe="1h",
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 12, 1),
        epochs=50,
    )
    cloud_config = CloudTrainingConfig(
        training_config=training_config,
        storage_config=CloudStorageConfig(s3_bucket="test-bucket"),
    )
    return CloudTrainingOrchestrator(cloud_config, MagicMock())


@pytest.mark.fast
class TestSyncLocalArtifactsCollision:
    """Version collisions must never destroy an existing model bundle."""

    def test_existing_version_dir_is_never_deleted(self, tmp_path: Path) -> None:
        orchestrator = _make_orchestrator()
        registry = tmp_path / "registry"
        existing = registry / "BTCUSDT" / "price" / "2026-07-05_10h_v1"
        existing.mkdir(parents=True)
        (existing / "model.onnx").write_text("first-job-model")

        artifact = tmp_path / "artifact"
        artifact.mkdir()
        (artifact / "model.onnx").write_text("second-job-model")

        result = orchestrator._sync_local_artifacts(
            artifact_path=artifact,
            local_registry=registry,
            symbol="BTCUSDT",
            model_type="price",
            version_id="2026-07-05_10h_v1",
        )

        # The first bundle survives untouched.
        assert (existing / "model.onnx").read_text() == "first-job-model"
        # The second bundle lands in a distinct sibling directory.
        assert result != existing
        assert result.parent == existing.parent
        assert (result / "model.onnx").read_text() == "second-job-model"

    def test_collision_updates_latest_to_new_bundle(self, tmp_path: Path) -> None:
        orchestrator = _make_orchestrator()
        registry = tmp_path / "registry"
        existing = registry / "BTCUSDT" / "price" / "2026-07-05_10h_v1"
        existing.mkdir(parents=True)
        (existing / "model.onnx").write_text("first")

        artifact = tmp_path / "artifact"
        artifact.mkdir()
        (artifact / "model.onnx").write_text("second")

        result = orchestrator._sync_local_artifacts(
            artifact_path=artifact,
            local_registry=registry,
            symbol="BTCUSDT",
            model_type="price",
            version_id="2026-07-05_10h_v1",
        )

        latest = registry / "BTCUSDT" / "price" / "latest"
        assert latest.is_symlink()
        assert latest.resolve() == result.resolve()

    def test_repeated_collisions_pick_next_free_suffix(self, tmp_path: Path) -> None:
        orchestrator = _make_orchestrator()
        registry = tmp_path / "registry"
        base = registry / "BTCUSDT" / "price"
        (base / "2026-07-05_10h_v1").mkdir(parents=True)
        (base / "2026-07-05_10h_v1-2").mkdir()

        artifact = tmp_path / "artifact"
        artifact.mkdir()
        (artifact / "model.onnx").write_text("third")

        result = orchestrator._sync_local_artifacts(
            artifact_path=artifact,
            local_registry=registry,
            symbol="BTCUSDT",
            model_type="price",
            version_id="2026-07-05_10h_v1",
        )

        assert result == base / "2026-07-05_10h_v1-3"

    def test_sync_updates_latest_via_atomic_helper(self, tmp_path: Path) -> None:
        """The sync path must use the shared atomic symlink helper.

        A plain unlink-then-symlink leaves a window where readers see no
        'latest' at all.
        """
        orchestrator = _make_orchestrator()
        registry = tmp_path / "registry"
        artifact = tmp_path / "artifact"
        artifact.mkdir()
        (artifact / "model.onnx").write_text("model")

        with patch("src.ml.cloud.orchestrator.update_latest_symlink") as mock_update:
            result = orchestrator._sync_local_artifacts(
                artifact_path=artifact,
                local_registry=registry,
                symbol="BTCUSDT",
                model_type="price",
                version_id="2026-07-06_10h00m00s_v1",
            )

        mock_update.assert_called_once_with(
            registry / "BTCUSDT" / "price", "2026-07-06_10h00m00s_v1"
        )
        assert result == registry / "BTCUSDT" / "price" / "2026-07-06_10h00m00s_v1"


@pytest.mark.fast
class TestSyncArtifactsUsesMetadataSymbol:
    """_sync_artifacts must trust the bundle's own symbol over the config."""

    def test_symbol_from_metadata_wins(self, tmp_path: Path) -> None:
        orchestrator = _make_orchestrator(symbol="BTCUSDT")
        metadata = {
            "symbol": "ETHUSDT",
            "model_type": "price",
            "version_id": "2026-07-05_10h00m00s_v1",
        }

        def fake_download(job_id: str, temp_dir: Path) -> Path:
            temp_dir.mkdir(parents=True, exist_ok=True)
            (temp_dir / "metadata.json").write_text(json.dumps(metadata))
            (temp_dir / "model.onnx").write_text("model-bytes")
            return temp_dir

        orchestrator.provider.download_artifacts.side_effect = fake_download

        with patch("src.ml.cloud.orchestrator.get_project_root", return_value=tmp_path):
            result = orchestrator._sync_artifacts("job-1", "s3://bucket/out/model.tar.gz")

        expected = (
            tmp_path / "src" / "ml" / "models" / "ETHUSDT" / "price" / "2026-07-05_10h00m00s_v1"
        )
        assert result == expected
        assert (expected / "model.onnx").exists()

    def test_falls_back_to_config_symbol_without_metadata(self, tmp_path: Path) -> None:
        orchestrator = _make_orchestrator(symbol="BTCUSDT")

        def fake_download(job_id: str, temp_dir: Path) -> Path:
            temp_dir.mkdir(parents=True, exist_ok=True)
            (temp_dir / "model.onnx").write_text("model-bytes")
            return temp_dir

        orchestrator.provider.download_artifacts.side_effect = fake_download

        with patch("src.ml.cloud.orchestrator.get_project_root", return_value=tmp_path):
            result = orchestrator._sync_artifacts("job-2", "s3://bucket/out/model.tar.gz")

        assert result.parent.parent.name == "BTCUSDT"


@pytest.mark.fast
class TestSyncArtifactsPublic:
    """Tests for the public sync_artifacts entry point (cloud-status --sync)."""

    def test_raises_for_unsuccessful_job(self) -> None:
        orchestrator = _make_orchestrator()
        orchestrator.provider.get_job_status.return_value = TrainingJobStatus(
            job_name="job-1",
            status="InProgress",
        )

        with pytest.raises(ArtifactSyncError, match="state"):
            orchestrator.sync_artifacts("job-1")

    def test_syncs_completed_job(self) -> None:
        orchestrator = _make_orchestrator()
        orchestrator.provider.get_job_status.return_value = TrainingJobStatus(
            job_name="job-1",
            status="Completed",
            output_s3_path="s3://bucket/out/model.tar.gz",
        )

        with patch.object(
            orchestrator, "_sync_artifacts", return_value=Path("/synced/path")
        ) as mock_sync:
            result = orchestrator.sync_artifacts("job-1")

        assert result == Path("/synced/path")
        mock_sync.assert_called_once_with("job-1", "s3://bucket/out/model.tar.gz")
