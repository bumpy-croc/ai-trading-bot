"""Cloud training orchestrator.

High-level coordination of cloud training workflow:
upload data → submit job → wait → download artifacts → sync to registry.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import re
import shutil
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

from src.infrastructure.runtime.paths import get_project_root
from src.ml.cloud.artifacts.s3_manager import S3ArtifactManager
from src.ml.cloud.config import CloudTrainingConfig
from src.ml.cloud.exceptions import (
    ArtifactSyncError,
    CloudTrainingError,
    JobSubmissionError,
    JobTimeoutError,
)
from src.ml.cloud.providers.base import (
    CloudTrainingProvider,
    TrainingJobSpec,
    TrainingJobStatus,
)

logger = logging.getLogger(__name__)

# Default poll interval for checking job status (seconds)
DEFAULT_POLL_INTERVAL = 60

# Spot instance interruption allowance multiplier (AWS recommends 2x for spot)
# SageMaker spot training may be interrupted and restarted, requiring extended wait time
SPOT_WAIT_TIME_MULTIPLIER = 2

# Maximum length for job ID suffix used in temporary directory naming
# Prevents excessively long paths while allowing for timestamp-based job IDs
MAX_JOB_SUFFIX_LENGTH = 100

# Maximum number of files to search when looking for nested metadata.json
# Prevents slow searches on deeply nested directories or large artifact trees
MAX_METADATA_SEARCH_DEPTH = 100

# Maximum suffix counter when a synced version collides with an existing bundle
MAX_VERSION_SUFFIX_ATTEMPTS = 1000


def _validate_registry_component(value: str, field: str) -> str:
    """Validate a path component used to build local model-registry paths.

    ``version_id``/``model_type`` originate from a downloaded ``metadata.json``
    (outside this process) and are used to build registry paths that are later
    targets of rmtree/copytree/symlink. A value containing ``..`` or a path
    separator would allow writing or deleting files outside the registry, so we
    only accept simple identifiers.
    """
    text = str(value)
    if text in (".", "..") or not re.match(r"^[\w\-.]+$", text):
        raise ArtifactSyncError(f"Refusing to use unsafe {field} from metadata: {value!r}")
    return text


class CloudTrainingOrchestrator:
    """Orchestrates cloud training workflow.

    Coordinates the complete cloud training lifecycle:
    1. Prepare training data locally (download from Binance/sentiment APIs)
    2. Upload data to S3
    3. Submit training job to cloud provider
    4. Wait for job completion (with polling)
    5. Download trained model artifacts from S3
    6. Sync to local model registry

    Example:
        >>> from src.ml.cloud.orchestrator import CloudTrainingOrchestrator
        >>> from src.ml.cloud.config import CloudTrainingConfig
        >>> from src.ml.cloud.providers import get_provider
        >>>
        >>> config = CloudTrainingConfig.from_env(training_config)
        >>> provider = get_provider("sagemaker")
        >>> orchestrator = CloudTrainingOrchestrator(config, provider)
        >>> result = orchestrator.run_training()
    """

    def __init__(
        self,
        config: CloudTrainingConfig,
        provider: CloudTrainingProvider,
        s3_manager: S3ArtifactManager | None = None,
    ) -> None:
        """Initialize orchestrator.

        Args:
            config: Cloud training configuration
            provider: Cloud training provider (SageMaker, local, etc.)
            s3_manager: S3 artifact manager (created from config if not provided)
        """
        self.config = config
        self.provider = provider
        self.s3_manager = s3_manager or S3ArtifactManager(
            bucket_name=config.storage_config.s3_bucket,
        )

    def _create_failure_result(self, error_message: str, start_time: float) -> CloudTrainingResult:
        """Create a failure result with consistent structure.

        Avoids duplication of error result creation (CODE.md line 62).

        Args:
            error_message: Error description
            start_time: Training start time from perf_counter()

        Returns:
            CloudTrainingResult indicating failure
        """
        return CloudTrainingResult(
            success=False,
            job_id=None,
            job_status="Failed",
            provider=self.provider.provider_name,
            error=error_message,
            duration_seconds=perf_counter() - start_time,
        )

    def _prepare_training_data(self) -> str | None:
        """Download training data locally and upload to S3.

        This step is required because Binance API blocks requests from AWS IP addresses.
        Data is downloaded fresh from Binance (no cache) and uploaded to S3
        for SageMaker to use as input channel.

        Returns:
            S3 URI of uploaded data, or None if input_data_s3_uri was already set

        Raises:
            RuntimeError: If data download or upload fails
        """
        # Skip if S3 URI already provided by user
        if self.config.input_data_s3_uri:
            logger.info(f"Using provided S3 data URI: {self.config.input_data_s3_uri}")
            return None

        from cli.commands.data import _download

        tc = self.config.training_config

        logger.info(f"Downloading {tc.symbol} data from Binance (fresh, no cache)...")

        # Create temp directory for downloads
        with tempfile.TemporaryDirectory() as tmpdir:
            # Download data from Binance using existing CLI function
            args = argparse.Namespace(
                symbol=tc.symbol,
                timeframe=tc.timeframe,
                start_date=tc.start_date.strftime("%Y-%m-%d"),
                end_date=tc.end_date.strftime("%Y-%m-%d"),
                output_dir=tmpdir,
                format="csv",
            )

            if _download(args) != 0:
                raise RuntimeError("Failed to download training data from Binance")

            # Find downloaded file
            data_files = list(Path(tmpdir).glob("*.csv"))
            if not data_files:
                raise RuntimeError("No data files found after download")

            logger.info(f"Downloaded {len(data_files)} file(s)")

            # Upload to S3 using existing artifact manager
            logger.info(
                f"Uploading training data to S3 bucket {self.config.storage_config.s3_bucket}..."
            )
            s3_uri = self.s3_manager.upload_training_data(
                symbol=tc.symbol,
                timeframe=tc.timeframe,
                data_files=data_files,
            )

            logger.info(f"Training data uploaded to {s3_uri}")
            return s3_uri

    def run_training(self, wait: bool = True) -> CloudTrainingResult:
        """Execute complete cloud training workflow.

        Args:
            wait: If True, wait for job completion and sync artifacts.
                  If False, submit job and return immediately.

        Returns:
            CloudTrainingResult with job status and artifact paths
        """
        start_time = perf_counter()

        try:
            # Steps 0-2: Prepare data, upload to S3, submit job (shared with --no-wait)
            job_id = self.submit_job()

            if not wait:
                return CloudTrainingResult(
                    success=True,
                    job_id=job_id,
                    job_status="InProgress",
                    provider=self.provider.provider_name,
                    message="Job submitted. Use 'atb train cloud status' to check progress.",
                    duration_seconds=perf_counter() - start_time,
                )

            # Step 3: Wait for completion
            logger.info("Step 3/5: Waiting for job completion...")
            status = self._wait_for_completion(job_id)

            if not status.is_successful:
                raise JobSubmissionError(
                    f"Training job failed: {status.failure_reason or 'Unknown error'}"
                )

            # Step 4: Sync artifacts to local registry
            logger.info("Step 4/5: Syncing artifacts to local registry...")
            artifact_path = self._sync_artifacts(job_id, status.output_s3_path)

            duration = perf_counter() - start_time

            return CloudTrainingResult(
                success=True,
                job_id=job_id,
                job_status=status.status,
                provider=self.provider.provider_name,
                artifact_path=artifact_path,
                metrics=status.metrics,
                duration_seconds=duration,
                message=f"Training completed in {duration:.1f}s",
            )

        except CloudTrainingError as exc:
            logger.error(f"Cloud training failed: {exc}")
            return self._create_failure_result(str(exc), start_time)
        except Exception as exc:
            logger.exception("Unexpected error during cloud training")
            return self._create_failure_result(f"Unexpected error: {exc}", start_time)

    def submit_job(self) -> str:
        """Prepare the data channel and submit a training job without waiting.

        Runs the same data-upload step as the blocking workflow: Binance blocks
        AWS IPs, so a job submitted without an uploaded data channel fails at
        the in-container fetch.

        Returns:
            Job identifier for status checking

        Raises:
            JobSubmissionError: If submission fails
            RuntimeError: If data download or upload fails
        """
        logger.info("Step 0/5: Preparing training data...")
        s3_uri = self._prepare_training_data()
        if s3_uri:
            self.config.input_data_s3_uri = s3_uri

        logger.info("Step 1/5: Building training job specification...")
        job_spec = self._build_job_spec()

        logger.info(f"Step 2/5: Submitting job to {self.provider.provider_name}...")
        job_id = self.provider.submit_training_job(job_spec)
        logger.info(f"Job submitted: {job_id}")
        return job_id

    def sync_artifacts(self, job_id: str) -> Path:
        """Download and sync artifacts for a completed job.

        Completes the round trip for jobs submitted with --no-wait
        (used by ``atb train cloud-status --sync``).

        Args:
            job_id: Job identifier from submit_job()

        Returns:
            Path to the synced model directory

        Raises:
            ArtifactSyncError: If the job is not in a successful terminal state
                or the download/sync fails
        """
        status = self.provider.get_job_status(job_id)
        if not status.is_successful:
            raise ArtifactSyncError(f"Cannot sync artifacts for job in state: {status.status}")
        return self._sync_artifacts(job_id, status.output_s3_path)

    def check_status(self, job_id: str) -> CloudTrainingResult:
        """Check status of a submitted training job.

        Args:
            job_id: Job identifier from submit_job()

        Returns:
            CloudTrainingResult with current status
        """
        try:
            status = self.provider.get_job_status(job_id)

            result = CloudTrainingResult(
                success=status.is_successful,
                job_id=job_id,
                job_status=status.status,
                provider=self.provider.provider_name,
                metrics=status.metrics,
            )

            if status.is_successful and self.config.auto_sync_artifacts:
                artifact_path = self._sync_artifacts(job_id, status.output_s3_path)
                result.artifact_path = artifact_path

            return result

        except Exception as exc:
            return CloudTrainingResult(
                success=False,
                job_id=job_id,
                job_status="Unknown",
                provider=self.provider.provider_name,
                error=str(exc),
            )

    def _build_job_spec(self) -> TrainingJobSpec:
        """Build TrainingJobSpec from configuration."""
        tc = self.config.training_config
        ic = self.config.instance_config
        sc = self.config.storage_config

        return TrainingJobSpec(
            symbol=tc.symbol,
            timeframe=tc.timeframe,
            start_date=tc.start_date.isoformat(),
            end_date=tc.end_date.isoformat(),
            epochs=tc.epochs,
            batch_size=tc.batch_size,
            sequence_length=tc.sequence_length,
            instance_type=ic.instance_type,
            use_spot_instances=ic.use_spot_instances,
            max_runtime_seconds=self.config.max_runtime_seconds,
            output_s3_path=sc.model_uri,
            input_data_s3_uri=self.config.input_data_s3_uri,
            hyperparameters={
                "force_sentiment": str(tc.force_sentiment).lower(),
                "force_price_only": str(tc.force_price_only).lower(),
                "mixed_precision": str(tc.mixed_precision).lower(),
            },
        )

    def _wait_for_completion(
        self,
        job_id: str,
        poll_interval: int = DEFAULT_POLL_INTERVAL,
    ) -> TrainingJobStatus:
        """Wait for training job to complete.

        Args:
            job_id: Job identifier
            poll_interval: Seconds between status checks

        Returns:
            Final job status

        Raises:
            JobTimeoutError: If job exceeds max runtime
        """

        max_wait = self.config.max_runtime_seconds * SPOT_WAIT_TIME_MULTIPLIER
        elapsed = 0

        while elapsed < max_wait:
            status = self.provider.get_job_status(job_id)

            if status.is_terminal:
                return status

            # Log progress
            logger.info(
                f"Job status: {status.status} | Elapsed: {elapsed}s | " f"Metrics: {status.metrics}"
            )

            time.sleep(poll_interval)
            elapsed += poll_interval

        raise JobTimeoutError(
            f"Training job exceeded timeout of {max_wait}s. "
            "Consider increasing max_runtime_hours or using a faster instance type."
        )

    def _sync_artifacts(
        self,
        job_id: str,
        s3_output_path: str | None,
    ) -> Path:
        """Sync trained model artifacts to local registry.

        Args:
            job_id: Job identifier
            s3_output_path: S3 path to model artifacts (or local path for local provider)

        Returns:
            Path to local model directory
        """
        if not s3_output_path:
            raise ArtifactSyncError("No output path found for job")

        # Extract and validate job_id suffix for temp directory naming
        job_suffix = job_id.split("/")[-1] if job_id else ""
        # Validate: alphanumeric/hyphens/underscores only, reasonable length
        # Note: regex ^[\w\-]+$ already excludes path separators (/, \)
        if (
            not job_suffix
            or not re.match(r"^[\w\-]+$", job_suffix)
            or len(job_suffix) > MAX_JOB_SUFFIX_LENGTH
        ):
            job_suffix = "download"

        temp_dir: Path | None = None
        try:
            # Create temp directory inside try block (cross-platform)
            temp_dir = Path(tempfile.gettempdir()) / f"model-download-{job_suffix}"

            # Download from provider (S3 for cloud, local path for local provider)
            artifact_path = self.provider.download_artifacts(job_id, temp_dir)

            # SageMaker training creates nested structure: SYMBOL/TYPE/VERSION/*
            # Find the actual model artifacts directory
            actual_artifact_path = self._find_artifacts_root(artifact_path)
            logger.info(f"Found artifacts at: {actual_artifact_path}")

            # Determine version ID, model type, and symbol from metadata
            metadata_path = actual_artifact_path / "metadata.json"
            metadata: dict = {}
            if metadata_path.exists():
                with open(metadata_path) as f:
                    metadata = json.load(f)
            fallback_version = datetime.now(UTC).strftime("%Y-%m-%d_%Hh%Mm%Ss_v1")
            version_id = metadata.get("version_id", fallback_version)
            model_type = metadata.get("model_type", "basic")

            # Validate metadata-supplied path components (see helper docstring).
            version_id = _validate_registry_component(version_id, "version_id")
            model_type = _validate_registry_component(model_type, "model_type")

            # Prefer the bundle's own symbol: for cloud-status --sync the config
            # symbol is only a placeholder parsed from the job name.
            config_symbol = self.config.training_config.symbol.upper()
            metadata_symbol = metadata.get("symbol")
            if metadata_symbol:
                symbol = _validate_registry_component(str(metadata_symbol).upper(), "symbol")
                if symbol != config_symbol:
                    logger.warning(
                        "Metadata symbol %s differs from configured symbol %s; "
                        "syncing under metadata symbol",
                        symbol,
                        config_symbol,
                    )
            else:
                symbol = config_symbol

            # Sync to local registry
            local_registry = get_project_root() / "src" / "ml" / "models"

            # Artifacts are already extracted locally by provider.download_artifacts()
            # Use _sync_local_artifacts for both local and cloud providers
            final_path = self._sync_local_artifacts(
                artifact_path=actual_artifact_path,
                local_registry=local_registry,
                symbol=symbol,
                model_type=model_type,
                version_id=version_id,
            )

            return final_path

        finally:
            # Clean up temp directory
            if temp_dir is not None and temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)

    def _find_artifacts_root(self, artifact_path: Path) -> Path:
        """Find the actual model artifacts directory.

        SageMaker training creates a nested structure: SYMBOL/TYPE/VERSION/*
        This method finds the directory containing the actual model files.

        Args:
            artifact_path: Extracted artifacts path (may be nested)

        Returns:
            Path to the directory containing model files
        """
        # Check if metadata.json exists at the root (local provider or flat structure)
        if (artifact_path / "metadata.json").exists():
            return artifact_path

        # Look for nested structure: SYMBOL/TYPE/VERSION/*
        # This happens when SageMaker saves with full registry path
        # Limit search depth to prevent slow searches on deep directories
        for candidate in itertools.islice(
            artifact_path.rglob("metadata.json"), MAX_METADATA_SEARCH_DEPTH
        ):
            # Find the parent directory that contains model files
            parent = candidate.parent
            if (parent / "model.keras").exists() or (parent / "model.onnx").exists():
                logger.info(f"Found nested artifacts at: {parent}")
                return parent

        # No valid structure found, return as-is
        logger.warning(f"Could not find metadata.json in artifacts, using root: {artifact_path}")
        return artifact_path

    def _sync_local_artifacts(
        self,
        artifact_path: Path,
        local_registry: Path,
        symbol: str,
        model_type: str,
        version_id: str,
    ) -> Path:
        """Sync downloaded artifacts to model registry.

        Used for both local and cloud providers after artifacts are downloaded
        and extracted to a local directory (by provider.download_artifacts()).

        Args:
            artifact_path: Local path to trained artifacts
            local_registry: Local model registry root
            symbol: Trading symbol
            model_type: Model type (basic, sentiment, etc.)
            version_id: Version identifier

        Returns:
            Path to synced model directory
        """
        # Create expected registry directory path
        version_dir = local_registry / symbol / model_type / version_id

        # Defense in depth: ensure the resolved target never escapes the registry
        # before any destructive copytree/symlink operation runs.
        registry_root = local_registry.resolve()
        if not version_dir.resolve().is_relative_to(registry_root):
            raise ArtifactSyncError(f"Computed model path escapes registry root: {version_dir}")

        # If artifact is already in the registry at the correct location, just update the symlink
        if artifact_path.resolve() == version_dir.resolve():
            logger.info(f"Artifacts already at correct registry location: {version_dir}")
        else:
            # Never overwrite an existing bundle: a same-name collision (e.g. two
            # cloud jobs producing identical version IDs) syncs to a suffixed
            # sibling instead of destroying the earlier model.
            if version_dir.exists():
                version_dir = self._next_free_version_dir(version_dir)
                logger.warning(
                    "Version %s already exists in registry; syncing to %s instead",
                    version_id,
                    version_dir.name,
                )
                version_id = version_dir.name
            version_dir.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(artifact_path, version_dir)
            logger.info(f"Copied artifacts to {version_dir}")

        # Update 'latest' symlink
        latest_link = local_registry / symbol / model_type / "latest"
        if latest_link.exists() or latest_link.is_symlink():
            latest_link.unlink()
        latest_link.symlink_to(version_id)
        logger.info(f"Updated latest symlink: {latest_link} -> {version_id}")

        return version_dir

    @staticmethod
    def _next_free_version_dir(version_dir: Path) -> Path:
        """Return the first non-existing ``{version_dir}-{n}`` sibling (n >= 2)."""
        for counter in range(2, MAX_VERSION_SUFFIX_ATTEMPTS + 1):
            candidate = version_dir.with_name(f"{version_dir.name}-{counter}")
            if not candidate.exists():
                return candidate
        raise ArtifactSyncError(
            f"Could not find a free version directory after "
            f"{MAX_VERSION_SUFFIX_ATTEMPTS} attempts for {version_dir}"
        )


class CloudTrainingResult:
    """Result of a cloud training operation.

    Provides a unified result type for cloud training operations,
    similar to TrainingResult from the local pipeline.
    """

    def __init__(
        self,
        success: bool,
        job_id: str | None,
        job_status: str,
        provider: str,
        artifact_path: Path | None = None,
        metrics: dict[str, float] | None = None,
        error: str | None = None,
        message: str | None = None,
        duration_seconds: float = 0.0,
    ) -> None:
        """Initialize result.

        Args:
            success: Whether training completed successfully
            job_id: Cloud provider job identifier
            job_status: Current/final job status
            provider: Provider name (sagemaker, local, etc.)
            artifact_path: Path to synced model artifacts
            metrics: Training metrics (loss, accuracy, etc.)
            error: Error message if failed
            message: Human-readable status message
            duration_seconds: Total operation duration
        """
        self.success = success
        self.job_id = job_id
        self.job_status = job_status
        self.provider = provider
        self.artifact_path = artifact_path
        self.metrics = metrics or {}
        self.error = error
        self.message = message
        self.duration_seconds = duration_seconds

    def to_dict(self) -> dict:
        """Convert result to dictionary for JSON serialization."""
        return {
            "success": self.success,
            "job_id": self.job_id,
            "job_status": self.job_status,
            "provider": self.provider,
            "artifact_path": str(self.artifact_path) if self.artifact_path else None,
            "metrics": self.metrics,
            "error": self.error,
            "message": self.message,
            "duration_seconds": self.duration_seconds,
        }
