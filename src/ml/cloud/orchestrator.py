"""Cloud training orchestrator.

High-level coordination of cloud training workflow:
upload data → submit job → wait → download artifacts → sync to registry.
"""

from __future__ import annotations

import logging
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
            # Step 1: Build job specification
            logger.info("Step 1/4: Building training job specification...")
            job_spec = self._build_job_spec()

            # Step 2: Submit training job
            logger.info(f"Step 2/4: Submitting job to {self.provider.provider_name}...")
            job_id = self.provider.submit_training_job(job_spec)
            logger.info(f"Job submitted: {job_id}")

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
            logger.info("Step 3/4: Waiting for job completion...")
            status = self._wait_for_completion(job_id)

            if not status.is_successful:
                raise JobSubmissionError(
                    f"Training job failed: {status.failure_reason or 'Unknown error'}"
                )

            # Step 4: Sync artifacts to local registry
            logger.info("Step 4/4: Syncing artifacts to local registry...")
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
            return CloudTrainingResult(
                success=False,
                job_id=None,
                job_status="Failed",
                provider=self.provider.provider_name,
                error=str(exc),
                duration_seconds=perf_counter() - start_time,
            )
        except Exception as exc:
            logger.exception("Unexpected error during cloud training")
            return CloudTrainingResult(
                success=False,
                job_id=None,
                job_status="Failed",
                provider=self.provider.provider_name,
                error=f"Unexpected error: {exc}",
                duration_seconds=perf_counter() - start_time,
            )

    def submit_job(self) -> str:
        """Submit training job without waiting.

        Returns:
            Job identifier for status checking

        Raises:
            JobSubmissionError: If submission fails
        """
        job_spec = self._build_job_spec()
        return self.provider.submit_training_job(job_spec)

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

        max_wait = self.config.max_runtime_seconds * 2  # Allow extra time for spot interruptions
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
        import re
        import tempfile

        job_suffix = job_id.split("/")[-1] if job_id else ""
        # Validate: alphanumeric/hyphens/underscores, max 100 chars, no path separators
        if (
            not job_suffix
            or not re.match(r"^[\w\-]+$", job_suffix)
            or len(job_suffix) > 100
            or "/" in job_suffix
            or "\\" in job_suffix
        ):
            job_suffix = "download"

        temp_dir: Path | None = None
        try:
            # Create temp directory inside try block (cross-platform)
            temp_dir = Path(tempfile.gettempdir()) / f"model-download-{job_suffix}"

            # Download from provider (S3 for cloud, local path for local provider)
            artifact_path = self.provider.download_artifacts(job_id, temp_dir)

            # Determine version ID from metadata
            metadata_path = artifact_path / "metadata.json"
            if metadata_path.exists():
                import json

                with open(metadata_path) as f:
                    metadata = json.load(f)
                version_id = metadata.get(
                    "version_id", datetime.now(UTC).strftime("%Y-%m-%d_%Hh_v1")
                )
                model_type = metadata.get("model_type", "basic")
            else:
                version_id = datetime.now(UTC).strftime("%Y-%m-%d_%Hh_v1")
                model_type = "basic"

            # Sync to local registry
            local_registry = get_project_root() / "src" / "ml" / "models"
            symbol = self.config.training_config.symbol.upper()

            # For local provider, artifacts are already on the filesystem
            if self.provider.provider_name == "local":
                final_path = self._sync_local_artifacts(
                    artifact_path=artifact_path,
                    local_registry=local_registry,
                    symbol=symbol,
                    model_type=model_type,
                    version_id=version_id,
                )
            else:
                final_path = self.s3_manager.sync_to_local_registry(
                    s3_uri=s3_output_path,
                    local_registry=local_registry,
                    symbol=symbol,
                    model_type=model_type,
                    version_id=version_id,
                    update_latest=True,
                )

            return final_path

        finally:
            # Clean up temp directory
            if temp_dir is not None and temp_dir.exists():
                import shutil

                shutil.rmtree(temp_dir, ignore_errors=True)

    def _sync_local_artifacts(
        self,
        artifact_path: Path,
        local_registry: Path,
        symbol: str,
        model_type: str,
        version_id: str,
    ) -> Path:
        """Sync locally trained artifacts to model registry.

        Used when training on local provider (no S3 needed).

        Args:
            artifact_path: Local path to trained artifacts
            local_registry: Local model registry root
            symbol: Trading symbol
            model_type: Model type (basic, sentiment, etc.)
            version_id: Version identifier

        Returns:
            Path to synced model directory
        """
        import shutil

        # Create expected registry directory path
        version_dir = local_registry / symbol / model_type / version_id

        # If artifact is already in the registry at the correct location, just update the symlink
        if artifact_path.resolve() == version_dir.resolve():
            logger.info(f"Artifacts already at correct registry location: {version_dir}")
        else:
            # Copy artifacts to registry
            version_dir.parent.mkdir(parents=True, exist_ok=True)
            if version_dir.exists():
                shutil.rmtree(version_dir)
            shutil.copytree(artifact_path, version_dir)
            logger.info(f"Copied artifacts to {version_dir}")

        # Update 'latest' symlink
        latest_link = local_registry / symbol / model_type / "latest"
        if latest_link.exists() or latest_link.is_symlink():
            latest_link.unlink()
        latest_link.symlink_to(version_id)
        logger.info(f"Updated latest symlink: {latest_link} -> {version_id}")

        return version_dir


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
