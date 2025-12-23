"""Local training provider for testing cloud training workflow.

Implements CloudTrainingProvider by running the actual training pipeline locally,
allowing full end-to-end testing of the cloud training workflow without AWS.
"""

from __future__ import annotations

import logging
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.ml.cloud.exceptions import ArtifactSyncError
from src.ml.cloud.providers.base import (
    CloudTrainingProvider,
    TrainingJobSpec,
    TrainingJobStatus,
)

logger = logging.getLogger(__name__)


class LocalProvider(CloudTrainingProvider):
    """Local fallback provider for testing cloud training workflow.

    Runs the actual training pipeline locally in a background thread,
    simulating the async behavior of cloud training jobs.

    Useful for:
    - Testing the cloud training CLI without AWS account
    - CI/CD pipeline testing
    - Development environment fallback
    """

    def __init__(self) -> None:
        """Initialize local provider."""
        self._jobs: dict[str, TrainingJobStatus] = {}
        self._job_results: dict[str, Any] = {}
        self._lock = threading.Lock()

    def is_available(self) -> bool:
        """Local provider is always available."""
        return True

    def submit_training_job(self, spec: TrainingJobSpec) -> str:
        """Submit training job by running local training in background thread.

        Args:
            spec: Training job specification

        Returns:
            Local job identifier
        """
        job_id = f"local-{uuid.uuid4().hex[:8]}"

        with self._lock:
            self._jobs[job_id] = TrainingJobStatus(
                job_name=job_id,
                status="InProgress",
                start_time=datetime.now(UTC),
                end_time=None,
                failure_reason=None,
                output_s3_path=None,
                metrics={},
            )

        # Run training in background thread
        thread = threading.Thread(
            target=self._run_local_training,
            args=(job_id, spec),
            daemon=True,
        )
        thread.start()

        logger.info(
            "Submitted local training job",
            extra={
                "job_id": job_id,
                "symbol": spec.symbol,
                "epochs": spec.epochs,
            },
        )

        return job_id

    def get_job_status(self, job_id: str) -> TrainingJobStatus:
        """Get current status of local training job.

        Args:
            job_id: Job identifier

        Returns:
            Current job status
        """
        with self._lock:
            if job_id not in self._jobs:
                raise ValueError(f"Job not found: {job_id}")
            return self._jobs[job_id]

    def cancel_job(self, job_id: str) -> None:
        """Cancel local training job.

        Note: Local provider doesn't support true cancellation,
        but will mark the job as stopped.
        """
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].status = "Stopped"
                self._jobs[job_id].end_time = datetime.now(UTC)
                logger.info(f"Marked local job as stopped: {job_id}")

    def download_artifacts(self, job_id: str, local_path: Path) -> Path:
        """Get path to locally trained model artifacts.

        For local provider, artifacts are already on the local filesystem.

        Args:
            job_id: Job identifier
            local_path: Ignored for local provider

        Returns:
            Path to trained model artifacts
        """
        status = self.get_job_status(job_id)

        if not status.is_successful:
            raise ArtifactSyncError(f"Cannot get artifacts for job in state: {status.status}")

        if not status.output_s3_path:
            raise ArtifactSyncError("Job completed but no output path found")

        # For local provider, output_s3_path is actually a local path
        artifact_path = Path(status.output_s3_path)
        if not artifact_path.exists():
            raise ArtifactSyncError(f"Artifact path does not exist: {artifact_path}")

        return artifact_path

    @property
    def provider_name(self) -> str:
        """Return provider name."""
        return "local"

    def _run_local_training(self, job_id: str, spec: TrainingJobSpec) -> None:
        """Run actual training pipeline locally.

        This method runs in a background thread.

        Args:
            job_id: Job identifier
            spec: Training specification
        """
        try:
            # Import here to avoid circular imports and heavy dependencies at module level
            from src.ml.training_pipeline.config import (
                DiagnosticsOptions,
                TrainingConfig,
                TrainingContext,
            )
            from src.ml.training_pipeline.pipeline import run_training_pipeline

            # Convert spec to TrainingConfig
            config = TrainingConfig(
                symbol=spec.symbol,
                timeframe=spec.timeframe,
                start_date=datetime.fromisoformat(spec.start_date),
                end_date=datetime.fromisoformat(spec.end_date),
                epochs=spec.epochs,
                batch_size=spec.batch_size,
                sequence_length=spec.sequence_length,
                # Skip plots and robustness for faster local testing
                diagnostics=DiagnosticsOptions(
                    generate_plots=False,
                    evaluate_robustness=False,
                    convert_to_onnx=True,
                ),
            )

            ctx = TrainingContext(config=config)
            result = run_training_pipeline(ctx)

            with self._lock:
                # Check for success with valid artifact paths
                has_valid_artifacts = (
                    result.success
                    and result.artifact_paths is not None
                    and hasattr(result.artifact_paths, "directory")
                    and result.artifact_paths.directory is not None
                )
                if has_valid_artifacts:
                    self._jobs[job_id].status = "Completed"
                    self._jobs[job_id].end_time = datetime.now(UTC)
                    self._jobs[job_id].output_s3_path = str(result.artifact_paths.directory)
                    self._jobs[job_id].metrics = result.metadata.get("evaluation_results", {})
                    self._job_results[job_id] = result
                    logger.info(f"Local training completed: {job_id}")
                else:
                    self._jobs[job_id].status = "Failed"
                    self._jobs[job_id].end_time = datetime.now(UTC)
                    self._jobs[job_id].failure_reason = result.metadata.get(
                        "error", "Unknown error"
                    )
                    logger.error(f"Local training failed: {job_id}")

        except Exception as exc:
            with self._lock:
                self._jobs[job_id].status = "Failed"
                self._jobs[job_id].end_time = datetime.now(UTC)
                self._jobs[job_id].failure_reason = str(exc)
            logger.exception(f"Local training failed with exception: {job_id}")
