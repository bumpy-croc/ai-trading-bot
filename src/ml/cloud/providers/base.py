"""Abstract base class for cloud training providers.

Defines the interface that all cloud training providers must implement,
allowing the orchestrator to work with any provider (SageMaker, Vertex AI, etc.).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class TrainingJobSpec:
    """Specification for a cloud training job.

    Contains all parameters needed to submit a training job to any cloud provider.
    Provider-specific details are handled in the provider implementation.

    Attributes:
        symbol: Trading symbol (e.g., BTCUSDT)
        timeframe: Data timeframe (e.g., 1h, 4h, 1d)
        start_date: Training data start date (ISO format)
        end_date: Training data end date (ISO format)
        epochs: Number of training epochs
        batch_size: Training batch size
        sequence_length: LSTM sequence length
        instance_type: Cloud instance type (e.g., ml.g4dn.xlarge)
        use_spot_instances: Use spot/preemptible instances for cost savings
        max_runtime_seconds: Maximum job runtime before timeout
        output_s3_path: S3 path for model output artifacts
        hyperparameters: Additional hyperparameters to pass to training script
    """

    symbol: str
    timeframe: str
    start_date: str
    end_date: str
    epochs: int
    batch_size: int
    sequence_length: int
    instance_type: str
    use_spot_instances: bool
    max_runtime_seconds: int
    output_s3_path: str
    hyperparameters: dict[str, Any] = field(default_factory=dict)

    def to_hyperparameters(self) -> dict[str, str]:
        """Convert spec to SageMaker-compatible hyperparameters dict.

        SageMaker requires all hyperparameter values to be strings.

        Returns:
            Dictionary with string values for all parameters
        """
        params = {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "epochs": str(self.epochs),
            "batch_size": str(self.batch_size),
            "sequence_length": str(self.sequence_length),
        }
        # Add any additional hyperparameters
        for key, value in self.hyperparameters.items():
            params[key] = str(value)
        return params


@dataclass
class TrainingJobStatus:
    """Status of a cloud training job.

    Provides a unified view of job status across different cloud providers.

    Attributes:
        job_name: Unique job identifier
        status: Current status (Pending, InProgress, Completed, Failed, Stopped)
        start_time: Job start timestamp (ISO format or None)
        end_time: Job completion timestamp (ISO format or None)
        failure_reason: Error message if job failed
        output_s3_path: S3 path to model artifacts (populated on completion)
        metrics: Training metrics extracted from job (loss, accuracy, etc.)
    """

    job_name: str
    status: str  # Pending | InProgress | Completed | Failed | Stopped
    start_time: datetime | None = None
    end_time: datetime | None = None
    failure_reason: str | None = None
    output_s3_path: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def is_terminal(self) -> bool:
        """Check if job has reached a terminal state."""
        return self.status in ("Completed", "Failed", "Stopped")

    @property
    def is_successful(self) -> bool:
        """Check if job completed successfully."""
        return self.status == "Completed"

    @property
    def duration_seconds(self) -> float | None:
        """Calculate job duration in seconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None


class CloudTrainingProvider(ABC):
    """Abstract base class for cloud training providers.

    Defines the contract that all cloud training providers must implement.
    This allows the CloudTrainingOrchestrator to work with any provider
    through a consistent interface.

    Implementations:
    - SageMakerProvider: AWS SageMaker training jobs
    - LocalProvider: Local training for testing (no cloud required)
    - VertexAIProvider: GCP Vertex AI (future)
    """

    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is configured and available.

        Validates that all required credentials and configuration
        are present for this provider to function.

        Returns:
            True if provider is ready to accept training jobs
        """
        pass

    @abstractmethod
    def submit_training_job(self, spec: TrainingJobSpec) -> str:
        """Submit a training job to the cloud provider.

        Args:
            spec: Training job specification

        Returns:
            Unique job identifier (ARN for SageMaker, job ID for others)

        Raises:
            JobSubmissionError: If job submission fails
            ProviderNotAvailableError: If provider is not configured
        """
        pass

    @abstractmethod
    def get_job_status(self, job_id: str) -> TrainingJobStatus:
        """Get current status of a training job.

        Args:
            job_id: Job identifier returned from submit_training_job

        Returns:
            Current job status including metrics if available

        Raises:
            CloudTrainingError: If status check fails
        """
        pass

    @abstractmethod
    def cancel_job(self, job_id: str) -> None:
        """Cancel a running training job.

        Args:
            job_id: Job identifier to cancel

        Raises:
            CloudTrainingError: If cancellation fails
        """
        pass

    @abstractmethod
    def download_artifacts(self, job_id: str, local_path: Path) -> Path:
        """Download trained model artifacts to local path.

        Args:
            job_id: Job identifier for completed job
            local_path: Local directory to download artifacts to

        Returns:
            Path to downloaded artifacts directory

        Raises:
            ArtifactSyncError: If download fails
        """
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return provider name for logging and identification.

        Returns:
            Provider identifier (e.g., 'sagemaker', 'vertex_ai', 'local')
        """
        pass
