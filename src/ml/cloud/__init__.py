"""Cloud training module for running ML training on cloud infrastructure.

This module provides abstractions for training models on cloud platforms
(AWS SageMaker, GCP Vertex AI, etc.) with automatic artifact syncing to S3.
"""

from src.ml.cloud.config import CloudInstanceConfig, CloudStorageConfig, CloudTrainingConfig
from src.ml.cloud.exceptions import (
    ArtifactSyncError,
    CloudTrainingError,
    JobSubmissionError,
    JobTimeoutError,
    ProviderNotAvailableError,
)
from src.ml.cloud.orchestrator import CloudTrainingOrchestrator

__all__ = [
    # Config
    "CloudTrainingConfig",
    "CloudInstanceConfig",
    "CloudStorageConfig",
    # Exceptions
    "CloudTrainingError",
    "ProviderNotAvailableError",
    "JobSubmissionError",
    "ArtifactSyncError",
    "JobTimeoutError",
    # Orchestrator
    "CloudTrainingOrchestrator",
]
