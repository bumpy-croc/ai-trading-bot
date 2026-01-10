"""Configuration dataclasses for cloud training.

Provides structured configuration for cloud training jobs,
including instance settings, storage paths, and provider selection.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from src.ml.training_pipeline.config import TrainingConfig

# Default cloud instance configuration values
DEFAULT_INSTANCE_TYPE = "ml.g4dn.xlarge"  # T4 GPU, good balance of cost/performance
DEFAULT_MAX_RUNTIME_HOURS = 4  # Maximum training time before timeout
DEFAULT_VOLUME_SIZE_GB = 30  # EBS volume size for training data


@dataclass
class CloudInstanceConfig:
    """Cloud compute instance configuration.

    Attributes:
        instance_type: SageMaker/cloud instance type (e.g., ml.g4dn.xlarge for T4 GPU)
        use_spot_instances: Use spot/preemptible instances for cost savings (70% cheaper)
        max_runtime_hours: Maximum training runtime before timeout
        volume_size_gb: EBS volume size for training data and artifacts
    """

    instance_type: str = DEFAULT_INSTANCE_TYPE
    use_spot_instances: bool = True
    max_runtime_hours: int = DEFAULT_MAX_RUNTIME_HOURS
    volume_size_gb: int = DEFAULT_VOLUME_SIZE_GB


@dataclass
class CloudStorageConfig:
    """Cloud storage configuration for training artifacts.

    Attributes:
        s3_bucket: S3 bucket name for storing training data and model artifacts
        data_prefix: S3 prefix for training data uploads
        model_prefix: S3 prefix for trained model outputs
        enable_versioning: Enable S3 versioning for model artifacts (recommended)
    """

    s3_bucket: str
    data_prefix: str = "training-data"
    model_prefix: str = "models"
    enable_versioning: bool = True

    @property
    def data_uri(self) -> str:
        """Return S3 URI for training data uploads."""
        return f"s3://{self.s3_bucket}/{self.data_prefix}"

    @property
    def model_uri(self) -> str:
        """Return S3 URI for model artifact outputs."""
        return f"s3://{self.s3_bucket}/{self.model_prefix}"


@dataclass
class CloudTrainingConfig:
    """Complete configuration for cloud-based training jobs.

    Combines training parameters (epochs, batch_size, etc.) with
    cloud-specific settings (instance type, S3 bucket, provider).

    Attributes:
        training_config: Core training parameters (reuses existing TrainingConfig)
        instance_config: Cloud compute instance settings
        storage_config: S3 bucket and path configuration
        provider: Cloud provider name (sagemaker, vertex_ai, local)
        job_name_prefix: Prefix for job names (helps with filtering in console)
        auto_sync_artifacts: Automatically sync artifacts to local registry on completion
        docker_image_uri: ECR image URI for training container (provider-specific)
    """

    training_config: TrainingConfig
    storage_config: CloudStorageConfig
    instance_config: CloudInstanceConfig = field(default_factory=CloudInstanceConfig)
    provider: str = "sagemaker"
    job_name_prefix: str = "atb-training"
    auto_sync_artifacts: bool = True
    docker_image_uri: str | None = None

    @classmethod
    def from_env(cls, training_config: TrainingConfig) -> CloudTrainingConfig:
        """Create CloudTrainingConfig from environment variables.

        Reads cloud configuration from environment:
        - SAGEMAKER_S3_BUCKET: S3 bucket for artifacts (required)
        - CLOUD_TRAINING_PROVIDER: Provider name (default: sagemaker)
        - SAGEMAKER_INSTANCE_TYPE: Instance type (default: ml.g4dn.xlarge)
        - SAGEMAKER_USE_SPOT: Use spot instances (default: true)
        - SAGEMAKER_DOCKER_IMAGE: ECR image URI (optional)

        Args:
            training_config: Core training configuration

        Returns:
            CloudTrainingConfig populated from environment
        """
        s3_bucket = os.getenv("SAGEMAKER_S3_BUCKET", "")
        if not s3_bucket:
            raise ValueError(
                "SAGEMAKER_S3_BUCKET environment variable is required for cloud training. "
                "Set it to your S3 bucket name for storing training artifacts."
            )

        storage_config = CloudStorageConfig(s3_bucket=s3_bucket)

        instance_config = CloudInstanceConfig(
            instance_type=os.getenv("SAGEMAKER_INSTANCE_TYPE", DEFAULT_INSTANCE_TYPE),
            use_spot_instances=os.getenv("SAGEMAKER_USE_SPOT", "true").lower() == "true",
            max_runtime_hours=int(
                os.getenv("SAGEMAKER_MAX_RUNTIME_HOURS", str(DEFAULT_MAX_RUNTIME_HOURS))
            ),
        )

        return cls(
            training_config=training_config,
            storage_config=storage_config,
            instance_config=instance_config,
            provider=os.getenv("CLOUD_TRAINING_PROVIDER", "sagemaker"),
            docker_image_uri=os.getenv("SAGEMAKER_DOCKER_IMAGE"),
        )

    @property
    def max_runtime_seconds(self) -> int:
        """Return maximum runtime in seconds for job timeout."""
        return self.instance_config.max_runtime_hours * 3600
