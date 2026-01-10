"""Cloud training exceptions.

Provides domain-specific error types for cloud training operations,
following the same pattern as src/prediction/exceptions.py.
"""


class CloudTrainingError(Exception):
    """Base exception for cloud training errors."""

    pass


class ProviderNotAvailableError(CloudTrainingError):
    """Raised when cloud provider is not configured or unavailable.

    Common causes:
    - Missing environment variables (SAGEMAKER_ROLE_ARN, AWS_REGION)
    - Invalid credentials
    - Provider-specific setup not completed
    """

    pass


class JobSubmissionError(CloudTrainingError):
    """Raised when training job submission fails.

    Common causes:
    - SageMaker quota limits exceeded
    - Invalid hyperparameters
    - Docker image not found in ECR
    - IAM permission issues
    """

    pass


class ArtifactSyncError(CloudTrainingError):
    """Raised when artifact upload or download fails.

    Common causes:
    - S3 bucket not accessible
    - Insufficient permissions
    - Network connectivity issues
    - Model artifacts corrupted
    """

    pass


class JobTimeoutError(CloudTrainingError):
    """Raised when training job exceeds maximum runtime.

    Common causes:
    - Epochs too high for instance type
    - Dataset too large
    - Spot instance interruption without recovery
    """

    pass
