"""AWS SageMaker training provider.

Implements the CloudTrainingProvider interface for AWS SageMaker,
with support for spot instances, managed checkpointing, and S3 artifacts.
"""

from __future__ import annotations

import logging
import os
import tarfile
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.ml.cloud.exceptions import (
    ArtifactSyncError,
    JobSubmissionError,
    ProviderNotAvailableError,
)
from src.ml.cloud.providers.base import (
    CloudTrainingProvider,
    TrainingJobSpec,
    TrainingJobStatus,
)

if TYPE_CHECKING:
    from mypy_boto3_s3 import S3Client
    from mypy_boto3_sagemaker import SageMakerClient

logger = logging.getLogger(__name__)


class SageMakerProvider(CloudTrainingProvider):
    """AWS SageMaker training provider with spot instance support.

    Uses SageMaker Training Jobs to run model training on GPU instances.
    Supports managed spot training for up to 70% cost savings.

    Environment variables:
        SAGEMAKER_ROLE_ARN: IAM role ARN for SageMaker execution
        SAGEMAKER_S3_BUCKET: S3 bucket for training artifacts
        AWS_REGION: AWS region (default: us-east-1)
        SAGEMAKER_DOCKER_IMAGE: ECR image URI for training container
    """

    def __init__(
        self,
        role_arn: str | None = None,
        s3_bucket: str | None = None,
        region: str | None = None,
        docker_image_uri: str | None = None,
    ) -> None:
        """Initialize SageMaker provider.

        Args:
            role_arn: IAM role ARN (or from SAGEMAKER_ROLE_ARN env var)
            s3_bucket: S3 bucket name (or from SAGEMAKER_S3_BUCKET env var)
            region: AWS region (or from AWS_REGION env var)
            docker_image_uri: ECR image URI (or from SAGEMAKER_DOCKER_IMAGE env var)
        """
        self._role_arn = role_arn or os.getenv("SAGEMAKER_ROLE_ARN")
        self._s3_bucket = s3_bucket or os.getenv("SAGEMAKER_S3_BUCKET")
        self._region = region or os.getenv("AWS_REGION", "us-east-1")
        self._docker_image_uri = docker_image_uri or os.getenv("SAGEMAKER_DOCKER_IMAGE")

        self._sagemaker_client: SageMakerClient | None = None
        self._s3_client: S3Client | None = None

    def _ensure_clients(self) -> None:
        """Lazily initialize boto3 clients on first use."""
        if self._sagemaker_client is None:
            try:
                import boto3
                from botocore.config import Config

                # Configure timeouts for external API calls (CODE.md: line 178)
                # connect_timeout: time to establish connection
                # read_timeout: time to read response from server
                config = Config(
                    connect_timeout=10,  # 10 seconds to connect
                    read_timeout=60,  # 60 seconds to read response
                    retries={"max_attempts": 3, "mode": "standard"},  # Retry with backoff
                )

                self._sagemaker_client = boto3.client(
                    "sagemaker", region_name=self._region, config=config
                )
                self._s3_client = boto3.client("s3", region_name=self._region, config=config)
            except ImportError as exc:
                raise ProviderNotAvailableError(
                    "boto3 is required for SageMaker provider. "
                    "Install with: pip install '.[cloud]'"
                ) from exc

    def is_available(self) -> bool:
        """Check if SageMaker credentials are configured."""
        if not self._role_arn or not self._s3_bucket:
            logger.debug(
                f"SageMaker not available: role_arn={self._role_arn}, "
                f"s3_bucket={self._s3_bucket}"
            )
            return False

        try:
            self._ensure_clients()
            # Verify credentials by making a lightweight API call
            if self._sagemaker_client is None:
                return False
            self._sagemaker_client.list_training_jobs(MaxResults=1)
            return True
        except Exception as exc:
            logger.debug(f"SageMaker availability check failed: {exc}")
            return False

    def submit_training_job(self, spec: TrainingJobSpec) -> str:
        """Submit training job to SageMaker with spot instances.

        Args:
            spec: Training job specification

        Returns:
            SageMaker training job ARN

        Raises:
            JobSubmissionError: If job submission fails
            ProviderNotAvailableError: If SageMaker is not configured
        """
        if not self.is_available():
            raise ProviderNotAvailableError(
                "SageMaker is not configured. Set SAGEMAKER_ROLE_ARN and SAGEMAKER_S3_BUCKET."
            )

        self._ensure_clients()
        if self._sagemaker_client is None:
            raise ProviderNotAvailableError("SageMaker client not initialized")

        job_name = self._generate_job_name(spec.symbol, spec.timeframe)

        try:
            training_params = self._build_training_params(job_name, spec)
            response = self._sagemaker_client.create_training_job(**training_params)

            job_arn = response["TrainingJobArn"]
            logger.info(
                "Submitted SageMaker training job",
                extra={
                    "job_name": job_name,
                    "job_arn": job_arn,
                    "symbol": spec.symbol,
                    "instance_type": spec.instance_type,
                    "spot_instances": spec.use_spot_instances,
                },
            )
            return job_arn

        except Exception as exc:
            error_msg = str(exc)
            logger.error(f"Raw SageMaker error: {exc.__class__.__name__}: {error_msg}")
            if "ResourceLimitExceeded" in error_msg:
                raise JobSubmissionError(
                    f"SageMaker quota limit exceeded for {spec.instance_type}. "
                    "Request a quota increase or use a different instance type."
                ) from exc
            if "ValidationError" in error_msg:
                raise JobSubmissionError(f"Invalid training parameters: {error_msg}") from exc
            raise JobSubmissionError(f"SageMaker job submission failed: {error_msg}") from exc

    def get_job_status(self, job_id: str) -> TrainingJobStatus:
        """Get current status of a SageMaker training job.

        Args:
            job_id: Job ARN or job name

        Returns:
            Current job status with metrics
        """
        self._ensure_clients()
        if self._sagemaker_client is None:
            raise ProviderNotAvailableError("SageMaker client not initialized")

        job_name = self._extract_job_name(job_id)

        try:
            response = self._sagemaker_client.describe_training_job(TrainingJobName=job_name)
            return self._parse_job_response(response)
        except Exception as exc:
            raise JobSubmissionError(f"Failed to get job status: {exc}") from exc

    def cancel_job(self, job_id: str) -> None:
        """Cancel a running SageMaker training job.

        Args:
            job_id: Job ARN or job name
        """
        self._ensure_clients()
        if self._sagemaker_client is None:
            raise ProviderNotAvailableError("SageMaker client not initialized")

        job_name = self._extract_job_name(job_id)

        try:
            self._sagemaker_client.stop_training_job(TrainingJobName=job_name)
            logger.info(f"Cancelled SageMaker training job: {job_name}")
        except Exception as exc:
            logger.warning(f"Failed to cancel job {job_name}: {exc}")

    def download_artifacts(self, job_id: str, local_path: Path) -> Path:
        """Download trained model artifacts from S3.

        SageMaker outputs a model.tar.gz file to the specified S3 path.
        This method downloads and extracts it to the local path.

        Args:
            job_id: Job ARN or job name
            local_path: Local directory to download artifacts to

        Returns:
            Path to extracted artifacts directory
        """
        self._ensure_clients()
        if self._s3_client is None:
            raise ProviderNotAvailableError("S3 client not initialized")

        status = self.get_job_status(job_id)
        if not status.is_successful:
            raise ArtifactSyncError(f"Cannot download artifacts for job in state: {status.status}")

        if not status.output_s3_path:
            raise ArtifactSyncError("Job completed but no output S3 path found")

        try:
            # Parse S3 URI: s3://bucket/prefix/model.tar.gz
            s3_uri = status.output_s3_path
            bucket, key = self._parse_s3_uri(s3_uri)

            # Ensure output directory exists
            local_path.mkdir(parents=True, exist_ok=True)

            # Download model.tar.gz
            with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp_file:
                tmp_path = Path(tmp_file.name)

            self._s3_client.download_file(bucket, key, str(tmp_path))
            logger.info(f"Downloaded artifacts from s3://{bucket}/{key}")

            # Extract tarball
            with tarfile.open(tmp_path, "r:gz") as tar:
                tar.extractall(path=local_path)

            # Clean up temp file
            tmp_path.unlink()

            return local_path

        except Exception as exc:
            raise ArtifactSyncError(f"Failed to download artifacts: {exc}") from exc

    @property
    def provider_name(self) -> str:
        """Return provider name."""
        return "sagemaker"

    def _generate_job_name(self, symbol: str, timeframe: str) -> str:
        """Generate unique job name for SageMaker.

        SageMaker job names must be unique and match pattern: ^[a-zA-Z0-9](-*[a-zA-Z0-9])*
        Max length: 63 characters
        """
        timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        return f"atb-{symbol.lower()}-{timeframe}-{timestamp}"

    def _extract_job_name(self, job_id: str) -> str:
        """Extract job name from ARN or return as-is if already a name."""
        if job_id.startswith("arn:aws:sagemaker"):
            # ARN format: arn:aws:sagemaker:region:account:training-job/job-name
            return job_id.split("/")[-1]
        return job_id

    def _build_training_params(self, job_name: str, spec: TrainingJobSpec) -> dict[str, Any]:
        """Build SageMaker CreateTrainingJob parameters."""
        params: dict[str, Any] = {
            "TrainingJobName": job_name,
            "RoleArn": self._role_arn,
            "AlgorithmSpecification": {
                "TrainingImage": self._get_training_image(),
                "TrainingInputMode": "File",
                "ContainerEntrypoint": ["python", "-m", "src.ml.cloud.entrypoint"],
            },
            "OutputDataConfig": {
                "S3OutputPath": spec.output_s3_path,
            },
            "ResourceConfig": {
                "InstanceType": spec.instance_type,
                "InstanceCount": 1,
                "VolumeSizeInGB": 50,
            },
            "StoppingCondition": {
                "MaxRuntimeInSeconds": spec.max_runtime_seconds,
            },
            "HyperParameters": spec.to_hyperparameters(),
            "Environment": {
                "PYTHONPATH": "/opt/ml/code",
            },
        }

        # Enable managed spot training for cost savings
        if spec.use_spot_instances:
            params["EnableManagedSpotTraining"] = True
            # Allow 2x runtime for spot interruptions
            params["StoppingCondition"]["MaxWaitTimeInSeconds"] = spec.max_runtime_seconds * 2

        return params

    def _get_training_image(self) -> str:
        """Get ECR image URI for training container."""
        if self._docker_image_uri:
            return self._docker_image_uri

        # Fallback to official TensorFlow GPU image
        # Using TensorFlow 2.15.1 (verified available) with Python 3.11 and CUDA 12.1
        # Note: Using -ec2 tag as -sagemaker tags are not consistently available
        return f"763104351884.dkr.ecr.{self._region}.amazonaws.com/tensorflow-training:2.15.1-gpu-py311-cu121-ubuntu22.04-ec2"

    def _parse_job_response(self, response: dict[str, Any]) -> TrainingJobStatus:
        """Parse SageMaker DescribeTrainingJob response."""
        # Extract metrics from training job output
        metrics: dict[str, float] = {}
        if "FinalMetricDataList" in response:
            for metric in response["FinalMetricDataList"]:
                metrics[metric["MetricName"]] = metric["Value"]

        # Get output path
        output_path = None
        if "ModelArtifacts" in response:
            output_path = response["ModelArtifacts"].get("S3ModelArtifacts")

        return TrainingJobStatus(
            job_name=response["TrainingJobName"],
            status=response["TrainingJobStatus"],
            start_time=response.get("TrainingStartTime"),
            end_time=response.get("TrainingEndTime"),
            failure_reason=response.get("FailureReason"),
            output_s3_path=output_path,
            metrics=metrics,
        )

    def _parse_s3_uri(self, s3_uri: str) -> tuple[str, str]:
        """Parse S3 URI into bucket and key."""
        if not s3_uri.startswith("s3://"):
            raise ValueError(f"Invalid S3 URI: {s3_uri}")
        # Remove s3:// prefix
        path = s3_uri[5:]
        bucket, _, key = path.partition("/")
        return bucket, key
