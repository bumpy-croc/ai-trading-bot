"""Unit tests for cloud training configuration."""

from datetime import datetime
from unittest.mock import patch

import pytest

from src.ml.cloud.config import (
    DEFAULT_INSTANCE_TYPE,
    DEFAULT_MAX_RUNTIME_HOURS,
    DEFAULT_VOLUME_SIZE_GB,
    CloudInstanceConfig,
    CloudStorageConfig,
    CloudTrainingConfig,
)
from src.ml.training_pipeline.config import TrainingConfig


class TestCloudInstanceConfig:
    """Tests for CloudInstanceConfig dataclass."""

    def test_default_values(self) -> None:
        """Verify default configuration values match constants."""
        config = CloudInstanceConfig()

        assert config.instance_type == DEFAULT_INSTANCE_TYPE
        assert config.max_runtime_hours == DEFAULT_MAX_RUNTIME_HOURS
        assert config.volume_size_gb == DEFAULT_VOLUME_SIZE_GB
        assert config.use_spot_instances is True

    def test_custom_values(self) -> None:
        """Verify custom values override defaults."""
        config = CloudInstanceConfig(
            instance_type="ml.p3.2xlarge",
            use_spot_instances=False,
            max_runtime_hours=8,
            volume_size_gb=100,
        )

        assert config.instance_type == "ml.p3.2xlarge"
        assert config.use_spot_instances is False
        assert config.max_runtime_hours == 8
        assert config.volume_size_gb == 100


class TestCloudStorageConfig:
    """Tests for CloudStorageConfig dataclass."""

    def test_required_bucket(self) -> None:
        """Verify s3_bucket is required."""
        config = CloudStorageConfig(s3_bucket="my-bucket")
        assert config.s3_bucket == "my-bucket"

    def test_default_prefixes(self) -> None:
        """Verify default prefix values."""
        config = CloudStorageConfig(s3_bucket="my-bucket")

        assert config.data_prefix == "training-data"
        assert config.model_prefix == "models"
        assert config.enable_versioning is True

    def test_data_uri_property(self) -> None:
        """Verify data_uri property construction."""
        config = CloudStorageConfig(s3_bucket="my-bucket", data_prefix="custom-data")
        assert config.data_uri == "s3://my-bucket/custom-data"

    def test_model_uri_property(self) -> None:
        """Verify model_uri property construction."""
        config = CloudStorageConfig(s3_bucket="my-bucket", model_prefix="custom-models")
        assert config.model_uri == "s3://my-bucket/custom-models"


class TestCloudTrainingConfig:
    """Tests for CloudTrainingConfig dataclass."""

    @pytest.fixture
    def training_config(self) -> TrainingConfig:
        """Create a sample TrainingConfig for testing."""
        return TrainingConfig(
            symbol="BTCUSDT",
            timeframe="1h",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 12, 1),
            epochs=100,
        )

    def test_max_runtime_seconds(self, training_config: TrainingConfig) -> None:
        """Verify max_runtime_seconds calculation."""
        storage = CloudStorageConfig(s3_bucket="test-bucket")
        instance = CloudInstanceConfig(max_runtime_hours=2)

        config = CloudTrainingConfig(
            training_config=training_config,
            storage_config=storage,
            instance_config=instance,
        )

        assert config.max_runtime_seconds == 7200  # 2 hours * 3600 seconds

    def test_default_values(self, training_config: TrainingConfig) -> None:
        """Verify default configuration values."""
        storage = CloudStorageConfig(s3_bucket="test-bucket")

        config = CloudTrainingConfig(
            training_config=training_config,
            storage_config=storage,
        )

        assert config.provider == "sagemaker"
        assert config.job_name_prefix == "atb-training"
        assert config.auto_sync_artifacts is True
        assert config.docker_image_uri is None

    @patch.dict(
        "os.environ",
        {
            "SAGEMAKER_S3_BUCKET": "env-bucket",
            "SAGEMAKER_INSTANCE_TYPE": "ml.p3.2xlarge",
            "SAGEMAKER_USE_SPOT": "false",
            "SAGEMAKER_MAX_RUNTIME_HOURS": "6",
            "CLOUD_TRAINING_PROVIDER": "local",
            "SAGEMAKER_DOCKER_IMAGE": "123456.ecr.amazonaws.com/training:v1",
        },
    )
    def test_from_env(self, training_config: TrainingConfig) -> None:
        """Verify from_env loads configuration from environment."""
        config = CloudTrainingConfig.from_env(training_config)

        assert config.storage_config.s3_bucket == "env-bucket"
        assert config.instance_config.instance_type == "ml.p3.2xlarge"
        assert config.instance_config.use_spot_instances is False
        assert config.instance_config.max_runtime_hours == 6
        assert config.provider == "local"
        assert config.docker_image_uri == "123456.ecr.amazonaws.com/training:v1"

    @patch.dict("os.environ", {}, clear=True)
    def test_from_env_requires_s3_bucket(self, training_config: TrainingConfig) -> None:
        """Verify from_env raises ValueError when SAGEMAKER_S3_BUCKET is not set."""
        with pytest.raises(ValueError, match="SAGEMAKER_S3_BUCKET environment variable is required"):
            CloudTrainingConfig.from_env(training_config)

    @patch.dict(
        "os.environ",
        {"SAGEMAKER_S3_BUCKET": "test-bucket"},
        clear=True,
    )
    def test_from_env_defaults(self, training_config: TrainingConfig) -> None:
        """Verify from_env uses defaults when only required env vars are set."""
        config = CloudTrainingConfig.from_env(training_config)

        assert config.storage_config.s3_bucket == "test-bucket"
        assert config.instance_config.instance_type == DEFAULT_INSTANCE_TYPE
        assert config.instance_config.use_spot_instances is True
        assert config.instance_config.max_runtime_hours == DEFAULT_MAX_RUNTIME_HOURS
        assert config.provider == "sagemaker"
