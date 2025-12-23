"""S3 artifact manager for cloud training.

Handles upload/download of training data and model artifacts to/from S3,
with support for syncing trained models to the local model registry.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from src.ml.cloud.exceptions import ArtifactSyncError

if TYPE_CHECKING:
    from mypy_boto3_s3 import S3Client

logger = logging.getLogger(__name__)


class S3ArtifactManager:
    """Manages upload/download of training artifacts to/from S3.

    Handles:
    - Uploading training data (OHLCV CSV, sentiment CSV) to S3
    - Downloading trained model artifacts from S3
    - Syncing models to local registry with atomic symlink updates

    Environment variables:
        AWS_REGION: AWS region (default: us-east-1)
    """

    def __init__(self, bucket_name: str, region: str | None = None) -> None:
        """Initialize S3 artifact manager.

        Args:
            bucket_name: S3 bucket name for artifacts
            region: AWS region (default: from AWS_REGION env var or us-east-1)
        """
        self.bucket_name = bucket_name
        self.region = region or os.getenv("AWS_REGION", "us-east-1")
        self._s3_client: S3Client | None = None

    def _ensure_client(self) -> None:
        """Lazily initialize S3 client."""
        if self._s3_client is None:
            try:
                import boto3

                self._s3_client = boto3.client("s3", region_name=self.region)
            except ImportError as exc:
                raise ArtifactSyncError(
                    "boto3 is required for S3 operations. "
                    "Install with: pip install '.[cloud]'"
                ) from exc

    def upload_training_data(
        self,
        symbol: str,
        timeframe: str,
        data_files: list[Path],
    ) -> str:
        """Upload training data files to S3.

        Creates a timestamped directory in S3 with all training data files.

        Args:
            symbol: Trading symbol (e.g., BTCUSDT)
            timeframe: Data timeframe (e.g., 1h, 4h, 1d)
            data_files: List of local data file paths to upload

        Returns:
            S3 URI to the uploaded data directory

        Raises:
            ArtifactSyncError: If upload fails
        """
        self._ensure_client()
        assert self._s3_client is not None

        timestamp = datetime.now(UTC).strftime("%Y-%m-%d_%H%M%S")
        s3_prefix = f"training-data/{symbol}/{timeframe}/{timestamp}"

        try:
            for file_path in data_files:
                if not file_path.exists():
                    logger.warning(f"Skipping non-existent file: {file_path}")
                    continue

                s3_key = f"{s3_prefix}/{file_path.name}"
                self._s3_client.upload_file(
                    str(file_path),
                    self.bucket_name,
                    s3_key,
                )
                logger.info(f"Uploaded {file_path.name} to s3://{self.bucket_name}/{s3_key}")

            s3_uri = f"s3://{self.bucket_name}/{s3_prefix}"
            logger.info(f"Training data uploaded to {s3_uri}")
            return s3_uri

        except Exception as exc:
            raise ArtifactSyncError(f"Failed to upload training data: {exc}") from exc

    def download_model_artifacts(
        self,
        s3_uri: str,
        local_dir: Path,
    ) -> Path:
        """Download model artifacts from S3 to local directory.

        Downloads all files from the S3 prefix to the local directory,
        preserving the directory structure.

        Args:
            s3_uri: S3 URI (e.g., s3://bucket/models/BTCUSDT/basic/v1/)
            local_dir: Local directory to download to

        Returns:
            Path to downloaded artifacts directory

        Raises:
            ArtifactSyncError: If download fails
        """
        self._ensure_client()
        assert self._s3_client is not None

        bucket, prefix = self._parse_s3_uri(s3_uri)

        try:
            local_dir.mkdir(parents=True, exist_ok=True)

            # List all objects in the S3 prefix
            paginator = self._s3_client.get_paginator("list_objects_v2")
            pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

            file_count = 0
            for page in pages:
                for obj in page.get("Contents", []):
                    s3_key = obj["Key"]
                    # Get relative path from prefix
                    relative_path = s3_key[len(prefix) :].lstrip("/")
                    if not relative_path:
                        continue

                    local_file = local_dir / relative_path
                    local_file.parent.mkdir(parents=True, exist_ok=True)

                    self._s3_client.download_file(bucket, s3_key, str(local_file))
                    file_count += 1

            logger.info(f"Downloaded {file_count} files from {s3_uri} to {local_dir}")
            return local_dir

        except Exception as exc:
            raise ArtifactSyncError(f"Failed to download artifacts: {exc}") from exc

    def sync_to_local_registry(
        self,
        s3_uri: str,
        local_registry: Path,
        symbol: str,
        model_type: str,
        version_id: str,
        update_latest: bool = True,
    ) -> Path:
        """Sync trained model from S3 to local model registry.

        Downloads model artifacts and updates the 'latest' symlink atomically.

        Args:
            s3_uri: S3 URI to model artifacts
            local_registry: Local model registry root (e.g., src/ml/models/)
            symbol: Trading symbol (e.g., BTCUSDT)
            model_type: Model type (e.g., basic, sentiment)
            version_id: Version identifier (e.g., 2025-12-23_14h_v1)
            update_latest: Update 'latest' symlink to point to new version

        Returns:
            Path to synced model directory

        Raises:
            ArtifactSyncError: If sync fails
        """
        # Build local path
        version_dir = local_registry / symbol.upper() / model_type / version_id
        version_dir.mkdir(parents=True, exist_ok=True)

        # Download artifacts
        self.download_model_artifacts(s3_uri, version_dir)

        # Validate artifacts
        self._validate_artifacts(version_dir)

        # Update latest symlink atomically
        if update_latest:
            self._update_latest_symlink(version_dir)

        logger.info(f"Synced model to local registry: {version_dir}")
        return version_dir

    def upload_model_version(
        self,
        local_model_dir: Path,
        symbol: str,
        model_type: str,
        version_id: str,
    ) -> str:
        """Upload trained model to S3 with versioning.

        Args:
            local_model_dir: Local directory containing model artifacts
            symbol: Trading symbol
            model_type: Model type (basic, sentiment)
            version_id: Version identifier

        Returns:
            S3 URI of uploaded model

        Raises:
            ArtifactSyncError: If upload fails
        """
        self._ensure_client()
        assert self._s3_client is not None

        s3_prefix = f"models/{symbol}/{model_type}/{version_id}"

        try:
            for file_path in local_model_dir.rglob("*"):
                if file_path.is_file():
                    relative_path = file_path.relative_to(local_model_dir)
                    s3_key = f"{s3_prefix}/{relative_path}"

                    self._s3_client.upload_file(
                        str(file_path),
                        self.bucket_name,
                        s3_key,
                    )
                    logger.debug(f"Uploaded {relative_path} to s3://{self.bucket_name}/{s3_key}")

            s3_uri = f"s3://{self.bucket_name}/{s3_prefix}"
            logger.info(f"Model uploaded to {s3_uri}")
            return s3_uri

        except Exception as exc:
            raise ArtifactSyncError(f"Failed to upload model: {exc}") from exc

    def list_model_versions(
        self,
        symbol: str,
        model_type: str,
    ) -> list[str]:
        """List available model versions in S3.

        Args:
            symbol: Trading symbol
            model_type: Model type (basic, sentiment)

        Returns:
            List of version IDs sorted by date (newest first)
        """
        self._ensure_client()
        assert self._s3_client is not None

        prefix = f"models/{symbol}/{model_type}/"

        try:
            paginator = self._s3_client.get_paginator("list_objects_v2")
            pages = paginator.paginate(Bucket=self.bucket_name, Prefix=prefix, Delimiter="/")

            versions = []
            for page in pages:
                for common_prefix in page.get("CommonPrefixes", []):
                    # Extract version ID from prefix
                    version_path = common_prefix["Prefix"]
                    version_id = version_path.rstrip("/").split("/")[-1]
                    versions.append(version_id)

            # Sort by version (assumes date-based naming: YYYY-MM-DD_HHh_vN)
            versions.sort(reverse=True)
            return versions

        except Exception as exc:
            logger.warning(f"Failed to list model versions: {exc}")
            return []

    def _parse_s3_uri(self, s3_uri: str) -> tuple[str, str]:
        """Parse S3 URI into bucket and prefix.

        Args:
            s3_uri: S3 URI (e.g., s3://bucket/prefix/path)

        Returns:
            Tuple of (bucket, prefix)

        Raises:
            ValueError: If URI is invalid or contains path traversal
        """
        if not s3_uri.startswith("s3://"):
            raise ValueError(f"Invalid S3 URI: {s3_uri}")
        path = s3_uri[5:]
        bucket, _, prefix = path.partition("/")

        # Validate bucket name
        if not bucket:
            raise ValueError(f"Invalid S3 URI: missing bucket name in {s3_uri}")

        # Check for path traversal attempts
        if ".." in prefix:
            raise ValueError(f"Invalid S3 URI: path traversal detected in {s3_uri}")

        return bucket, prefix

    def _validate_artifacts(self, model_dir: Path) -> None:
        """Validate that required model artifacts exist.

        Args:
            model_dir: Directory containing model artifacts

        Raises:
            ArtifactSyncError: If required artifacts are missing
        """
        required_files = ["metadata.json"]
        model_files = ["model.keras", "model.onnx"]

        # Check required files
        for filename in required_files:
            if not (model_dir / filename).exists():
                raise ArtifactSyncError(f"Required artifact missing: {filename}")

        # Check for at least one model file
        has_model = any((model_dir / f).exists() for f in model_files)
        if not has_model:
            raise ArtifactSyncError(f"No model file found. Expected one of: {model_files}")

        # Validate metadata JSON
        metadata_path = model_dir / "metadata.json"
        try:
            with open(metadata_path) as f:
                metadata = json.load(f)
            if "symbol" not in metadata or "training_date" not in metadata:
                raise ArtifactSyncError("Invalid metadata.json: missing required fields")
        except json.JSONDecodeError as exc:
            raise ArtifactSyncError(f"Invalid metadata.json: {exc}") from exc

    def _update_latest_symlink(self, version_dir: Path) -> None:
        """Atomically update 'latest' symlink to point to new version.

        Uses the same atomic update pattern as artifacts.py:313-337.

        Args:
            version_dir: Path to new version directory
        """
        type_dir = version_dir.parent
        latest_link = type_dir / "latest"
        temp_link = type_dir / f".latest.{version_dir.name}.tmp"

        try:
            # Clean up any stale temp symlink
            if temp_link.exists() or temp_link.is_symlink():
                temp_link.unlink()

            # Create new symlink with temporary name
            temp_link.symlink_to(version_dir.name)

            # Atomically replace old symlink (os.replace is atomic on POSIX)
            os.replace(str(temp_link), str(latest_link))

            logger.info(f"Updated 'latest' symlink to {version_dir.name}")

        except OSError as exc:
            # Clean up temp symlink on failure
            if temp_link.exists() or temp_link.is_symlink():
                try:
                    temp_link.unlink()
                except OSError:
                    pass
            logger.error(f"Failed to update 'latest' symlink: {exc}")
            raise ArtifactSyncError(f"Failed to update 'latest' symlink: {exc}") from exc
