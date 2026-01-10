"""Artifact management for cloud training.

Provides utilities for uploading training data to S3 and
syncing trained models back to the local registry.
"""

from src.ml.cloud.artifacts.s3_manager import S3ArtifactManager

__all__ = ["S3ArtifactManager"]
