"""Security regression test for S3 artifact download path traversal.

S3 object keys may legally contain ``..`` segments; download must skip any key
whose resolved local path escapes the target directory.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.ml.cloud.artifacts.s3_manager import S3ArtifactManager


def _make_manager(contents, downloaded):
    """Build an S3ArtifactManager with a mocked S3 client (bypassing __init__)."""
    mgr = object.__new__(S3ArtifactManager)

    def _download_file(bucket, key, dest):
        Path(dest).write_text("data")
        downloaded.append((key, dest))

    client = MagicMock()
    paginator = MagicMock()
    paginator.paginate.return_value = [{"Contents": contents}]
    client.get_paginator.return_value = paginator
    client.download_file.side_effect = _download_file
    mgr._s3_client = client
    return mgr


@pytest.mark.fast
def test_skips_keys_escaping_local_dir(tmp_path):
    local_dir = tmp_path / "out"
    downloaded: list = []
    contents = [
        {"Key": "prefix/sub/model.onnx"},  # normal -> downloaded
        {"Key": "prefix/../../escape.txt"},  # traversal -> skipped
    ]
    mgr = _make_manager(contents, downloaded)

    mgr.download_model_artifacts("s3://bucket/prefix", local_dir)

    # Normal key landed inside local_dir; traversal key was skipped.
    assert (local_dir / "sub" / "model.onnx").exists()
    assert not (tmp_path / "escape.txt").exists()
    downloaded_keys = {k for k, _ in downloaded}
    assert "prefix/sub/model.onnx" in downloaded_keys
    assert "prefix/../../escape.txt" not in downloaded_keys
