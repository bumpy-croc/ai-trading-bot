"""Tests for safe tar extraction in the SageMaker provider.

Locks in protection against path-traversal ("zip-slip") and symlink members in
downloaded model archives (CVE-2007-4559 class).
"""

from __future__ import annotations

import io
import tarfile
from pathlib import Path

import pytest

from src.ml.cloud.exceptions import ArtifactSyncError
from src.ml.cloud.providers.sagemaker import _safe_extract_tar


def _make_tar(path: Path, members: dict[str, bytes]) -> None:
    with tarfile.open(path, "w:gz") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))


@pytest.mark.fast
def test_safe_extract_allows_normal_members(tmp_path):
    archive = tmp_path / "ok.tar.gz"
    _make_tar(archive, {"model.onnx": b"abc", "sub/metadata.json": b"{}"})
    dest = tmp_path / "out"
    dest.mkdir()
    with tarfile.open(archive, "r:gz") as tar:
        _safe_extract_tar(tar, dest)
    assert (dest / "model.onnx").read_bytes() == b"abc"
    assert (dest / "sub" / "metadata.json").exists()


@pytest.mark.fast
def test_safe_extract_rejects_path_traversal(tmp_path):
    archive = tmp_path / "evil.tar.gz"
    _make_tar(archive, {"../../escape.txt": b"pwned"})
    dest = tmp_path / "out"
    dest.mkdir()
    with tarfile.open(archive, "r:gz") as tar, pytest.raises(ArtifactSyncError):
        _safe_extract_tar(tar, dest)
    assert not (tmp_path / "escape.txt").exists()


@pytest.mark.fast
def test_safe_extract_rejects_escaping_symlink(tmp_path):
    archive = tmp_path / "evil-link.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        info = tarfile.TarInfo(name="link")
        info.type = tarfile.SYMTYPE
        info.linkname = "../../../etc/passwd"
        tar.addfile(info)
    dest = tmp_path / "out"
    dest.mkdir()
    with tarfile.open(archive, "r:gz") as tar, pytest.raises(ArtifactSyncError):
        _safe_extract_tar(tar, dest)
