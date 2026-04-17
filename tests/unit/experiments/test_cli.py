"""Tests for the experiment CLI helpers (safety + resolution)."""

from __future__ import annotations

from pathlib import Path

import pytest

from cli.commands.experiment import _safe_artifacts_dir

pytestmark = pytest.mark.fast


def test_safe_artifacts_dir_rejects_traversal(tmp_path: Path) -> None:
    root = tmp_path
    with pytest.raises(ValueError, match="outside"):
        _safe_artifacts_dir(root, "..", "run1")


def test_safe_artifacts_dir_rejects_absolute_suite_id(tmp_path: Path) -> None:
    root = tmp_path
    with pytest.raises(ValueError):
        _safe_artifacts_dir(root, "/etc/passwd", "run1")


def test_safe_artifacts_dir_accepts_legal_paths(tmp_path: Path) -> None:
    root = tmp_path
    path = _safe_artifacts_dir(root, "suite_x", "run1")
    assert str(path).startswith(str(tmp_path.resolve()))
