"""Behavioural tests for ``bin/run_mypy.py``, the runner behind the CI type-check job (GH #1111).

The gate is only worth having if it can fail, so the load-bearing test is
``test_fails_on_type_error_*``: a clean tree passes, a tree with one bad annotation does not.
Each test builds a throwaway tree containing ``src/``, ``cli/`` and the real ``mypy.ini`` and runs
the shipped script against it.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = REPO_ROOT / "bin" / "run_mypy.py"

CLEAN_MODULE = "def double(x: int) -> int:\n    return x * 2\n"
BROKEN_MODULE = "def double(x: int) -> int:\n    return 'two'\n"


def _make_tree(path: Path, src_module: str, cli_module: str = CLEAN_MODULE) -> Path:
    for package, body in (("src", src_module), ("cli", cli_module)):
        (path / package).mkdir()
        (path / package / "__init__.py").write_text("")
        (path / package / "mod.py").write_text(body)
    shutil.copy(REPO_ROOT / "mypy.ini", path / "mypy.ini")
    return path


def _run(tree: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(RUNNER), *args],
        cwd=tree,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_passes_on_clean_tree(tmp_path: Path) -> None:
    result = _run(_make_tree(tmp_path, CLEAN_MODULE))
    assert result.returncode == 0, result.stdout + result.stderr


def test_fails_on_type_error_in_cli_and_checks_cli_by_default(tmp_path: Path) -> None:
    result = _run(_make_tree(tmp_path, CLEAN_MODULE, cli_module=BROKEN_MODULE))
    assert result.returncode == 1
    assert "cli/mod.py" in result.stdout


def test_targets_limit_the_check_to_named_files(tmp_path: Path) -> None:
    tree = _make_tree(tmp_path, CLEAN_MODULE, cli_module=BROKEN_MODULE)
    result = _run(tree, "--targets", "src/mod.py")
    assert result.returncode == 0, result.stdout + result.stderr
