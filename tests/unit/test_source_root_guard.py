"""Regression tests for the worktree import-isolation defect (GH #1070).

The defect: ``pip install -e .`` writes a ``sys.meta_path`` finder whose MAPPING hardcodes the
install-time checkout's absolute path. Because that finder sits *after* ``PathFinder``, it only
loses when a ``sys.path`` entry already contains ``src``/``cli`` — true for ``python -c`` and
``python <repo-root>/x.py``, false for console scripts and ``python experiments/x.py``. Those
invocations silently imported the install-time checkout from inside any git worktree.

These tests build synthetic checkouts under ``tmp_path`` and drive real subprocesses so the
actual import machinery is exercised, not a mock of it.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from src.utils.source_root import (
    SourceRootMismatchError,
    find_repo_root,
    source_root,
    verify_source_root,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SHIM_SOURCE = REPO_ROOT / "tools" / "atb_worktree_shim.py"

pytestmark = pytest.mark.fast


def _load_shim_module():
    """Import the shim straight from ``tools/`` without installing it into site-packages."""
    spec = importlib.util.spec_from_file_location("_atb_worktree_shim_under_test", SHIM_SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_fake_checkout(root: Path, marker: str, *, guarded: bool = False) -> Path:
    """Create a minimal but genuine ai-trading-bot checkout that reports ``marker``.

    With ``guarded=True`` the checkout carries a real copy of ``src/utils/source_root.py`` and a
    ``src/__init__.py`` that invokes it, mirroring this repo's own layout so a subprocess
    exercises the production guard rather than a re-implementation of it.
    """
    root.mkdir(parents=True, exist_ok=True)
    (root / "pyproject.toml").write_text('[project]\nname = "ai-trading-bot"\n', encoding="utf-8")
    for pkg in ("src", "cli"):
        pkg_dir = root / pkg
        pkg_dir.mkdir(exist_ok=True)
        (pkg_dir / "__init__.py").write_text(f'CHECKOUT = "{marker}"\n', encoding="utf-8")
    (root / "experiments").mkdir(exist_ok=True)

    if guarded:
        utils = root / "src" / "utils"
        utils.mkdir(exist_ok=True)
        (utils / "__init__.py").touch()
        (utils / "source_root.py").write_text(
            (REPO_ROOT / "src" / "utils" / "source_root.py").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        (root / "src" / "__init__.py").write_text(
            f'CHECKOUT = "{marker}"\n'
            "from src.utils.source_root import verify_source_root\n"
            "verify_source_root()\n",
            encoding="utf-8",
        )
    return root


def _run_python(
    code: str, *, cwd: Path, env_extra: dict[str, str] | None = None, script: Path | None = None
):
    env = {**os.environ, "PYTHONPATH": ""}
    env.pop("ATB_DISABLE_WORKTREE_SHIM", None)
    env.update(env_extra or {})
    if script is not None:
        script.write_text(code, encoding="utf-8")
        argv = [sys.executable, str(script)]
    else:
        argv = [sys.executable, "-c", code]
    return subprocess.run(argv, cwd=cwd, env=env, capture_output=True, text=True, timeout=120)


# Emulates the console-script / `python experiments/x.py` shape: an editable-install finder
# pointing at checkout A is on sys.meta_path, and cwd is nowhere on sys.path.
_PROBE_PREAMBLE = textwrap.dedent(
    """
    import sys
    from importlib.machinery import PathFinder

    STALE_ROOT = {stale!r}

    # The test venv already has this repo installed editable; drop its finder (and the shim,
    # if a developer has it installed) so the synthetic checkouts below are the only sources.
    def _origin(finder):
        # pip appends the finder CLASS, not an instance, so check both forms.
        return (getattr(finder, "__module__", "") or type(finder).__module__).lower()


    sys.meta_path[:] = [
        f for f in sys.meta_path
        if "editable" not in _origin(f) and "atb_worktree_shim" not in _origin(f)
    ]


    class StaleEditableFinder:
        '''Stand-in for __editable___*_finder._EditableFinder.'''

        def find_spec(self, fullname, path=None, target=None):
            if fullname in ("src", "cli"):
                return PathFinder.find_spec(fullname, path=[STALE_ROOT])
            return None


    # pip appends its finder AFTER PathFinder, exactly as the real editable install does.
    sys.meta_path.append(StaleEditableFinder())
    """
)

# The shim normally arrives via a .pth in site-packages; here it is loaded straight from
# tools/ and the loader path is removed again so it cannot influence resolution itself.
_PROBE_SHIM_SETUP = textwrap.dedent(
    """
    sys.path.insert(0, {tools_dir!r})
    import atb_worktree_shim
    sys.path.remove({tools_dir!r})
    atb_worktree_shim.install_quietly()
    """
)

_PROBE_BODY = textwrap.dedent(
    """
    import src
    print(src.CHECKOUT)
    """
)


def _probe_code(stale_root: Path, *, with_shim: bool) -> str:
    parts = [_PROBE_PREAMBLE.format(stale=str(stale_root))]
    if with_shim:
        parts.append(_PROBE_SHIM_SETUP.format(tools_dir=str(SHIM_SOURCE.parent)))
    parts.append(_PROBE_BODY)
    return "".join(parts)


class TestDefectReproduction:
    def test_without_shim_a_worktree_run_imports_the_stale_checkout(self, tmp_path: Path) -> None:
        """The bug, pinned: same cwd, wrong code. If this ever passes, the premise changed."""
        primary = _make_fake_checkout(tmp_path / "primary", "primary")
        worktree = _make_fake_checkout(tmp_path / "worktree", "worktree")

        # Script lives in experiments/ so sys.path[0] is NOT the repo root - the real shape.
        result = _run_python(
            _probe_code(primary, with_shim=False),
            cwd=worktree,
            script=worktree / "experiments" / "probe.py",
        )

        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "primary"

    def test_shim_makes_a_worktree_run_import_that_worktree(self, tmp_path: Path) -> None:
        primary = _make_fake_checkout(tmp_path / "primary", "primary")
        worktree = _make_fake_checkout(tmp_path / "worktree", "worktree")

        result = _run_python(
            _probe_code(primary, with_shim=True),
            cwd=worktree,
            script=worktree / "experiments" / "probe.py",
        )

        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "worktree"

    def test_shim_leaves_the_primary_checkout_working_normally(self, tmp_path: Path) -> None:
        """The single-checkout case must be a semantic no-op."""
        primary = _make_fake_checkout(tmp_path / "primary", "primary")

        result = _run_python(
            _probe_code(primary, with_shim=True),
            cwd=primary,
            script=primary / "experiments" / "probe.py",
        )

        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "primary"

    def test_shim_is_a_no_op_outside_any_checkout(self, tmp_path: Path) -> None:
        primary = _make_fake_checkout(tmp_path / "primary", "primary")
        elsewhere = tmp_path / "elsewhere"
        (elsewhere / "experiments").mkdir(parents=True)

        result = _run_python(
            _probe_code(primary, with_shim=True),
            cwd=elsewhere,
            script=elsewhere / "experiments" / "probe.py",
        )

        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "primary"

    def test_shim_respects_the_disable_env_var(self, tmp_path: Path) -> None:
        primary = _make_fake_checkout(tmp_path / "primary", "primary")
        worktree = _make_fake_checkout(tmp_path / "worktree", "worktree")

        result = _run_python(
            _probe_code(primary, with_shim=True),
            cwd=worktree,
            script=worktree / "experiments" / "probe.py",
            env_extra={"ATB_DISABLE_WORKTREE_SHIM": "1"},
        )

        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "primary"


class TestRootDetection:
    def test_finds_root_from_a_nested_directory(self, tmp_path: Path) -> None:
        root = _make_fake_checkout(tmp_path / "repo", "repo")
        nested = root / "src" / "deeply" / "nested"
        nested.mkdir(parents=True)

        assert find_repo_root(nested) == root.resolve()

    def test_returns_none_outside_a_checkout(self, tmp_path: Path) -> None:
        stray = tmp_path / "stray"
        stray.mkdir()

        assert find_repo_root(stray) is None

    def test_ignores_a_pyproject_belonging_to_another_project(self, tmp_path: Path) -> None:
        impostor = tmp_path / "impostor"
        (impostor / "src").mkdir(parents=True)
        (impostor / "cli").mkdir()
        (impostor / "src" / "__init__.py").touch()
        (impostor / "cli" / "__init__.py").touch()
        (impostor / "pyproject.toml").write_text('[project]\nname = "something-else"\n')

        assert find_repo_root(impostor) is None

    def test_ignores_a_marker_without_the_packages(self, tmp_path: Path) -> None:
        bare = tmp_path / "bare"
        bare.mkdir()
        (bare / "pyproject.toml").write_text('[project]\nname = "ai-trading-bot"\n')

        assert find_repo_root(bare) is None

    def test_guard_and_shim_agree_on_every_root(self, tmp_path: Path) -> None:
        """Pins the deliberate duplication: the shim cannot import src, so the logic is
        implemented twice. Any drift between the two is a bug."""
        shim = _load_shim_module()
        checkout = _make_fake_checkout(tmp_path / "repo", "repo")
        cases = [checkout, checkout / "experiments", tmp_path, Path(__file__).parent]

        for case in cases:
            assert shim.find_repo_root(case) == find_repo_root(case), case


class TestVerifySourceRoot:
    def test_passes_when_invoked_from_the_real_repo_root(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(REPO_ROOT)

        assert verify_source_root() == source_root()

    def test_passes_when_cwd_is_outside_any_checkout(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)

        assert verify_source_root() == source_root()

    def test_raises_on_mismatch_with_a_copy_pasteable_remedy(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        other = _make_fake_checkout(tmp_path / "other", "other")
        monkeypatch.chdir(other)

        with pytest.raises(SourceRootMismatchError) as excinfo:
            verify_source_root()

        message = str(excinfo.value)
        assert str(source_root()) in message
        assert str(other.resolve()) in message
        assert "python tools/install_worktree_shim.py" in message
        assert 'PYTHONPATH="$(pwd)"' in message

    def test_ignores_a_non_editable_site_packages_install(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Production containers `pip install .` into site-packages and run from /app; that is
        not a mismatch, and refusing to boot there would be far worse than the bug."""
        other = _make_fake_checkout(tmp_path / "app", "app")
        monkeypatch.chdir(other)
        fake_site_packages = tmp_path / "site-packages"
        monkeypatch.setattr(
            "src.utils.source_root.source_root", lambda: fake_site_packages, raising=True
        )

        assert verify_source_root() == fake_site_packages

    def test_escape_hatch_downgrades_to_a_stderr_warning(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        other = _make_fake_checkout(tmp_path / "other", "other")
        monkeypatch.chdir(other)
        monkeypatch.setenv("ATB_ALLOW_SOURCE_ROOT_MISMATCH", "1")

        assert verify_source_root() == source_root()
        assert "ATB SOURCE ROOT MISMATCH" in capsys.readouterr().err


class TestGuardFiresOnBareImport:
    """`src/__init__.py` must make the mismatch loud for entry points nobody remembers to edit."""

    def test_bare_python_script_aborts_instead_of_importing_the_stale_checkout(
        self, tmp_path: Path
    ) -> None:
        """`python experiments/x.py` — the shape that produced the +114.69% / -28.29% pair."""
        primary = _make_fake_checkout(tmp_path / "primary", "primary", guarded=True)
        worktree = _make_fake_checkout(tmp_path / "worktree", "worktree", guarded=True)

        result = _run_python(
            _probe_code(primary, with_shim=False),
            cwd=worktree,
            script=worktree / "experiments" / "probe.py",
        )

        assert result.returncode != 0
        assert "ATB SOURCE ROOT MISMATCH" in result.stderr
        assert "SourceRootMismatchError" in result.stderr
        assert result.stdout.strip() == ""

    def test_shim_installed_means_the_guard_never_trips(self, tmp_path: Path) -> None:
        """Belt and braces together: the shim corrects the import, the guard then agrees."""
        primary = _make_fake_checkout(tmp_path / "primary", "primary", guarded=True)
        worktree = _make_fake_checkout(tmp_path / "worktree", "worktree", guarded=True)

        result = _run_python(
            _probe_code(primary, with_shim=True),
            cwd=worktree,
            script=worktree / "experiments" / "probe.py",
        )

        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "worktree"

    def test_single_checkout_import_is_unaffected(self, tmp_path: Path) -> None:
        """No shim, no worktree, no mismatch: the guard must stay entirely out of the way."""
        primary = _make_fake_checkout(tmp_path / "primary", "primary", guarded=True)

        result = _run_python(
            _probe_code(primary, with_shim=False),
            cwd=primary,
            script=primary / "experiments" / "probe.py",
        )

        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "primary"

    def test_this_repo_imports_src_cleanly_from_its_own_root(self) -> None:
        """End-to-end on the real checkout, with the shim explicitly disabled."""
        result = subprocess.run(
            [sys.executable, "-c", "import src; print(src.__file__)"],
            cwd=REPO_ROOT,
            env={**os.environ, "ATB_DISABLE_WORKTREE_SHIM": "1", "PYTHONPATH": str(REPO_ROOT)},
            capture_output=True,
            text=True,
            timeout=120,
        )

        assert result.returncode == 0, result.stderr
        assert (
            Path(result.stdout.strip()).resolve() == (REPO_ROOT / "src" / "__init__.py").resolve()
        )


class TestAtbEntryPointIsGuarded:
    def test_cli_main_guards_before_importing_src_modules(self) -> None:
        """The guard must precede every other src import, or it guards nothing."""
        lines = (REPO_ROOT / "cli" / "__main__.py").read_text(encoding="utf-8").splitlines()
        guard_line = next(i for i, ln in enumerate(lines) if ln.startswith("    import src"))
        first_other = min(
            i for i, ln in enumerate(lines) if ln.startswith(("from src.", "import src."))
        )

        assert guard_line < first_other
