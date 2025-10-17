import os
import sys
from pathlib import Path
from shutil import which

# Ensure we always execute under Python 3.11+ even if the system `python3` points
# to an older interpreter (e.g., the macOS default 3.9). This repo depends on
# 3.11 features like PEP 604 unions, so re-exec into a 3.11 binary when required.
if sys.version_info < (3, 11):
    candidates: list[str] = []

    explicit_env = os.environ.get("PYTHON311")
    if explicit_env:
        candidates.append(explicit_env)

    which_candidate = which("python3.11")
    if which_candidate:
        candidates.append(which_candidate)

    default_homebrew = Path("/opt/homebrew/opt/python@3.11/bin/python3.11")
    candidates.append(str(default_homebrew))

    for candidate in candidates:
        if candidate and Path(candidate).exists():
            os.execv(candidate, [candidate, *sys.argv])

    raise RuntimeError(
        "Python 3.11 interpreter is required but could not be located. "
        "Set PYTHON311 to the desired executable or install python@3.11."
    )

# If running under pytest, avoid altering sys.path/PYTHONPATH to keep imports predictable
if os.environ.get("PYTEST_CURRENT_TEST"):
    # Do not modify path during tests; rely on pytest.ini pythonpath=src
    pass
else:
    # Ensure both the project root and the `src` directory are on the Python path so all modules
    # can be imported from anywhere in the project (tests, scripts, notebooks, etc.).
    _project_root = Path(__file__).resolve().parent
    _src_path = _project_root / "src"

    # Insert project root first to allow `import src` to resolve properly
    project_root_str = str(_project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)

    if _src_path.exists():
        # Insert `src` path after project root so it has priority over any site-packages that may
        # contain similarly named modules, while keeping root available for `import src`.
        src_path_str = str(_src_path)
        if src_path_str not in sys.path:
            sys.path.insert(1, src_path_str)

    # Also add to PYTHONPATH environment variable for subprocess calls
    existing_pythonpath = os.environ.get("PYTHONPATH", "")
    paths_to_add = [project_root_str, str(_src_path)]

    # Build a colon-separated PYTHONPATH ensuring no duplicates and preserving order
    new_pythonpath_parts = []
    for p in paths_to_add:
        if p and p not in existing_pythonpath and p not in new_pythonpath_parts:
            new_pythonpath_parts.append(p)
    if existing_pythonpath:
        new_pythonpath_parts.append(existing_pythonpath)

    os.environ["PYTHONPATH"] = ":".join(new_pythonpath_parts)

# ---------------------------------------------------------------------------
# Removed pandas fallback – real pandas will be installed in the
# production/test environment (Python 3.11 container).
# ---------------------------------------------------------------------------
