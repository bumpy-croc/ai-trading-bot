import os
import sys
from pathlib import Path

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
