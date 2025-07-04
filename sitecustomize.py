import sys
from pathlib import Path

# Ensure that the `src` directory is on the Python path so all modules inside it
# can be imported from anywhere in the project (tests, scripts, notebooks, etc.).
# This approach avoids the need to modify existing import statements after the
# project was reorganised into a `src/` layout.
_project_root = Path(__file__).resolve().parent
_src_path = _project_root / "src"

if _src_path.exists():
    # Insert at the beginning so it has priority over any site-packages that may
    # contain similarly named modules.
    sys.path.insert(0, str(_src_path))

# ---------------------------------------------------------------------------
# Removed pandas fallback – real pandas will be installed in the
# production/test environment (Python 3.11 container).
# ---------------------------------------------------------------------------