import sys
import os
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
    src_path_str = str(_src_path)
    if src_path_str not in sys.path:
        sys.path.insert(0, src_path_str)

# Also add to PYTHONPATH environment variable for subprocess calls
if 'PYTHONPATH' in os.environ:
    if str(_src_path) not in os.environ['PYTHONPATH']:
        os.environ['PYTHONPATH'] = str(_src_path) + ':' + os.environ['PYTHONPATH']
else:
    os.environ['PYTHONPATH'] = str(_src_path)

# ---------------------------------------------------------------------------
# Removed pandas fallback – real pandas will be installed in the
# production/test environment (Python 3.11 container).
# ---------------------------------------------------------------------------