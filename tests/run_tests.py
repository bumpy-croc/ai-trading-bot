#!/usr/bin/env python3
"""Convenience wrapper to run the test suite.

Examples
--------
# Run full suite
python tests/run_tests.py

# Run smoke subset
python tests/run_tests.py smoke
"""

import sys
import subprocess
from pathlib import Path

import pytest  # noqa: add to requirements if missing


def main():
    """Proxy CLI args directly to pytest."""
    args = sys.argv[1:]
    # Default to tests directory at project root
    tests_dir = Path(__file__).parent
    if not args or args[0] in {"smoke", "all", "critical"}:
        # Legacy shortcuts: map keywords to markers
        keyword = args[0] if args else "all"
        marker_map = {
            "smoke": "smoke",
            "critical": "critical",
            "all": ""
        }
        marker = marker_map.get(keyword, "")
        cmd = ["pytest", str(tests_dir)]
        if marker:
            cmd.extend(["-m", marker])
    else:
        # Pass custom args through
        cmd = ["pytest"] + args

    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()