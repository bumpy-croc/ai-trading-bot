"""
CLI command for running the test suite via tests/run_tests.py
"""

import argparse
import subprocess
import sys
from pathlib import Path


def _handle_test(ns: argparse.Namespace) -> int:
    """Run tests using the tests/run_tests.py script."""
    project_root = Path(__file__).parent.parent.parent
    test_runner = project_root / "tests" / "run_tests.py"

    if not test_runner.exists():
        print(f"Error: Test runner not found at {test_runner}", file=sys.stderr)
        return 1

    # Build command
    cmd = [sys.executable, str(test_runner)]

    # Add the test category if provided
    if ns.category:
        cmd.append(ns.category)

    # Add other flags first (before pytest args)
    if ns.file:
        cmd.extend(["--file", ns.file])
    if ns.markers:
        cmd.extend(["--markers", ns.markers])
    if ns.coverage:
        cmd.append("--coverage")
    if ns.verbose:
        cmd.append("--verbose")
    if ns.quiet:
        cmd.append("--quiet")

    # Add pytest args last (REMAINDER consumes everything after it)
    if ns.pytest_args:
        cmd.append("--pytest-args")
        cmd.extend(ns.pytest_args)

    # Run the test runner
    try:
        result = subprocess.run(cmd, cwd=project_root, check=False)
        return result.returncode
    except KeyboardInterrupt:
        print("\nTest run interrupted by user", file=sys.stderr)
        return 1
    except FileNotFoundError as e:
        print(f"Error: Test runner not found: {e}", file=sys.stderr)
        return 1
    except PermissionError as e:
        print(f"Error: Permission denied running tests: {e}", file=sys.stderr)
        return 1


def register(subparsers: argparse._SubParsersAction) -> None:
    """Register the test command."""
    p = subparsers.add_parser(
        "test",
        help="Run test suite",
        description="Run the test suite using the project's test runner",
    )

    # Test category (optional positional argument)
    p.add_argument(
        "category",
        nargs="?",
        choices=[
            "smoke",
            "critical",
            "unit",
            "integration",
            "database",
            "coverage",
            "all",
            "validate",
            "fast",
            "slow",
            "grouped",
            "benchmark",
        ],
        help="Test category to run (default: interactive mode if not specified)",
    )

    # Optional arguments
    p.add_argument("--file", "-f", help="Run specific test file")
    p.add_argument("--markers", "-m", help="Run tests with specific pytest markers")
    p.add_argument("--coverage", "-c", action="store_true", help="Run with coverage analysis")
    p.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    p.add_argument("--quiet", "-q", action="store_true", help="Quiet output")

    # Pass-through pytest arguments
    p.add_argument(
        "--pytest-args",
        nargs=argparse.REMAINDER,
        help="Additional arguments to pass to pytest",
    )

    p.set_defaults(func=_handle_test)
