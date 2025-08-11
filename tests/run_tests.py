#!/usr/bin/env python3
"""
Trading Bot Test Runner

A simple script to run various test suites for the crypto trading bot.
This provides quick access to common testing scenarios.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

# Ensure we can import from the project
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))
sys.path.insert(0, str(project_root))

# Set PYTHONPATH for subprocess calls
if "PYTHONPATH" in os.environ:
    os.environ["PYTHONPATH"] = str(src_path) + ":" + os.environ["PYTHONPATH"]
else:
    os.environ["PYTHONPATH"] = str(src_path)


def get_worker_count():
    """Get appropriate worker count based on environment"""
    # Check if running in CI environment
    ci_indicators = [
        "CI",
        "GITHUB_ACTIONS",
        "TRAVIS",
        "CIRCLECI",
        "JENKINS",
        "GITLAB_CI",
        "BITBUCKET_BUILD_NUMBER",
        "BUILDKITE",
    ]

    is_ci = any(os.getenv(indicator) for indicator in ci_indicators)

    if is_ci:
        # Use limited workers in CI to prevent resource exhaustion
        # CI environments often have limited resources and can be slower
        return "2"
    else:
        # Use 4 workers locally for good performance while preventing CPU overload
        return "4"


class Colors:
    """ANSI color codes for terminal output"""

    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def print_header(text):
    """Print colored header"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text.center(60)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}\n")


def print_subheader(text):
    """Print a formatted subheader"""
    print(f"\n{Colors.OKBLUE}{Colors.BOLD}{text}{Colors.ENDC}")
    print(f"{Colors.OKBLUE}{'-' * len(text)}{Colors.ENDC}")


def print_success(text):
    """Print success message"""
    print(f"{Colors.OKGREEN}✅ {text}{Colors.ENDC}")


def print_warning(text):
    """Print warning message"""
    print(f"{Colors.WARNING}⚠️  {text}{Colors.ENDC}")


def print_error(text):
    """Print error message"""
    print(f"{Colors.FAIL}❌ {text}{Colors.ENDC}")


def print_progress(text):
    """Print progress message"""
    print(f"{Colors.OKCYAN}🔄 {text}{Colors.ENDC}")


def print_progress_bar(current, total, width=50):
    """Print a simple progress bar"""
    progress = int(width * current / total) if total > 0 else 0
    bar = "█" * progress + "░" * (width - progress)
    percentage = int(100 * current / total) if total > 0 else 0
    print(
        f"\r{Colors.OKCYAN}[{bar}] {percentage}% ({current}/{total}){Colors.ENDC}",
        end="",
        flush=True,
    )


def run_command(cmd, description, show_progress=True):
    """Run a command and return success status"""
    print(f"{Colors.OKBLUE}Running: {description}{Colors.ENDC}")
    print(f"DEBUG: Command execution started at {time.strftime('%H:%M:%S')}")

    if show_progress:
        print(f"Command: {' '.join(cmd)}")
        # Show environment detection
        worker_count = get_worker_count()
        ci_indicators = [
            "CI",
            "GITHUB_ACTIONS",
            "TRAVIS",
            "CIRCLECI",
            "JENKINS",
            "GITLAB_CI",
            "BITBUCKET_BUILD_NUMBER",
            "BUILDKITE",
        ]
        is_ci = any(os.getenv(indicator) for indicator in ci_indicators)
        env_type = "CI" if is_ci else "Local"
        print(
            f"{Colors.OKCYAN}🖥️  Environment: {env_type} (using {worker_count} workers){Colors.ENDC}"
        )
        print_progress("Starting test execution...")
        print(
            f"{Colors.OKCYAN}🔄 Tests are running - you should see live output below...{Colors.ENDC}"
        )

    start_time = time.time()

    try:
        print(f"DEBUG: About to start subprocess at {time.strftime('%H:%M:%S')}")

        if show_progress:
            # For Cursor's embedded terminal, use direct output
            # This allows real-time display in the embedded terminal
            process = subprocess.Popen(
                cmd,
                stdout=None,  # Use current terminal
                stderr=None,  # Use current terminal
                text=True,
            )

            print(
                f"DEBUG: Subprocess started with PID {process.pid} at {time.strftime('%H:%M:%S')}"
            )
            print("DEBUG: Waiting for subprocess to complete...")

            return_code = process.wait()

            print(
                f"DEBUG: Subprocess completed with return code {return_code} at {time.strftime('%H:%M:%S')}"
            )
        else:
            # Fallback to captured output for non-progress commands
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            return_code = result.returncode

        duration = time.time() - start_time

        if return_code == 0:
            print_success(f"{description} completed successfully in {duration:.2f} seconds")
            return True
        else:
            print_error(f"{description} failed after {duration:.2f} seconds")
            return False
    except Exception as e:
        duration = time.time() - start_time
        print_error(f"Failed to run {description} after {duration:.2f} seconds: {e}")
        print(f"DEBUG: Exception occurred at {time.strftime('%H:%M:%S')}")
        return False


def check_dependencies():
    """Check if required dependencies are installed"""
    print_header("Checking Dependencies")

    required_packages = ["pytest", "pandas", "numpy"]
    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
            print_success(f"{package} is installed")
        except ImportError:
            missing_packages.append(package)
            print_error(f"{package} is NOT installed")

    if missing_packages:
        print_warning(f"Please install missing packages: pip install {' '.join(missing_packages)}")
        return False

    return True


def run_critical_tests():
    """Run critical tests (live trading + risk management)"""
    print_header("Running Critical Tests")

    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-m",
        "(live_trading or risk_management) and not slow",
        "-v",
        "--tb=short",
        "-n",
        get_worker_count(),
        "--dist=loadgroup",  # Dynamic worker count based on environment
    ]

    return run_command(cmd, "Critical Tests (Live Trading + Risk Management)")


def run_unit_tests():
    """Run all unit tests"""
    print_header("Running Unit Tests")

    # Check if running in CI environment
    ci_indicators = [
        "CI",
        "GITHUB_ACTIONS",
        "TRAVIS",
        "CIRCLECI",
        "JENKINS",
        "GITLAB_CI",
        "BITBUCKET_BUILD_NUMBER",
        "BUILDKITE",
    ]
    is_ci = any(os.getenv(indicator) for indicator in ci_indicators)

    print(f"DEBUG: CI environment detected: {is_ci}")
    print(f"DEBUG: Worker count: {get_worker_count()}")
    print(f"DEBUG: Current working directory: {os.getcwd()}")
    print(f"DEBUG: Python executable: {sys.executable}")
    print(f"DEBUG: Python version: {sys.version}")

    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/",
        "-v",
        "--tb=short",
        "-n",
        get_worker_count(),
        "--dist=loadgroup",  # Dynamic worker count based on environment
        "-m",
        "not integration and not slow",
        "--color=yes",  # Enable colored output for better visibility in Cursor
        "--timeout=300",  # 5 minute timeout per test to prevent hanging
    ]

    # In CI, skip the heaviest tests to prevent timeouts
    if is_ci:
        cmd.extend(
            [
                "-k",
                "not test_very_large_dataset and not test_ml_basic_backtest_2024_smoke and not test_position_sizing and not test_dynamic_stop_loss and not test_market_regime_detection and not test_volatility_calculations and not test_entry_conditions_crisis and not test_ml_predictions",
            ]
        )
        print_warning("CI environment detected - skipping heaviest tests to prevent timeouts")

    print(f"DEBUG: Final command: {' '.join(cmd)}")
    print(f"DEBUG: About to execute command at {time.strftime('%H:%M:%S')}")

    return run_command(cmd, "Unit Tests")


def run_integration_tests():
    """Run integration tests"""
    print_header("Running Integration Tests")

    # Set environment variable to enable integration test mode (PostgreSQL containers)
    original_env = os.environ.get("ENABLE_INTEGRATION_TESTS", "0")
    os.environ["ENABLE_INTEGRATION_TESTS"] = "1"

    try:
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            "-m",
            "integration",
            "-v",
            "--tb=short",
            # No parallelization for integration tests - they need sequential DB access
        ]

        return run_command(cmd, "Integration Tests")
    finally:
        # Restore original environment
        os.environ["ENABLE_INTEGRATION_TESTS"] = original_env


def run_coverage_analysis():
    """Run tests with coverage analysis"""
    print_header("Running Coverage Analysis")

    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "--cov=ai-trading-bot",
        "--cov-report=term-missing",
        "--cov-report=html",
        "-n",
        get_worker_count(),
        "--dist=loadgroup",  # Dynamic worker count based on environment
        "tests/",
    ]

    success = run_command(cmd, "Coverage Analysis")

    if success:
        print_success("Coverage report generated in htmlcov/index.html")

    return success


def run_specific_test_file(test_file):
    """Run a specific test file"""
    print_header(f"Running {test_file}")

    # Handle both with and without tests/ prefix
    if not test_file.startswith("tests/"):
        test_file = f"tests/{test_file}"

    cmd = [
        sys.executable,
        "-m",
        "pytest",
        test_file,
        "-v",
        "--tb=short",
        "-n",
        get_worker_count(),
        "--dist=loadgroup",  # Dynamic worker count based on environment
    ]

    return run_command(cmd, f"Tests in {test_file}")


def run_quick_smoke_test():
    """Run a quick smoke test to verify basic functionality"""
    print_header("Running Quick Smoke Test")

    # Test basic imports
    try:
        from risk.risk_manager import RiskManager
        from strategies.ml_adaptive import MlAdaptive

        print_success("All core modules import successfully")

        # Test basic functionality
        _strategy = MlAdaptive()
        _risk_manager = RiskManager()
        print_success("Core objects can be instantiated")

        return True
    except Exception as e:
        print_error(f"Smoke test failed: {e}")
        return False


def validate_test_environment():
    """Validate the test environment"""
    print_header("Validating Test Environment")

    # Check if tests directory exists
    if not Path("tests").exists():
        print_error("Tests directory not found")
        return False

    # Check pytest.ini or setup.cfg
    has_pytest_config = any(
        [Path("pytest.ini").exists(), Path("setup.cfg").exists(), Path("pyproject.toml").exists()]
    )

    if not has_pytest_config:
        print_warning(
            "No pytest configuration file found (pytest.ini, setup.cfg, or pyproject.toml)"
        )

    print_success("Test environment validation passed")
    return True


def run_database_tests():
    """Run database tests only"""
    print_header("Running Database Tests")

    cmd = [sys.executable, "-m", "pytest", "tests/test_database.py", "-v", "--tb=short"]

    return run_command(cmd, "Database Tests")


def run_fast_tests():
    """Run fast tests only (< 1 second each)"""
    print_header("Running Fast Tests")

    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-m",
        "(fast or mock_only) and not slow",
        "-v",
        "--tb=short",
        "-n",
        get_worker_count(),
        "--dist=loadgroup",  # Dynamic worker count based on environment
    ]

    return run_command(cmd, "Fast Tests")


def run_slow_tests():
    """Run slow tests only"""
    print_header("Running Slow Tests")

    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-m",
        "slow or computation",
        "-v",
        "--tb=short",
        # No parallelization for slow tests to avoid resource conflicts
    ]

    return run_command(cmd, "Slow Tests")


def run_grouped_tests():
    """Run tests in optimized groups for better parallelization"""
    print_header("Running Tests in Optimized Groups")

    # Group 1: Fast unit tests (parallel)
    print(f"{Colors.OKBLUE}Group 1: Fast Unit Tests{Colors.ENDC}")
    fast_cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/test_indicators.py",
        "tests/test_returns_calculations.py",
        "tests/test_smoke.py",
        "tests/test_coinbase_provider.py",
        "tests/test_dashboard_balance.py",
        "-v",
        "--tb=short",
        "-n",
        get_worker_count(),
        "--dist=loadgroup",  # Dynamic worker count based on environment
    ]

    fast_success = run_command(fast_cmd, "Fast Unit Tests Group")

    # Group 2: Medium unit tests (parallel)
    print(f"{Colors.OKBLUE}Group 2: Medium Unit Tests{Colors.ENDC}")
    medium_cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/test_strategies.py",
        "tests/test_risk_management.py",
        "tests/test_performance.py",
        "tests/test_config_system.py",
        "-v",
        "--tb=short",
        "-n",
        get_worker_count(),
        "--dist=loadgroup",  # Dynamic worker count based on environment
    ]

    medium_success = run_command(medium_cmd, "Medium Unit Tests Group")

    # Group 3: Computation-heavy tests (limited parallelization)
    print(f"{Colors.OKBLUE}Group 3: Computation-Heavy Tests{Colors.ENDC}")
    compute_cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/test_backtesting.py",
        "tests/test_ml_adaptive.py",
        "tests/test_monitoring.py",
        "-v",
        "--tb=short",
        "-n",
        "2",  # Limited parallelization for heavy tests
    ]

    compute_success = run_command(compute_cmd, "Computation-Heavy Tests Group")

    return fast_success and medium_success and compute_success


def run_performance_benchmark():
    """Run performance benchmark of test suite"""
    print_header("Running Performance Benchmark")

    cmd = [sys.executable, "tests/performance_benchmark.py"]

    return run_command(cmd, "Performance Benchmark")


# Available commands
AVAILABLE_COMMANDS = [
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
]


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Trading Bot Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_tests.py smoke                    # Quick smoke test
  python run_tests.py unit                     # Run unit tests
  python run_tests.py database                 # Run database tests only
  python run_tests.py critical                 # Run critical tests
  python run_tests.py all                      # Run all tests
  python run_tests.py test_strategies.py       # Run specific test file
  python run_tests.py --file test_data_providers.py  # Run specific file
  python run_tests.py --markers "not integration"    # Run tests with specific markers
  python run_tests.py --coverage               # Run with coverage
        """,
    )

    # Main test command
    parser.add_argument(
        "command", nargs="?", choices=AVAILABLE_COMMANDS, help="Test command to run"
    )

    # Alternative ways to specify tests
    parser.add_argument("--file", "-f", help="Run specific test file (e.g., test_strategies.py)")

    parser.add_argument(
        "--markers", "-m", help='Run tests with specific pytest markers (e.g., "not integration")'
    )

    parser.add_argument(
        "--coverage", "-c", action="store_true", help="Run tests with coverage analysis"
    )

    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    parser.add_argument("--quiet", "-q", action="store_true", help="Quiet output (less verbose)")

    parser.add_argument("--no-deps-check", action="store_true", help="Skip dependency checking")

    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Force interactive mode (ignore command line args)",
    )

    return parser.parse_args()


def run_custom_pytest_command(markers=None, coverage=False, verbose=False, quiet=False):
    """Run custom pytest command with specified options"""
    print_header("Running Custom Tests")

    cmd = [sys.executable, "-m", "pytest", "tests/"]

    # Enable parallel execution with pytest-xdist if available
    # Use loadgroup for better test distribution
    if not markers or "integration" not in markers:
        cmd.extend(["-n", get_worker_count(), "--dist=loadgroup"])

    if markers:
        cmd.extend(["-m", markers])

    if coverage:
        cmd.extend(["--cov=ai-trading-bot", "--cov-report=term-missing", "--cov-report=html"])

    if verbose:
        cmd.append("-v")
    elif quiet:
        cmd.append("-q")
    else:
        cmd.append("-v")  # Default to verbose

    cmd.append("--tb=short")

    description = "Custom Tests"
    if markers:
        description += f" (markers: {markers})"
    if coverage:
        description += " with Coverage"

    success = run_command(cmd, description)

    if success and coverage:
        print_success("Coverage report generated in htmlcov/index.html")

    return success


def interactive_mode():
    """Run in interactive mode"""
    print_header("Trading Bot Test Runner - Interactive Mode")

    print("Available test commands:")
    print("1. smoke    - Quick smoke test")
    print("2. critical - Critical tests (live trading + risk)")
    print("3. unit     - All unit tests")
    print("4. integration - Integration tests")
    print("5. database - Database tests only")
    print("6. coverage - Tests with coverage analysis")
    print("7. all      - All tests")
    print("8. validate - Validate test environment")
    print("9. fast     - Fast tests only")
    print("10. slow    - Slow tests only")
    print("11. grouped - Optimized test groups")
    print("12. benchmark - Performance benchmark")
    print()

    command = input("Enter command (or test file name): ").strip().lower()
    return command


def run_all_tests(coverage=False, verbose=False, quiet=False):
    """Run the full test suite (all tests in the tests/ directory)."""
    print_header("Running All Tests")

    # Run unit tests first (these can run in parallel)
    print_subheader("Phase 1: Unit Tests")
    unit_success = run_unit_tests()

    if not unit_success:
        print_error("Unit tests failed. Stopping before integration tests.")
        return False

    # Run integration tests (these need sequential execution)
    print_subheader("Phase 2: Integration Tests")
    integration_success = run_integration_tests()

    return unit_success and integration_success


def main():
    """Main test runner function"""
    main_start_time = time.time()

    args = parse_arguments()

    # Handle interactive mode
    if args.interactive or (
        not args.command and not args.file and not args.markers and not args.coverage
    ):
        command = interactive_mode()
    else:
        command = args.command

    # Validate environment first
    if not validate_test_environment():
        sys.exit(1)

    # Check dependencies for most commands (unless explicitly skipped)
    if not args.no_deps_check and command not in ["smoke", "validate"]:
        if not check_dependencies():
            print_error("Missing dependencies. Please install them first.")
            print_warning("Use --no-deps-check to skip this check")
            sys.exit(1)

    success = True

    # Handle specific file argument
    if args.file:
        success = run_specific_test_file(args.file)
    # Handle coverage flag
    elif args.coverage:
        success = run_coverage_analysis()
    # Handle custom markers
    elif args.markers:
        success = run_custom_pytest_command(
            markers=args.markers, coverage=args.coverage, verbose=args.verbose, quiet=args.quiet
        )
    # Handle main commands
    elif command == "smoke":
        success = run_quick_smoke_test()
    elif command == "critical":
        success = run_critical_tests()
    elif command == "unit":
        success = run_unit_tests()
    elif command == "integration":
        success = run_integration_tests()
    elif command == "database":
        success = run_database_tests()
    elif command == "coverage":
        success = run_coverage_analysis()
    elif command == "fast":
        success = run_fast_tests()
    elif command == "slow":
        success = run_slow_tests()
    elif command == "grouped":
        success = run_grouped_tests()
    elif command == "benchmark":
        success = run_performance_benchmark()
    elif command == "all":
        success = run_all_tests(coverage=args.coverage, verbose=args.verbose, quiet=args.quiet)
    elif command == "validate":
        success = validate_test_environment()
    elif command and command.endswith(".py"):
        success = run_specific_test_file(command)
    elif command:
        print_error(f"Unknown command: {command}")
        print("Available commands: " + ", ".join(AVAILABLE_COMMANDS))
        print("Or use --help for more options")
        sys.exit(1)
    else:
        print_error("No command specified")
        print("Use --help for available options")
        sys.exit(1)

    # Summary
    total_duration = time.time() - main_start_time
    print_header("Test Summary")

    print(f"{Colors.BOLD}Total Execution Time: {total_duration:.2f} seconds{Colors.ENDC}")

    if success:
        print_success("All tests completed successfully! ✨")
        sys.exit(0)
    else:
        print_error("Some tests failed. Please review the output above.")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{Colors.WARNING}Test run interrupted by user{Colors.ENDC}")
        sys.exit(1)
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        sys.exit(1)
