#!/usr/bin/env python3
"""
Account Synchronization Test Runner

This script runs the account synchronization tests to verify the functionality
of the account sync system.
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def run_tests(test_type="all", verbose=False, coverage=False):
    """
    Run account synchronization tests.
    
    Args:
        test_type: Type of tests to run ('unit', 'integration', 'all')
        verbose: Whether to run tests in verbose mode
        coverage: Whether to generate coverage report
    """
    
    # Get the test file path
    test_file = Path(__file__).parent.parent / "tests" / "test_account_sync.py"
    
    if not test_file.exists():
        print(f"❌ Test file not found: {test_file}")
        return False
    
    # Build pytest command
    cmd = ["python", "-m", "pytest", str(test_file)]
    
    # Add test type filters
    if test_type == "unit":
        cmd.extend(["-k", "not integration"])
    elif test_type == "integration":
        cmd.extend(["-k", "integration"])
    
    # Add verbose flag
    if verbose:
        cmd.append("-v")
    
    # Add coverage if requested
    if coverage:
        cmd.extend(["--cov=src.live.account_sync", "--cov-report=html", "--cov-report=term"])
    
    # Add some default flags for better output
    cmd.extend(["--tb=short", "--strict-markers"])
    
    print(f"🚀 Running account sync tests ({test_type})...")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 80)
    
    try:
        # Run the tests
        result = subprocess.run(cmd, check=False, capture_output=False)
        
        if result.returncode == 0:
            print("-" * 80)
            print("✅ All account sync tests passed!")
            return True
        else:
            print("-" * 80)
            print("❌ Some account sync tests failed!")
            return False
            
    except FileNotFoundError:
        print("❌ pytest not found. Please install pytest: pip install pytest")
        return False
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return False

def main():
    """Main function to parse arguments and run tests."""
    parser = argparse.ArgumentParser(
        description="Run account synchronization tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_account_sync_tests.py                    # Run all tests
  python scripts/run_account_sync_tests.py --type unit        # Run unit tests only
  python scripts/run_account_sync_tests.py --type integration # Run integration tests only
  python scripts/run_account_sync_tests.py --verbose          # Run with verbose output
  python scripts/run_account_sync_tests.py --coverage         # Run with coverage report
        """
    )
    
    parser.add_argument(
        "--type", "-t",
        choices=["unit", "integration", "all"],
        default="all",
        help="Type of tests to run (default: all)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Run tests in verbose mode"
    )
    
    parser.add_argument(
        "--coverage", "-c",
        action="store_true",
        help="Generate coverage report"
    )
    
    args = parser.parse_args()
    
    # Run the tests
    success = run_tests(
        test_type=args.type,
        verbose=args.verbose,
        coverage=args.coverage
    )
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()