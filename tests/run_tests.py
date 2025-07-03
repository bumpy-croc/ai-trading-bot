#!/usr/bin/env python3
"""
Trading Bot Test Runner

A simple script to run various test suites for the crypto trading bot.
This provides quick access to common testing scenarios.
"""

import sys
import subprocess
import os
import argparse
from pathlib import Path

# Ensure we can import from the project
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

class Colors:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_header(text):
    """Print colored header"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text.center(60)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}\n")

def print_success(text):
    """Print success message"""
    print(f"{Colors.OKGREEN}✅ {text}{Colors.ENDC}")

def print_warning(text):
    """Print warning message"""
    print(f"{Colors.WARNING}⚠️  {text}{Colors.ENDC}")

def print_error(text):
    """Print error message"""
    print(f"{Colors.FAIL}❌ {text}{Colors.ENDC}")

def run_command(cmd, description):
    """Run a command and return success status"""
    print(f"{Colors.OKBLUE}Running: {description}{Colors.ENDC}")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        
        if result.returncode == 0:
            print_success(f"{description} completed successfully")
            if result.stdout.strip():
                print(f"{Colors.OKCYAN}{result.stdout}{Colors.ENDC}")
            return True
        else:
            print_error(f"{description} failed")
            if result.stderr.strip():
                print(f"{Colors.FAIL}{result.stderr}{Colors.ENDC}")
            if result.stdout.strip():
                print(f"{Colors.WARNING}{result.stdout}{Colors.ENDC}")
            return False
    except Exception as e:
        print_error(f"Failed to run {description}: {e}")
        return False

def check_dependencies():
    """Check if required dependencies are installed"""
    print_header("Checking Dependencies")
    
    required_packages = ['pytest', 'pandas', 'numpy']
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
        sys.executable, '-m', 'pytest', 
        '-m', 'live_trading or risk_management',
        '-v', '--tb=short'
    ]
    
    return run_command(cmd, "Critical Tests (Live Trading + Risk Management)")

def run_unit_tests():
    """Run all unit tests"""
    print_header("Running Unit Tests")
    
    cmd = [
        sys.executable, '-m', 'pytest',
        'tests/',
        '-v', '--tb=short',
        '-m', 'not integration'
    ]
    
    return run_command(cmd, "Unit Tests")

def run_integration_tests():
    """Run integration tests"""
    print_header("Running Integration Tests")
    
    cmd = [
        sys.executable, '-m', 'pytest',
        '-m', 'integration',
        '-v', '--tb=short'
    ]
    
    return run_command(cmd, "Integration Tests")

def run_coverage_analysis():
    """Run tests with coverage analysis"""
    print_header("Running Coverage Analysis")
    
    cmd = [
        sys.executable, '-m', 'pytest',
        '--cov=ai-trading-bot',
        '--cov-report=term-missing',
        '--cov-report=html',
        'tests/'
    ]
    
    success = run_command(cmd, "Coverage Analysis")
    
    if success:
        print_success("Coverage report generated in htmlcov/index.html")
    
    return success

def run_specific_test_file(test_file):
    """Run a specific test file"""
    print_header(f"Running {test_file}")
    
    # Handle both with and without tests/ prefix
    if not test_file.startswith('tests/'):
        test_file = f'tests/{test_file}'
    
    cmd = [
        sys.executable, '-m', 'pytest',
        test_file,
        '-v', '--tb=short'
    ]
    
    return run_command(cmd, f"Tests in {test_file}")

def run_quick_smoke_test():
    """Run a quick smoke test to verify basic functionality"""
    print_header("Running Quick Smoke Test")
    
    # Test basic imports
    try:
        from strategies.adaptive import AdaptiveStrategy
        from core.risk.risk_manager import RiskManager
        from live.trading_engine import LiveTradingEngine
        print_success("All core modules import successfully")
        
        # Test basic functionality
        strategy = AdaptiveStrategy()
        risk_manager = RiskManager()
        print_success("Core objects can be instantiated")
        
        return True
    except Exception as e:
        print_error(f"Smoke test failed: {e}")
        return False

def validate_test_environment():
    """Validate the test environment"""
    print_header("Validating Test Environment")
    
    # Check if tests directory exists
    if not Path('tests').exists():
        print_error("Tests directory not found")
        return False
    
    # Check pytest.ini or setup.cfg
    has_pytest_config = any([
        Path('pytest.ini').exists(),
        Path('setup.cfg').exists(),
        Path('pyproject.toml').exists()
    ])
    
    if not has_pytest_config:
        print_warning("No pytest configuration file found (pytest.ini, setup.cfg, or pyproject.toml)")
    
    print_success("Test environment validation passed")
    return True

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Trading Bot Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_tests.py smoke                    # Quick smoke test
  python run_tests.py unit                     # Run unit tests
  python run_tests.py critical                 # Run critical tests
  python run_tests.py all                      # Run all tests
  python run_tests.py test_strategies.py       # Run specific test file
  python run_tests.py --file test_data_providers.py  # Run specific file
  python run_tests.py --markers "not integration"    # Run tests with specific markers
  python run_tests.py --coverage               # Run with coverage
        """
    )
    
    # Main test command
    parser.add_argument(
        'command',
        nargs='?',
        choices=['smoke', 'critical', 'unit', 'integration', 'coverage', 'all', 'validate'],
        help='Test command to run'
    )
    
    # Alternative ways to specify tests
    parser.add_argument(
        '--file', '-f',
        help='Run specific test file (e.g., test_strategies.py)'
    )
    
    parser.add_argument(
        '--markers', '-m',
        help='Run tests with specific pytest markers (e.g., "not integration")'
    )
    
    parser.add_argument(
        '--coverage', '-c',
        action='store_true',
        help='Run tests with coverage analysis'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Quiet output (less verbose)'
    )
    
    parser.add_argument(
        '--no-deps-check',
        action='store_true',
        help='Skip dependency checking'
    )
    
    parser.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Force interactive mode (ignore command line args)'
    )
    
    return parser.parse_args()

def run_custom_pytest_command(markers=None, coverage=False, verbose=False, quiet=False):
    """Run custom pytest command with specified options"""
    print_header("Running Custom Tests")
    
    cmd = [sys.executable, '-m', 'pytest', 'tests/']
    
    if markers:
        cmd.extend(['-m', markers])
    
    if coverage:
        cmd.extend([
            '--cov=ai-trading-bot',
            '--cov-report=term-missing',
            '--cov-report=html'
        ])
    
    if verbose:
        cmd.append('-v')
    elif quiet:
        cmd.append('-q')
    else:
        cmd.append('-v')  # Default to verbose
    
    cmd.append('--tb=short')
    
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
    print("5. coverage - Tests with coverage analysis")
    print("6. all      - All tests")
    print("7. validate - Validate test environment")
    print()
    
    command = input("Enter command (or test file name): ").strip().lower()
    return command

def main():
    """Main test runner function"""
    args = parse_arguments()
    
    # Handle interactive mode
    if args.interactive or (not args.command and not args.file and not args.markers and not args.coverage):
        command = interactive_mode()
    else:
        command = args.command
    
    # Validate environment first
    if not validate_test_environment():
        sys.exit(1)
    
    # Check dependencies for most commands (unless explicitly skipped)
    if not args.no_deps_check and command not in ['smoke', 'validate']:
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
            markers=args.markers,
            coverage=args.coverage,
            verbose=args.verbose,
            quiet=args.quiet
        )
    # Handle main commands
    elif command == 'smoke':
        success = run_quick_smoke_test()
    elif command == 'critical':
        success = run_critical_tests()
    elif command == 'unit':
        success = run_unit_tests()
    elif command == 'integration':
        success = run_integration_tests()
    elif command == 'coverage':
        success = run_coverage_analysis()
    elif command == 'all':
        success = (
            run_quick_smoke_test() and
            run_critical_tests() and
            run_unit_tests() and
            run_integration_tests()
        )
    elif command == 'validate':
        success = validate_test_environment()
    elif command and command.endswith('.py'):
        success = run_specific_test_file(command)
    elif command:
        print_error(f"Unknown command: {command}")
        print("Available commands: smoke, critical, unit, integration, coverage, all, validate")
        print("Or use --help for more options")
        sys.exit(1)
    else:
        print_error("No command specified")
        print("Use --help for available options")
        sys.exit(1)
    
    # Summary
    print_header("Test Summary")
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