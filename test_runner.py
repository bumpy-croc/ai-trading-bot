#!/usr/bin/env python3
"""
Trading Bot Test Runner

A simple script to run various test suites for the crypto trading bot.
This provides quick access to common testing scenarios.
"""

import sys
import subprocess
import os
from pathlib import Path

# Ensure we can import from the project
project_root = Path(__file__).parent
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
        '--cov=bottrade',
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
    
    cmd = [
        sys.executable, '-m', 'pytest',
        f'tests/{test_file}',
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
    
    # Check if we're in the right directory
    if not (Path.cwd() / 'bottrade').exists():
        print_error("Not in the correct project directory")
        print_warning("Please run this script from the project root directory")
        return False
    
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

def main():
    """Main test runner function"""
    print_header("Trading Bot Test Runner")
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
    else:
        # Interactive mode
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
    
    # Validate environment first
    if not validate_test_environment():
        sys.exit(1)
    
    # Check dependencies for most commands
    if command not in ['smoke', 'validate']:
        if not check_dependencies():
            print_error("Missing dependencies. Please install them first.")
            sys.exit(1)
    
    success = True
    
    if command == 'smoke':
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
    elif command.endswith('.py'):
        success = run_specific_test_file(command)
    else:
        print_error(f"Unknown command: {command}")
        print("Available commands: smoke, critical, unit, integration, coverage, all, validate")
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