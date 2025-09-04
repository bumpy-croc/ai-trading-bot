#!/usr/bin/env python3
"""
Setup script for Railway Log Analysis System

Configures the environment and validates prerequisites for the
automated Railway log analysis and error fixing system.
"""

import os
import subprocess
import sys
from pathlib import Path


def check_railway_cli():
    """Check if Railway CLI is installed and authenticated."""
    try:
        result = subprocess.run(["railway", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Railway CLI is installed")
            
            # * Check authentication
            auth_result = subprocess.run(["railway", "whoami"], capture_output=True, text=True)
            if auth_result.returncode == 0:
                print("✅ Railway CLI is authenticated")
                return True
            else:
                print("❌ Railway CLI not authenticated. Run 'railway login'")
                return False
        else:
            print("❌ Railway CLI not found")
            return False
    except FileNotFoundError:
        print("❌ Railway CLI not installed")
        print("   Install with: curl -fsSL https://railway.app/install.sh | sh")
        return False


def check_github_cli():
    """Check if GitHub CLI is installed and authenticated."""
    try:
        result = subprocess.run(["gh", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ GitHub CLI is installed")
            
            # * Check authentication
            auth_result = subprocess.run(["gh", "auth", "status"], capture_output=True, text=True)
            if auth_result.returncode == 0:
                print("✅ GitHub CLI is authenticated")
                return True
            else:
                print("❌ GitHub CLI not authenticated. Run 'gh auth login'")
                return False
        else:
            print("❌ GitHub CLI not found")
            return False
    except FileNotFoundError:
        print("❌ GitHub CLI not installed")
        print("   Install from: https://cli.github.com/")
        return False


def check_environment_variables():
    """Check required environment variables."""
    required_vars = [
        "RAILWAY_PROJECT_ID",
    ]
    
    optional_vars = [
        "RAILWAY_SERVICE_ID",
        "GITHUB_TOKEN",
    ]
    
    all_good = True
    
    print("\n--- Environment Variables ---")
    
    for var in required_vars:
        if os.getenv(var):
            print(f"✅ {var} is set")
        else:
            print(f"❌ {var} is required but not set")
            all_good = False
    
    for var in optional_vars:
        if os.getenv(var):
            print(f"✅ {var} is set")
        else:
            print(f"⚠️  {var} is optional but recommended")
    
    return all_good


def create_directories():
    """Create required directories for log storage."""
    directories = [
        "logs/railway/production",
        "logs/railway/staging", 
        "logs/railway/development",
        "logs/analysis_reports",
    ]
    
    print("\n--- Creating Directories ---")
    
    for dir_path in directories:
        path = Path(dir_path)
        path.mkdir(parents=True, exist_ok=True)
        print(f"✅ Created: {dir_path}")


def test_railway_connection():
    """Test Railway connection and log access."""
    print("\n--- Testing Railway Connection ---")
    
    try:
        # * Test basic Railway status
        result = subprocess.run(["railway", "status"], capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print("✅ Railway connection successful")
            
            # * Test log access (fetch last 10 minutes)
            log_result = subprocess.run(
                ["railway", "logs", "--since", "10m"], 
                capture_output=True, text=True, timeout=60
            )
            if log_result.returncode == 0:
                log_lines = len(log_result.stdout.splitlines())
                print(f"✅ Log access successful ({log_lines} lines retrieved)")
                return True
            else:
                print(f"❌ Log access failed: {log_result.stderr}")
                return False
        else:
            print(f"❌ Railway status check failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Railway connection timed out")
        return False
    except Exception as e:
        print(f"❌ Railway connection test failed: {e}")
        return False


def test_log_analysis():
    """Test log analysis with sample data."""
    print("\n--- Testing Log Analysis ---")
    
    try:
        # * Create sample log content
        sample_logs = '{"timestamp": "2025-01-09T12:00:00Z", "level": "ERROR", "message": "Test error for setup validation"}'
        
        # * Test analysis
        from unittest.mock import Mock

        from src.monitoring.log_analyzer import RailwayLogAnalyzer
        
        analyzer = RailwayLogAnalyzer(db_manager=Mock())
        report = analyzer.analyze_logs(sample_logs, 1)
        
        if report.total_entries > 0:
            print("✅ Log analysis working correctly")
            print(f"   Analyzed {report.total_entries} entries")
            return True
        else:
            print("❌ Log analysis failed to process sample data")
            return False
            
    except Exception as e:
        print(f"❌ Log analysis test failed: {e}")
        return False


def main():
    """Main setup function."""
    print("🚀 Railway Log Analysis System Setup")
    print("=" * 40)
    
    all_checks_passed = True
    
    # * Check prerequisites
    all_checks_passed &= check_railway_cli()
    all_checks_passed &= check_github_cli()
    all_checks_passed &= check_environment_variables()
    
    # * Create required directories
    create_directories()
    
    # * Test connections
    all_checks_passed &= test_railway_connection()
    all_checks_passed &= test_log_analysis()
    
    print("\n" + "=" * 40)
    
    if all_checks_passed:
        print("✅ Setup completed successfully!")
        print("\nNext steps:")
        print("1. Test manual log fetching: atb logs fetch --environment staging --hours 1 --save")
        print("2. Test log analysis: atb logs analyze --environment staging --hours 1")
        print("3. Test full pipeline: atb logs daily --environment staging --hours 1 --dry-run")
        print("4. Configure GitHub Actions secrets for automated execution")
        return 0
    else:
        print("❌ Setup incomplete - please fix the issues above")
        print("\nCommon solutions:")
        print("- Install Railway CLI: curl -fsSL https://railway.app/install.sh | sh")
        print("- Install GitHub CLI: https://cli.github.com/")
        print("- Authenticate: railway login && gh auth login")
        print("- Set environment variables in .env file")
        return 1


if __name__ == "__main__":
    sys.exit(main())