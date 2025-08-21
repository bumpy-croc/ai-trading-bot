#!/usr/bin/env python3
"""
Setup script for pre-commit hooks.
Run this script to install pre-commit hooks for code quality checks.
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"* {description}...")
    try:
        subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"  ✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ✗ {description} failed:")
        print(f"    {e.stderr}")
        return False


def main():
    """Main setup function."""
    print("Setting up pre-commit hooks for code quality...")
    print()

    # Check if we're in the project root
    if not Path(".pre-commit-config.yaml").exists():
        print(
            "Error: .pre-commit-config.yaml not found. Please run this script from the project root."
        )
        sys.exit(1)

    # Install pre-commit if not already installed
    if not run_command("python3 -m pip install pre-commit", "Installing pre-commit"):
        print("Failed to install pre-commit. Please install it manually:")
        print("  python3 -m pip install pre-commit")
        sys.exit(1)

    # Install pre-commit hooks
    if not run_command("pre-commit install", "Installing pre-commit hooks"):
        print("Failed to install pre-commit hooks.")
        sys.exit(1)

    # Run initial check
    print()
    print("Running initial pre-commit check...")
    if run_command("pre-commit run --all-files", "Running initial pre-commit check"):
        print()
        print("✓ Pre-commit setup completed successfully!")
        print()
        print("Your pre-commit hooks are now active. They will run automatically on every commit.")
        print("To run checks manually:")
        print("  pre-commit run --all-files  # Check all files")
        print("  pre-commit run              # Check staged files only")
    else:
        print()
        print("⚠ Pre-commit setup completed, but initial check had issues.")
        print("Please review and fix any linting errors before committing.")


if __name__ == "__main__":
    main()
