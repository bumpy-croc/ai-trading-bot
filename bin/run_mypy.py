#!/usr/bin/env python3
"""
Script to run mypy on all Python files in the src directory.
This handles the project structure issues that prevent direct mypy usage.

Supports:
- Running on all src/ files (default)
- Running on specific files via --targets flag
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile


def find_python_files(directory: str) -> list[str]:
    """Find all Python files in the given directory recursively."""
    python_files = []
    for root, _dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                python_files.append(os.path.join(root, file))
    return sorted(python_files)


def filter_target_files(targets: list[str], base_dir: str) -> list[str]:
    """Filter and validate target files for mypy checking.

    Args:
        targets: List of file/directory paths (relative to project root).
        base_dir: Base directory to resolve paths from.

    Returns:
        List of valid Python file paths (relative to base_dir).
    """
    valid_files = []

    for target in targets:
        # Convert to absolute path for validation
        target_path = os.path.abspath(target)

        if not os.path.exists(target_path):
            print(f"Warning: Target path does not exist: {target}")
            continue

        if os.path.isfile(target_path):
            if target.endswith(".py"):
                # Get path relative to base_dir
                rel_path = os.path.relpath(target_path, base_dir)
                valid_files.append(rel_path)
            else:
                print(f"Warning: Skipping non-Python file: {target}")
        elif os.path.isdir(target_path):
            # Recursively find Python files in directory
            for root, _dirs, files in os.walk(target_path):
                for file in files:
                    if file.endswith(".py"):
                        full_path = os.path.join(root, file)
                        rel_path = os.path.relpath(full_path, base_dir)
                        valid_files.append(rel_path)

    return sorted(set(valid_files))


def run_mypy_on_files(files: list[str], config_file: str) -> int:
    """Run mypy on a list of files and return exit code."""
    print(f"Running mypy on {len(files)} Python files...")

    total_errors = 0
    files_with_errors = 0

    for file in files:
        print(f"Checking {file}...", end=" ")

        # Run mypy on individual file
        cmd = ["mypy", "--config-file", config_file, file]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print("✅")
        else:
            print("❌")
            files_with_errors += 1
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(result.stderr)
            total_errors += 1

    print(f"\nSummary: {files_with_errors} files with errors out of {len(files)} files checked")

    return 1 if total_errors > 0 else 0


def main():
    """Main function to run mypy on all src files or specific targets."""
    parser = argparse.ArgumentParser(
        description="Run mypy type checking on src files or specific targets."
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        help="Specific files or directories to check (relative to project root)",
    )
    args = parser.parse_args()

    src_dir = "src"
    original_dir = os.getcwd()

    # Determine which files to check
    if args.targets:
        # Use provided targets
        target_files = filter_target_files(args.targets, original_dir)
        if not target_files:
            print("No valid Python files found in targets")
            return 1
        print(f"Checking {len(target_files)} targeted file(s)")
    else:
        # Default: check all src files
        if not os.path.exists(src_dir):
            print(f"Error: {src_dir} directory not found")
            return 1
        target_files = None  # Will use find_python_files on temp src

    # Create a temporary directory with a valid package name
    with tempfile.TemporaryDirectory(prefix="mypy_check_") as temp_dir:
        # Copy src directory to temp directory (always needed for dependencies)
        temp_src = os.path.join(temp_dir, "src")
        shutil.copytree(src_dir, temp_src)

        # Copy cli directory if it exists
        if os.path.exists("cli"):
            temp_cli = os.path.join(temp_dir, "cli")
            shutil.copytree("cli", temp_cli)

        # Copy mypy config to temp directory
        shutil.copy("mypy.ini", temp_dir)

        # Change to temp directory
        os.chdir(temp_dir)

        try:
            if args.targets:
                # Map target files to temp directory structure
                python_files = []
                for f in target_files:
                    temp_path = os.path.join(temp_dir, f)
                    if os.path.exists(temp_path):
                        python_files.append(f)
                    else:
                        print(f"Warning: File not found in temp structure: {f}")
            else:
                # Change to src directory within temp directory
                os.chdir("src")
                python_files = find_python_files(".")
                # Prepend "src/" prefix for proper config reference
                python_files = [os.path.join("src", f) for f in python_files]
                os.chdir("..")  # Back to temp root for config access

            if not python_files:
                print("No Python files to check")
                return 1

            print(f"Running mypy on {len(python_files)} file(s)")

            # Run mypy on all files (config file is in current directory)
            exit_code = run_mypy_on_files(python_files, "mypy.ini")

            if exit_code == 0:
                print("✅ Mypy check passed!")
            else:
                print("❌ Mypy check failed!")

            return exit_code

        finally:
            # Change back to original directory
            os.chdir(original_dir)


if __name__ == "__main__":
    sys.exit(main())
