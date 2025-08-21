#!/usr/bin/env python3
"""
Script to run mypy on all Python files in the src directory.
This handles the project structure issues that prevent direct mypy usage.
"""

import subprocess
import sys
import os
import shutil
import tempfile
from pathlib import Path


def find_python_files(directory: str) -> list[str]:
    """Find all Python files in the given directory recursively."""
    python_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    return sorted(python_files)


def run_mypy_on_files(files: list[str], config_file: str) -> int:
    """Run mypy on a list of files and return exit code."""
    print(f"Running mypy on {len(files)} Python files...")
    
    total_errors = 0
    files_with_errors = 0
    
    for file in files:
        print(f"Checking {file}...", end=" ")
        
        # Run mypy on individual file
        cmd = ['mypy', '--config-file', config_file, file]
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
    """Main function to run mypy on all src files."""
    src_dir = "src"
    
    if not os.path.exists(src_dir):
        print(f"Error: {src_dir} directory not found")
        return 1
    
    # Create a temporary directory with a valid package name
    with tempfile.TemporaryDirectory(prefix="mypy_check_") as temp_dir:
        # Copy src directory to temp directory
        temp_src = os.path.join(temp_dir, "src")
        shutil.copytree(src_dir, temp_src)
        
        # Copy mypy config to temp directory
        shutil.copy("mypy.ini", temp_dir)
        
        # Change to temp directory
        original_dir = os.getcwd()
        os.chdir(temp_dir)
        
        try:
            # Change to src directory within temp directory
            os.chdir("src")
            
            python_files = find_python_files(".")
            
            if not python_files:
                print(f"No Python files found in {src_dir} directory")
                return 1
            
            print(f"Found {len(python_files)} Python files in {src_dir}/")
            
            # Run mypy on all files (config file is in parent directory)
            exit_code = run_mypy_on_files(python_files, "../mypy.ini")
            
            if exit_code == 0:
                print("✅ Mypy check passed!")
            else:
                print("❌ Mypy check failed!")
            
            return exit_code
            
        finally:
            # Change back to original directory
            os.chdir(original_dir)


if __name__ == '__main__':
    sys.exit(main())
