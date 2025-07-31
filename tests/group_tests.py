#!/usr/bin/env python3
"""
Test grouping script for GitHub Actions Matrix Strategy.

This script analyzes test files and groups them for parallel execution
across multiple GitHub Actions runners to avoid pytest-xdist hanging issues.
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple


def get_test_files() -> List[str]:
    """Get all test files in the tests directory."""
    test_files = []
    tests_dir = Path('tests')
    
    for test_file in tests_dir.rglob('test_*.py'):
        if test_file.is_file():
            test_files.append(str(test_file))
    
    return sorted(test_files)


def estimate_test_duration(test_file: str) -> float:
    """Estimate test duration based on file size and complexity."""
    try:
        file_path = Path(test_file)
        if not file_path.exists():
            return 1.0
        
        # Get file size in KB
        size_kb = file_path.stat().st_size / 1024
        
        # Count lines of code
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Count test functions
        test_functions = sum(1 for line in lines if line.strip().startswith('def test_'))
        
        # Estimate duration based on size and complexity
        # Base duration: 1 second per test function
        # Size factor: 0.1 seconds per KB
        estimated_duration = test_functions + (size_kb * 0.1)
        
        # Cap at reasonable limits
        return min(max(estimated_duration, 1.0), 60.0)
        
    except Exception:
        return 1.0


def group_tests_by_duration(test_files: List[str], num_groups: int = 4) -> List[List[str]]:
    """Group tests by estimated duration to balance load across groups."""
    # Calculate estimated duration for each test file
    test_durations = [(test_file, estimate_test_duration(test_file)) for test_file in test_files]
    
    # Sort by duration (heaviest first)
    test_durations.sort(key=lambda x: x[1], reverse=True)
    
    # Initialize groups
    groups = [[] for _ in range(num_groups)]
    group_durations = [0.0] * num_groups
    
    # Distribute tests using greedy approach (heaviest first)
    for test_file, duration in test_durations:
        # Find group with minimum total duration
        min_group_idx = group_durations.index(min(group_durations))
        groups[min_group_idx].append(test_file)
        group_durations[min_group_idx] += duration
    
    return groups


def create_test_groups(num_groups: int = 4) -> Dict[str, List[str]]:
    """Create test groups for matrix execution."""
    test_files = get_test_files()
    groups = group_tests_by_duration(test_files, num_groups)
    
    result = {}
    for i, group in enumerate(groups):
        group_name = f"group-{i+1}"
        result[group_name] = group
    
    return result


def get_test_patterns_for_group(group_name: str) -> str:
    """Get pytest patterns for a specific test group."""
    groups = create_test_groups()
    
    if group_name not in groups:
        print(f"Error: Unknown group '{group_name}'", file=sys.stderr)
        sys.exit(1)
    
    test_files = groups[group_name]
    
    # Convert file paths to pytest patterns
    patterns = []
    for test_file in test_files:
        # Convert path to module pattern
        # e.g., tests/test_something.py -> tests/test_something.py
        patterns.append(test_file)
    
    return ' '.join(patterns)


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python group_tests.py <command> [args...]")
        print("Commands:")
        print("  list-groups                    - List all test groups")
        print("  get-patterns <group-name>      - Get pytest patterns for a group")
        print("  run-group <group-name>         - Run tests for a specific group")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "list-groups":
        groups = create_test_groups()
        print("Available test groups:")
        for group_name, test_files in groups.items():
            print(f"  {group_name}: {len(test_files)} test files")
            for test_file in test_files[:3]:  # Show first 3 files
                print(f"    - {test_file}")
            if len(test_files) > 3:
                print(f"    ... and {len(test_files) - 3} more")
            print()
    
    elif command == "get-patterns":
        if len(sys.argv) < 3:
            print("Error: Group name required", file=sys.stderr)
            sys.exit(1)
        group_name = sys.argv[2]
        patterns = get_test_patterns_for_group(group_name)
        print(patterns)
    
    elif command == "run-group":
        if len(sys.argv) < 3:
            print("Error: Group name required", file=sys.stderr)
            sys.exit(1)
        group_name = sys.argv[2]
        patterns = get_test_patterns_for_group(group_name)
        
        # Run pytest with the patterns
        cmd = [
            sys.executable, '-m', 'pytest',
            '-v', '--tb=short',
            '--timeout=300',
            '-m', 'not integration'
        ] + patterns.split()
        
        print(f"Running test group: {group_name}")
        print(f"Command: {' '.join(cmd)}")
        
        result = subprocess.run(cmd)
        sys.exit(result.returncode)
    
    else:
        print(f"Error: Unknown command '{command}'", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main() 