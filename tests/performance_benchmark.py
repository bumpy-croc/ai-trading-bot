#!/usr/bin/env python3
"""
Performance Benchmark Script for Test Suite

This script measures and tracks test suite performance over time.
It provides baseline metrics and helps identify performance regressions.
"""

import json
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


class TestPerformanceBenchmark:
    """Benchmark test suite performance"""

    def __init__(self):
        self.results_file = Path("tests/performance_baseline.json")
        self.baseline_data = self.load_baseline()

    def load_baseline(self) -> Dict[str, Any]:
        """Load existing baseline data"""
        if self.results_file.exists():
            try:
                with open(self.results_file) as f:
                    return json.load(f)
            except (OSError, json.JSONDecodeError):
                pass

        return {"benchmarks": [], "averages": {}, "last_updated": None}

    def save_baseline(self):
        """Save baseline data to file"""
        self.baseline_data["last_updated"] = datetime.now().isoformat()
        with open(self.results_file, "w") as f:
            json.dump(self.baseline_data, f, indent=2)

    def run_test_command(self, command: List[str], name: str) -> Dict[str, Any]:
        """Run a test command and measure performance"""
        print(f"📊 Benchmarking: {name}")
        print(f"Command: {' '.join(command)}")

        start_time = time.time()

        try:
            result = subprocess.run(command, capture_output=True, text=True, check=False)

            duration = time.time() - start_time

            # Extract test count from pytest output
            test_count = self.extract_test_count(result.stdout)

            success = result.returncode == 0

            benchmark_result = {
                "name": name,
                "duration": round(duration, 2),
                "test_count": test_count,
                "success": success,
                "timestamp": datetime.now().isoformat(),
                "tests_per_second": (
                    round(test_count / duration, 2) if duration > 0 and test_count > 0 else 0
                ),
            }

            print(
                f"✅ {name}: {duration:.2f}s ({test_count} tests, {benchmark_result['tests_per_second']:.1f} tests/sec)"
            )

            return benchmark_result

        except Exception as e:
            print(f"❌ {name} failed: {e}")
            return {
                "name": name,
                "duration": time.time() - start_time,
                "test_count": 0,
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "tests_per_second": 0,
            }

    def extract_test_count(self, output: str) -> int:
        """Extract test count from pytest output"""
        lines = output.split("\n")
        for line in lines:
            if "passed" in line and (
                "failed" in line or "error" in line or line.strip().endswith("passed")
            ):
                # Look for patterns like "25 passed", "20 passed, 2 failed", etc.
                words = line.split()
                for i, word in enumerate(words):
                    if word in ["passed", "failed", "error", "skipped"]:
                        if i > 0 and words[i - 1].isdigit():
                            return int(words[i - 1])
                        # If we find "passed" but no number before it, look for total
                        elif "passed" in line:
                            # Try to extract from patterns like "===== 25 passed in 2.34s ====="
                            import re

                            match = re.search(r"(\d+)\s+passed", line)
                            if match:
                                return int(match.group(1))
        return 0

    def run_full_benchmark(self):
        """Run complete performance benchmark suite"""
        print("🚀 Starting Test Suite Performance Benchmark")
        print("=" * 60)

        benchmarks = []

        # 1. Fast unit tests
        fast_result = self.run_test_command(
            [
                sys.executable,
                "-m",
                "pytest",
                "tests/test_indicators.py",
                "tests/test_returns_calculations.py",
                "tests/test_smoke.py",
                "-v",
                "--tb=short",
                "-q",
                "-n",
                "auto",
                "--dist=loadgroup",
            ],
            "Fast Unit Tests",
        )
        benchmarks.append(fast_result)

        # 2. Medium unit tests
        medium_result = self.run_test_command(
            [
                sys.executable,
                "-m",
                "pytest",
                "tests/test_strategies.py",
                "tests/test_risk_management.py",
                "tests/test_performance.py",
                "-v",
                "--tb=short",
                "-q",
                "-n",
                "auto",
                "--dist=loadgroup",
                "-m",
                "not integration",
            ],
            "Medium Unit Tests",
        )
        benchmarks.append(medium_result)

        # 3. Computation-heavy tests
<<<<<<< HEAD
        compute_result = self.run_test_command(
            [
                sys.executable,
                "-m",
                "pytest",
                "tests/test_backtesting.py",
                "tests/test_ml_adaptive.py",
                "-v",
                "--tb=short",
                "-q",
                "-n",
                "2",
                "-m",
                "not integration",
            ],
            "Computation Tests",
        )
=======
        compute_result = self.run_test_command([
            sys.executable, "-m", "pytest", 
            "tests/test_backtesting.py",
            "-v", "--tb=short", "-q",
            "-n", "2",
            "-m", "not integration"
        ], "Computation Tests")
>>>>>>> origin/develop
        benchmarks.append(compute_result)

        # 4. All unit tests (for comparison)
        unit_result = self.run_test_command(
            [
                sys.executable,
                "-m",
                "pytest",
                "-m",
                "not integration",
                "-v",
                "--tb=short",
                "-q",
                "-n",
                "auto",
                "--dist=loadgroup",
            ],
            "All Unit Tests",
        )
        benchmarks.append(unit_result)

        # 5. Database setup benchmark (if integration tests enabled)
        if os.getenv("ENABLE_INTEGRATION_TESTS") == "1":
            integration_result = self.run_test_command(
                [sys.executable, "-m", "pytest", "-m", "integration", "-v", "--tb=short", "-q"],
                "Integration Tests",
            )
            benchmarks.append(integration_result)

        # Save results
        self.baseline_data["benchmarks"].extend(benchmarks)
        self.update_averages(benchmarks)
        self.save_baseline()

        # Print summary
        self.print_summary(benchmarks)

        return benchmarks

    def update_averages(self, new_benchmarks: List[Dict[str, Any]]):
        """Update running averages"""
        for benchmark in new_benchmarks:
            name = benchmark["name"]
            if name not in self.baseline_data["averages"]:
                self.baseline_data["averages"][name] = {
                    "duration": [],
                    "tests_per_second": [],
                    "test_count": [],
                }

            # Keep last 10 runs for rolling average
            avg_data = self.baseline_data["averages"][name]
            avg_data["duration"].append(benchmark["duration"])
            avg_data["tests_per_second"].append(benchmark["tests_per_second"])
            avg_data["test_count"].append(benchmark["test_count"])

            # Keep only last 10 measurements
            for key in avg_data:
                avg_data[key] = avg_data[key][-10:]

    def print_summary(self, benchmarks: List[Dict[str, Any]]):
        """Print benchmark summary"""
        print("\n" + "=" * 60)
        print("📈 PERFORMANCE BENCHMARK SUMMARY")
        print("=" * 60)

        total_duration = sum(b["duration"] for b in benchmarks if b["success"])
        total_tests = sum(b["test_count"] for b in benchmarks if b["success"])

        print(f"Total Duration: {total_duration:.2f} seconds")
        print(f"Total Tests: {total_tests}")
        print(f"Overall Rate: {total_tests/total_duration:.1f} tests/second")
        print()

        print("Individual Results:")
        print("-" * 40)
        for benchmark in benchmarks:
            status = "✅" if benchmark["success"] else "❌"
            print(
                f"{status} {benchmark['name']:<20} {benchmark['duration']:>6.2f}s ({benchmark['test_count']:>3} tests)"
            )

        # Compare with averages if available
        print("\nComparison with Historical Average:")
        print("-" * 40)
        for benchmark in benchmarks:
            name = benchmark["name"]
            if name in self.baseline_data["averages"]:
                avg_duration = statistics.mean(self.baseline_data["averages"][name]["duration"])
                diff = benchmark["duration"] - avg_duration
                diff_pct = (diff / avg_duration) * 100 if avg_duration > 0 else 0

                if abs(diff_pct) < 5:
                    trend = "≈"
                elif diff_pct > 0:
                    trend = f"↗ +{diff_pct:.1f}%"
                else:
                    trend = f"↘ {diff_pct:.1f}%"

                print(f"{name:<20} {trend}")

        print("\n💡 Performance Tips:")
        print("- Run 'python tests/run_tests.py grouped' for optimized test execution")
        print("- Use 'python tests/run_tests.py fast' for quick feedback during development")
        print("- Integration tests are automatically run nightly in CI")

    def compare_with_baseline(self, current_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compare current results with baseline"""
        comparison = {"improved": [], "regressed": [], "new": [], "overall_change": 0}

        for result in current_results:
            name = result["name"]
            if name in self.baseline_data["averages"]:
                avg_duration = statistics.mean(self.baseline_data["averages"][name]["duration"])
                change_pct = ((result["duration"] - avg_duration) / avg_duration) * 100

                if change_pct < -5:  # 5% faster
                    comparison["improved"].append({"name": name, "change": change_pct})
                elif change_pct > 10:  # 10% slower
                    comparison["regressed"].append({"name": name, "change": change_pct})
            else:
                comparison["new"].append(name)

        return comparison


def main():
    """Run performance benchmark"""
    benchmark = TestPerformanceBenchmark()

    if len(sys.argv) > 1 and sys.argv[1] == "--compare":
        # Just show comparison with existing baseline
        if benchmark.baseline_data["benchmarks"]:
            print("📊 Latest Benchmark Results:")
            latest_benchmarks = benchmark.baseline_data["benchmarks"][-4:]  # Last 4 benchmark types
            benchmark.print_summary(latest_benchmarks)
        else:
            print("No baseline data found. Run without --compare to create baseline.")
    else:
        # Run full benchmark
        _results = benchmark.run_full_benchmark()

        print(f"\n💾 Results saved to: {benchmark.results_file}")
        print("Run 'python tests/performance_benchmark.py --compare' to compare future runs")


if __name__ == "__main__":
    main()
