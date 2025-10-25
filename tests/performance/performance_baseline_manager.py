"""
Performance Baseline Management System

Manages performance baselines, tracks regression trends, and provides
automated performance monitoring for the component system.
"""

import json
import statistics
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd


class PerformanceBaselineManager:
    """Manage performance baselines and regression detection"""

    def __init__(self, baseline_file: str = "tests/performance/component_baselines.json"):
        self.baseline_file = Path(baseline_file)
        self.baselines = self.load_baselines()

    def load_baselines(self) -> Dict[str, Any]:
        """Load existing baselines from file"""
        if self.baseline_file.exists():
            try:
                with open(self.baseline_file, "r") as f:
                    return json.load(f)
            except (OSError, json.JSONDecodeError):
                pass

        return self.create_default_baselines()

    def create_default_baselines(self) -> Dict[str, Any]:
        """Create default baseline structure"""
        return {
            "metadata": {
                "created": datetime.now().isoformat(),
                "last_updated": None,
                "version": "1.0",
                "description": "Component system performance baselines",
            },
            "baselines": {
                "signal_generation": {
                    "target_ms": 5.0,
                    "warning_ms": 8.0,
                    "critical_ms": 15.0,
                    "samples": [],
                    "trend": "stable",
                },
                "risk_calculation": {
                    "target_ms": 2.0,
                    "warning_ms": 4.0,
                    "critical_ms": 8.0,
                    "samples": [],
                    "trend": "stable",
                },
                "position_sizing": {
                    "target_ms": 1.0,
                    "warning_ms": 2.0,
                    "critical_ms": 5.0,
                    "samples": [],
                    "trend": "stable",
                },
                "complete_decision": {
                    "target_ms": 15.0,
                    "warning_ms": 25.0,
                    "critical_ms": 50.0,
                    "samples": [],
                    "trend": "stable",
                },
                "batch_processing_100": {
                    "target_ms": 500.0,
                    "warning_ms": 800.0,
                    "critical_ms": 1500.0,
                    "samples": [],
                    "trend": "stable",
                },
                "memory_usage": {
                    "target_mb": 30.0,
                    "warning_mb": 50.0,
                    "critical_mb": 100.0,
                    "samples": [],
                    "trend": "stable",
                },
            },
            "regression_thresholds": {
                "warning_pct": 20.0,
                "critical_pct": 50.0,
                "trend_window": 10,
                "min_samples": 5,
            },
        }

    def save_baselines(self):
        """Save baselines to file"""
        self.baselines["metadata"]["last_updated"] = datetime.now().isoformat()

        # Ensure directory exists
        self.baseline_file.parent.mkdir(parents=True, exist_ok=True)

        with open(self.baseline_file, "w") as f:
            json.dump(self.baselines, f, indent=2)

    def record_measurement(
        self, test_name: str, value: float, metadata: Optional[Dict[str, Any]] = None
    ):
        """Record a performance measurement"""
        if test_name not in self.baselines["baselines"]:
            # Create new baseline entry
            self.baselines["baselines"][test_name] = {
                "target_ms": value * 1.2,  # 20% buffer
                "warning_ms": value * 1.5,
                "critical_ms": value * 2.0,
                "samples": [],
                "trend": "new",
            }

        baseline = self.baselines["baselines"][test_name]

        # Add sample
        sample = {
            "value": value,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {},
        }

        baseline["samples"].append(sample)

        # Keep only recent samples (last 50)
        baseline["samples"] = baseline["samples"][-50:]

        # Update trend analysis
        self.update_trend_analysis(test_name)

        # Save updated baselines
        self.save_baselines()

    def update_trend_analysis(self, test_name: str):
        """Update trend analysis for a test"""
        baseline = self.baselines["baselines"][test_name]
        samples = baseline["samples"]

        if len(samples) < 5:
            baseline["trend"] = "insufficient_data"
            return

        # Get recent values
        recent_values = [s["value"] for s in samples[-10:]]

        if len(recent_values) < 5:
            baseline["trend"] = "insufficient_data"
            return

        # Calculate trend
        x = list(range(len(recent_values)))
        y = recent_values

        # Simple linear regression
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x2 = sum(x[i] ** 2 for i in range(n))

        if n * sum_x2 - sum_x**2 == 0:
            baseline["trend"] = "stable"
            return

        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x**2)

        # Determine trend
        avg_value = statistics.mean(recent_values)
        slope_pct = (slope / avg_value) * 100 if avg_value > 0 else 0

        if abs(slope_pct) < 2:
            baseline["trend"] = "stable"
        elif slope_pct > 5:
            baseline["trend"] = "degrading"
        elif slope_pct < -5:
            baseline["trend"] = "improving"
        else:
            baseline["trend"] = "stable"

    def check_performance(self, test_name: str, value: float) -> Dict[str, Any]:
        """Check performance against baseline"""
        if test_name not in self.baselines["baselines"]:
            return {
                "status": "no_baseline",
                "value": value,
                "message": f"No baseline exists for {test_name}",
            }

        baseline = self.baselines["baselines"][test_name]
        target = (
            baseline["target_ms"]
            if "ms" in test_name
            else baseline.get("target_mb", baseline["target_ms"])
        )
        warning = (
            baseline["warning_ms"]
            if "ms" in test_name
            else baseline.get("warning_mb", baseline["warning_ms"])
        )
        critical = (
            baseline["critical_ms"]
            if "ms" in test_name
            else baseline.get("critical_mb", baseline["critical_ms"])
        )

        if value <= target:
            status = "excellent"
            message = f"Performance excellent: {value:.2f} <= {target:.2f}"
        elif value <= warning:
            status = "good"
            message = f"Performance good: {value:.2f} <= {warning:.2f}"
        elif value <= critical:
            status = "warning"
            message = f"Performance warning: {value:.2f} > {warning:.2f}"
        else:
            status = "critical"
            message = f"Performance critical: {value:.2f} > {critical:.2f}"

        # Check for regression
        regression_info = self.check_regression(test_name, value)

        return {
            "status": status,
            "value": value,
            "target": target,
            "warning": warning,
            "critical": critical,
            "message": message,
            "trend": baseline.get("trend", "unknown"),
            "regression": regression_info,
        }

    def check_regression(self, test_name: str, current_value: float) -> Dict[str, Any]:
        """Check for performance regression"""
        if test_name not in self.baselines["baselines"]:
            return {"detected": False, "reason": "no_baseline"}

        baseline = self.baselines["baselines"][test_name]
        samples = baseline["samples"]

        if len(samples) < self.baselines["regression_thresholds"]["min_samples"]:
            return {"detected": False, "reason": "insufficient_samples"}

        # Calculate historical average
        window = self.baselines["regression_thresholds"]["trend_window"]
        recent_samples = samples[-window:] if len(samples) >= window else samples
        historical_avg = statistics.mean([s["value"] for s in recent_samples])

        # Calculate regression percentage
        if historical_avg > 0:
            regression_pct = ((current_value - historical_avg) / historical_avg) * 100
        else:
            regression_pct = 0

        warning_threshold = self.baselines["regression_thresholds"]["warning_pct"]
        critical_threshold = self.baselines["regression_thresholds"]["critical_pct"]

        if regression_pct > critical_threshold:
            return {
                "detected": True,
                "severity": "critical",
                "percentage": regression_pct,
                "historical_avg": historical_avg,
                "current": current_value,
            }
        elif regression_pct > warning_threshold:
            return {
                "detected": True,
                "severity": "warning",
                "percentage": regression_pct,
                "historical_avg": historical_avg,
                "current": current_value,
            }
        else:
            return {
                "detected": False,
                "percentage": regression_pct,
                "historical_avg": historical_avg,
                "current": current_value,
            }

    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_tests": len(self.baselines["baselines"]),
                "tests_with_data": 0,
                "trending_up": 0,
                "trending_down": 0,
                "stable": 0,
            },
            "tests": {},
        }

        for test_name, baseline in self.baselines["baselines"].items():
            samples = baseline["samples"]

            if samples:
                report["summary"]["tests_with_data"] += 1

                recent_values = [s["value"] for s in samples[-10:]]

                test_report = {
                    "trend": baseline.get("trend", "unknown"),
                    "sample_count": len(samples),
                    "recent_avg": statistics.mean(recent_values) if recent_values else 0,
                    "recent_min": min(recent_values) if recent_values else 0,
                    "recent_max": max(recent_values) if recent_values else 0,
                    "target": baseline.get("target_ms", baseline.get("target_mb", 0)),
                    "warning": baseline.get("warning_ms", baseline.get("warning_mb", 0)),
                    "critical": baseline.get("critical_ms", baseline.get("critical_mb", 0)),
                }

                # Update summary counts
                trend = baseline.get("trend", "unknown")
                if trend == "improving":
                    report["summary"]["trending_down"] += 1
                elif trend == "degrading":
                    report["summary"]["trending_up"] += 1
                else:
                    report["summary"]["stable"] += 1

                report["tests"][test_name] = test_report

        return report

    def export_performance_data(self, output_file: str):
        """Export performance data to CSV for analysis"""
        data = []

        for test_name, baseline in self.baselines["baselines"].items():
            for sample in baseline["samples"]:
                data.append(
                    {
                        "test_name": test_name,
                        "timestamp": sample["timestamp"],
                        "value": sample["value"],
                        "target": baseline.get("target_ms", baseline.get("target_mb", 0)),
                        "warning": baseline.get("warning_ms", baseline.get("warning_mb", 0)),
                        "critical": baseline.get("critical_ms", baseline.get("critical_mb", 0)),
                    }
                )

        if data:
            df = pd.DataFrame(data)
            df.to_csv(output_file, index=False)
            return len(data)

        return 0

    def cleanup_old_data(self, days_to_keep: int = 30):
        """Clean up old performance data"""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        cutoff_iso = cutoff_date.isoformat()

        cleaned_count = 0

        for test_name, baseline in self.baselines["baselines"].items():
            original_count = len(baseline["samples"])

            # Keep only recent samples
            baseline["samples"] = [s for s in baseline["samples"] if s["timestamp"] > cutoff_iso]

            cleaned_count += original_count - len(baseline["samples"])

        if cleaned_count > 0:
            self.save_baselines()

        return cleaned_count


class PerformanceRegressionDetector:
    """Automated performance regression detection"""

    def __init__(self, baseline_manager: PerformanceBaselineManager):
        self.baseline_manager = baseline_manager

    def run_regression_analysis(self) -> Dict[str, Any]:
        """Run comprehensive regression analysis"""
        analysis = {
            "timestamp": datetime.now().isoformat(),
            "regressions_detected": [],
            "warnings": [],
            "improvements": [],
            "summary": {
                "critical_regressions": 0,
                "warning_regressions": 0,
                "improvements": 0,
                "stable_tests": 0,
            },
        }

        for test_name, baseline in self.baseline_manager.baselines["baselines"].items():
            if len(baseline["samples"]) < 5:
                continue

            # Get recent performance
            recent_values = [s["value"] for s in baseline["samples"][-5:]]
            current_avg = statistics.mean(recent_values)

            # Check regression
            regression_info = self.baseline_manager.check_regression(test_name, current_avg)

            if regression_info["detected"]:
                regression_data = {
                    "test_name": test_name,
                    "severity": regression_info["severity"],
                    "percentage": regression_info["percentage"],
                    "current_avg": current_avg,
                    "historical_avg": regression_info["historical_avg"],
                    "trend": baseline.get("trend", "unknown"),
                }

                if regression_info["severity"] == "critical":
                    analysis["regressions_detected"].append(regression_data)
                    analysis["summary"]["critical_regressions"] += 1
                else:
                    analysis["warnings"].append(regression_data)
                    analysis["summary"]["warning_regressions"] += 1

            elif regression_info["percentage"] < -10:  # Significant improvement
                analysis["improvements"].append(
                    {
                        "test_name": test_name,
                        "improvement_pct": abs(regression_info["percentage"]),
                        "current_avg": current_avg,
                        "historical_avg": regression_info["historical_avg"],
                    }
                )
                analysis["summary"]["improvements"] += 1

            else:
                analysis["summary"]["stable_tests"] += 1

        return analysis

    def generate_regression_report(self, analysis: Dict[str, Any]) -> str:
        """Generate human-readable regression report"""
        report = []
        report.append("Performance Regression Analysis Report")
        report.append("=" * 50)
        report.append(f"Generated: {analysis['timestamp']}")
        report.append("")

        # Summary
        summary = analysis["summary"]
        report.append("Summary:")
        report.append(f"  Critical Regressions: {summary['critical_regressions']}")
        report.append(f"  Warning Regressions:  {summary['warning_regressions']}")
        report.append(f"  Improvements:         {summary['improvements']}")
        report.append(f"  Stable Tests:         {summary['stable_tests']}")
        report.append("")

        # Critical regressions
        if analysis["regressions_detected"]:
            report.append("🚨 CRITICAL REGRESSIONS:")
            for reg in analysis["regressions_detected"]:
                report.append(
                    f"  - {reg['test_name']}: {reg['percentage']:+.1f}% "
                    f"({reg['historical_avg']:.2f} → {reg['current_avg']:.2f})"
                )
            report.append("")

        # Warnings
        if analysis["warnings"]:
            report.append("⚠️  WARNING REGRESSIONS:")
            for warn in analysis["warnings"]:
                report.append(
                    f"  - {warn['test_name']}: {warn['percentage']:+.1f}% "
                    f"({warn['historical_avg']:.2f} → {warn['current_avg']:.2f})"
                )
            report.append("")

        # Improvements
        if analysis["improvements"]:
            report.append("✅ IMPROVEMENTS:")
            for imp in analysis["improvements"]:
                report.append(
                    f"  - {imp['test_name']}: {imp['improvement_pct']:.1f}% faster "
                    f"({imp['historical_avg']:.2f} → {imp['current_avg']:.2f})"
                )
            report.append("")

        return "\n".join(report)


def main():
    """Main function for command-line usage"""
    import argparse

    parser = argparse.ArgumentParser(description="Performance Baseline Manager")
    parser.add_argument("--report", action="store_true", help="Generate performance report")
    parser.add_argument("--regression-check", action="store_true", help="Run regression analysis")
    parser.add_argument("--export", type=str, help="Export data to CSV file")
    parser.add_argument("--cleanup", type=int, help="Clean up data older than N days")

    args = parser.parse_args()

    manager = PerformanceBaselineManager()

    if args.report:
        report = manager.get_performance_report()
        print(json.dumps(report, indent=2))

    if args.regression_check:
        detector = PerformanceRegressionDetector(manager)
        analysis = detector.run_regression_analysis()
        report = detector.generate_regression_report(analysis)
        print(report)

    if args.export:
        count = manager.export_performance_data(args.export)
        print(f"Exported {count} performance records to {args.export}")

    if args.cleanup:
        count = manager.cleanup_old_data(args.cleanup)
        print(f"Cleaned up {count} old performance records")


if __name__ == "__main__":
    main()
