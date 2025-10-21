#!/usr/bin/env python3
"""
Automated Performance Monitor

Runs performance tests automatically and tracks regression trends.
Can be integrated into CI/CD pipeline for continuous performance monitoring.
"""

import subprocess
import sys
import time
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

from performance_baseline_manager import PerformanceBaselineManager, PerformanceRegressionDetector


class AutomatedPerformanceMonitor:
    """Automated performance monitoring system"""
    
    def __init__(self, baseline_file: str = "tests/performance/component_baselines.json"):
        self.baseline_manager = PerformanceBaselineManager(baseline_file)
        self.regression_detector = PerformanceRegressionDetector(self.baseline_manager)
        self.results_dir = Path("tests/performance/results")
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def run_performance_tests(self, test_pattern: str = "test_component_performance_regression.py") -> Dict[str, Any]:
        """Run performance tests and collect results"""
        print(f"🚀 Running performance tests: {test_pattern}")
        
        # Run pytest with performance markers
        cmd = [
            sys.executable, "-m", "pytest",
            f"tests/performance/{test_pattern}",
            "-v", "--tb=short",
            "-m", "performance",
            "--json-report",
            f"--json-report-file={self.results_dir}/latest_results.json"
        ]
        
        start_time = time.time()
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            execution_time = time.time() - start_time
            
            # Parse results
            test_results = self.parse_test_results(result)
            
            return {
                "timestamp": datetime.now().isoformat(),
                "execution_time": execution_time,
                "return_code": result.returncode,
                "tests_passed": test_results["passed"],
                "tests_failed": test_results["failed"],
                "tests_total": test_results["total"],
                "stdout": result.stdout,
                "stderr": result.stderr,
                "performance_data": self.extract_performance_data(result.stdout)
            }
        
        except Exception as e:
            return {
                "timestamp": datetime.now().isoformat(),
                "execution_time": time.time() - start_time,
                "error": str(e),
                "return_code": -1
            }
    
    def parse_test_results(self, result: subprocess.CompletedProcess) -> Dict[str, int]:
        """Parse pytest results"""
        output = result.stdout
        
        # Extract test counts from pytest output
        passed = output.count(" PASSED")
        failed = output.count(" FAILED")
        skipped = output.count(" SKIPPED")
        
        return {
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
            "total": passed + failed + skipped
        }
    
    def extract_performance_data(self, output: str) -> Dict[str, float]:
        """Extract performance measurements from test output"""
        performance_data = {}
        
        lines = output.split('\n')
        for line in lines:
            # Look for performance measurement patterns
            if "Avg:" in line and "ms" in line:
                # Parse lines like "MLBasicSignalGenerator - Avg: 2.34ms, Max: 5.67ms, Status: excellent"
                parts = line.split(" - ")
                if len(parts) >= 2:
                    test_name = parts[0].strip()
                    metrics_part = parts[1]
                    
                    # Extract average time
                    if "Avg:" in metrics_part:
                        try:
                            avg_start = metrics_part.find("Avg:") + 4
                            avg_end = metrics_part.find("ms", avg_start)
                            avg_time = float(metrics_part[avg_start:avg_end].strip())
                            performance_data[f"{test_name}_avg_ms"] = avg_time
                        except (ValueError, IndexError):
                            pass
                    
                    # Extract max time
                    if "Max:" in metrics_part:
                        try:
                            max_start = metrics_part.find("Max:") + 4
                            max_end = metrics_part.find("ms", max_start)
                            max_time = float(metrics_part[max_start:max_end].strip())
                            performance_data[f"{test_name}_max_ms"] = max_time
                        except (ValueError, IndexError):
                            pass
            
            # Look for other performance patterns
            elif "Total:" in line and "ms" in line:
                # Parse batch processing results
                try:
                    total_start = line.find("Total:") + 6
                    total_end = line.find("ms", total_start)
                    total_time = float(line[total_start:total_end].strip())
                    performance_data["batch_processing_total_ms"] = total_time
                except (ValueError, IndexError):
                    pass
            
            elif "Memory Usage" in line and "MB" in line:
                # Parse memory usage results
                try:
                    increase_start = line.find("Increase:") + 9
                    increase_end = line.find("MB", increase_start)
                    memory_increase = float(line[increase_start:increase_end].strip())
                    performance_data["memory_increase_mb"] = memory_increase
                except (ValueError, IndexError):
                    pass
        
        return performance_data
    
    def record_performance_measurements(self, performance_data: Dict[str, float]):
        """Record performance measurements in baseline manager"""
        for test_name, value in performance_data.items():
            # Map test names to baseline categories
            if "signal" in test_name.lower() and "avg_ms" in test_name:
                self.baseline_manager.record_measurement("signal_generation", value)
            elif "risk" in test_name.lower() and "avg_ms" in test_name:
                self.baseline_manager.record_measurement("risk_calculation", value)
            elif "position" in test_name.lower() and "avg_ms" in test_name:
                self.baseline_manager.record_measurement("position_sizing", value)
            elif "decision" in test_name.lower() and "avg_ms" in test_name:
                self.baseline_manager.record_measurement("complete_decision", value)
            elif "batch" in test_name.lower() and "total_ms" in test_name:
                self.baseline_manager.record_measurement("batch_processing_100", value)
            elif "memory" in test_name.lower() and "mb" in test_name:
                self.baseline_manager.record_measurement("memory_usage", value)
    
    def run_regression_analysis(self) -> Dict[str, Any]:
        """Run regression analysis and generate report"""
        print("🔍 Running regression analysis...")
        
        analysis = self.regression_detector.run_regression_analysis()
        report = self.regression_detector.generate_regression_report(analysis)
        
        # Save analysis results
        analysis_file = self.results_dir / f"regression_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        # Save report
        report_file = self.results_dir / f"regression_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        
        return analysis
    
    def generate_performance_dashboard(self) -> str:
        """Generate performance dashboard HTML"""
        report = self.baseline_manager.get_performance_report()
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Component Performance Dashboard</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .summary {{ display: flex; gap: 20px; margin: 20px 0; }}
                .metric {{ background-color: #e8f4f8; padding: 15px; border-radius: 5px; flex: 1; }}
                .test-results {{ margin: 20px 0; }}
                .test-item {{ margin: 10px 0; padding: 10px; border-left: 4px solid #ccc; }}
                .excellent {{ border-left-color: #4CAF50; }}
                .good {{ border-left-color: #8BC34A; }}
                .warning {{ border-left-color: #FF9800; }}
                .critical {{ border-left-color: #F44336; }}
                .timestamp {{ color: #666; font-size: 0.9em; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Component Performance Dashboard</h1>
                <p class="timestamp">Generated: {report['timestamp']}</p>
            </div>
            
            <div class="summary">
                <div class="metric">
                    <h3>Total Tests</h3>
                    <p>{report['summary']['total_tests']}</p>
                </div>
                <div class="metric">
                    <h3>Tests with Data</h3>
                    <p>{report['summary']['tests_with_data']}</p>
                </div>
                <div class="metric">
                    <h3>Stable</h3>
                    <p>{report['summary']['stable']}</p>
                </div>
                <div class="metric">
                    <h3>Improving</h3>
                    <p>{report['summary']['trending_down']}</p>
                </div>
                <div class="metric">
                    <h3>Degrading</h3>
                    <p>{report['summary']['trending_up']}</p>
                </div>
            </div>
            
            <div class="test-results">
                <h2>Test Results</h2>
        """
        
        for test_name, test_data in report['tests'].items():
            trend_icon = {
                'improving': '📈',
                'degrading': '📉',
                'stable': '➡️'
            }.get(test_data['trend'], '❓')
            
            # Determine status class
            if test_data['recent_avg'] <= test_data['target']:
                status_class = 'excellent'
            elif test_data['recent_avg'] <= test_data['warning']:
                status_class = 'good'
            elif test_data['recent_avg'] <= test_data['critical']:
                status_class = 'warning'
            else:
                status_class = 'critical'
            
            html += f"""
                <div class="test-item {status_class}">
                    <h3>{trend_icon} {test_name}</h3>
                    <p>Recent Average: {test_data['recent_avg']:.2f} (Target: {test_data['target']:.2f})</p>
                    <p>Range: {test_data['recent_min']:.2f} - {test_data['recent_max']:.2f}</p>
                    <p>Samples: {test_data['sample_count']}, Trend: {test_data['trend']}</p>
                </div>
            """
        
        html += """
            </div>
        </body>
        </html>
        """
        
        # Save dashboard
        dashboard_file = self.results_dir / "performance_dashboard.html"
        with open(dashboard_file, 'w') as f:
            f.write(html)
        
        return str(dashboard_file)
    
    def run_full_monitoring_cycle(self) -> Dict[str, Any]:
        """Run complete monitoring cycle"""
        print("🔄 Starting automated performance monitoring cycle...")
        
        cycle_results = {
            "start_time": datetime.now().isoformat(),
            "steps": []
        }
        
        # Step 1: Run performance tests
        print("\n📊 Step 1: Running performance tests")
        test_results = self.run_performance_tests()
        cycle_results["steps"].append({
            "step": "performance_tests",
            "status": "success" if test_results["return_code"] == 0 else "failed",
            "details": test_results
        })
        
        if test_results["return_code"] != 0:
            print(f"❌ Performance tests failed with return code {test_results['return_code']}")
            print("STDERR:", test_results.get("stderr", ""))
        else:
            print(f"✅ Performance tests completed: {test_results['tests_passed']} passed, {test_results['tests_failed']} failed")
        
        # Step 2: Record measurements
        if "performance_data" in test_results and test_results["performance_data"]:
            print("\n📝 Step 2: Recording performance measurements")
            self.record_performance_measurements(test_results["performance_data"])
            cycle_results["steps"].append({
                "step": "record_measurements",
                "status": "success",
                "measurements_count": len(test_results["performance_data"])
            })
            print(f"✅ Recorded {len(test_results['performance_data'])} performance measurements")
        else:
            print("\n⚠️  Step 2: No performance data to record")
            cycle_results["steps"].append({
                "step": "record_measurements",
                "status": "skipped",
                "reason": "no_performance_data"
            })
        
        # Step 3: Run regression analysis
        print("\n🔍 Step 3: Running regression analysis")
        try:
            regression_analysis = self.run_regression_analysis()
            cycle_results["steps"].append({
                "step": "regression_analysis",
                "status": "success",
                "critical_regressions": regression_analysis["summary"]["critical_regressions"],
                "warning_regressions": regression_analysis["summary"]["warning_regressions"],
                "improvements": regression_analysis["summary"]["improvements"]
            })
            
            if regression_analysis["summary"]["critical_regressions"] > 0:
                print(f"🚨 Found {regression_analysis['summary']['critical_regressions']} critical regressions!")
            elif regression_analysis["summary"]["warning_regressions"] > 0:
                print(f"⚠️  Found {regression_analysis['summary']['warning_regressions']} warning regressions")
            else:
                print("✅ No significant regressions detected")
            
            if regression_analysis["summary"]["improvements"] > 0:
                print(f"🎉 Found {regression_analysis['summary']['improvements']} performance improvements!")
        
        except Exception as e:
            print(f"❌ Regression analysis failed: {e}")
            cycle_results["steps"].append({
                "step": "regression_analysis",
                "status": "failed",
                "error": str(e)
            })
        
        # Step 4: Generate dashboard
        print("\n📊 Step 4: Generating performance dashboard")
        try:
            dashboard_file = self.generate_performance_dashboard()
            cycle_results["steps"].append({
                "step": "generate_dashboard",
                "status": "success",
                "dashboard_file": dashboard_file
            })
            print(f"✅ Dashboard generated: {dashboard_file}")
        except Exception as e:
            print(f"❌ Dashboard generation failed: {e}")
            cycle_results["steps"].append({
                "step": "generate_dashboard",
                "status": "failed",
                "error": str(e)
            })
        
        cycle_results["end_time"] = datetime.now().isoformat()
        cycle_results["success"] = all(
            step["status"] in ["success", "skipped"] 
            for step in cycle_results["steps"]
        )
        
        # Save cycle results
        cycle_file = self.results_dir / f"monitoring_cycle_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(cycle_file, 'w') as f:
            json.dump(cycle_results, f, indent=2)
        
        print(f"\n🏁 Monitoring cycle completed. Results saved to: {cycle_file}")
        
        return cycle_results


def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Automated Performance Monitor")
    parser.add_argument("--test-pattern", default="test_component_performance_regression.py",
                       help="Test file pattern to run")
    parser.add_argument("--baseline-file", default="tests/performance/component_baselines.json",
                       help="Baseline file path")
    parser.add_argument("--full-cycle", action="store_true",
                       help="Run full monitoring cycle")
    parser.add_argument("--tests-only", action="store_true",
                       help="Run tests only")
    parser.add_argument("--analysis-only", action="store_true",
                       help="Run regression analysis only")
    parser.add_argument("--dashboard-only", action="store_true",
                       help="Generate dashboard only")
    
    args = parser.parse_args()
    
    monitor = AutomatedPerformanceMonitor(args.baseline_file)
    
    if args.full_cycle:
        results = monitor.run_full_monitoring_cycle()
        sys.exit(0 if results["success"] else 1)
    
    elif args.tests_only:
        results = monitor.run_performance_tests(args.test_pattern)
        sys.exit(results["return_code"])
    
    elif args.analysis_only:
        analysis = monitor.run_regression_analysis()
        critical_count = analysis["summary"]["critical_regressions"]
        sys.exit(1 if critical_count > 0 else 0)
    
    elif args.dashboard_only:
        dashboard_file = monitor.generate_performance_dashboard()
        print(f"Dashboard generated: {dashboard_file}")
        sys.exit(0)
    
    else:
        # Default: run full cycle
        results = monitor.run_full_monitoring_cycle()
        sys.exit(0 if results["success"] else 1)


if __name__ == "__main__":
    main()