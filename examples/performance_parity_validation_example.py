#!/usr/bin/env python3
"""
Performance Parity Validation Example

This example demonstrates how to use the performance parity validation system
to compare two trading strategies and ensure they maintain equivalent performance.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Any, Dict

from src.strategies.components.testing import (
    PerformanceParityValidator,
    ToleranceConfig,
    ValidationResult,
    quick_strategy_comparison,
    validate_migration_readiness,
)


def create_sample_backtest_results(
    initial_balance: float = 10000.0,
    days: int = 365,
    daily_return_mean: float = 0.001,
    daily_return_std: float = 0.02,
    seed: int = 42
) -> pd.DataFrame:
    """Create sample backtest results for demonstration."""
    
    np.random.seed(seed)
    
    # Generate dates
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(days)]
    
    # Generate daily returns
    daily_returns = np.random.normal(daily_return_mean, daily_return_std, days)
    
    # Calculate cumulative balance
    balance = initial_balance * np.cumprod(1 + daily_returns)
    
    # Generate some trade PnL data
    trade_pnl = np.random.normal(50, 100, days)
    
    return pd.DataFrame({
        'timestamp': dates,
        'balance': balance,
        'trade_pnl': trade_pnl
    })


def example_basic_validation():
    """Example of basic performance parity validation."""
    
    print("=" * 80)
    print("BASIC PERFORMANCE PARITY VALIDATION EXAMPLE")
    print("=" * 80)
    
    # Create sample data for two similar strategies
    print("Creating sample backtest data...")
    legacy_results = create_sample_backtest_results(
        daily_return_mean=0.001,
        daily_return_std=0.02,
        seed=42
    )
    
    new_results = create_sample_backtest_results(
        daily_return_mean=0.0012,  # Slightly different performance
        daily_return_std=0.021,
        seed=43
    )
    
    print(f"Legacy strategy final balance: ${legacy_results['balance'].iloc[-1]:,.2f}")
    print(f"New strategy final balance: ${new_results['balance'].iloc[-1]:,.2f}")
    
    # Create validator with default tolerances
    validator = PerformanceParityValidator()
    
    print("\nRunning performance parity validation...")
    report = validator.validate_performance_parity(
        legacy_results=legacy_results,
        new_results=new_results,
        strategy_name="ML Basic Strategy",
        legacy_strategy_id="ml_basic_v1",
        new_strategy_id="ml_basic_v2",
        comparison_period="2023_full_year"
    )
    
    # Display results
    print(f"\nValidation Result: {report.overall_result.value.upper()}")
    print(f"Certified: {'YES' if report.certified else 'NO'}")
    print(f"Total Metrics Tested: {report.total_metrics_tested}")
    print(f"Metrics Passed: {report.metrics_passed}")
    print(f"Metrics Failed: {report.metrics_failed}")
    print(f"Equity Curve Correlation: {report.equity_curve_correlation:.4f}")
    
    print("\nDetailed Metric Results:")
    print("-" * 60)
    for comparison in report.metric_comparisons:
        status_symbol = "✓" if comparison.result == ValidationResult.PASS else "✗"
        print(f"{status_symbol} {comparison.metric_name}: "
              f"Legacy={comparison.legacy_value:.6f}, "
              f"New={comparison.new_value:.6f}, "
              f"Diff={comparison.difference:.6f}")
    
    # Generate certification report
    print("\n" + "=" * 80)
    print("CERTIFICATION REPORT")
    print("=" * 80)
    cert_report = validator.generate_certification_report(report)
    print(cert_report)
    
    return report


def example_custom_tolerances():
    """Example with custom tolerance configuration."""
    
    print("\n" + "=" * 80)
    print("CUSTOM TOLERANCE VALIDATION EXAMPLE")
    print("=" * 80)
    
    # Create data with larger differences
    legacy_results = create_sample_backtest_results(
        daily_return_mean=0.001,
        daily_return_std=0.02,
        seed=42
    )
    
    new_results = create_sample_backtest_results(
        daily_return_mean=0.004,  # Much higher return
        daily_return_std=0.025,   # Higher volatility
        seed=43
    )
    
    print(f"Legacy strategy final balance: ${legacy_results['balance'].iloc[-1]:,.2f}")
    print(f"New strategy final balance: ${new_results['balance'].iloc[-1]:,.2f}")
    
    # Create custom tolerance configuration (more lenient)
    custom_tolerances = ToleranceConfig(
        total_return_tolerance=0.10,  # Allow 10% difference in total return
        sharpe_ratio_tolerance=0.5,   # Allow 0.5 difference in Sharpe ratio
        minimum_correlation=0.85,     # Require 85% correlation (vs default 95%)
        volatility_tolerance=0.20     # Allow 20% difference in volatility
    )
    
    validator = PerformanceParityValidator(custom_tolerances)
    
    print("\nRunning validation with custom (lenient) tolerances...")
    report = validator.validate_performance_parity(
        legacy_results=legacy_results,
        new_results=new_results,
        strategy_name="High Performance Strategy",
        legacy_strategy_id="strategy_v1",
        new_strategy_id="strategy_v2"
    )
    
    print(f"\nValidation Result: {report.overall_result.value.upper()}")
    print(f"With lenient tolerances, validation {'PASSED' if report.certified else 'FAILED'}")
    
    # Now try with strict tolerances
    strict_tolerances = ToleranceConfig(
        total_return_tolerance=0.01,  # Allow only 1% difference
        sharpe_ratio_tolerance=0.05,  # Allow only 0.05 difference in Sharpe
        minimum_correlation=0.98      # Require 98% correlation
    )
    
    strict_validator = PerformanceParityValidator(strict_tolerances)
    
    print("\nRunning validation with strict tolerances...")
    strict_report = strict_validator.validate_performance_parity(
        legacy_results=legacy_results,
        new_results=new_results,
        strategy_name="High Performance Strategy",
        legacy_strategy_id="strategy_v1",
        new_strategy_id="strategy_v2"
    )
    
    print(f"Validation Result: {strict_report.overall_result.value.upper()}")
    print(f"With strict tolerances, validation {'PASSED' if strict_report.certified else 'FAILED'}")
    
    return report, strict_report


def example_statistical_analysis():
    """Example of statistical analysis capabilities."""
    
    print("\n" + "=" * 80)
    print("STATISTICAL ANALYSIS EXAMPLE")
    print("=" * 80)
    
    from src.strategies.components.testing.statistical_tests import (
        FinancialStatisticalTests,
        EquivalenceTests,
        format_test_results
    )
    
    # Create two return series
    np.random.seed(42)
    returns1 = pd.Series(np.random.normal(0.001, 0.02, 252))  # 1 year of daily returns
    returns2 = pd.Series(np.random.normal(0.0012, 0.021, 252))  # Slightly different
    
    print("Performing comprehensive statistical analysis...")
    
    # Initialize test engines
    stat_tests = FinancialStatisticalTests(significance_level=0.05)
    equiv_tests = EquivalenceTests(equivalence_margin=0.01)
    
    # Run comprehensive comparison
    results = stat_tests.comprehensive_comparison(
        returns1, returns2, "Legacy Strategy", "New Strategy"
    )
    
    # Run equivalence test
    tost_result = equiv_tests.two_one_sided_test(returns1, returns2)
    
    # Display results
    print("\nStatistical Test Results:")
    print(format_test_results(results))
    
    print("\nEquivalence Test Result:")
    print(f"Test: {tost_result.test_name}")
    print(f"Statistic: {tost_result.statistic:.6f}")
    print(f"P-value: {tost_result.p_value:.6f}")
    print(f"Conclusion: {tost_result.interpretation}")
    
    return results, tost_result


def example_export_functionality():
    """Example of exporting validation results."""
    
    print("\n" + "=" * 80)
    print("EXPORT FUNCTIONALITY EXAMPLE")
    print("=" * 80)
    
    from src.strategies.components.testing.performance_parity_validator import (
        PerformanceParityReporter
    )
    import tempfile
    import os
    
    # Create sample validation report
    legacy_results = create_sample_backtest_results(seed=42)
    new_results = create_sample_backtest_results(seed=43)
    
    validator = PerformanceParityValidator()
    report = validator.validate_performance_parity(
        legacy_results, new_results,
        "Export Example Strategy", "legacy", "new"
    )
    
    # Create temporary directory for exports
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Exporting results to: {temp_dir}")
        
        # Export to CSV
        csv_path = os.path.join(temp_dir, "validation_metrics.csv")
        PerformanceParityReporter.export_to_csv(report, csv_path)
        print(f"✓ Exported metrics to CSV: {csv_path}")
        
        # Export to JSON
        json_path = os.path.join(temp_dir, "validation_report.json")
        PerformanceParityReporter.export_to_json(report, json_path)
        print(f"✓ Exported full report to JSON: {json_path}")
        
        # Show file sizes
        csv_size = os.path.getsize(csv_path)
        json_size = os.path.getsize(json_path)
        print(f"CSV file size: {csv_size} bytes")
        print(f"JSON file size: {json_size} bytes")
        
        # Read back and display sample
        import json
        with open(json_path, 'r') as f:
            exported_data = json.load(f)
        
        print(f"\nExported report contains {len(exported_data['metric_comparisons'])} metric comparisons")
        print(f"Overall result: {exported_data['overall_result']}")
    
    return report


def main():
    """Run all examples."""
    
    print("Performance Parity Validation System Examples")
    print("=" * 80)
    print("This example demonstrates the comprehensive performance validation")
    print("system for comparing trading strategies during migration.")
    print()
    
    try:
        # Run examples
        basic_report = example_basic_validation()
        custom_report, strict_report = example_custom_tolerances()
        stat_results, equiv_result = example_statistical_analysis()
        export_report = example_export_functionality()
        
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print("All examples completed successfully!")
        print()
        print("Key Features Demonstrated:")
        print("✓ Basic performance parity validation")
        print("✓ Custom tolerance configuration")
        print("✓ Comprehensive statistical testing")
        print("✓ Equivalence testing (TOST)")
        print("✓ Export functionality (CSV/JSON)")
        print("✓ Certification reporting")
        print()
        print("The performance parity validation system provides:")
        print("• Comprehensive metric comparison")
        print("• Statistical significance testing")
        print("• Configurable tolerance levels")
        print("• Detailed reporting and certification")
        print("• Export capabilities for further analysis")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()