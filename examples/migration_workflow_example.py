"""
Complete Migration Workflow Example

DEPRECATED: This example is no longer functional as the migration has been completed
and all legacy code has been removed. This file is kept for historical reference only.

This example demonstrated the complete workflow for migrating legacy strategies
to the new component-based architecture, including conversion, validation,
cross-validation testing, and rollback capabilities.

The migration has been completed and the following modules have been removed:
- src.strategies.base (BaseStrategy)
- src.strategies.adapters (LegacyStrategyAdapter, adapter_factory)
- src.strategies.migration (all migration utilities)

For current strategy development, see:
- src/strategies/README.md - Component-based strategy guide
- src/strategies/components/ - Strategy components
- examples/component_testing_example.py - Component testing examples
"""

# This file is deprecated and will not run
import sys
sys.exit("This migration example is deprecated. The migration has been completed and legacy code has been removed.")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_data(length: int = 500) -> pd.DataFrame:
    """Create sample market data for testing"""
    import numpy as np
    
    np.random.seed(42)
    
    # Generate synthetic OHLCV data
    dates = pd.date_range(start='2023-01-01', periods=length, freq='1H')
    
    base_price = 100.0
    prices = [base_price]
    
    for i in range(1, length):
        change = np.random.normal(0, 0.02)  # 2% volatility
        new_price = prices[-1] * (1 + change)
        new_price = max(new_price, 1.0)  # Ensure positive prices
        prices.append(new_price)
    
    # Create OHLCV data
    data = []
    for i, price in enumerate(prices):
        high = price * (1 + abs(np.random.normal(0, 0.01)))
        low = price * (1 - abs(np.random.normal(0, 0.01)))
        open_price = prices[i-1] if i > 0 else price
        volume = np.random.uniform(1000, 10000)
        
        data.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': price,
            'volume': volume
        })
    
    df = pd.DataFrame(data)
    df.index = dates
    
    return df


def demonstrate_complete_migration_workflow():
    """Demonstrate the complete migration workflow"""
    
    print("=" * 80)
    print("COMPLETE STRATEGY MIGRATION WORKFLOW DEMONSTRATION")
    print("=" * 80)
    
    # Initialize components
    converter = StrategyConverter()
    cross_validator = CrossValidationTester()
    regression_tester = RegressionTester()
    difference_analyzer = DifferenceAnalyzer()
    rollback_manager = RollbackManager()
    rollback_validator = RollbackValidator()
    audit_manager = AuditTrailManager()
    
    # Create sample data
    print("\n1. Creating sample market data...")
    test_data = create_sample_data(500)
    print(f"   Created {len(test_data)} rows of sample data")
    
    # Create legacy strategy
    print("\n2. Creating legacy strategy...")
    try:
        legacy_strategy = MlBasic(name="TestMlBasic")
        print(f"   Created legacy strategy: {legacy_strategy.name}")
    except Exception as e:
        print(f"   Warning: Could not create MlBasic strategy: {e}")
        print("   This is expected if ONNX model files are not available")
        print("   Continuing with demonstration using mock strategy...")
        
        # Create a mock strategy for demonstration
        from src.strategies.base import BaseStrategy
        
        class MockStrategy(BaseStrategy):
            def __init__(self, name="MockStrategy"):
                super().__init__(name)
                self.model_path = "mock_model.onnx"
                self.sequence_length = 120
                self.stop_loss_pct = 0.02
                
            def calculate_indicators(self, df):
                return df.copy()
            
            def check_entry_conditions(self, df, index):
                return index % 10 == 0  # Mock entry every 10 periods
            
            def check_exit_conditions(self, df, index, entry_price):
                return index % 15 == 0  # Mock exit every 15 periods
            
            def calculate_position_size(self, df, index, balance):
                return balance * 0.1
            
            def calculate_stop_loss(self, df, index, price, side):
                return price * 0.95 if side == 'long' else price * 1.05
            
            def get_parameters(self):
                return {
                    'model_path': self.model_path,
                    'sequence_length': self.sequence_length,
                    'stop_loss_pct': self.stop_loss_pct
                }
        
        legacy_strategy = MockStrategy()
        print(f"   Created mock legacy strategy: {legacy_strategy.name}")
    
    # Start audit session
    print("\n3. Starting audit session...")
    session_id = audit_manager.start_migration_session("demo_user", "demonstration")
    print(f"   Started audit session: {session_id}")
    
    # Create rollback point
    print("\n4. Creating rollback point...")
    rollback_id = rollback_manager.create_rollback_point(
        strategy_name=legacy_strategy.name,
        legacy_strategy=legacy_strategy,
        migration_metadata={
            'migration_type': 'demonstration',
            'user': 'demo_user',
            'reason': 'workflow_demonstration'
        }
    )
    print(f"   Created rollback point: {rollback_id}")
    
    # Convert strategy
    print("\n5. Converting strategy to component-based architecture...")
    try:
        converted_strategy, conversion_report = converter.convert_strategy(
            legacy_strategy,
            target_name=f"converted_{legacy_strategy.name}",
            validate_conversion=True
        )
        
        print(f"   Conversion successful: {conversion_report.success}")
        print(f"   Target components: {conversion_report.target_components}")
        
        if conversion_report.warnings:
            print(f"   Warnings: {len(conversion_report.warnings)}")
        
        if conversion_report.errors:
            print(f"   Errors: {len(conversion_report.errors)}")
        
        # Log conversion to audit trail
        audit_manager.log_conversion(conversion_report, "demo_user")
        
    except Exception as e:
        print(f"   Conversion failed: {e}")
        print("   This is expected in the demonstration environment")
        converted_strategy = None
        conversion_report = None
    
    if converted_strategy:
        # Perform cross-validation
        print("\n6. Performing cross-validation testing...")
        try:
            cross_validation_report = cross_validator.run_cross_validation(
                legacy_strategy=legacy_strategy,
                converted_strategy=converted_strategy,
                test_data=test_data,
                test_balance=10000.0
            )
            
            print(f"   Cross-validation compatibility: {cross_validation_report.overall_compatibility:.1f}%")
            print(f"   Total comparisons: {len(cross_validation_report.comparison_results)}")
            
            successful_comparisons = sum(
                1 for result in cross_validation_report.comparison_results 
                if result.within_tolerance
            )
            print(f"   Successful comparisons: {successful_comparisons}")
            
        except Exception as e:
            print(f"   Cross-validation failed: {e}")
            cross_validation_report = None
        
        # Perform regression testing
        print("\n7. Running regression tests...")
        try:
            regression_results = regression_tester.run_regression_tests(
                legacy_strategy=legacy_strategy,
                converted_strategy=converted_strategy,
                test_suite_name="basic"
            )
            
            print(f"   Regression tests passed: {regression_results['passed_tests']}/{regression_results['total_tests']}")
            print(f"   Success rate: {regression_results['success_rate']:.1f}%")
            print(f"   Critical failures: {regression_results['critical_failures']}")
            
        except Exception as e:
            print(f"   Regression testing failed: {e}")
            regression_results = None
        
        # Perform difference analysis
        print("\n8. Analyzing differences...")
        try:
            difference_report = difference_analyzer.analyze_differences(
                legacy_strategy=legacy_strategy,
                converted_strategy=converted_strategy,
                test_data=test_data,
                sample_size=50
            )
            
            print(f"   Difference metrics analyzed: {len(difference_report.difference_metrics)}")
            print(f"   Severity breakdown: {difference_report.severity_breakdown}")
            
            critical_differences = sum(
                1 for metric in difference_report.difference_metrics 
                if metric.severity == 'critical'
            )
            print(f"   Critical differences: {critical_differences}")
            
        except Exception as e:
            print(f"   Difference analysis failed: {e}")
            difference_report = None
        
        # Validate rollback
        print("\n9. Validating rollback capabilities...")
        try:
            rollback_point = rollback_manager.rollback_points[rollback_id]
            rollback_validation = rollback_validator.validate_rollback_safety(
                rollback_point=rollback_point,
                current_strategy=converted_strategy
            )
            
            print(f"   Rollback validation status: {rollback_validation.overall_status}")
            print(f"   Safety checks: {rollback_validation.safety_checks['overall_safety']}")
            print(f"   Risk level: {rollback_validation.risk_assessment['overall_risk_level']}")
            print(f"   Recommendations: {len(rollback_validation.recommendations)}")
            
        except Exception as e:
            print(f"   Rollback validation failed: {e}")
            rollback_validation = None
    
    else:
        print("\n6-9. Skipping validation steps due to conversion failure")
    
    # Generate reports
    print("\n10. Generating comprehensive reports...")
    
    # Conversion summary
    conversion_summary = converter.generate_conversion_summary()
    print(f"    Conversion Summary:")
    print(f"      Total conversions: {conversion_summary['total_conversions']}")
    print(f"      Success rate: {conversion_summary['success_rate']:.1f}%")
    
    # Cross-validation summary
    cross_validation_summary = cross_validator.generate_test_summary()
    print(f"    Cross-Validation Summary:")
    print(f"      Total tests: {cross_validation_summary['total_tests']}")
    print(f"      Average compatibility: {cross_validation_summary['average_compatibility']:.1f}%")
    
    # Rollback summary
    rollback_report = rollback_manager.generate_rollback_report()
    print(f"    Rollback Summary:")
    print(f"      Total rollback points: {rollback_report['total_rollback_points']}")
    print(f"      Backup directory size: {rollback_report['backup_directory_size']['total_size_mb']:.1f} MB")
    
    # Audit summary
    audit_stats = audit_manager.get_audit_statistics()
    print(f"    Audit Summary:")
    print(f"      Total events: {audit_stats['total_events']}")
    print(f"      Success rate: {audit_stats['success_rate']:.1f}%")
    
    # End audit session
    print("\n11. Ending audit session...")
    audit_manager.end_migration_session(session_id)
    print(f"    Audit session ended: {session_id}")
    
    # Demonstrate emergency rollback procedure
    print("\n12. Demonstrating emergency rollback procedure...")
    try:
        emergency_validation = rollback_validator.validate_emergency_rollback(
            strategy_name=legacy_strategy.name,
            trigger_condition="performance_degradation"
        )
        
        print(f"    Emergency procedure found: {emergency_validation['emergency_procedure_found']}")
        print(f"    Risk level: {emergency_validation['risk_level']}")
        print(f"    Estimated recovery time: {emergency_validation['estimated_recovery_time']} minutes")
        
    except Exception as e:
        print(f"    Emergency rollback validation failed: {e}")
    
    # Final recommendations
    print("\n13. Final Migration Recommendations:")
    
    if converted_strategy and conversion_report and conversion_report.success:
        print("    ✓ Strategy conversion completed successfully")
        
        if cross_validation_report and cross_validation_report.overall_compatibility > 80:
            print("    ✓ Cross-validation shows good compatibility")
        else:
            print("    ⚠ Cross-validation shows compatibility issues")
        
        if regression_results and regression_results['critical_failures'] == 0:
            print("    ✓ No critical regression test failures")
        else:
            print("    ⚠ Critical regression test failures detected")
        
        if rollback_validation and rollback_validation.overall_status == 'pass':
            print("    ✓ Rollback capabilities validated")
        else:
            print("    ⚠ Rollback validation has concerns")
        
        print("\n    RECOMMENDATION: Review validation results before production deployment")
        
    else:
        print("    ✗ Strategy conversion failed")
        print("    RECOMMENDATION: Address conversion issues before proceeding")
    
    print("\n" + "=" * 80)
    print("MIGRATION WORKFLOW DEMONSTRATION COMPLETED")
    print("=" * 80)
    
    # Cleanup
    print("\nCleaning up demonstration data...")
    try:
        rollback_manager.delete_rollback_point(rollback_id)
        audit_manager.clear_audit_statistics()
        print("Cleanup completed successfully")
    except Exception as e:
        print(f"Cleanup warning: {e}")


if __name__ == "__main__":
    demonstrate_complete_migration_workflow()