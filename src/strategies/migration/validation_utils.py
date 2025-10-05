"""
Validation Utilities for Strategy Migration

This module provides utilities for validating converted strategies, including
functional testing, parameter validation, and compatibility checks.
"""

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.strategies.base import BaseStrategy
from src.strategies.adapters.legacy_adapter import LegacyStrategyAdapter

# Constants for validation thresholds
SLOW_TEST_THRESHOLD_MS = 1000
PERFORMANCE_THRESHOLD_MS = 100
DEFAULT_TOLERANCE = 0.01


@dataclass
class ValidationResult:
    """
    Result of a validation test
    
    Attributes:
        test_name: Name of the validation test
        passed: Whether the test passed
        message: Description of the test result
        details: Additional details about the test
        execution_time_ms: Time taken to execute the test
        error: Exception if test failed with error
    """
    test_name: str
    passed: bool
    message: str
    details: dict[str, Any]
    execution_time_ms: float
    error: Optional[Exception] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "test_name": self.test_name,
            "passed": self.passed,
            "message": self.message,
            "details": self.details,
            "execution_time_ms": self.execution_time_ms,
            "error": str(self.error) if self.error else None
        }


@dataclass
class ValidationReport:
    """
    Comprehensive validation report for strategy conversion
    
    Attributes:
        strategy_name: Name of the validated strategy
        validation_timestamp: When validation was performed
        total_tests: Total number of tests performed
        passed_tests: Number of tests that passed
        failed_tests: Number of tests that failed
        test_results: List of individual test results
        overall_success: Whether validation was successful overall
        recommendations: List of recommendations based on validation
        performance_metrics: Performance comparison metrics
    """
    strategy_name: str
    validation_timestamp: datetime
    total_tests: int
    passed_tests: int
    failed_tests: int
    test_results: List[ValidationResult]
    overall_success: bool
    recommendations: List[str]
    performance_metrics: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "strategy_name": self.strategy_name,
            "validation_timestamp": self.validation_timestamp.isoformat(),
            "total_tests": self.total_tests,
            "passed_tests": self.passed_tests,
            "failed_tests": self.failed_tests,
            "success_rate": (self.passed_tests / self.total_tests) * 100 if self.total_tests > 0 else 0,
            "test_results": [result.to_dict() for result in self.test_results],
            "overall_success": self.overall_success,
            "recommendations": self.recommendations,
            "performance_metrics": self.performance_metrics
        }


class StrategyValidator:
    """
    Comprehensive validator for converted strategies
    
    This class provides various validation tests to ensure that converted strategies
    maintain compatibility and functionality with the original legacy strategies.
    """

    def __init__(self, tolerance: float = 0.01):
        """
        Initialize the strategy validator
        
        Args:
            tolerance: Tolerance for numerical comparisons (default 1%)
        """
        self.logger = logging.getLogger("StrategyValidator")
        self.tolerance = tolerance

        # Track validation history
        self.validation_history: List[ValidationReport] = []

    def validate_converted_strategy(self, legacy_strategy: BaseStrategy,
                                  converted_strategy: LegacyStrategyAdapter,
                                  test_data: Optional[pd.DataFrame] = None,
                                  test_balance: float = 10000.0) -> ValidationReport:
        """
        Perform comprehensive validation of a converted strategy
        
        Args:
            legacy_strategy: Original legacy strategy
            converted_strategy: Converted component-based strategy
            test_data: Optional test data for functional testing
            test_balance: Test balance for position sizing tests
            
        Returns:
            ValidationReport with detailed results
        """
        start_time = datetime.now()
        test_results = []

        self.logger.info(f"Starting validation of converted strategy: {converted_strategy.name}")

        # Test 1: Interface Compatibility
        test_results.append(self._test_interface_compatibility(legacy_strategy, converted_strategy))

        # Test 2: Parameter Consistency
        test_results.append(self._test_parameter_consistency(legacy_strategy, converted_strategy))

        # Test 3: Trading Pair Consistency
        test_results.append(self._test_trading_pair_consistency(legacy_strategy, converted_strategy))

        # Test 4: Component Creation
        test_results.append(self._test_component_creation(converted_strategy))

        # Test 5: Basic Functionality
        if test_data is not None:
            test_results.extend(self._test_basic_functionality(
                legacy_strategy, converted_strategy, test_data, test_balance
            ))

        # Test 6: Error Handling
        test_results.append(self._test_error_handling(converted_strategy))

        # Test 7: Performance Metrics
        test_results.append(self._test_performance_metrics(converted_strategy))

        # Calculate summary statistics
        total_tests = len(test_results)
        passed_tests = sum(1 for result in test_results if result.passed)
        failed_tests = total_tests - passed_tests
        overall_success = failed_tests == 0

        # Generate recommendations
        recommendations = self._generate_recommendations(test_results, legacy_strategy, converted_strategy)

        # Calculate performance metrics
        performance_metrics = self._calculate_performance_metrics(test_results)

        # Create validation report
        report = ValidationReport(
            strategy_name=converted_strategy.name,
            validation_timestamp=start_time,
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            test_results=test_results,
            overall_success=overall_success,
            recommendations=recommendations,
            performance_metrics=performance_metrics
        )

        # Store in history
        self.validation_history.append(report)

        self.logger.info(f"Validation completed: {passed_tests}/{total_tests} tests passed")

        return report

    def _test_interface_compatibility(self, legacy_strategy: BaseStrategy,
                                    converted_strategy: LegacyStrategyAdapter) -> ValidationResult:
        """Test that the converted strategy implements the required interface"""
        start_time = time.time()

        try:
            required_methods = [
                "calculate_indicators",
                "check_entry_conditions",
                "check_exit_conditions",
                "calculate_position_size",
                "calculate_stop_loss",
                "get_parameters"
            ]

            missing_methods = []
            for method in required_methods:
                if not hasattr(converted_strategy, method):
                    missing_methods.append(method)

            # Test that methods are callable
            callable_methods = []
            for method in required_methods:
                if hasattr(converted_strategy, method) and callable(getattr(converted_strategy, method)):
                    callable_methods.append(method)

            passed = len(missing_methods) == 0 and len(callable_methods) == len(required_methods)

            details = {
                "required_methods": required_methods,
                "missing_methods": missing_methods,
                "callable_methods": callable_methods,
                "interface_complete": passed
            }

            message = "Interface compatibility test passed" if passed else f"Missing methods: {missing_methods}"

            return ValidationResult(
                test_name="interface_compatibility",
                passed=passed,
                message=message,
                details=details,
                execution_time_ms=(time.time() - start_time) * 1000
            )

        except Exception as e:
            return ValidationResult(
                test_name="interface_compatibility",
                passed=False,
                message=f"Interface compatibility test failed: {e}",
                details={"error": str(e)},
                execution_time_ms=(time.time() - start_time) * 1000,
                error=e
            )

    def _test_parameter_consistency(self, legacy_strategy: BaseStrategy,
                                  converted_strategy: LegacyStrategyAdapter) -> ValidationResult:
        """Test that parameters are consistently mapped"""
        start_time = time.time()

        try:
            # Get parameters from both strategies
            legacy_params = {}
            if hasattr(legacy_strategy, "get_parameters"):
                legacy_params = legacy_strategy.get_parameters() or {}

            converted_params = converted_strategy.get_parameters() or {}

            # Check for key parameter preservation
            key_params = ["trading_pair", "name"]
            preserved_params = []
            missing_params = []

            for param in key_params:
                legacy_value = getattr(legacy_strategy, param, None) or legacy_params.get(param)
                converted_value = getattr(converted_strategy, param, None) or converted_params.get(param)

                if legacy_value is not None and converted_value is not None:
                    if legacy_value == converted_value:
                        preserved_params.append(param)
                    else:
                        missing_params.append(f"{param}: {legacy_value} != {converted_value}")
                elif legacy_value is not None:
                    missing_params.append(f"{param}: missing in converted strategy")

            passed = len(missing_params) == 0

            details = {
                "legacy_param_count": len(legacy_params),
                "converted_param_count": len(converted_params),
                "preserved_params": preserved_params,
                "missing_params": missing_params,
                "legacy_params": legacy_params,
                "converted_params": converted_params
            }

            message = "Parameter consistency test passed" if passed else f"Parameter inconsistencies: {missing_params}"

            return ValidationResult(
                test_name="parameter_consistency",
                passed=passed,
                message=message,
                details=details,
                execution_time_ms=(time.time() - start_time) * 1000
            )

        except Exception as e:
            return ValidationResult(
                test_name="parameter_consistency",
                passed=False,
                message=f"Parameter consistency test failed: {e}",
                details={"error": str(e)},
                execution_time_ms=(time.time() - start_time) * 1000,
                error=e
            )

    def _test_trading_pair_consistency(self, legacy_strategy: BaseStrategy,
                                     converted_strategy: LegacyStrategyAdapter) -> ValidationResult:
        """Test that trading pair is preserved"""
        start_time = time.time()

        try:
            legacy_pair = legacy_strategy.get_trading_pair()
            converted_pair = converted_strategy.get_trading_pair()

            passed = legacy_pair == converted_pair

            details = {
                "legacy_trading_pair": legacy_pair,
                "converted_trading_pair": converted_pair,
                "consistent": passed
            }

            message = f"Trading pair consistency: {legacy_pair} -> {converted_pair}" if passed else f"Trading pair mismatch: {legacy_pair} != {converted_pair}"

            return ValidationResult(
                test_name="trading_pair_consistency",
                passed=passed,
                message=message,
                details=details,
                execution_time_ms=(time.time() - start_time) * 1000
            )

        except Exception as e:
            return ValidationResult(
                test_name="trading_pair_consistency",
                passed=False,
                message=f"Trading pair consistency test failed: {e}",
                details={"error": str(e)},
                execution_time_ms=(time.time() - start_time) * 1000,
                error=e
            )

    def _test_component_creation(self, converted_strategy: LegacyStrategyAdapter) -> ValidationResult:
        """Test that all components were created successfully"""
        start_time = time.time()

        try:
            component_status = converted_strategy.get_component_status()

            required_components = ["signal_generator", "risk_manager", "position_sizer", "regime_detector"]
            created_components = []
            missing_components = []

            for component in required_components:
                if component in component_status and "none" not in component_status[component].lower():
                    created_components.append(component)
                else:
                    missing_components.append(component)

            passed = len(missing_components) == 0

            details = {
                "required_components": required_components,
                "created_components": created_components,
                "missing_components": missing_components,
                "component_status": component_status
            }

            message = "All components created successfully" if passed else f"Missing components: {missing_components}"

            return ValidationResult(
                test_name="component_creation",
                passed=passed,
                message=message,
                details=details,
                execution_time_ms=(time.time() - start_time) * 1000
            )

        except Exception as e:
            return ValidationResult(
                test_name="component_creation",
                passed=False,
                message=f"Component creation test failed: {e}",
                details={"error": str(e)},
                execution_time_ms=(time.time() - start_time) * 1000,
                error=e
            )

    def _test_basic_functionality(self, legacy_strategy: BaseStrategy,
                                converted_strategy: LegacyStrategyAdapter,
                                test_data: pd.DataFrame,
                                test_balance: float) -> List[ValidationResult]:
        """Test basic functionality with sample data"""
        results = []

        # Prepare test data
        if len(test_data) < 100:
            # Create minimal test data if not provided
            test_data = self._create_test_data()

        # Test indicator calculation
        results.append(self._test_indicator_calculation(legacy_strategy, converted_strategy, test_data))

        # Test entry conditions
        results.append(self._test_entry_conditions(legacy_strategy, converted_strategy, test_data))

        # Test position sizing
        results.append(self._test_position_sizing(legacy_strategy, converted_strategy, test_data, test_balance))

        # Test stop loss calculation
        results.append(self._test_stop_loss_calculation(legacy_strategy, converted_strategy, test_data))

        return results

    def _test_indicator_calculation(self, legacy_strategy: BaseStrategy,
                                  converted_strategy: LegacyStrategyAdapter,
                                  test_data: pd.DataFrame) -> ValidationResult:
        """Test indicator calculation functionality"""
        start_time = time.time()

        try:
            # Test with sample data
            legacy_result = legacy_strategy.calculate_indicators(test_data.copy())
            converted_result = converted_strategy.calculate_indicators(test_data.copy())

            # Check that both return DataFrames
            legacy_is_df = isinstance(legacy_result, pd.DataFrame)
            converted_is_df = isinstance(converted_result, pd.DataFrame)

            # Check that results have same shape
            same_shape = legacy_is_df and converted_is_df and legacy_result.shape == converted_result.shape

            passed = legacy_is_df and converted_is_df and same_shape

            details = {
                "legacy_is_dataframe": legacy_is_df,
                "converted_is_dataframe": converted_is_df,
                "legacy_shape": legacy_result.shape if legacy_is_df else None,
                "converted_shape": converted_result.shape if converted_is_df else None,
                "same_shape": same_shape
            }

            message = "Indicator calculation test passed" if passed else "Indicator calculation test failed"

            return ValidationResult(
                test_name="indicator_calculation",
                passed=passed,
                message=message,
                details=details,
                execution_time_ms=(time.time() - start_time) * 1000
            )

        except Exception as e:
            return ValidationResult(
                test_name="indicator_calculation",
                passed=False,
                message=f"Indicator calculation test failed: {e}",
                details={"error": str(e)},
                execution_time_ms=(time.time() - start_time) * 1000,
                error=e
            )

    def _test_entry_conditions(self, legacy_strategy: BaseStrategy,
                             converted_strategy: LegacyStrategyAdapter,
                             test_data: pd.DataFrame) -> ValidationResult:
        """Test entry condition checking"""
        start_time = time.time()

        try:
            # Prepare data
            legacy_data = legacy_strategy.calculate_indicators(test_data.copy())
            converted_data = converted_strategy.calculate_indicators(test_data.copy())

            # Test entry conditions at multiple points
            test_indices = [50, 75, 100] if len(test_data) > 100 else [len(test_data) - 1]

            entry_results = []
            for idx in test_indices:
                if idx < len(legacy_data) and idx < len(converted_data):
                    try:
                        legacy_entry = legacy_strategy.check_entry_conditions(legacy_data, idx)
                        converted_entry = converted_strategy.check_entry_conditions(converted_data, idx)

                        entry_results.append({
                            "index": idx,
                            "legacy_entry": legacy_entry,
                            "converted_entry": converted_entry,
                            "consistent": legacy_entry == converted_entry
                        })
                    except Exception as e:
                        entry_results.append({
                            "index": idx,
                            "error": str(e)
                        })

            # Check consistency
            consistent_results = [r for r in entry_results if r.get("consistent", False)]
            passed = len(consistent_results) > 0 or len(entry_results) == 0

            details = {
                "test_indices": test_indices,
                "entry_results": entry_results,
                "consistent_count": len(consistent_results),
                "total_tests": len(entry_results)
            }

            message = f"Entry conditions test: {len(consistent_results)}/{len(entry_results)} consistent"

            return ValidationResult(
                test_name="entry_conditions",
                passed=passed,
                message=message,
                details=details,
                execution_time_ms=(time.time() - start_time) * 1000
            )

        except Exception as e:
            return ValidationResult(
                test_name="entry_conditions",
                passed=False,
                message=f"Entry conditions test failed: {e}",
                details={"error": str(e)},
                execution_time_ms=(time.time() - start_time) * 1000,
                error=e
            )

    def _test_position_sizing(self, legacy_strategy: BaseStrategy,
                            converted_strategy: LegacyStrategyAdapter,
                            test_data: pd.DataFrame,
                            test_balance: float) -> ValidationResult:
        """Test position sizing calculation"""
        start_time = time.time()

        try:
            # Prepare data
            legacy_data = legacy_strategy.calculate_indicators(test_data.copy())
            converted_data = converted_strategy.calculate_indicators(test_data.copy())

            # Test position sizing at multiple points
            test_indices = [50, 75, 100] if len(test_data) > 100 else [len(test_data) - 1]

            sizing_results = []
            for idx in test_indices:
                if idx < len(legacy_data) and idx < len(converted_data):
                    try:
                        legacy_size = legacy_strategy.calculate_position_size(legacy_data, idx, test_balance)
                        converted_size = converted_strategy.calculate_position_size(converted_data, idx, test_balance)

                        # Check if sizes are within tolerance
                        size_diff = abs(legacy_size - converted_size) / max(legacy_size, converted_size, 1e-8)
                        within_tolerance = size_diff <= self.tolerance

                        sizing_results.append({
                            "index": idx,
                            "legacy_size": legacy_size,
                            "converted_size": converted_size,
                            "difference_pct": size_diff * 100,
                            "within_tolerance": within_tolerance
                        })
                    except Exception as e:
                        sizing_results.append({
                            "index": idx,
                            "error": str(e)
                        })

            # Check tolerance
            within_tolerance_count = sum(1 for r in sizing_results if r.get("within_tolerance", False))
            passed = within_tolerance_count > 0 or len(sizing_results) == 0

            details = {
                "test_indices": test_indices,
                "sizing_results": sizing_results,
                "within_tolerance_count": within_tolerance_count,
                "total_tests": len(sizing_results),
                "tolerance_pct": self.tolerance * 100
            }

            message = f"Position sizing test: {within_tolerance_count}/{len(sizing_results)} within tolerance"

            return ValidationResult(
                test_name="position_sizing",
                passed=passed,
                message=message,
                details=details,
                execution_time_ms=(time.time() - start_time) * 1000
            )

        except Exception as e:
            return ValidationResult(
                test_name="position_sizing",
                passed=False,
                message=f"Position sizing test failed: {e}",
                details={"error": str(e)},
                execution_time_ms=(time.time() - start_time) * 1000,
                error=e
            )

    def _test_stop_loss_calculation(self, legacy_strategy: BaseStrategy,
                                  converted_strategy: LegacyStrategyAdapter,
                                  test_data: pd.DataFrame) -> ValidationResult:
        """Test stop loss calculation"""
        start_time = time.time()

        try:
            # Test stop loss calculation with sample prices
            test_prices = [100.0, 50.0, 200.0]
            test_sides = ["long", "short"]

            stop_loss_results = []
            for price in test_prices:
                for side in test_sides:
                    try:
                        legacy_stop = legacy_strategy.calculate_stop_loss(test_data, len(test_data) - 1, price, side)
                        converted_stop = converted_strategy.calculate_stop_loss(test_data, len(test_data) - 1, price, side)

                        # Check if stop losses are within tolerance
                        stop_diff = abs(legacy_stop - converted_stop) / max(legacy_stop, converted_stop, 1e-8)
                        within_tolerance = stop_diff <= self.tolerance

                        stop_loss_results.append({
                            "price": price,
                            "side": side,
                            "legacy_stop": legacy_stop,
                            "converted_stop": converted_stop,
                            "difference_pct": stop_diff * 100,
                            "within_tolerance": within_tolerance
                        })
                    except Exception as e:
                        stop_loss_results.append({
                            "price": price,
                            "side": side,
                            "error": str(e)
                        })

            # Check tolerance
            within_tolerance_count = sum(1 for r in stop_loss_results if r.get("within_tolerance", False))
            passed = within_tolerance_count > 0 or len(stop_loss_results) == 0

            details = {
                "test_prices": test_prices,
                "test_sides": test_sides,
                "stop_loss_results": stop_loss_results,
                "within_tolerance_count": within_tolerance_count,
                "total_tests": len(stop_loss_results),
                "tolerance_pct": self.tolerance * 100
            }

            message = f"Stop loss test: {within_tolerance_count}/{len(stop_loss_results)} within tolerance"

            return ValidationResult(
                test_name="stop_loss_calculation",
                passed=passed,
                message=message,
                details=details,
                execution_time_ms=(time.time() - start_time) * 1000
            )

        except Exception as e:
            return ValidationResult(
                test_name="stop_loss_calculation",
                passed=False,
                message=f"Stop loss calculation test failed: {e}",
                details={"error": str(e)},
                execution_time_ms=(time.time() - start_time) * 1000,
                error=e
            )

    def _test_error_handling(self, converted_strategy: LegacyStrategyAdapter) -> ValidationResult:
        """Test error handling capabilities"""
        start_time = time.time()

        try:
            error_tests = []

            # Test with invalid data
            try:
                empty_df = pd.DataFrame()
                converted_strategy.calculate_indicators(empty_df)
                error_tests.append({"test": "empty_dataframe", "handled": True})
            except Exception:
                error_tests.append({"test": "empty_dataframe", "handled": False})

            # Test with invalid index
            try:
                test_data = self._create_test_data()
                converted_strategy.check_entry_conditions(test_data, -1)
                error_tests.append({"test": "invalid_index", "handled": True})
            except Exception:
                error_tests.append({"test": "invalid_index", "handled": False})

            # Test with invalid balance
            try:
                test_data = self._create_test_data()
                converted_strategy.calculate_position_size(test_data, 50, -100.0)
                error_tests.append({"test": "negative_balance", "handled": True})
            except Exception:
                error_tests.append({"test": "negative_balance", "handled": False})

            handled_count = sum(1 for test in error_tests if test["handled"])
            passed = handled_count >= len(error_tests) * 0.5  # At least 50% should be handled gracefully

            details = {
                "error_tests": error_tests,
                "handled_count": handled_count,
                "total_tests": len(error_tests)
            }

            message = f"Error handling test: {handled_count}/{len(error_tests)} errors handled gracefully"

            return ValidationResult(
                test_name="error_handling",
                passed=passed,
                message=message,
                details=details,
                execution_time_ms=(time.time() - start_time) * 1000
            )

        except Exception as e:
            return ValidationResult(
                test_name="error_handling",
                passed=False,
                message=f"Error handling test failed: {e}",
                details={"error": str(e)},
                execution_time_ms=(time.time() - start_time) * 1000,
                error=e
            )

    def _test_performance_metrics(self, converted_strategy: LegacyStrategyAdapter) -> ValidationResult:
        """Test performance metrics collection"""
        start_time = time.time()

        try:
            # Get performance metrics
            metrics = converted_strategy.get_performance_metrics()

            required_metrics = [
                "signals_generated",
                "entry_conditions_checked",
                "exit_conditions_checked",
                "position_sizes_calculated",
                "component_errors"
            ]

            available_metrics = []
            missing_metrics = []

            for metric in required_metrics:
                if metric in metrics:
                    available_metrics.append(metric)
                else:
                    missing_metrics.append(metric)

            passed = len(missing_metrics) == 0

            details = {
                "required_metrics": required_metrics,
                "available_metrics": available_metrics,
                "missing_metrics": missing_metrics,
                "metrics": metrics
            }

            message = "Performance metrics test passed" if passed else f"Missing metrics: {missing_metrics}"

            return ValidationResult(
                test_name="performance_metrics",
                passed=passed,
                message=message,
                details=details,
                execution_time_ms=(time.time() - start_time) * 1000
            )

        except Exception as e:
            return ValidationResult(
                test_name="performance_metrics",
                passed=False,
                message=f"Performance metrics test failed: {e}",
                details={"error": str(e)},
                execution_time_ms=(time.time() - start_time) * 1000,
                error=e
            )

    def _create_test_data(self, length: int = 200) -> pd.DataFrame:
        """Create synthetic test data for validation"""
        np.random.seed(42)  # For reproducible results

        # Generate synthetic OHLCV data
        dates = pd.date_range(start="2023-01-01", periods=length, freq="1H")

        # Generate price data with some trend and volatility
        base_price = 100.0
        price_changes = np.random.normal(0, 0.02, length)  # 2% volatility
        prices = [base_price]

        for change in price_changes[1:]:
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, 1.0))  # Ensure positive prices

        # Create OHLCV data
        data = []
        for i, price in enumerate(prices):
            high = price * (1 + abs(np.random.normal(0, 0.01)))
            low = price * (1 - abs(np.random.normal(0, 0.01)))
            open_price = prices[i-1] if i > 0 else price
            close_price = price
            volume = np.random.uniform(1000, 10000)

            data.append({
                "timestamp": dates[i],
                "open": open_price,
                "high": high,
                "low": low,
                "close": close_price,
                "volume": volume
            })

        df = pd.DataFrame(data)
        df.set_index("timestamp", inplace=True)

        return df

    def _generate_recommendations(self, test_results: List[ValidationResult],
                                legacy_strategy: BaseStrategy,
                                converted_strategy: LegacyStrategyAdapter) -> List[str]:
        """Generate recommendations based on validation results"""
        recommendations = []

        # Check for failed tests
        failed_tests = [result for result in test_results if not result.passed]

        if not failed_tests:
            recommendations.append(
                "All validation tests passed. The converted strategy appears to be working correctly."
            )
        else:
            recommendations.append(
                f"{len(failed_tests)} validation tests failed. Review the following issues:"
            )

            recommendations.extend([
                f"- {failed_test.test_name}: {failed_test.message}"
                for failed_test in failed_tests
            ])

        # Check for performance issues
        slow_tests = [result for result in test_results if result.execution_time_ms > SLOW_TEST_THRESHOLD_MS]
        if slow_tests:
            recommendations.append(
                f"Performance concern: {len(slow_tests)} tests took longer than 1 second to execute."
            )

        # Check component status
        try:
            component_status = converted_strategy.get_component_status()
            if "none" in str(component_status).lower():
                recommendations.append("Some components may not be properly initialized. Check component creation.")
        except Exception as e:
            recommendations.append(f"Unable to check component status: {e}")

        # Check error handling
        error_handling_test = next((r for r in test_results if r.test_name == "error_handling"), None)
        if error_handling_test and not error_handling_test.passed:
            recommendations.append("Improve error handling to gracefully handle edge cases and invalid inputs.")

        return recommendations

    def _calculate_performance_metrics(self, test_results: List[ValidationResult]) -> Dict[str, Any]:
        """Calculate performance metrics from test results"""
        if not test_results:
            return {}

        execution_times = [result.execution_time_ms for result in test_results]

        return {
            "total_execution_time_ms": sum(execution_times),
            "average_execution_time_ms": sum(execution_times) / len(execution_times),
            "max_execution_time_ms": max(execution_times),
            "min_execution_time_ms": min(execution_times),
            "tests_over_100ms": sum(1 for t in execution_times if t > PERFORMANCE_THRESHOLD_MS),
            "tests_over_1000ms": sum(1 for t in execution_times if t > SLOW_TEST_THRESHOLD_MS)
        }

    def get_validation_history(self) -> List[ValidationReport]:
        """Get history of all validations performed"""
        return self.validation_history.copy()

    def clear_validation_history(self) -> None:
        """Clear the validation history"""
        self.validation_history.clear()
        self.logger.info("Validation history cleared")

    def generate_validation_summary(self) -> Dict[str, Any]:
        """Generate a summary of all validations performed"""
        if not self.validation_history:
            return {
                "total_validations": 0,
                "successful_validations": 0,
                "failed_validations": 0,
                "success_rate": 0.0,
                "average_tests_per_validation": 0.0,
                "common_failures": []
            }

        total = len(self.validation_history)
        successful = sum(1 for report in self.validation_history if report.overall_success)
        failed = total - successful

        # Calculate average tests per validation
        total_tests = sum(report.total_tests for report in self.validation_history)
        avg_tests = total_tests / total if total > 0 else 0

        # Collect common failures
        all_failures = []
        for report in self.validation_history:
            failed_tests = [result for result in report.test_results if not result.passed]
            all_failures.extend([test.test_name for test in failed_tests])

        # Count failure frequency
        failure_counts = {}
        for failure in all_failures:
            failure_counts[failure] = failure_counts.get(failure, 0) + 1

        common_failures = sorted(failure_counts.items(), key=lambda x: x[1], reverse=True)[:5]

        return {
            "total_validations": total,
            "successful_validations": successful,
            "failed_validations": failed,
            "success_rate": (successful / total) * 100 if total > 0 else 0.0,
            "average_tests_per_validation": avg_tests,
            "common_failures": [{"test": test, "count": count} for test, count in common_failures],
            "validation_timeline": [
                {
                    "timestamp": report.validation_timestamp.isoformat(),
                    "strategy_name": report.strategy_name,
                    "success": report.overall_success,
                    "tests_passed": report.passed_tests,
                    "tests_total": report.total_tests
                }
                for report in self.validation_history
            ]
        }
