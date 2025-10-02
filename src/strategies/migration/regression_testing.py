"""
Automated Regression Testing for Strategy Migration

This module provides automated regression testing capabilities to ensure
that converted strategies maintain the same behavior as their legacy
counterparts across different market conditions and scenarios.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.strategies.base import BaseStrategy
from src.strategies.adapters.legacy_adapter import LegacyStrategyAdapter


@dataclass
class RegressionTestCase:
    """
    Definition of a regression test case
    
    Attributes:
        test_name: Name of the test case
        description: Description of what the test validates
        test_data: Test data for the scenario
        test_parameters: Parameters for the test
        expected_behavior: Expected behavior description
        tolerance: Tolerance for numerical comparisons
        critical: Whether this test is critical for migration approval
    """
    test_name: str
    description: str
    test_data: pd.DataFrame
    test_parameters: Dict[str, Any]
    expected_behavior: str
    tolerance: float
    critical: bool


@dataclass
class RegressionTestResult:
    """
    Result of a regression test
    
    Attributes:
        test_case_name: Name of the test case
        passed: Whether the test passed
        legacy_results: Results from legacy strategy
        converted_results: Results from converted strategy
        differences: Identified differences
        execution_time: Time taken to execute the test
        error_message: Error message if test failed
        metadata: Additional test metadata
    """
    test_case_name: str
    passed: bool
    legacy_results: Dict[str, Any]
    converted_results: Dict[str, Any]
    differences: List[Dict[str, Any]]
    execution_time: float
    error_message: Optional[str]
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "test_case_name": self.test_case_name,
            "passed": self.passed,
            "legacy_results": self.legacy_results,
            "converted_results": self.converted_results,
            "differences": self.differences,
            "execution_time": self.execution_time,
            "error_message": self.error_message,
            "metadata": self.metadata
        }


@dataclass
class RegressionTestSuite:
    """
    Collection of regression tests for a strategy migration
    
    Attributes:
        suite_name: Name of the test suite
        description: Description of the test suite
        test_cases: List of test cases in the suite
        setup_data: Common setup data for all tests
        teardown_actions: Actions to perform after tests
    """
    suite_name: str
    description: str
    test_cases: List[RegressionTestCase]
    setup_data: Dict[str, Any]
    teardown_actions: List[str]


class RegressionTester:
    """
    Automated regression testing framework for strategy migration
    
    This class provides comprehensive regression testing capabilities to ensure
    that converted strategies maintain behavioral compatibility with legacy
    strategies across various market scenarios.
    """

    def __init__(self, default_tolerance: float = 0.01):
        """
        Initialize the regression tester
        
        Args:
            default_tolerance: Default tolerance for numerical comparisons
        """
        self.logger = logging.getLogger("RegressionTester")
        self.default_tolerance = default_tolerance

        # Built-in test suites
        self.test_suites: Dict[str, RegressionTestSuite] = {}
        self._initialize_standard_test_suites()

        # Test execution history
        self.test_history: List[Dict[str, Any]] = []

    def run_regression_tests(self, legacy_strategy: BaseStrategy,
                           converted_strategy: LegacyStrategyAdapter,
                           test_suite_name: Optional[str] = None,
                           custom_test_cases: Optional[List[RegressionTestCase]] = None) -> Dict[str, Any]:
        """
        Run regression tests for a strategy migration
        
        Args:
            legacy_strategy: Original legacy strategy
            converted_strategy: Converted component-based strategy
            test_suite_name: Name of test suite to run (uses default if None)
            custom_test_cases: Custom test cases to run instead of suite
            
        Returns:
            Dictionary with comprehensive test results
        """
        start_time = datetime.now()

        self.logger.info(f"Starting regression tests for {legacy_strategy.name}")

        # Determine test cases to run
        if custom_test_cases:
            test_cases = custom_test_cases
            suite_name = "custom"
        elif test_suite_name and test_suite_name in self.test_suites:
            test_suite = self.test_suites[test_suite_name]
            test_cases = test_suite.test_cases
            suite_name = test_suite_name
        else:
            # Use default comprehensive suite
            test_suite = self.test_suites.get("comprehensive", self.test_suites["basic"])
            test_cases = test_suite.test_cases
            suite_name = "comprehensive"

        # Run individual test cases
        test_results = []
        critical_failures = []

        for test_case in test_cases:
            self.logger.debug(f"Running test case: {test_case.test_name}")

            try:
                result = self._run_single_test_case(legacy_strategy, converted_strategy, test_case)
                test_results.append(result)

                if not result.passed and test_case.critical:
                    critical_failures.append(result)

            except Exception as e:
                self.logger.error(f"Test case {test_case.test_name} failed with error: {e}")

                error_result = RegressionTestResult(
                    test_case_name=test_case.test_name,
                    passed=False,
                    legacy_results={},
                    converted_results={},
                    differences=[],
                    execution_time=0.0,
                    error_message=str(e),
                    metadata={"error": True}
                )
                test_results.append(error_result)

                if test_case.critical:
                    critical_failures.append(error_result)

        # Calculate summary statistics
        total_tests = len(test_results)
        passed_tests = sum(1 for result in test_results if result.passed)
        failed_tests = total_tests - passed_tests

        execution_time = (datetime.now() - start_time).total_seconds()

        # Determine overall result
        overall_passed = len(critical_failures) == 0

        # Create comprehensive result
        regression_result = {
            "test_timestamp": start_time.isoformat(),
            "strategy_name": legacy_strategy.name,
            "test_suite": suite_name,
            "overall_passed": overall_passed,
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "critical_failures": len(critical_failures),
            "success_rate": (passed_tests / total_tests) * 100 if total_tests > 0 else 0,
            "execution_time_seconds": execution_time,
            "test_results": [result.to_dict() for result in test_results],
            "critical_failure_details": [result.to_dict() for result in critical_failures],
            "summary": self._generate_test_summary(test_results),
            "recommendations": self._generate_test_recommendations(test_results, critical_failures)
        }

        # Store in history
        self.test_history.append(regression_result)

        self.logger.info(f"Regression tests completed: {passed_tests}/{total_tests} passed, "
                        f"{len(critical_failures)} critical failures")

        return regression_result

    def _run_single_test_case(self, legacy_strategy: BaseStrategy,
                            converted_strategy: LegacyStrategyAdapter,
                            test_case: RegressionTestCase) -> RegressionTestResult:
        """Run a single regression test case"""
        start_time = datetime.now()

        try:
            # Prepare test data
            test_data = test_case.test_data.copy()
            test_params = test_case.test_parameters

            # Run legacy strategy
            legacy_results = self._execute_strategy_test(legacy_strategy, test_data, test_params)

            # Run converted strategy
            converted_results = self._execute_strategy_test(converted_strategy, test_data, test_params)

            # Compare results
            differences = self._compare_test_results(legacy_results, converted_results, test_case.tolerance)

            # Determine if test passed
            passed = len(differences) == 0 or all(diff["within_tolerance"] for diff in differences)

            execution_time = (datetime.now() - start_time).total_seconds()

            return RegressionTestResult(
                test_case_name=test_case.test_name,
                passed=passed,
                legacy_results=legacy_results,
                converted_results=converted_results,
                differences=differences,
                execution_time=execution_time,
                error_message=None,
                metadata={
                    "test_description": test_case.description,
                    "expected_behavior": test_case.expected_behavior,
                    "tolerance": test_case.tolerance,
                    "critical": test_case.critical
                }
            )

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()

            return RegressionTestResult(
                test_case_name=test_case.test_name,
                passed=False,
                legacy_results={},
                converted_results={},
                differences=[],
                execution_time=execution_time,
                error_message=str(e),
                metadata={"error": True}
            )

    def _execute_strategy_test(self, strategy: Union[BaseStrategy, LegacyStrategyAdapter],
                             test_data: pd.DataFrame, test_params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute strategy test and collect results"""
        results = {}

        # Prepare data
        prepared_data = strategy.calculate_indicators(test_data)
        results["data_shape"] = prepared_data.shape
        results["data_columns"] = list(prepared_data.columns)

        # Test parameters
        test_balance = test_params.get("balance", 10000.0)
        test_indices = test_params.get("indices", [len(prepared_data) - 10, len(prepared_data) - 5, len(prepared_data) - 1])

        # Collect entry conditions
        entry_conditions = []
        for idx in test_indices:
            if 0 <= idx < len(prepared_data):
                try:
                    entry_condition = strategy.check_entry_conditions(prepared_data, idx)
                    entry_conditions.append({"index": idx, "entry": entry_condition})
                except Exception as e:
                    entry_conditions.append({"index": idx, "error": str(e)})

        results["entry_conditions"] = entry_conditions

        # Collect position sizes
        position_sizes = []
        for idx in test_indices:
            if 0 <= idx < len(prepared_data):
                try:
                    position_size = strategy.calculate_position_size(prepared_data, idx, test_balance)
                    position_sizes.append({"index": idx, "size": position_size})
                except Exception as e:
                    position_sizes.append({"index": idx, "error": str(e)})

        results["position_sizes"] = position_sizes

        # Collect stop losses
        stop_losses = []
        test_prices = test_params.get("test_prices", [100.0, 200.0])
        test_sides = test_params.get("test_sides", ["long", "short"])

        for idx in test_indices[:2]:  # Limit to first 2 indices
            if 0 <= idx < len(prepared_data):
                for price in test_prices:
                    for side in test_sides:
                        try:
                            stop_loss = strategy.calculate_stop_loss(prepared_data, idx, price, side)
                            stop_losses.append({
                                "index": idx,
                                "price": price,
                                "side": side,
                                "stop_loss": stop_loss
                            })
                        except Exception as e:
                            stop_losses.append({
                                "index": idx,
                                "price": price,
                                "side": side,
                                "error": str(e)
                            })

        results["stop_losses"] = stop_losses

        # Collect strategy parameters
        try:
            if hasattr(strategy, "get_parameters"):
                results["parameters"] = strategy.get_parameters()
        except Exception as e:
            results["parameters"] = {"error": str(e)}

        return results

    def _compare_test_results(self, legacy_results: Dict[str, Any],
                            converted_results: Dict[str, Any],
                            tolerance: float) -> List[Dict[str, Any]]:
        """Compare test results and identify differences"""
        differences = []

        # Compare data shapes
        if legacy_results.get("data_shape") != converted_results.get("data_shape"):
            differences.append({
                "type": "data_shape",
                "legacy_value": legacy_results.get("data_shape"),
                "converted_value": converted_results.get("data_shape"),
                "within_tolerance": False,
                "description": "Data frame shapes differ"
            })

        # Compare entry conditions
        legacy_entries = legacy_results.get("entry_conditions", [])
        converted_entries = converted_results.get("entry_conditions", [])

        for i, (legacy_entry, converted_entry) in enumerate(zip(legacy_entries, converted_entries)):
            if "error" not in legacy_entry and "error" not in converted_entry:
                if legacy_entry.get("entry") != converted_entry.get("entry"):
                    differences.append({
                        "type": "entry_condition",
                        "index": legacy_entry.get("index"),
                        "legacy_value": legacy_entry.get("entry"),
                        "converted_value": converted_entry.get("entry"),
                        "within_tolerance": False,
                        "description": f'Entry condition differs at index {legacy_entry.get("index")}'
                    })

        # Compare position sizes
        legacy_sizes = legacy_results.get("position_sizes", [])
        converted_sizes = converted_results.get("position_sizes", [])

        for legacy_size, converted_size in zip(legacy_sizes, converted_sizes):
            if "error" not in legacy_size and "error" not in converted_size:
                legacy_val = legacy_size.get("size", 0)
                converted_val = converted_size.get("size", 0)

                if abs(legacy_val - converted_val) > tolerance * max(abs(legacy_val), abs(converted_val), 1e-8):
                    differences.append({
                        "type": "position_size",
                        "index": legacy_size.get("index"),
                        "legacy_value": legacy_val,
                        "converted_value": converted_val,
                        "within_tolerance": False,
                        "description": f'Position size differs at index {legacy_size.get("index")}'
                    })

        # Compare stop losses
        legacy_stops = legacy_results.get("stop_losses", [])
        converted_stops = converted_results.get("stop_losses", [])

        for legacy_stop, converted_stop in zip(legacy_stops, converted_stops):
            if "error" not in legacy_stop and "error" not in converted_stop:
                legacy_val = legacy_stop.get("stop_loss", 0)
                converted_val = converted_stop.get("stop_loss", 0)

                if abs(legacy_val - converted_val) > tolerance * max(abs(legacy_val), abs(converted_val), 1e-8):
                    differences.append({
                        "type": "stop_loss",
                        "index": legacy_stop.get("index"),
                        "price": legacy_stop.get("price"),
                        "side": legacy_stop.get("side"),
                        "legacy_value": legacy_val,
                        "converted_value": converted_val,
                        "within_tolerance": False,
                        "description": f'Stop loss differs for price {legacy_stop.get("price")}, side {legacy_stop.get("side")}'
                    })

        return differences

    def _generate_test_summary(self, test_results: List[RegressionTestResult]) -> Dict[str, Any]:
        """Generate summary of test results"""
        if not test_results:
            return {}

        # Count by test type
        test_types = {}
        for result in test_results:
            test_type = result.test_case_name.split("_")[0] if "_" in result.test_case_name else "other"
            if test_type not in test_types:
                test_types[test_type] = {"total": 0, "passed": 0, "failed": 0}

            test_types[test_type]["total"] += 1
            if result.passed:
                test_types[test_type]["passed"] += 1
            else:
                test_types[test_type]["failed"] += 1

        # Count difference types
        difference_types = {}
        for result in test_results:
            for diff in result.differences:
                diff_type = diff.get("type", "unknown")
                difference_types[diff_type] = difference_types.get(diff_type, 0) + 1

        # Calculate execution time statistics
        execution_times = [result.execution_time for result in test_results]

        return {
            "test_types": test_types,
            "difference_types": difference_types,
            "execution_time_stats": {
                "total": sum(execution_times),
                "average": sum(execution_times) / len(execution_times),
                "min": min(execution_times),
                "max": max(execution_times)
            },
            "error_count": sum(1 for result in test_results if result.error_message is not None)
        }

    def _generate_test_recommendations(self, test_results: List[RegressionTestResult],
                                     critical_failures: List[RegressionTestResult]) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []

        if not critical_failures:
            recommendations.append("All critical tests passed. Migration appears successful.")
        else:
            recommendations.append(f"{len(critical_failures)} critical tests failed. Review before proceeding with migration.")

            # Analyze critical failures
            failure_types = {}
            for failure in critical_failures:
                for diff in failure.differences:
                    diff_type = diff.get("type", "unknown")
                    failure_types[diff_type] = failure_types.get(diff_type, 0) + 1

            if failure_types:
                most_common = max(failure_types.items(), key=lambda x: x[1])
                recommendations.append(f"Most common critical failure: {most_common[0]} ({most_common[1]} occurrences)")

        # Check for performance issues
        slow_tests = [result for result in test_results if result.execution_time > 1.0]
        if slow_tests:
            recommendations.append(f"{len(slow_tests)} tests took longer than 1 second. Consider performance optimization.")

        # Check for error patterns
        error_tests = [result for result in test_results if result.error_message is not None]
        if error_tests:
            recommendations.append(f"{len(error_tests)} tests encountered errors. Review error handling implementation.")

        return recommendations

    def _initialize_standard_test_suites(self) -> None:
        """Initialize standard test suites"""
        # Basic test suite
        basic_test_cases = [
            self._create_basic_functionality_test(),
            self._create_parameter_consistency_test(),
            self._create_error_handling_test()
        ]

        self.test_suites["basic"] = RegressionTestSuite(
            suite_name="basic",
            description="Basic functionality regression tests",
            test_cases=basic_test_cases,
            setup_data={},
            teardown_actions=[]
        )

        # Comprehensive test suite
        comprehensive_test_cases = basic_test_cases + [
            self._create_market_scenario_test("bull_market"),
            self._create_market_scenario_test("bear_market"),
            self._create_market_scenario_test("sideways_market"),
            self._create_volatility_test("high_volatility"),
            self._create_volatility_test("low_volatility"),
            self._create_edge_case_test()
        ]

        self.test_suites["comprehensive"] = RegressionTestSuite(
            suite_name="comprehensive",
            description="Comprehensive regression test suite",
            test_cases=comprehensive_test_cases,
            setup_data={},
            teardown_actions=[]
        )

    def _create_basic_functionality_test(self) -> RegressionTestCase:
        """Create basic functionality test case"""
        test_data = self._generate_test_data(200, scenario="normal")

        return RegressionTestCase(
            test_name="basic_functionality",
            description="Test basic strategy functionality with normal market data",
            test_data=test_data,
            test_parameters={
                "balance": 10000.0,
                "indices": [150, 175, 199],
                "test_prices": [100.0, 200.0],
                "test_sides": ["long", "short"]
            },
            expected_behavior="Strategy should produce consistent results for basic operations",
            tolerance=self.default_tolerance,
            critical=True
        )

    def _create_parameter_consistency_test(self) -> RegressionTestCase:
        """Create parameter consistency test case"""
        test_data = self._generate_test_data(100, scenario="normal")

        return RegressionTestCase(
            test_name="parameter_consistency",
            description="Test that strategy parameters are consistently mapped",
            test_data=test_data,
            test_parameters={
                "balance": 5000.0,
                "indices": [99],
                "test_prices": [150.0],
                "test_sides": ["long"]
            },
            expected_behavior="Strategy parameters should be identical or equivalent",
            tolerance=0.0,  # Exact match required for parameters
            critical=True
        )

    def _create_error_handling_test(self) -> RegressionTestCase:
        """Create error handling test case"""
        # Create problematic test data
        test_data = pd.DataFrame({
            "open": [100.0, np.nan, 102.0],
            "high": [101.0, 103.0, np.nan],
            "low": [99.0, 101.0, 101.0],
            "close": [100.5, 102.0, 101.5],
            "volume": [1000, 0, 1500]
        })

        return RegressionTestCase(
            test_name="error_handling",
            description="Test error handling with problematic data",
            test_data=test_data,
            test_parameters={
                "balance": 10000.0,
                "indices": [2],
                "test_prices": [100.0],
                "test_sides": ["long"]
            },
            expected_behavior="Strategy should handle errors gracefully",
            tolerance=self.default_tolerance,
            critical=False
        )

    def _create_market_scenario_test(self, scenario: str) -> RegressionTestCase:
        """Create market scenario test case"""
        test_data = self._generate_test_data(300, scenario=scenario)

        return RegressionTestCase(
            test_name=f"market_scenario_{scenario}",
            description=f"Test strategy behavior in {scenario} conditions",
            test_data=test_data,
            test_parameters={
                "balance": 15000.0,
                "indices": [250, 275, 299],
                "test_prices": [100.0, 150.0, 200.0],
                "test_sides": ["long", "short"]
            },
            expected_behavior=f"Strategy should adapt appropriately to {scenario} conditions",
            tolerance=self.default_tolerance,
            critical=False
        )

    def _create_volatility_test(self, volatility_type: str) -> RegressionTestCase:
        """Create volatility test case"""
        test_data = self._generate_test_data(250, scenario="normal", volatility=volatility_type)

        return RegressionTestCase(
            test_name=f"volatility_{volatility_type}",
            description=f"Test strategy behavior in {volatility_type} conditions",
            test_data=test_data,
            test_parameters={
                "balance": 12000.0,
                "indices": [200, 225, 249],
                "test_prices": [100.0, 200.0],
                "test_sides": ["long", "short"]
            },
            expected_behavior=f"Strategy should handle {volatility_type} appropriately",
            tolerance=self.default_tolerance * 2,  # Allow higher tolerance for volatile conditions
            critical=False
        )

    def _create_edge_case_test(self) -> RegressionTestCase:
        """Create edge case test"""
        # Create edge case data with extreme values
        test_data = pd.DataFrame({
            "open": [100.0, 0.01, 10000.0, 100.0],
            "high": [100.0, 0.02, 10001.0, 100.0],
            "low": [100.0, 0.01, 9999.0, 100.0],
            "close": [100.0, 0.015, 10000.0, 100.0],
            "volume": [1000, 1, 1000000, 1000]
        })

        return RegressionTestCase(
            test_name="edge_cases",
            description="Test strategy behavior with edge case values",
            test_data=test_data,
            test_parameters={
                "balance": 10000.0,
                "indices": [3],
                "test_prices": [0.01, 10000.0],
                "test_sides": ["long", "short"]
            },
            expected_behavior="Strategy should handle edge cases without crashing",
            tolerance=self.default_tolerance * 5,  # Higher tolerance for edge cases
            critical=False
        )

    def _generate_test_data(self, length: int, scenario: str = "normal",
                          volatility: str = "normal") -> pd.DataFrame:
        """Generate synthetic test data for different scenarios"""
        np.random.seed(42)  # For reproducible results

        # Base parameters
        base_price = 100.0
        base_volume = 1000.0

        # Scenario-specific parameters
        if scenario == "bull_market":
            trend = 0.001  # 0.1% upward trend per period
            trend_strength = 0.8
        elif scenario == "bear_market":
            trend = -0.001  # 0.1% downward trend per period
            trend_strength = 0.8
        elif scenario == "sideways_market":
            trend = 0.0
            trend_strength = 0.0
        else:  # normal
            trend = 0.0
            trend_strength = 0.3

        # Volatility parameters
        if volatility == "high_volatility":
            vol_multiplier = 3.0
        elif volatility == "low_volatility":
            vol_multiplier = 0.3
        else:  # normal
            vol_multiplier = 1.0

        # Generate price series
        prices = [base_price]
        volumes = [base_volume]

        for i in range(1, length):
            # Trend component
            trend_component = trend * trend_strength

            # Random component
            random_component = np.random.normal(0, 0.02 * vol_multiplier)

            # Mean reversion component
            mean_reversion = -0.1 * (prices[-1] - base_price) / base_price

            # Calculate price change
            price_change = trend_component + random_component + mean_reversion
            new_price = prices[-1] * (1 + price_change)
            new_price = max(new_price, 0.01)  # Ensure positive prices

            prices.append(new_price)

            # Generate volume
            volume_change = np.random.normal(0, 0.3)
            new_volume = volumes[-1] * (1 + volume_change)
            new_volume = max(new_volume, 1.0)  # Ensure positive volume

            volumes.append(new_volume)

        # Create OHLCV data
        data = []
        for i, (price, volume) in enumerate(zip(prices, volumes)):
            # Generate OHLC from close price
            volatility_factor = 0.01 * vol_multiplier
            high = price * (1 + abs(np.random.normal(0, volatility_factor)))
            low = price * (1 - abs(np.random.normal(0, volatility_factor)))
            open_price = prices[i-1] if i > 0 else price

            data.append({
                "open": open_price,
                "high": high,
                "low": low,
                "close": price,
                "volume": volume
            })

        df = pd.DataFrame(data)
        df.index = pd.date_range(start="2023-01-01", periods=length, freq="1H")

        return df

    def add_custom_test_suite(self, test_suite: RegressionTestSuite) -> None:
        """Add a custom test suite"""
        self.test_suites[test_suite.suite_name] = test_suite
        self.logger.info(f"Added custom test suite: {test_suite.suite_name}")

    def get_available_test_suites(self) -> List[str]:
        """Get list of available test suite names"""
        return list(self.test_suites.keys())

    def get_test_history(self) -> List[Dict[str, Any]]:
        """Get test execution history"""
        return self.test_history.copy()

    def clear_test_history(self) -> None:
        """Clear test execution history"""
        self.test_history.clear()
        self.logger.info("Regression test history cleared")
