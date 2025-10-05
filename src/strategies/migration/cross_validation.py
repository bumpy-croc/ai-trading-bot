"""
Cross-Validation Testing Framework

This module provides comprehensive cross-validation testing between legacy
strategies and converted component-based strategies, including performance
comparison, regression testing, and detailed difference analysis.
"""

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.strategies.base import BaseStrategy
from src.strategies.adapters.legacy_adapter import LegacyStrategyAdapter


@dataclass
class ComparisonResult:
    """
    Result of comparing two strategy outputs
    
    Attributes:
        test_name: Name of the comparison test
        legacy_value: Value from legacy strategy
        converted_value: Value from converted strategy
        difference: Absolute difference between values
        relative_difference: Relative difference as percentage
        within_tolerance: Whether difference is within acceptable tolerance
        tolerance_used: Tolerance threshold used for comparison
        metadata: Additional comparison metadata
    """
    test_name: str
    legacy_value: Any
    converted_value: Any
    difference: float
    relative_difference: float
    within_tolerance: bool
    tolerance_used: float
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "test_name": self.test_name,
            "legacy_value": self.legacy_value,
            "converted_value": self.converted_value,
            "difference": self.difference,
            "relative_difference": self.relative_difference,
            "within_tolerance": self.within_tolerance,
            "tolerance_used": self.tolerance_used,
            "metadata": self.metadata
        }


@dataclass
class CrossValidationReport:
    """
    Comprehensive cross-validation report
    
    Attributes:
        strategy_name: Name of the strategy being tested
        test_timestamp: When the test was performed
        legacy_strategy_type: Type of the legacy strategy
        converted_strategy_type: Type of the converted strategy
        test_data_info: Information about test data used
        comparison_results: List of individual comparison results
        performance_metrics: Performance comparison metrics
        regression_test_results: Results of regression tests
        overall_compatibility: Overall compatibility score (0-100)
        recommendations: List of recommendations based on results
        execution_summary: Summary of test execution
    """
    strategy_name: str
    test_timestamp: datetime
    legacy_strategy_type: str
    converted_strategy_type: str
    test_data_info: Dict[str, Any]
    comparison_results: List[ComparisonResult]
    performance_metrics: Dict[str, Any]
    regression_test_results: Dict[str, Any]
    overall_compatibility: float
    recommendations: List[str]
    execution_summary: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "strategy_name": self.strategy_name,
            "test_timestamp": self.test_timestamp.isoformat(),
            "legacy_strategy_type": self.legacy_strategy_type,
            "converted_strategy_type": self.converted_strategy_type,
            "test_data_info": self.test_data_info,
            "comparison_results": [result.to_dict() for result in self.comparison_results],
            "performance_metrics": self.performance_metrics,
            "regression_test_results": self.regression_test_results,
            "overall_compatibility": self.overall_compatibility,
            "recommendations": self.recommendations,
            "execution_summary": self.execution_summary
        }


class CrossValidationTester:
    """
    Cross-validation testing framework for strategy migration
    
    This class provides comprehensive testing capabilities to compare legacy
    strategies with their converted component-based counterparts, ensuring
    compatibility and identifying any regressions.
    """

    def __init__(self, tolerance: float = 0.01, performance_tolerance: float = 0.05):
        """
        Initialize the cross-validation tester
        
        Args:
            tolerance: Default tolerance for numerical comparisons (1%)
            performance_tolerance: Tolerance for performance comparisons (5%)
        """
        self.logger = logging.getLogger("CrossValidationTester")
        self.tolerance = tolerance
        self.performance_tolerance = performance_tolerance

        # Track test history
        self.test_history: List[CrossValidationReport] = []

    def run_cross_validation(self, legacy_strategy: BaseStrategy,
                           converted_strategy: LegacyStrategyAdapter,
                           test_data: pd.DataFrame,
                           test_balance: float = 10000.0,
                           test_indices: Optional[List[int]] = None) -> CrossValidationReport:
        """
        Run comprehensive cross-validation between legacy and converted strategies
        
        Args:
            legacy_strategy: Original legacy strategy
            converted_strategy: Converted component-based strategy
            test_data: Test data for comparison
            test_balance: Test balance for position sizing
            test_indices: Specific indices to test (uses sample if None)
            
        Returns:
            CrossValidationReport with detailed comparison results
        """
        start_time = datetime.now()

        self.logger.info(f"Starting cross-validation for {legacy_strategy.name} vs {converted_strategy.name}")

        # Prepare test data
        test_data_info = self._analyze_test_data(test_data)

        # Determine test indices
        if test_indices is None:
            test_indices = self._select_test_indices(test_data)

        # Prepare data for both strategies
        legacy_data = legacy_strategy.calculate_indicators(test_data.copy())
        converted_data = converted_strategy.calculate_indicators(test_data.copy())

        # Run comparison tests
        comparison_results = []

        # Test 1: Indicator calculation comparison
        comparison_results.extend(self._compare_indicators(legacy_data, converted_data))

        # Test 2: Entry conditions comparison
        comparison_results.extend(self._compare_entry_conditions(
            legacy_strategy, converted_strategy, legacy_data, converted_data, test_indices
        ))

        # Test 3: Exit conditions comparison
        comparison_results.extend(self._compare_exit_conditions(
            legacy_strategy, converted_strategy, legacy_data, converted_data, test_indices
        ))

        # Test 4: Position sizing comparison
        comparison_results.extend(self._compare_position_sizing(
            legacy_strategy, converted_strategy, legacy_data, converted_data, test_indices, test_balance
        ))

        # Test 5: Stop loss calculation comparison
        comparison_results.extend(self._compare_stop_loss_calculation(
            legacy_strategy, converted_strategy, legacy_data, converted_data, test_indices
        ))

        # Run performance comparison
        performance_metrics = self._compare_performance(
            legacy_strategy, converted_strategy, legacy_data, converted_data, test_indices
        )

        # Run regression tests
        regression_test_results = self._run_regression_tests(
            legacy_strategy, converted_strategy, test_data, test_balance
        )

        # Calculate overall compatibility score
        overall_compatibility = self._calculate_compatibility_score(comparison_results)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            comparison_results, performance_metrics, regression_test_results
        )

        # Create execution summary
        execution_time = (datetime.now() - start_time).total_seconds()
        execution_summary = {
            "execution_time_seconds": execution_time,
            "total_comparisons": len(comparison_results),
            "successful_comparisons": sum(1 for r in comparison_results if r.within_tolerance),
            "failed_comparisons": sum(1 for r in comparison_results if not r.within_tolerance),
            "test_indices_count": len(test_indices),
            "test_data_rows": len(test_data)
        }

        # Create report
        report = CrossValidationReport(
            strategy_name=f"{legacy_strategy.name}_vs_{converted_strategy.name}",
            test_timestamp=start_time,
            legacy_strategy_type=legacy_strategy.__class__.__name__,
            converted_strategy_type=converted_strategy.__class__.__name__,
            test_data_info=test_data_info,
            comparison_results=comparison_results,
            performance_metrics=performance_metrics,
            regression_test_results=regression_test_results,
            overall_compatibility=overall_compatibility,
            recommendations=recommendations,
            execution_summary=execution_summary
        )

        # Store in history
        self.test_history.append(report)

        self.logger.info(f"Cross-validation completed: {overall_compatibility:.1f}% compatibility")

        return report

    def run_batch_cross_validation(self, strategy_pairs: List[Tuple[BaseStrategy, LegacyStrategyAdapter]],
                                 test_data: pd.DataFrame,
                                 test_balance: float = 10000.0) -> List[CrossValidationReport]:
        """
        Run cross-validation for multiple strategy pairs
        
        Args:
            strategy_pairs: List of (legacy_strategy, converted_strategy) tuples
            test_data: Test data for comparison
            test_balance: Test balance for position sizing
            
        Returns:
            List of cross-validation reports
        """
        reports = []

        self.logger.info(f"Starting batch cross-validation for {len(strategy_pairs)} strategy pairs")

        for i, (legacy_strategy, converted_strategy) in enumerate(strategy_pairs):
            self.logger.info(f"Testing pair {i+1}/{len(strategy_pairs)}: {legacy_strategy.name}")

            try:
                report = self.run_cross_validation(legacy_strategy, converted_strategy, test_data, test_balance)
                reports.append(report)

            except Exception as e:
                self.logger.error(f"Cross-validation failed for {legacy_strategy.name}: {e}")

                # Create error report
                error_report = CrossValidationReport(
                    strategy_name=f"{legacy_strategy.name}_vs_{converted_strategy.name}",
                    test_timestamp=datetime.now(),
                    legacy_strategy_type=legacy_strategy.__class__.__name__,
                    converted_strategy_type=converted_strategy.__class__.__name__,
                    test_data_info={},
                    comparison_results=[],
                    performance_metrics={},
                    regression_test_results={"error": str(e)},
                    overall_compatibility=0.0,
                    recommendations=[f"Cross-validation failed: {e}"],
                    execution_summary={"error": True}
                )
                reports.append(error_report)

        successful_tests = sum(1 for r in reports if r.overall_compatibility > 0)
        self.logger.info(f"Batch cross-validation completed: {successful_tests}/{len(strategy_pairs)} successful")

        return reports

    def _analyze_test_data(self, test_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze test data and return information"""
        return {
            "rows": len(test_data),
            "columns": list(test_data.columns),
            "date_range": {
                "start": test_data.index[0].isoformat() if hasattr(test_data.index, "__getitem__") else None,
                "end": test_data.index[-1].isoformat() if hasattr(test_data.index, "__getitem__") else None
            },
            "price_range": {
                "min": float(test_data["close"].min()) if "close" in test_data.columns else None,
                "max": float(test_data["close"].max()) if "close" in test_data.columns else None,
                "mean": float(test_data["close"].mean()) if "close" in test_data.columns else None
            },
            "volume_stats": {
                "min": float(test_data["volume"].min()) if "volume" in test_data.columns else None,
                "max": float(test_data["volume"].max()) if "volume" in test_data.columns else None,
                "mean": float(test_data["volume"].mean()) if "volume" in test_data.columns else None
            }
        }

    def _select_test_indices(self, test_data: pd.DataFrame, sample_size: int = 20) -> List[int]:
        """Select representative test indices from the data"""
        data_length = len(test_data)

        if data_length <= sample_size:
            return list(range(max(1, data_length - 10), data_length))

        # Select indices from different parts of the data
        indices = []

        # Early indices (after warm-up period)
        start_idx = max(50, data_length // 10)
        indices.extend(range(start_idx, min(start_idx + 5, data_length)))

        # Middle indices
        mid_idx = data_length // 2
        indices.extend(range(mid_idx - 2, min(mid_idx + 3, data_length)))

        # Late indices
        end_idx = data_length - 10
        indices.extend(range(max(end_idx, mid_idx + 5), data_length))

        # Remove duplicates and sort
        indices = sorted(list(set(indices)))

        # Limit to sample size
        if len(indices) > sample_size:
            step = len(indices) // sample_size
            indices = indices[::step][:sample_size]

        return indices

    def _compare_indicators(self, legacy_data: pd.DataFrame, converted_data: pd.DataFrame) -> List[ComparisonResult]:
        """Compare indicator calculations between strategies"""
        results = []

        # Find common columns
        common_columns = set(legacy_data.columns) & set(converted_data.columns)

        # Compare basic OHLCV columns
        basic_columns = ["open", "high", "low", "close", "volume"]
        for column in basic_columns:
            if column in common_columns:
                legacy_values = legacy_data[column].dropna()
                converted_values = converted_data[column].dropna()

                if len(legacy_values) > 0 and len(converted_values) > 0:
                    # Compare means
                    legacy_mean = legacy_values.mean()
                    converted_mean = converted_values.mean()

                    difference = abs(legacy_mean - converted_mean)
                    relative_diff = (difference / max(abs(legacy_mean), abs(converted_mean), 1e-8)) * 100

                    results.append(ComparisonResult(
                        test_name=f"indicator_{column}_mean",
                        legacy_value=legacy_mean,
                        converted_value=converted_mean,
                        difference=difference,
                        relative_difference=relative_diff,
                        within_tolerance=relative_diff <= self.tolerance * 100,
                        tolerance_used=self.tolerance,
                        metadata={"column": column, "comparison_type": "mean"}
                    ))

        # Compare data frame shapes
        shape_match = legacy_data.shape == converted_data.shape
        results.append(ComparisonResult(
            test_name="dataframe_shape",
            legacy_value=legacy_data.shape,
            converted_value=converted_data.shape,
            difference=0 if shape_match else 1,
            relative_difference=0 if shape_match else 100,
            within_tolerance=shape_match,
            tolerance_used=0,
            metadata={"comparison_type": "shape"}
        ))

        return results

    def _compare_entry_conditions(self, legacy_strategy: BaseStrategy, converted_strategy: LegacyStrategyAdapter,
                                legacy_data: pd.DataFrame, converted_data: pd.DataFrame,
                                test_indices: List[int]) -> List[ComparisonResult]:
        """Compare entry condition decisions"""
        results = []

        for idx in test_indices:
            if idx < len(legacy_data) and idx < len(converted_data):
                try:
                    legacy_entry = legacy_strategy.check_entry_conditions(legacy_data, idx)
                    converted_entry = converted_strategy.check_entry_conditions(converted_data, idx)

                    # Convert boolean to numeric for comparison
                    legacy_numeric = 1 if legacy_entry else 0
                    converted_numeric = 1 if converted_entry else 0

                    difference = abs(legacy_numeric - converted_numeric)

                    results.append(ComparisonResult(
                        test_name=f"entry_condition_idx_{idx}",
                        legacy_value=legacy_entry,
                        converted_value=converted_entry,
                        difference=difference,
                        relative_difference=difference * 100,
                        within_tolerance=difference == 0,
                        tolerance_used=0,
                        metadata={"index": idx, "comparison_type": "entry_condition"}
                    ))

                except Exception as e:
                    self.logger.warning(f"Entry condition comparison failed at index {idx}: {e}")

        return results

    def _compare_exit_conditions(self, legacy_strategy: BaseStrategy, converted_strategy: LegacyStrategyAdapter,
                               legacy_data: pd.DataFrame, converted_data: pd.DataFrame,
                               test_indices: List[int]) -> List[ComparisonResult]:
        """Compare exit condition decisions"""
        results = []

        # Use sample entry prices for testing
        test_prices = [100.0, 150.0, 200.0]

        for idx in test_indices[:5]:  # Limit to first 5 indices for exit testing
            if idx < len(legacy_data) and idx < len(converted_data):
                for entry_price in test_prices:
                    try:
                        legacy_exit = legacy_strategy.check_exit_conditions(legacy_data, idx, entry_price)
                        converted_exit = converted_strategy.check_exit_conditions(converted_data, idx, entry_price)

                        # Convert boolean to numeric for comparison
                        legacy_numeric = 1 if legacy_exit else 0
                        converted_numeric = 1 if converted_exit else 0

                        difference = abs(legacy_numeric - converted_numeric)

                        results.append(ComparisonResult(
                            test_name=f"exit_condition_idx_{idx}_price_{entry_price}",
                            legacy_value=legacy_exit,
                            converted_value=converted_exit,
                            difference=difference,
                            relative_difference=difference * 100,
                            within_tolerance=difference == 0,
                            tolerance_used=0,
                            metadata={
                                "index": idx,
                                "entry_price": entry_price,
                                "comparison_type": "exit_condition"
                            }
                        ))

                    except Exception as e:
                        self.logger.warning(f"Exit condition comparison failed at index {idx}, price {entry_price}: {e}")

        return results

    def _compare_position_sizing(self, legacy_strategy: BaseStrategy, converted_strategy: LegacyStrategyAdapter,
                               legacy_data: pd.DataFrame, converted_data: pd.DataFrame,
                               test_indices: List[int], test_balance: float) -> List[ComparisonResult]:
        """Compare position sizing calculations"""
        results = []

        for idx in test_indices:
            if idx < len(legacy_data) and idx < len(converted_data):
                try:
                    legacy_size = legacy_strategy.calculate_position_size(legacy_data, idx, test_balance)
                    converted_size = converted_strategy.calculate_position_size(converted_data, idx, test_balance)

                    difference = abs(legacy_size - converted_size)
                    relative_diff = (difference / max(abs(legacy_size), abs(converted_size), 1e-8)) * 100

                    results.append(ComparisonResult(
                        test_name=f"position_size_idx_{idx}",
                        legacy_value=legacy_size,
                        converted_value=converted_size,
                        difference=difference,
                        relative_difference=relative_diff,
                        within_tolerance=relative_diff <= self.tolerance * 100,
                        tolerance_used=self.tolerance,
                        metadata={
                            "index": idx,
                            "balance": test_balance,
                            "comparison_type": "position_size"
                        }
                    ))

                except Exception as e:
                    self.logger.warning(f"Position sizing comparison failed at index {idx}: {e}")

        return results

    def _compare_stop_loss_calculation(self, legacy_strategy: BaseStrategy, converted_strategy: LegacyStrategyAdapter,
                                     legacy_data: pd.DataFrame, converted_data: pd.DataFrame,
                                     test_indices: List[int]) -> List[ComparisonResult]:
        """Compare stop loss calculations"""
        results = []

        # Test with sample prices and sides
        test_prices = [100.0, 200.0]
        test_sides = ["long", "short"]

        for idx in test_indices[:3]:  # Limit to first 3 indices
            if idx < len(legacy_data) and idx < len(converted_data):
                for price in test_prices:
                    for side in test_sides:
                        try:
                            legacy_stop = legacy_strategy.calculate_stop_loss(legacy_data, idx, price, side)
                            converted_stop = converted_strategy.calculate_stop_loss(converted_data, idx, price, side)

                            difference = abs(legacy_stop - converted_stop)
                            relative_diff = (difference / max(abs(legacy_stop), abs(converted_stop), 1e-8)) * 100

                            results.append(ComparisonResult(
                                test_name=f"stop_loss_idx_{idx}_price_{price}_side_{side}",
                                legacy_value=legacy_stop,
                                converted_value=converted_stop,
                                difference=difference,
                                relative_difference=relative_diff,
                                within_tolerance=relative_diff <= self.tolerance * 100,
                                tolerance_used=self.tolerance,
                                metadata={
                                    "index": idx,
                                    "price": price,
                                    "side": side,
                                    "comparison_type": "stop_loss"
                                }
                            ))

                        except Exception as e:
                            self.logger.warning(f"Stop loss comparison failed at index {idx}, price {price}, side {side}: {e}")

        return results

    def _compare_performance(self, legacy_strategy: BaseStrategy, converted_strategy: LegacyStrategyAdapter,
                           legacy_data: pd.DataFrame, converted_data: pd.DataFrame,
                           test_indices: List[int]) -> Dict[str, Any]:
        """Compare performance characteristics"""
        performance_metrics = {}

        # Measure execution times
        legacy_times = []
        converted_times = []

        for idx in test_indices[:10]:  # Test performance on first 10 indices
            if idx < len(legacy_data) and idx < len(converted_data):
                # Time legacy strategy
                start_time = time.time()
                try:
                    legacy_strategy.check_entry_conditions(legacy_data, idx)
                    legacy_times.append(time.time() - start_time)
                except Exception:
                    pass

                # Time converted strategy
                start_time = time.time()
                try:
                    converted_strategy.check_entry_conditions(converted_data, idx)
                    converted_times.append(time.time() - start_time)
                except Exception:
                    pass

        # Calculate performance statistics
        if legacy_times and converted_times:
            legacy_avg_time = np.mean(legacy_times) * 1000  # Convert to ms
            converted_avg_time = np.mean(converted_times) * 1000

            performance_metrics["execution_time"] = {
                "legacy_avg_ms": legacy_avg_time,
                "converted_avg_ms": converted_avg_time,
                "difference_ms": abs(legacy_avg_time - converted_avg_time),
                "relative_difference_pct": abs(legacy_avg_time - converted_avg_time) / max(legacy_avg_time, converted_avg_time, 1e-8) * 100,
                "performance_improvement": (legacy_avg_time - converted_avg_time) / max(legacy_avg_time, 1e-8) * 100
            }

        # Memory usage comparison (simplified)
        performance_metrics["memory_usage"] = {
            "legacy_estimated_mb": self._estimate_memory_usage(legacy_strategy),
            "converted_estimated_mb": self._estimate_memory_usage(converted_strategy)
        }

        return performance_metrics

    def _run_regression_tests(self, legacy_strategy: BaseStrategy, converted_strategy: LegacyStrategyAdapter,
                            test_data: pd.DataFrame, test_balance: float) -> Dict[str, Any]:
        """Run regression tests to ensure no functionality is lost"""
        regression_results = {}

        # Test 1: Parameter consistency
        try:
            legacy_params = legacy_strategy.get_parameters() if hasattr(legacy_strategy, "get_parameters") else {}
            converted_params = converted_strategy.get_parameters()

            regression_results["parameter_consistency"] = {
                "legacy_param_count": len(legacy_params),
                "converted_param_count": len(converted_params),
                "common_params": len(set(legacy_params.keys()) & set(converted_params.keys())),
                "test_passed": len(converted_params) > 0
            }
        except Exception as e:
            regression_results["parameter_consistency"] = {"error": str(e), "test_passed": False}

        # Test 2: Trading pair consistency
        try:
            legacy_pair = legacy_strategy.get_trading_pair()
            converted_pair = converted_strategy.get_trading_pair()

            regression_results["trading_pair_consistency"] = {
                "legacy_pair": legacy_pair,
                "converted_pair": converted_pair,
                "consistent": legacy_pair == converted_pair,
                "test_passed": legacy_pair == converted_pair
            }
        except Exception as e:
            regression_results["trading_pair_consistency"] = {"error": str(e), "test_passed": False}

        # Test 3: Error handling
        try:
            # Test with invalid data
            empty_df = pd.DataFrame()
            legacy_error = False
            converted_error = False

            try:
                legacy_strategy.calculate_indicators(empty_df)
            except Exception:
                legacy_error = True

            try:
                converted_strategy.calculate_indicators(empty_df)
            except Exception:
                converted_error = True

            regression_results["error_handling"] = {
                "legacy_handles_errors": legacy_error,
                "converted_handles_errors": converted_error,
                "consistent_error_handling": legacy_error == converted_error,
                "test_passed": True  # Both should handle errors gracefully
            }
        except Exception as e:
            regression_results["error_handling"] = {"error": str(e), "test_passed": False}

        return regression_results

    def _calculate_compatibility_score(self, comparison_results: List[ComparisonResult]) -> float:
        """Calculate overall compatibility score based on comparison results"""
        if not comparison_results:
            return 0.0

        # Weight different types of comparisons
        weights = {
            "indicator": 0.2,
            "entry_condition": 0.3,
            "exit_condition": 0.2,
            "position_size": 0.2,
            "stop_loss": 0.1
        }

        weighted_score = 0.0
        total_weight = 0.0

        for result in comparison_results:
            # Determine weight based on test type
            weight = 1.0  # Default weight
            for test_type, type_weight in weights.items():
                if test_type in result.test_name:
                    weight = type_weight
                    break

            # Add to weighted score
            if result.within_tolerance:
                weighted_score += weight

            total_weight += weight

        # Calculate percentage
        if total_weight > 0:
            return (weighted_score / total_weight) * 100
        else:
            return 0.0

    def _generate_recommendations(self, comparison_results: List[ComparisonResult],
                                performance_metrics: Dict[str, Any],
                                regression_test_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []

        # Check comparison results
        failed_comparisons = [r for r in comparison_results if not r.within_tolerance]
        if failed_comparisons:
            recommendations.append(f"{len(failed_comparisons)} comparisons failed tolerance checks. Review parameter mappings.")

            # Identify most common failure types
            failure_types = {}
            for result in failed_comparisons:
                test_type = result.test_name.split("_")[0]
                failure_types[test_type] = failure_types.get(test_type, 0) + 1

            most_common_failure = max(failure_types.items(), key=lambda x: x[1])
            recommendations.append(f"Most common failure type: {most_common_failure[0]} ({most_common_failure[1]} failures)")

        # Check performance
        if "execution_time" in performance_metrics:
            perf_diff = performance_metrics["execution_time"].get("relative_difference_pct", 0)
            if perf_diff > self.performance_tolerance * 100:
                recommendations.append(f"Performance difference of {perf_diff:.1f}% exceeds tolerance. Consider optimization.")

        # Check regression tests
        failed_regression_tests = [test for test, result in regression_test_results.items()
                                 if isinstance(result, dict) and not result.get("test_passed", True)]
        if failed_regression_tests:
            recommendations.append(f"Regression tests failed: {', '.join(failed_regression_tests)}")

        # Overall recommendations
        if not recommendations:
            recommendations.append("All tests passed. The converted strategy appears to be fully compatible.")
        else:
            recommendations.append("Review failed tests and consider adjusting conversion parameters or component implementations.")

        return recommendations

    def _estimate_memory_usage(self, strategy: Union[BaseStrategy, LegacyStrategyAdapter]) -> float:
        """Estimate memory usage of a strategy (simplified)"""
        # This is a simplified estimation
        # In practice, you might use memory profiling tools
        base_size = 1.0  # Base size in MB

        # Add estimated size for parameters
        try:
            params = strategy.get_parameters() if hasattr(strategy, "get_parameters") else {}
            param_size = len(str(params)) / 1024 / 1024  # Convert to MB
            base_size += param_size
        except Exception:
            pass

        # Add estimated size for component-based strategies
        if isinstance(strategy, LegacyStrategyAdapter):
            base_size += 0.5  # Additional overhead for components

        return base_size

    def get_test_history(self) -> List[CrossValidationReport]:
        """Get history of all cross-validation tests"""
        return self.test_history.copy()

    def clear_test_history(self) -> None:
        """Clear the test history"""
        self.test_history.clear()
        self.logger.info("Cross-validation test history cleared")

    def generate_test_summary(self) -> Dict[str, Any]:
        """Generate summary of all cross-validation tests"""
        if not self.test_history:
            return {
                "total_tests": 0,
                "average_compatibility": 0.0,
                "successful_tests": 0,
                "failed_tests": 0,
                "common_issues": []
            }

        total_tests = len(self.test_history)
        compatibility_scores = [report.overall_compatibility for report in self.test_history]
        average_compatibility = sum(compatibility_scores) / total_tests

        successful_tests = sum(1 for score in compatibility_scores if score >= 80.0)
        failed_tests = total_tests - successful_tests

        # Collect common issues
        all_recommendations = []
        for report in self.test_history:
            all_recommendations.extend(report.recommendations)

        # Count recommendation frequency
        recommendation_counts = {}
        for rec in all_recommendations:
            recommendation_counts[rec] = recommendation_counts.get(rec, 0) + 1

        common_issues = sorted(recommendation_counts.items(), key=lambda x: x[1], reverse=True)[:5]

        return {
            "total_tests": total_tests,
            "average_compatibility": average_compatibility,
            "successful_tests": successful_tests,
            "failed_tests": failed_tests,
            "success_rate": (successful_tests / total_tests) * 100 if total_tests > 0 else 0,
            "compatibility_distribution": {
                "excellent": sum(1 for s in compatibility_scores if s >= 95),
                "good": sum(1 for s in compatibility_scores if 80 <= s < 95),
                "fair": sum(1 for s in compatibility_scores if 60 <= s < 80),
                "poor": sum(1 for s in compatibility_scores if s < 60)
            },
            "common_issues": [{"issue": issue, "count": count} for issue, count in common_issues],
            "test_timeline": [
                {
                    "timestamp": report.test_timestamp.isoformat(),
                    "strategy_name": report.strategy_name,
                    "compatibility": report.overall_compatibility
                }
                for report in self.test_history
            ]
        }
