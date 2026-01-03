"""
Performance Comparison Engine

This module provides a high-level interface for comparing strategy performance
with comprehensive validation, statistical testing, and reporting capabilities.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

from src.strategies.components import Strategy

# Note: Using a simplified interface for backtesting
# In production, this would integrate with the actual Backtester class


class BacktestEngineProtocol(Protocol):
    """Protocol for backtesting engines used in performance comparison."""

    def run_backtest(self, strategy: Any, data: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        """
        Run a backtest and return results DataFrame.

        Args:
            strategy: Strategy to backtest
            data: Market data for backtesting
            **kwargs: Additional configuration parameters

        Returns:
            DataFrame with columns: timestamp, balance, and optionally trade_pnl
        """
        ...


from src.strategies.components.testing.performance_parity_validator import (
    PerformanceComparisonReport,
    PerformanceParityValidator,
    ToleranceConfig,
    ValidationResult,
)
from src.strategies.components.testing.statistical_tests import (
    EquivalenceTests,
    FinancialStatisticalTests,
    StatisticalTestResult,
    format_test_results,
)

logger = logging.getLogger(__name__)


@dataclass
class ComparisonConfig:
    """Configuration for strategy performance comparison."""

    # Validation tolerances
    tolerance_config: ToleranceConfig = field(default_factory=ToleranceConfig)

    # Statistical test configuration
    statistical_significance_level: float = 0.05
    equivalence_margin: float = 0.01  # 1% equivalence margin

    # Backtesting configuration
    initial_balance: float = 10000.0
    commission_rate: float = 0.001  # 0.1% commission

    # Reporting configuration
    generate_detailed_report: bool = True
    export_results: bool = True
    export_directory: str | None = None

    # Validation requirements
    require_statistical_equivalence: bool = True
    require_performance_parity: bool = True
    minimum_correlation_threshold: float = 0.95


@dataclass
class StrategyComparisonResult:
    """Complete result of strategy comparison."""

    # Basic information
    comparison_id: str
    timestamp: datetime
    legacy_strategy_name: str
    new_strategy_name: str

    # Performance parity validation
    parity_report: PerformanceComparisonReport

    # Statistical test results
    statistical_tests: dict[str, list[StatisticalTestResult]]

    # Equivalence test results
    equivalence_tests: list[StatisticalTestResult]

    # Overall assessment
    overall_validation_result: ValidationResult
    certification_status: str
    recommendations: list[str] = field(default_factory=list)

    # Raw data for further analysis
    legacy_backtest_results: pd.DataFrame | None = None
    new_backtest_results: pd.DataFrame | None = None


class PerformanceComparisonEngine:
    """
    High-level engine for comprehensive strategy performance comparison.

    This engine orchestrates the entire comparison process including:
    - Running backtests for both strategies
    - Performing performance parity validation
    - Conducting statistical tests
    - Generating comprehensive reports
    - Providing certification recommendations
    """

    def __init__(
        self,
        config: ComparisonConfig | None = None,
        backtest_engine: BacktestEngineProtocol | None = None,
    ):
        """
        Initialize the performance comparison engine.

        Args:
            config: Configuration for comparison process
            backtest_engine: Backtesting engine (required, will raise ValueError if None when compare_strategies is called)
        """
        self.config = config or ComparisonConfig()
        self.backtest_engine = backtest_engine  # Will be provided by caller

        # Initialize validators and test engines
        self.parity_validator = PerformanceParityValidator(self.config.tolerance_config)
        self.statistical_tests = FinancialStatisticalTests(
            self.config.statistical_significance_level
        )
        self.equivalence_tests = EquivalenceTests(self.config.equivalence_margin)

        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def compare_strategies(
        self,
        legacy_strategy: Strategy,
        new_strategy: Strategy,
        market_data: pd.DataFrame,
        comparison_id: str | None = None,
    ) -> StrategyComparisonResult:
        """
        Perform comprehensive comparison between legacy and new strategies.

        Args:
            legacy_strategy: Legacy strategy implementation
            new_strategy: New strategy implementation
            market_data: Historical market data for backtesting
            comparison_id: Unique identifier for this comparison

        Returns:
            Complete comparison result with validation and recommendations
        """
        comparison_id = comparison_id or f"comparison_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"

        self.logger.info(f"Starting strategy comparison: {comparison_id}")

        try:
            # Check if backtest engine is available
            if self.backtest_engine is None:
                raise ValueError(
                    "No backtest engine provided. Please provide a backtest engine to run comparisons."
                )

            # Run backtests for both strategies
            self.logger.info("Running backtests...")
            legacy_results = self._run_backtest(legacy_strategy, market_data, "legacy")
            new_results = self._run_backtest(new_strategy, market_data, "new")

            # Perform performance parity validation
            self.logger.info("Performing performance parity validation...")
            parity_report = self.parity_validator.validate_performance_parity(
                legacy_results,
                new_results,
                f"{legacy_strategy.__class__.__name__} vs {new_strategy.__class__.__name__}",
                f"legacy_{legacy_strategy.__class__.__name__}",
                f"new_{new_strategy.__class__.__name__}",
                f"backtest_{datetime.now(UTC).strftime('%Y-%m-%d')}",
            )

            # Perform statistical tests
            self.logger.info("Performing statistical tests...")
            statistical_results = self._perform_statistical_analysis(legacy_results, new_results)

            # Perform equivalence tests
            self.logger.info("Performing equivalence tests...")
            equivalence_results = self._perform_equivalence_tests(legacy_results, new_results)

            # Create comparison result
            result = StrategyComparisonResult(
                comparison_id=comparison_id,
                timestamp=datetime.now(UTC),
                legacy_strategy_name=legacy_strategy.__class__.__name__,
                new_strategy_name=new_strategy.__class__.__name__,
                parity_report=parity_report,
                statistical_tests=statistical_results,
                equivalence_tests=equivalence_results,
                overall_validation_result=ValidationResult.INCONCLUSIVE,
                certification_status="Pending",
                legacy_backtest_results=legacy_results,
                new_backtest_results=new_results,
            )

            # Determine overall validation result and recommendations
            self._assess_overall_result(result)

            # Export results if configured
            if self.config.export_results:
                self._export_results(result)

            self.logger.info(f"Strategy comparison completed: {result.overall_validation_result}")
            return result

        except (ValueError, KeyError, TypeError) as e:
            self.logger.exception("Strategy comparison failed")
            # Return a failed result
            return StrategyComparisonResult(
                comparison_id=comparison_id,
                timestamp=datetime.now(UTC),
                legacy_strategy_name=legacy_strategy.__class__.__name__,
                new_strategy_name=new_strategy.__class__.__name__,
                parity_report=PerformanceComparisonReport(
                    strategy_name="Failed Comparison",
                    comparison_period="N/A",
                    legacy_strategy_id="N/A",
                    new_strategy_id="N/A",
                    overall_result=ValidationResult.FAIL,
                ),
                statistical_tests={},
                equivalence_tests=[],
                overall_validation_result=ValidationResult.FAIL,
                certification_status=f"Failed: {e!s}",
                recommendations=[f"Fix error: {e!s}"],
            )

    def _run_backtest(
        self, strategy: Strategy, market_data: pd.DataFrame, strategy_type: str
    ) -> pd.DataFrame:
        """Run backtest for a strategy and return results."""

        self.logger.debug(
            f"Running backtest for {strategy_type} strategy: {strategy.__class__.__name__}"
        )

        # Configure backtest parameters
        backtest_config = {
            "initial_balance": self.config.initial_balance,
            "commission_rate": self.config.commission_rate,
        }

        # Run the backtest
        # Note: This is a simplified interface - actual implementation would depend on BacktestEngine API
        results = self.backtest_engine.run_backtest(
            strategy=strategy, data=market_data, **backtest_config
        )

        # Ensure results have required columns
        if "balance" not in results.columns:
            raise ValueError(
                f"Backtest results missing 'balance' column for {strategy_type} strategy"
            )

        if "timestamp" not in results.columns:
            # Check if timestamp is in index names (handles both single index and MultiIndex)
            # For MultiIndex, index.name is None but index.names is a list containing level names
            if hasattr(results.index, "names") and "timestamp" in results.index.names:
                results = results.reset_index(level="timestamp")
                # Verify that reset_index actually created a timestamp column
                if "timestamp" not in results.columns:
                    raise ValueError(
                        f"Backtest results index named 'timestamp' but reset_index did not create 'timestamp' column for {strategy_type} strategy"
                    )
            else:
                raise ValueError(
                    f"Backtest results missing 'timestamp' column for {strategy_type} strategy"
                )

        return results

    def _perform_statistical_analysis(
        self, legacy_results: pd.DataFrame, new_results: pd.DataFrame
    ) -> dict[str, list[StatisticalTestResult]]:
        """Perform comprehensive statistical analysis."""

        # Extract return series
        legacy_returns = legacy_results["balance"].pct_change().dropna()
        new_returns = new_results["balance"].pct_change().dropna()

        # Perform comprehensive comparison
        return self.statistical_tests.comprehensive_comparison(
            legacy_returns, new_returns, "Legacy Strategy", "New Strategy"
        )

    def _perform_equivalence_tests(
        self, legacy_results: pd.DataFrame, new_results: pd.DataFrame
    ) -> list[StatisticalTestResult]:
        """Perform equivalence tests."""

        legacy_returns = legacy_results["balance"].pct_change().dropna()
        new_returns = new_results["balance"].pct_change().dropna()

        # Two One-Sided Test for mean equivalence
        tost_result = self.equivalence_tests.two_one_sided_test(legacy_returns, new_returns)

        return [tost_result]

    def _assess_overall_result(self, result: StrategyComparisonResult) -> None:
        """Assess overall validation result and generate recommendations."""

        # Start with parity validation result
        parity_result = result.parity_report.overall_result

        # Check statistical test results using list comprehension
        statistical_failures = [
            f"{category}: {test.test_name}"
            for category, tests in result.statistical_tests.items()
            for test in tests
            if test.reject_null and "equality" in test.test_name.lower()
        ]

        # Check equivalence test results
        # TOST (Two One-Sided Test) tests for equivalence, so we check if reject_null is True
        # reject_null=True means we reject the null hypothesis of non-equivalence (i.e., equivalence confirmed)
        equivalence_passed = any(test.reject_null for test in result.equivalence_tests)

        # Determine overall result
        if parity_result == ValidationResult.PASS:
            if not statistical_failures and equivalence_passed:
                result.overall_validation_result = ValidationResult.PASS
                result.certification_status = "CERTIFIED - Full Performance Parity"
                result.recommendations = [
                    "Migration can proceed with confidence",
                    "Performance parity validated across all metrics",
                    "Statistical equivalence confirmed",
                ]
            elif not statistical_failures:
                result.overall_validation_result = ValidationResult.WARNING
                result.certification_status = (
                    "CONDITIONAL - Performance Parity with Statistical Differences"
                )
                result.recommendations = [
                    "Migration can proceed with caution",
                    "Performance parity validated but some statistical differences detected",
                    "Monitor performance closely after migration",
                ]
            else:
                result.overall_validation_result = ValidationResult.WARNING
                result.certification_status = "CONDITIONAL - Performance Parity with Concerns"
                result.recommendations = [
                    "Migration requires careful consideration",
                    "Performance parity achieved but statistical tests show differences",
                    f"Statistical failures in: {', '.join(statistical_failures)}",
                ]
        else:
            result.overall_validation_result = ValidationResult.FAIL
            result.certification_status = "NOT CERTIFIED - Performance Parity Failed"
            result.recommendations = [
                "Migration should not proceed without addressing performance differences",
                "Investigate causes of performance discrepancies",
                "Consider adjusting strategy parameters or implementation",
            ]

        # Add specific recommendations based on parity report
        if result.parity_report.metrics_failed > 0:
            failed_metrics = [
                comp.metric_name
                for comp in result.parity_report.metric_comparisons
                if comp.result == ValidationResult.FAIL
            ]
            result.recommendations.append(
                f"Failed metrics requiring attention: {', '.join(failed_metrics)}"
            )

        # Add correlation-specific recommendations
        if (
            result.parity_report.equity_curve_correlation
            < self.config.minimum_correlation_threshold
        ):
            result.recommendations.append(
                f"Low equity curve correlation ({result.parity_report.equity_curve_correlation:.3f}) "
                f"indicates significant behavioral differences"
            )

    def _export_results(self, result: StrategyComparisonResult) -> None:
        """Export comparison results to files."""

        export_dir = Path(self.config.export_directory or "comparison_results")
        export_dir.mkdir(exist_ok=True)

        timestamp = result.timestamp.strftime("%Y%m%d_%H%M%S")
        base_filename = f"{result.comparison_id}_{timestamp}"

        try:
            # Export parity report
            from src.strategies.components.testing.performance_parity_validator import (
                PerformanceParityReporter,
            )

            # CSV export
            PerformanceParityReporter.export_to_csv(
                result.parity_report, str(export_dir / f"{base_filename}_parity_metrics.csv")
            )

            # JSON export
            PerformanceParityReporter.export_to_json(
                result.parity_report, str(export_dir / f"{base_filename}_parity_report.json")
            )

            # Generate text report
            text_report = self.generate_text_report(result)
            with open(export_dir / f"{base_filename}_full_report.txt", "w") as f:
                f.write(text_report)

            # Export raw backtest data if available
            if result.legacy_backtest_results is not None:
                result.legacy_backtest_results.to_csv(
                    export_dir / f"{base_filename}_legacy_backtest.csv", index=False
                )

            if result.new_backtest_results is not None:
                result.new_backtest_results.to_csv(
                    export_dir / f"{base_filename}_new_backtest.csv", index=False
                )

            self.logger.info(f"Results exported to {export_dir}")

        except (OSError, ValueError, KeyError) as e:
            self.logger.warning("Failed to export results: %s", e)

    def generate_text_report(self, result: StrategyComparisonResult) -> str:
        """Generate comprehensive text report."""

        lines = [
            "=" * 100,
            "COMPREHENSIVE STRATEGY COMPARISON REPORT",
            "=" * 100,
            f"Comparison ID: {result.comparison_id}",
            f"Timestamp: {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Legacy Strategy: {result.legacy_strategy_name}",
            f"New Strategy: {result.new_strategy_name}",
            "",
            f"OVERALL VALIDATION RESULT: {result.overall_validation_result.value.upper()}",
            f"CERTIFICATION STATUS: {result.certification_status}",
            "",
            "RECOMMENDATIONS:",
            "-" * 50,
        ]

        for i, rec in enumerate(result.recommendations, 1):
            lines.append(f"{i}. {rec}")

        lines.extend(["", "PERFORMANCE PARITY VALIDATION:", "-" * 50])

        # Add parity report summary
        parity_summary = [
            f"Overall Result: {result.parity_report.overall_result.value.upper()}",
            f"Metrics Tested: {result.parity_report.total_metrics_tested}",
            f"Metrics Passed: {result.parity_report.metrics_passed}",
            f"Metrics Failed: {result.parity_report.metrics_failed}",
            f"Metrics with Warnings: {result.parity_report.metrics_warning}",
            f"Equity Curve Correlation: {result.parity_report.equity_curve_correlation:.6f}",
            f"Certified: {'YES' if result.parity_report.certified else 'NO'}",
        ]

        lines.extend(parity_summary)

        # Add detailed metric comparisons
        lines.extend(["", "DETAILED METRIC COMPARISONS:", "-" * 50])

        for comp in result.parity_report.metric_comparisons:
            lines.extend(
                [
                    f"Metric: {comp.metric_name}",
                    f"  Legacy Value: {comp.legacy_value:.6f}",
                    f"  New Value: {comp.new_value:.6f}",
                    f"  Difference: {comp.difference:.6f}",
                    f"  Result: {comp.result.value.upper()}",
                    "",
                ]
            )

        # Add statistical test results
        lines.extend(["", "STATISTICAL TEST RESULTS:", "-" * 50])

        statistical_report = format_test_results(result.statistical_tests)
        lines.append(statistical_report)

        # Add equivalence test results
        lines.extend(["", "EQUIVALENCE TEST RESULTS:", "-" * 50])

        for test in result.equivalence_tests:
            lines.extend(
                [
                    f"Test: {test.test_name}",
                    f"  Statistic: {test.statistic:.6f}",
                    f"  P-value: {test.p_value:.6f}",
                    f"  Result: {'EQUIVALENT' if test.reject_null else 'NOT EQUIVALENT'}",
                    f"  Interpretation: {test.interpretation}",
                    "",
                ]
            )

        lines.extend(["", "=" * 100])

        return "\n".join(lines)


# Convenience functions for common use cases


def quick_strategy_comparison(
    legacy_strategy: Strategy,
    new_strategy: Strategy,
    market_data: pd.DataFrame,
    tolerance_config: ToleranceConfig | None = None,
) -> StrategyComparisonResult:
    """
    Quick strategy comparison with default settings.

    Args:
        legacy_strategy: Legacy strategy to compare
        new_strategy: New strategy to compare
        market_data: Market data for backtesting
        tolerance_config: Optional custom tolerance configuration

    Returns:
        Strategy comparison result
    """
    config = ComparisonConfig()
    if tolerance_config:
        config.tolerance_config = tolerance_config

    engine = PerformanceComparisonEngine(config)
    return engine.compare_strategies(legacy_strategy, new_strategy, market_data)


def validate_migration_readiness(
    legacy_strategy: Strategy,
    new_strategy: Strategy,
    market_data: pd.DataFrame,
    strict_validation: bool = True,
) -> tuple[bool, list[str]]:
    """
    Validate if a strategy migration is ready for production.

    Args:
        legacy_strategy: Legacy strategy
        new_strategy: New strategy
        market_data: Market data for testing
        strict_validation: Whether to use strict validation criteria

    Returns:
        Tuple of (is_ready, list_of_issues)
    """
    # Configure strict or lenient validation
    tolerance_config = ToleranceConfig()
    if not strict_validation:
        # Relax tolerances for lenient validation
        tolerance_config.total_return_tolerance = 0.05  # 5%
        tolerance_config.sharpe_ratio_tolerance = 0.2
        tolerance_config.minimum_correlation = 0.90

    config = ComparisonConfig(
        tolerance_config=tolerance_config,
        require_statistical_equivalence=strict_validation,
        require_performance_parity=True,
    )

    engine = PerformanceComparisonEngine(config)
    result = engine.compare_strategies(legacy_strategy, new_strategy, market_data)

    # Determine readiness
    is_ready = result.overall_validation_result in [ValidationResult.PASS, ValidationResult.WARNING]

    # Collect issues
    issues = []
    if result.overall_validation_result == ValidationResult.FAIL:
        issues.extend(result.recommendations)

    if result.parity_report.metrics_failed > 0:
        failed_metrics = [
            comp.metric_name
            for comp in result.parity_report.metric_comparisons
            if comp.result == ValidationResult.FAIL
        ]
        issues.append(f"Failed performance metrics: {', '.join(failed_metrics)}")

    return is_ready, issues
