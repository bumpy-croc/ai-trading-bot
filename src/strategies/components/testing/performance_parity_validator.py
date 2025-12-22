"""
Performance Parity Validation System

This module provides comprehensive performance comparison and validation tools
for ensuring that migrated strategies maintain performance parity with their
legacy counterparts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from src.performance.metrics import (
    cagr,
    max_drawdown,
    sharpe,
    total_return,
)

logger = logging.getLogger(__name__)


class ValidationResult(str, Enum):
    """Performance validation result status."""

    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"
    INCONCLUSIVE = "inconclusive"


class MetricType(str, Enum):
    """Types of performance metrics for validation."""

    RETURN = "return"
    RISK = "risk"
    EFFICIENCY = "efficiency"
    ACCURACY = "accuracy"
    TIMING = "timing"


@dataclass
class ToleranceConfig:
    """Configuration for acceptable performance differences."""

    # Return metrics tolerances (as decimal percentages)
    total_return_tolerance: float = 0.02  # 2% absolute difference
    cagr_tolerance: float = 0.02  # 2% absolute difference

    # Risk metrics tolerances
    max_drawdown_tolerance: float = 0.01  # 1% absolute difference
    volatility_tolerance: float = 0.05  # 5% relative difference

    # Efficiency metrics tolerances
    sharpe_ratio_tolerance: float = 0.1  # 0.1 absolute difference
    win_rate_tolerance: float = 0.05  # 5% absolute difference

    # Statistical significance levels
    statistical_significance_level: float = 0.05  # 5% significance level
    minimum_sample_size: int = 30  # Minimum trades for statistical tests

    # Correlation requirements
    minimum_correlation: float = 0.95  # 95% correlation between equity curves

    # Trade-level tolerances
    trade_count_tolerance: float = 0.1  # 10% difference in trade count
    avg_trade_duration_tolerance: float = 0.2  # 20% difference in duration


@dataclass
class MetricComparison:
    """Comparison result for a single metric."""

    metric_name: str
    metric_type: MetricType
    legacy_value: float
    new_value: float
    difference: float
    relative_difference: float
    tolerance: float
    result: ValidationResult
    p_value: float | None = None
    confidence_interval: tuple[float, float] | None = None
    notes: str = ""


@dataclass
class PerformanceComparisonReport:
    """Comprehensive performance comparison report."""

    strategy_name: str
    comparison_period: str
    legacy_strategy_id: str
    new_strategy_id: str

    # Overall validation result
    overall_result: ValidationResult

    # Individual metric comparisons
    metric_comparisons: list[MetricComparison] = field(default_factory=list)

    # Statistical tests
    equity_curve_correlation: float = 0.0
    kolmogorov_smirnov_test: tuple[float, float] | None = None
    mann_whitney_test: tuple[float, float] | None = None

    # Trade-level analysis
    trade_count_legacy: int = 0
    trade_count_new: int = 0
    trade_timing_correlation: float = 0.0

    # Summary statistics
    total_metrics_tested: int = 0
    metrics_passed: int = 0
    metrics_failed: int = 0
    metrics_warning: int = 0

    # Certification
    certified: bool = False
    certification_timestamp: datetime | None = None
    certification_notes: str = ""

    # Detailed analysis
    detailed_analysis: dict[str, Any] = field(default_factory=dict)


class PerformanceParityValidator:
    """
    Comprehensive performance parity validation system.

    This class provides tools to compare performance between legacy and new
    strategy implementations, ensuring that migrations maintain performance
    equivalence within acceptable tolerances.
    """

    def __init__(self, tolerance_config: ToleranceConfig | None = None):
        """
        Initialize the performance parity validator.

        Args:
            tolerance_config: Configuration for acceptable performance differences
        """
        self.tolerance_config = tolerance_config or ToleranceConfig()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def validate_performance_parity(
        self,
        legacy_results: pd.DataFrame,
        new_results: pd.DataFrame,
        strategy_name: str,
        legacy_strategy_id: str,
        new_strategy_id: str,
        comparison_period: str = "full_backtest",
    ) -> PerformanceComparisonReport:
        """
        Perform comprehensive performance parity validation.

        Args:
            legacy_results: Backtest results from legacy strategy
            new_results: Backtest results from new strategy
            strategy_name: Name of the strategy being compared
            legacy_strategy_id: Identifier for legacy strategy
            new_strategy_id: Identifier for new strategy
            comparison_period: Description of the comparison period

        Returns:
            Comprehensive performance comparison report
        """
        self.logger.info(f"Starting performance parity validation for {strategy_name}")

        # Initialize report
        report = PerformanceComparisonReport(
            strategy_name=strategy_name,
            comparison_period=comparison_period,
            legacy_strategy_id=legacy_strategy_id,
            new_strategy_id=new_strategy_id,
            overall_result=ValidationResult.INCONCLUSIVE,
        )

        try:
            # Validate input data
            self._validate_input_data(legacy_results, new_results)

            # Perform metric comparisons
            self._compare_return_metrics(legacy_results, new_results, report)
            self._compare_risk_metrics(legacy_results, new_results, report)
            self._compare_efficiency_metrics(legacy_results, new_results, report)
            self._compare_trade_metrics(legacy_results, new_results, report)

            # Perform statistical tests
            self._perform_statistical_tests(legacy_results, new_results, report)

            # Analyze equity curve correlation
            self._analyze_equity_curve_correlation(legacy_results, new_results, report)

            # Determine overall result
            self._determine_overall_result(report)

            # Generate detailed analysis
            self._generate_detailed_analysis(legacy_results, new_results, report)

            self.logger.info(f"Performance parity validation completed: {report.overall_result}")

        except Exception as e:
            self.logger.error(f"Error during performance parity validation: {e}")
            report.overall_result = ValidationResult.FAIL
            report.certification_notes = f"Validation failed due to error: {str(e)}"

        return report

    def _validate_input_data(self, legacy_results: pd.DataFrame, new_results: pd.DataFrame) -> None:
        """Validate that input data is suitable for comparison."""

        required_columns = ["balance", "timestamp"]

        for df, name in [(legacy_results, "legacy"), (new_results, "new")]:
            if df.empty:
                raise ValueError(f"{name} results DataFrame is empty")

            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                raise ValueError(f"{name} results missing columns: {missing_cols}")

        # Check for reasonable overlap in time periods
        legacy_start = legacy_results["timestamp"].min()
        legacy_end = legacy_results["timestamp"].max()
        new_start = new_results["timestamp"].min()
        new_end = new_results["timestamp"].max()

        overlap_start = max(legacy_start, new_start)
        overlap_end = min(legacy_end, new_end)

        if overlap_start >= overlap_end:
            raise ValueError("No temporal overlap between legacy and new results")

        self.logger.debug("Input data validation passed")

    def _compare_return_metrics(
        self,
        legacy_results: pd.DataFrame,
        new_results: pd.DataFrame,
        report: PerformanceComparisonReport,
    ) -> None:
        """Compare return-based performance metrics."""

        # Calculate return metrics for both strategies
        legacy_initial = legacy_results["balance"].iloc[0]
        legacy_final = legacy_results["balance"].iloc[-1]
        new_initial = new_results["balance"].iloc[0]
        new_final = new_results["balance"].iloc[-1]

        # Total return comparison
        legacy_total_return = total_return(legacy_initial, legacy_final) / 100.0
        new_total_return = total_return(new_initial, new_final) / 100.0

        report.metric_comparisons.append(
            self._create_metric_comparison(
                "Total Return",
                MetricType.RETURN,
                legacy_total_return,
                new_total_return,
                self.tolerance_config.total_return_tolerance,
            )
        )

        # CAGR comparison - compute period separately for each DataFrame
        legacy_days = (
            legacy_results["timestamp"].iloc[-1] - legacy_results["timestamp"].iloc[0]
        ).days
        new_days = (new_results["timestamp"].iloc[-1] - new_results["timestamp"].iloc[0]).days
        legacy_cagr = cagr(legacy_initial, legacy_final, legacy_days) / 100.0
        new_cagr = cagr(new_initial, new_final, new_days) / 100.0

        report.metric_comparisons.append(
            self._create_metric_comparison(
                "CAGR",
                MetricType.RETURN,
                legacy_cagr,
                new_cagr,
                self.tolerance_config.cagr_tolerance,
            )
        )

    def _compare_risk_metrics(
        self,
        legacy_results: pd.DataFrame,
        new_results: pd.DataFrame,
        report: PerformanceComparisonReport,
    ) -> None:
        """Compare risk-based performance metrics."""

        # Maximum drawdown comparison
        legacy_mdd = max_drawdown(legacy_results["balance"]) / 100.0
        new_mdd = max_drawdown(new_results["balance"]) / 100.0

        report.metric_comparisons.append(
            self._create_metric_comparison(
                "Maximum Drawdown",
                MetricType.RISK,
                legacy_mdd,
                new_mdd,
                self.tolerance_config.max_drawdown_tolerance,
            )
        )

        # Volatility comparison (daily returns standard deviation)
        legacy_daily_returns = legacy_results["balance"].pct_change().dropna()
        new_daily_returns = new_results["balance"].pct_change().dropna()

        legacy_volatility = legacy_daily_returns.std() * np.sqrt(365)
        new_volatility = new_daily_returns.std() * np.sqrt(365)

        report.metric_comparisons.append(
            self._create_metric_comparison(
                "Annualized Volatility",
                MetricType.RISK,
                legacy_volatility,
                new_volatility,
                self.tolerance_config.volatility_tolerance,
                use_relative_tolerance=True,
            )
        )

    def _compare_efficiency_metrics(
        self,
        legacy_results: pd.DataFrame,
        new_results: pd.DataFrame,
        report: PerformanceComparisonReport,
    ) -> None:
        """Compare efficiency-based performance metrics."""

        # Sharpe ratio comparison
        legacy_sharpe = sharpe(legacy_results["balance"])
        new_sharpe = sharpe(new_results["balance"])

        report.metric_comparisons.append(
            self._create_metric_comparison(
                "Sharpe Ratio",
                MetricType.EFFICIENCY,
                legacy_sharpe,
                new_sharpe,
                self.tolerance_config.sharpe_ratio_tolerance,
            )
        )

        # Win rate comparison (if trade data available)
        if "trade_pnl" in legacy_results.columns and "trade_pnl" in new_results.columns:
            legacy_trades = legacy_results["trade_pnl"].dropna()
            new_trades = new_results["trade_pnl"].dropna()

            if len(legacy_trades) > 0 and len(new_trades) > 0:
                legacy_win_rate = (legacy_trades > 0).mean()
                new_win_rate = (new_trades > 0).mean()

                report.metric_comparisons.append(
                    self._create_metric_comparison(
                        "Win Rate",
                        MetricType.EFFICIENCY,
                        legacy_win_rate,
                        new_win_rate,
                        self.tolerance_config.win_rate_tolerance,
                    )
                )

    def _compare_trade_metrics(
        self,
        legacy_results: pd.DataFrame,
        new_results: pd.DataFrame,
        report: PerformanceComparisonReport,
    ) -> None:
        """Compare trade-level metrics."""

        # Trade count comparison
        legacy_trades = legacy_results.get("trade_pnl", pd.Series()).dropna()
        new_trades = new_results.get("trade_pnl", pd.Series()).dropna()

        report.trade_count_legacy = len(legacy_trades)
        report.trade_count_new = len(new_trades)

        if len(legacy_trades) > 0 and len(new_trades) > 0:
            trade_count_diff = abs(len(legacy_trades) - len(new_trades)) / len(legacy_trades)

            result = (
                ValidationResult.PASS
                if trade_count_diff <= self.tolerance_config.trade_count_tolerance
                else ValidationResult.WARNING
            )

            report.metric_comparisons.append(
                MetricComparison(
                    metric_name="Trade Count",
                    metric_type=MetricType.TIMING,
                    legacy_value=len(legacy_trades),
                    new_value=len(new_trades),
                    difference=len(new_trades) - len(legacy_trades),
                    relative_difference=trade_count_diff,
                    tolerance=self.tolerance_config.trade_count_tolerance,
                    result=result,
                )
            )

    def _perform_statistical_tests(
        self,
        legacy_results: pd.DataFrame,
        new_results: pd.DataFrame,
        report: PerformanceComparisonReport,
    ) -> None:
        """Perform statistical tests for performance equivalence."""

        # Prepare return series for testing
        legacy_returns = legacy_results["balance"].pct_change().dropna()
        new_returns = new_results["balance"].pct_change().dropna()

        # Ensure we have enough data for statistical tests
        min_size = min(len(legacy_returns), len(new_returns))
        if min_size < self.tolerance_config.minimum_sample_size:
            self.logger.warning(
                f"Insufficient data for statistical tests: {min_size} < {self.tolerance_config.minimum_sample_size}"
            )
            return

        # Kolmogorov-Smirnov test for distribution similarity
        try:
            ks_statistic, ks_p_value = stats.ks_2samp(legacy_returns, new_returns)
            report.kolmogorov_smirnov_test = (ks_statistic, ks_p_value)

            # Add KS test result to comparisons
            ks_result = (
                ValidationResult.PASS
                if ks_p_value > self.tolerance_config.statistical_significance_level
                else ValidationResult.FAIL
            )

            report.metric_comparisons.append(
                MetricComparison(
                    metric_name="Distribution Similarity (KS Test)",
                    metric_type=MetricType.ACCURACY,
                    legacy_value=0.0,  # Not applicable
                    new_value=0.0,  # Not applicable
                    difference=ks_statistic,
                    relative_difference=0.0,
                    tolerance=self.tolerance_config.statistical_significance_level,
                    result=ks_result,
                    p_value=ks_p_value,
                    notes=f"KS statistic: {ks_statistic:.4f}",
                )
            )

        except Exception as e:
            self.logger.warning(f"Failed to perform KS test: {e}")

        # Mann-Whitney U test for median difference
        try:
            mw_statistic, mw_p_value = stats.mannwhitneyu(
                legacy_returns, new_returns, alternative="two-sided"
            )
            report.mann_whitney_test = (mw_statistic, mw_p_value)

            # Add MW test result to comparisons
            mw_result = (
                ValidationResult.PASS
                if mw_p_value > self.tolerance_config.statistical_significance_level
                else ValidationResult.FAIL
            )

            report.metric_comparisons.append(
                MetricComparison(
                    metric_name="Median Difference (Mann-Whitney Test)",
                    metric_type=MetricType.ACCURACY,
                    legacy_value=legacy_returns.median(),
                    new_value=new_returns.median(),
                    difference=new_returns.median() - legacy_returns.median(),
                    relative_difference=0.0,
                    tolerance=self.tolerance_config.statistical_significance_level,
                    result=mw_result,
                    p_value=mw_p_value,
                    notes=f"MW statistic: {mw_statistic:.4f}",
                )
            )

        except Exception as e:
            self.logger.warning(f"Failed to perform Mann-Whitney test: {e}")

    def _analyze_equity_curve_correlation(
        self,
        legacy_results: pd.DataFrame,
        new_results: pd.DataFrame,
        report: PerformanceComparisonReport,
    ) -> None:
        """Analyze correlation between equity curves."""

        try:
            # Align the data by timestamp for correlation analysis
            legacy_aligned = legacy_results.set_index("timestamp")["balance"]
            new_aligned = new_results.set_index("timestamp")["balance"]

            # Find common timestamps
            common_timestamps = legacy_aligned.index.intersection(new_aligned.index)

            if len(common_timestamps) < 10:
                self.logger.warning("Insufficient overlapping data for correlation analysis")
                return

            # Calculate correlation
            legacy_common = legacy_aligned.loc[common_timestamps]
            new_common = new_aligned.loc[common_timestamps]

            correlation = legacy_common.corr(new_common)
            report.equity_curve_correlation = correlation

            # Add correlation to metric comparisons
            corr_result = (
                ValidationResult.PASS
                if correlation >= self.tolerance_config.minimum_correlation
                else ValidationResult.FAIL
            )

            report.metric_comparisons.append(
                MetricComparison(
                    metric_name="Equity Curve Correlation",
                    metric_type=MetricType.ACCURACY,
                    legacy_value=1.0,  # Perfect correlation with itself
                    new_value=correlation,
                    difference=1.0 - correlation,
                    relative_difference=1.0 - correlation,
                    tolerance=1.0 - self.tolerance_config.minimum_correlation,
                    result=corr_result,
                    notes=f"Correlation: {correlation:.4f}",
                )
            )

        except Exception as e:
            self.logger.warning(f"Failed to calculate equity curve correlation: {e}")

    def _create_metric_comparison(
        self,
        metric_name: str,
        metric_type: MetricType,
        legacy_value: float,
        new_value: float,
        tolerance: float,
        use_relative_tolerance: bool = False,
    ) -> MetricComparison:
        """Create a metric comparison object."""

        difference = new_value - legacy_value

        if use_relative_tolerance and legacy_value != 0:
            relative_difference = abs(difference) / abs(legacy_value)
            within_tolerance = relative_difference <= tolerance
        else:
            relative_difference = abs(difference)
            within_tolerance = abs(difference) <= tolerance

        result = ValidationResult.PASS if within_tolerance else ValidationResult.FAIL

        return MetricComparison(
            metric_name=metric_name,
            metric_type=metric_type,
            legacy_value=legacy_value,
            new_value=new_value,
            difference=difference,
            relative_difference=relative_difference,
            tolerance=tolerance,
            result=result,
        )

    def _determine_overall_result(self, report: PerformanceComparisonReport) -> None:
        """Determine the overall validation result."""

        # Count results
        report.total_metrics_tested = len(report.metric_comparisons)
        report.metrics_passed = sum(
            1 for m in report.metric_comparisons if m.result == ValidationResult.PASS
        )
        report.metrics_failed = sum(
            1 for m in report.metric_comparisons if m.result == ValidationResult.FAIL
        )
        report.metrics_warning = sum(
            1 for m in report.metric_comparisons if m.result == ValidationResult.WARNING
        )

        # Determine overall result
        if report.metrics_failed == 0:
            if report.metrics_warning == 0:
                report.overall_result = ValidationResult.PASS
                report.certified = True
                report.certification_timestamp = datetime.now()
                report.certification_notes = "All metrics passed validation"
            else:
                report.overall_result = ValidationResult.WARNING
                report.certification_notes = f"{report.metrics_warning} metrics had warnings"
        else:
            report.overall_result = ValidationResult.FAIL
            report.certification_notes = f"{report.metrics_failed} metrics failed validation"

    def _generate_detailed_analysis(
        self,
        legacy_results: pd.DataFrame,
        new_results: pd.DataFrame,
        report: PerformanceComparisonReport,
    ) -> None:
        """Generate detailed analysis for the report."""

        report.detailed_analysis = {
            "data_quality": {
                "legacy_data_points": len(legacy_results),
                "new_data_points": len(new_results),
                "temporal_overlap": self._calculate_temporal_overlap(legacy_results, new_results),
            },
            "performance_summary": {
                "legacy_final_balance": legacy_results["balance"].iloc[-1],
                "new_final_balance": new_results["balance"].iloc[-1],
                "balance_difference": new_results["balance"].iloc[-1]
                - legacy_results["balance"].iloc[-1],
            },
            "risk_analysis": {
                "legacy_max_balance": legacy_results["balance"].max(),
                "new_max_balance": new_results["balance"].max(),
                "legacy_min_balance": legacy_results["balance"].min(),
                "new_min_balance": new_results["balance"].min(),
            },
        }

    def _calculate_temporal_overlap(
        self, legacy_results: pd.DataFrame, new_results: pd.DataFrame
    ) -> dict[str, Any]:
        """Calculate temporal overlap statistics."""

        legacy_start = legacy_results["timestamp"].min()
        legacy_end = legacy_results["timestamp"].max()
        new_start = new_results["timestamp"].min()
        new_end = new_results["timestamp"].max()

        overlap_start = max(legacy_start, new_start)
        overlap_end = min(legacy_end, new_end)

        legacy_duration = (legacy_end - legacy_start).total_seconds()
        new_duration = (new_end - new_start).total_seconds()
        overlap_duration = max(0, (overlap_end - overlap_start).total_seconds())

        return {
            "legacy_period": f"{legacy_start} to {legacy_end}",
            "new_period": f"{new_start} to {new_end}",
            "overlap_period": f"{overlap_start} to {overlap_end}",
            "overlap_percentage": overlap_duration / max(legacy_duration, new_duration) * 100,
        }

    def generate_certification_report(self, report: PerformanceComparisonReport) -> str:
        """Generate a human-readable certification report."""

        lines = [
            "=" * 80,
            "PERFORMANCE PARITY VALIDATION REPORT",
            "=" * 80,
            f"Strategy: {report.strategy_name}",
            f"Comparison Period: {report.comparison_period}",
            f"Legacy Strategy ID: {report.legacy_strategy_id}",
            f"New Strategy ID: {report.new_strategy_id}",
            f"Validation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            f"OVERALL RESULT: {report.overall_result.value.upper()}",
            f"Certified: {'YES' if report.certified else 'NO'}",
            "",
            "SUMMARY STATISTICS:",
            f"  Total Metrics Tested: {report.total_metrics_tested}",
            f"  Metrics Passed: {report.metrics_passed}",
            f"  Metrics Failed: {report.metrics_failed}",
            f"  Metrics with Warnings: {report.metrics_warning}",
            "",
            "DETAILED METRIC COMPARISONS:",
            "-" * 80,
        ]

        for comparison in report.metric_comparisons:
            lines.extend(
                [
                    f"Metric: {comparison.metric_name} ({comparison.metric_type.value})",
                    f"  Legacy Value: {comparison.legacy_value:.6f}",
                    f"  New Value: {comparison.new_value:.6f}",
                    f"  Difference: {comparison.difference:.6f}",
                    f"  Relative Difference: {comparison.relative_difference:.6f}",
                    f"  Tolerance: {comparison.tolerance:.6f}",
                    f"  Result: {comparison.result.value.upper()}",
                    f"  P-Value: {comparison.p_value:.6f}" if comparison.p_value else "",
                    f"  Notes: {comparison.notes}" if comparison.notes else "",
                    "",
                ]
            )

        lines.extend(
            [
                "STATISTICAL ANALYSIS:",
                f"  Equity Curve Correlation: {report.equity_curve_correlation:.6f}",
                (
                    f"  KS Test: {report.kolmogorov_smirnov_test}"
                    if report.kolmogorov_smirnov_test
                    else ""
                ),
                (
                    f"  Mann-Whitney Test: {report.mann_whitney_test}"
                    if report.mann_whitney_test
                    else ""
                ),
                "",
                "CERTIFICATION NOTES:",
                report.certification_notes,
                "",
                "=" * 80,
            ]
        )

        return "\n".join(lines)


class PerformanceParityReporter:
    """
    Reporting utilities for performance parity validation results.
    """

    @staticmethod
    def export_to_csv(report: PerformanceComparisonReport, filepath: str) -> None:
        """Export metric comparisons to CSV format."""

        data = []
        for comparison in report.metric_comparisons:
            data.append(
                {
                    "metric_name": comparison.metric_name,
                    "metric_type": comparison.metric_type.value,
                    "legacy_value": comparison.legacy_value,
                    "new_value": comparison.new_value,
                    "difference": comparison.difference,
                    "relative_difference": comparison.relative_difference,
                    "tolerance": comparison.tolerance,
                    "result": comparison.result.value,
                    "p_value": comparison.p_value,
                    "notes": comparison.notes,
                }
            )

        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)

    @staticmethod
    def export_to_json(report: PerformanceComparisonReport, filepath: str) -> None:
        """Export full report to JSON format."""

        import json
        from dataclasses import asdict

        # Convert dataclass to dictionary
        report_dict = asdict(report)

        # Handle datetime serialization
        if report_dict.get("certification_timestamp"):
            report_dict["certification_timestamp"] = report_dict[
                "certification_timestamp"
            ].isoformat()

        with open(filepath, "w") as f:
            json.dump(report_dict, f, indent=2, default=str)
