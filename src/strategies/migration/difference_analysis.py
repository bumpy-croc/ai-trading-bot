"""
Difference Analysis and Reporting

This module provides detailed analysis and reporting of differences between
legacy and converted strategies, including statistical analysis, visualization
preparation, and comprehensive reporting capabilities.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from src.strategies.base import BaseStrategy
from src.strategies.adapters.legacy_adapter import LegacyStrategyAdapter


@dataclass
class DifferenceMetric:
    """
    Metric for measuring differences between strategies
    
    Attributes:
        metric_name: Name of the metric
        legacy_value: Value from legacy strategy
        converted_value: Value from converted strategy
        absolute_difference: Absolute difference
        relative_difference: Relative difference as percentage
        statistical_significance: P-value if applicable
        confidence_interval: Confidence interval for the difference
        interpretation: Human-readable interpretation
        severity: Severity level (low, medium, high, critical)
    """
    metric_name: str
    legacy_value: Union[float, int, bool, str]
    converted_value: Union[float, int, bool, str]
    absolute_difference: float
    relative_difference: float
    statistical_significance: Optional[float]
    confidence_interval: Optional[Tuple[float, float]]
    interpretation: str
    severity: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "metric_name": self.metric_name,
            "legacy_value": self.legacy_value,
            "converted_value": self.converted_value,
            "absolute_difference": self.absolute_difference,
            "relative_difference": self.relative_difference,
            "statistical_significance": self.statistical_significance,
            "confidence_interval": list(self.confidence_interval) if self.confidence_interval else None,
            "interpretation": self.interpretation,
            "severity": self.severity
        }


@dataclass
class DifferenceAnalysisReport:
    """
    Comprehensive difference analysis report
    
    Attributes:
        analysis_timestamp: When the analysis was performed
        strategy_name: Name of the strategy being analyzed
        legacy_strategy_type: Type of legacy strategy
        converted_strategy_type: Type of converted strategy
        test_data_summary: Summary of test data used
        difference_metrics: List of difference metrics
        statistical_summary: Statistical summary of differences
        severity_breakdown: Breakdown of differences by severity
        recommendations: Recommendations based on analysis
        visualization_data: Data prepared for visualization
    """
    analysis_timestamp: datetime
    strategy_name: str
    legacy_strategy_type: str
    converted_strategy_type: str
    test_data_summary: Dict[str, Any]
    difference_metrics: List[DifferenceMetric]
    statistical_summary: Dict[str, Any]
    severity_breakdown: Dict[str, int]
    recommendations: List[str]
    visualization_data: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "analysis_timestamp": self.analysis_timestamp.isoformat(),
            "strategy_name": self.strategy_name,
            "legacy_strategy_type": self.legacy_strategy_type,
            "converted_strategy_type": self.converted_strategy_type,
            "test_data_summary": self.test_data_summary,
            "difference_metrics": [metric.to_dict() for metric in self.difference_metrics],
            "statistical_summary": self.statistical_summary,
            "severity_breakdown": self.severity_breakdown,
            "recommendations": self.recommendations,
            "visualization_data": self.visualization_data
        }


class DifferenceAnalyzer:
    """
    Comprehensive difference analyzer for strategy migration
    
    This class provides detailed statistical analysis of differences between
    legacy and converted strategies, including significance testing, trend
    analysis, and comprehensive reporting.
    """

    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize the difference analyzer
        
        Args:
            confidence_level: Confidence level for statistical tests (default 95%)
        """
        self.logger = logging.getLogger("DifferenceAnalyzer")
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level

        # Analysis history
        self.analysis_history: List[DifferenceAnalysisReport] = []

    def analyze_differences(self, legacy_strategy: BaseStrategy,
                          converted_strategy: LegacyStrategyAdapter,
                          test_data: pd.DataFrame,
                          test_balance: float = 10000.0,
                          sample_size: int = 100) -> DifferenceAnalysisReport:
        """
        Perform comprehensive difference analysis between strategies
        
        Args:
            legacy_strategy: Original legacy strategy
            converted_strategy: Converted component-based strategy
            test_data: Test data for analysis
            test_balance: Test balance for calculations
            sample_size: Number of samples to analyze
            
        Returns:
            DifferenceAnalysisReport with detailed analysis
        """
        start_time = datetime.now()

        self.logger.info(f"Starting difference analysis for {legacy_strategy.name}")

        # Prepare test data summary
        test_data_summary = self._summarize_test_data(test_data)

        # Generate sample indices for analysis
        sample_indices = self._generate_sample_indices(test_data, sample_size)

        # Collect strategy outputs
        legacy_outputs = self._collect_strategy_outputs(legacy_strategy, test_data, sample_indices, test_balance)
        converted_outputs = self._collect_strategy_outputs(converted_strategy, test_data, sample_indices, test_balance)

        # Calculate difference metrics
        difference_metrics = self._calculate_difference_metrics(legacy_outputs, converted_outputs)

        # Perform statistical analysis
        statistical_summary = self._perform_statistical_analysis(legacy_outputs, converted_outputs)

        # Calculate severity breakdown
        severity_breakdown = self._calculate_severity_breakdown(difference_metrics)

        # Generate recommendations
        recommendations = self._generate_recommendations(difference_metrics, statistical_summary)

        # Prepare visualization data
        visualization_data = self._prepare_visualization_data(legacy_outputs, converted_outputs, difference_metrics)

        # Create analysis report
        report = DifferenceAnalysisReport(
            analysis_timestamp=start_time,
            strategy_name=f"{legacy_strategy.name}_vs_{converted_strategy.name}",
            legacy_strategy_type=legacy_strategy.__class__.__name__,
            converted_strategy_type=converted_strategy.__class__.__name__,
            test_data_summary=test_data_summary,
            difference_metrics=difference_metrics,
            statistical_summary=statistical_summary,
            severity_breakdown=severity_breakdown,
            recommendations=recommendations,
            visualization_data=visualization_data
        )

        # Store in history
        self.analysis_history.append(report)

        self.logger.info(f"Difference analysis completed: {len(difference_metrics)} metrics analyzed")

        return report

    def _summarize_test_data(self, test_data: pd.DataFrame) -> Dict[str, Any]:
        """Summarize test data characteristics"""
        return {
            "rows": len(test_data),
            "columns": list(test_data.columns),
            "date_range": {
                "start": test_data.index[0].isoformat() if hasattr(test_data.index, "__getitem__") else None,
                "end": test_data.index[-1].isoformat() if hasattr(test_data.index, "__getitem__") else None
            },
            "price_statistics": {
                "min": float(test_data["close"].min()) if "close" in test_data.columns else None,
                "max": float(test_data["close"].max()) if "close" in test_data.columns else None,
                "mean": float(test_data["close"].mean()) if "close" in test_data.columns else None,
                "std": float(test_data["close"].std()) if "close" in test_data.columns else None
            },
            "volume_statistics": {
                "min": float(test_data["volume"].min()) if "volume" in test_data.columns else None,
                "max": float(test_data["volume"].max()) if "volume" in test_data.columns else None,
                "mean": float(test_data["volume"].mean()) if "volume" in test_data.columns else None,
                "std": float(test_data["volume"].std()) if "volume" in test_data.columns else None
            }
        }

    def _generate_sample_indices(self, test_data: pd.DataFrame, sample_size: int) -> List[int]:
        """Generate representative sample indices"""
        data_length = len(test_data)

        if data_length <= sample_size:
            return list(range(max(50, data_length // 4), data_length))

        # Generate stratified sample
        start_idx = max(50, data_length // 10)  # Skip initial warm-up period
        end_idx = data_length - 1

        # Create evenly spaced indices
        indices = np.linspace(start_idx, end_idx, sample_size, dtype=int)

        return list(indices)

    def _collect_strategy_outputs(self, strategy: Union[BaseStrategy, LegacyStrategyAdapter],
                                test_data: pd.DataFrame, sample_indices: List[int],
                                test_balance: float) -> Dict[str, List[Any]]:
        """Collect strategy outputs for analysis"""
        outputs = {
            "entry_conditions": [],
            "position_sizes": [],
            "stop_losses_long": [],
            "stop_losses_short": [],
            "execution_times": []
        }

        # Prepare data
        prepared_data = strategy.calculate_indicators(test_data)

        # Test prices for stop loss calculations
        test_prices = [100.0, 150.0, 200.0]

        for idx in sample_indices:
            if 0 <= idx < len(prepared_data):
                # Entry conditions
                try:
                    start_time = datetime.now()
                    entry_condition = strategy.check_entry_conditions(prepared_data, idx)
                    execution_time = (datetime.now() - start_time).total_seconds() * 1000

                    outputs["entry_conditions"].append(entry_condition)
                    outputs["execution_times"].append(execution_time)
                except Exception as e:
                    outputs["entry_conditions"].append(None)
                    outputs["execution_times"].append(0.0)
                    self.logger.warning(f"Entry condition failed at index {idx}: {e}")

                # Position sizes
                try:
                    position_size = strategy.calculate_position_size(prepared_data, idx, test_balance)
                    outputs["position_sizes"].append(position_size)
                except Exception as e:
                    outputs["position_sizes"].append(0.0)
                    self.logger.warning(f"Position size calculation failed at index {idx}: {e}")

                # Stop losses
                for price in test_prices:
                    try:
                        stop_loss_long = strategy.calculate_stop_loss(prepared_data, idx, price, "long")
                        stop_loss_short = strategy.calculate_stop_loss(prepared_data, idx, price, "short")

                        outputs["stop_losses_long"].append(stop_loss_long)
                        outputs["stop_losses_short"].append(stop_loss_short)
                    except Exception as e:
                        outputs["stop_losses_long"].append(price * 0.95)  # Default fallback
                        outputs["stop_losses_short"].append(price * 1.05)  # Default fallback
                        self.logger.warning(f"Stop loss calculation failed at index {idx}, price {price}: {e}")

        return outputs

    def _calculate_difference_metrics(self, legacy_outputs: Dict[str, List[Any]],
                                    converted_outputs: Dict[str, List[Any]]) -> List[DifferenceMetric]:
        """Calculate comprehensive difference metrics"""
        metrics = []

        # Entry conditions analysis
        metrics.extend(self._analyze_entry_conditions(
            legacy_outputs["entry_conditions"],
            converted_outputs["entry_conditions"]
        ))

        # Position sizes analysis
        metrics.extend(self._analyze_position_sizes(
            legacy_outputs["position_sizes"],
            converted_outputs["position_sizes"]
        ))

        # Stop losses analysis
        metrics.extend(self._analyze_stop_losses(
            legacy_outputs["stop_losses_long"],
            converted_outputs["stop_losses_long"],
            "long"
        ))

        metrics.extend(self._analyze_stop_losses(
            legacy_outputs["stop_losses_short"],
            converted_outputs["stop_losses_short"],
            "short"
        ))

        # Execution time analysis
        metrics.extend(self._analyze_execution_times(
            legacy_outputs["execution_times"],
            converted_outputs["execution_times"]
        ))

        return metrics

    def _analyze_entry_conditions(self, legacy_entries: List[Any],
                                converted_entries: List[Any]) -> List[DifferenceMetric]:
        """Analyze differences in entry conditions"""
        metrics = []

        # Filter out None values
        valid_pairs = [(l, c) for l, c in zip(legacy_entries, converted_entries)
                      if l is not None and c is not None]

        if not valid_pairs:
            return metrics

        legacy_vals = [l for l, c in valid_pairs]
        converted_vals = [c for l, c in valid_pairs]

        # Convert boolean to numeric for analysis
        legacy_numeric = [1 if val else 0 for val in legacy_vals]
        converted_numeric = [1 if val else 0 for val in converted_vals]

        # Agreement rate
        agreements = sum(1 for l, c in zip(legacy_numeric, converted_numeric) if l == c)
        agreement_rate = agreements / len(valid_pairs) if valid_pairs else 0

        # Statistical test (McNemar's test for paired binary data)
        try:
            # Create contingency table
            both_true = sum(1 for l, c in zip(legacy_numeric, converted_numeric) if l == 1 and c == 1)
            legacy_only = sum(1 for l, c in zip(legacy_numeric, converted_numeric) if l == 1 and c == 0)
            converted_only = sum(1 for l, c in zip(legacy_numeric, converted_numeric) if l == 0 and c == 1)
            both_false = sum(1 for l, c in zip(legacy_numeric, converted_numeric) if l == 0 and c == 0)

            # McNemar's test
            if legacy_only + converted_only > 0:
                mcnemar_stat = (abs(legacy_only - converted_only) - 1) ** 2 / (legacy_only + converted_only)
                p_value = 1 - stats.chi2.cdf(mcnemar_stat, 1)
            else:
                p_value = 1.0  # Perfect agreement
        except Exception:
            p_value = None

        # Determine severity
        if agreement_rate >= 0.95:
            severity = "low"
        elif agreement_rate >= 0.85:
            severity = "medium"
        elif agreement_rate >= 0.70:
            severity = "high"
        else:
            severity = "critical"

        metrics.append(DifferenceMetric(
            metric_name="entry_condition_agreement_rate",
            legacy_value=f"{sum(legacy_numeric)}/{len(legacy_numeric)} signals",
            converted_value=f"{sum(converted_numeric)}/{len(converted_numeric)} signals",
            absolute_difference=abs(sum(legacy_numeric) - sum(converted_numeric)),
            relative_difference=(1 - agreement_rate) * 100,
            statistical_significance=p_value,
            confidence_interval=None,
            interpretation=f"Strategies agree on {agreement_rate:.1%} of entry decisions",
            severity=severity
        ))

        return metrics

    def _analyze_position_sizes(self, legacy_sizes: List[float],
                              converted_sizes: List[float]) -> List[DifferenceMetric]:
        """Analyze differences in position sizes"""
        metrics = []

        # Filter out zero and invalid values
        valid_pairs = [(l, c) for l, c in zip(legacy_sizes, converted_sizes)
                      if l > 0 and c > 0 and np.isfinite(l) and np.isfinite(c)]

        if not valid_pairs:
            return metrics

        legacy_vals = np.array([l for l, c in valid_pairs])
        converted_vals = np.array([c for l, c in valid_pairs])

        # Statistical analysis
        mean_legacy = np.mean(legacy_vals)
        mean_converted = np.mean(converted_vals)

        # Paired t-test
        try:
            t_stat, p_value = stats.ttest_rel(legacy_vals, converted_vals)
        except Exception:
            t_stat, p_value = None, None

        # Calculate differences
        differences = converted_vals - legacy_vals
        relative_differences = (differences / legacy_vals) * 100

        mean_abs_diff = np.mean(np.abs(differences))
        mean_rel_diff = np.mean(np.abs(relative_differences))

        # Confidence interval for mean difference
        try:
            diff_mean = np.mean(differences)
            diff_std = np.std(differences, ddof=1)
            n = len(differences)
            t_critical = stats.t.ppf(1 - self.alpha/2, n-1)
            margin_error = t_critical * diff_std / np.sqrt(n)
            ci = (diff_mean - margin_error, diff_mean + margin_error)
        except Exception:
            ci = None

        # Determine severity
        if mean_rel_diff <= 5:
            severity = "low"
        elif mean_rel_diff <= 15:
            severity = "medium"
        elif mean_rel_diff <= 30:
            severity = "high"
        else:
            severity = "critical"

        metrics.append(DifferenceMetric(
            metric_name="position_size_mean_difference",
            legacy_value=mean_legacy,
            converted_value=mean_converted,
            absolute_difference=mean_abs_diff,
            relative_difference=mean_rel_diff,
            statistical_significance=p_value,
            confidence_interval=ci,
            interpretation=f"Average position size differs by {mean_rel_diff:.1f}%",
            severity=severity
        ))

        return metrics

    def _analyze_stop_losses(self, legacy_stops: List[float],
                           converted_stops: List[float],
                           side: str) -> List[DifferenceMetric]:
        """Analyze differences in stop loss calculations"""
        metrics = []

        # Filter out invalid values
        valid_pairs = [(l, c) for l, c in zip(legacy_stops, converted_stops)
                      if l > 0 and c > 0 and np.isfinite(l) and np.isfinite(c)]

        if not valid_pairs:
            return metrics

        legacy_vals = np.array([l for l, c in valid_pairs])
        converted_vals = np.array([c for l, c in valid_pairs])

        # Statistical analysis
        mean_legacy = np.mean(legacy_vals)
        mean_converted = np.mean(converted_vals)

        # Paired t-test
        try:
            t_stat, p_value = stats.ttest_rel(legacy_vals, converted_vals)
        except Exception:
            t_stat, p_value = None, None

        # Calculate relative differences
        relative_differences = np.abs((converted_vals - legacy_vals) / legacy_vals) * 100
        mean_rel_diff = np.mean(relative_differences)

        # Determine severity
        if mean_rel_diff <= 2:
            severity = "low"
        elif mean_rel_diff <= 5:
            severity = "medium"
        elif mean_rel_diff <= 10:
            severity = "high"
        else:
            severity = "critical"

        metrics.append(DifferenceMetric(
            metric_name=f"stop_loss_{side}_difference",
            legacy_value=mean_legacy,
            converted_value=mean_converted,
            absolute_difference=abs(mean_converted - mean_legacy),
            relative_difference=mean_rel_diff,
            statistical_significance=p_value,
            confidence_interval=None,
            interpretation=f"Stop loss ({side}) differs by {mean_rel_diff:.1f}% on average",
            severity=severity
        ))

        return metrics

    def _analyze_execution_times(self, legacy_times: List[float],
                               converted_times: List[float]) -> List[DifferenceMetric]:
        """Analyze differences in execution times"""
        metrics = []

        # Filter out invalid values
        valid_pairs = [(l, c) for l, c in zip(legacy_times, converted_times)
                      if l >= 0 and c >= 0 and np.isfinite(l) and np.isfinite(c)]

        if not valid_pairs:
            return metrics

        legacy_vals = np.array([l for l, c in valid_pairs])
        converted_vals = np.array([c for l, c in valid_pairs])

        # Statistical analysis
        mean_legacy = np.mean(legacy_vals)
        mean_converted = np.mean(converted_vals)

        # Calculate performance improvement
        if mean_legacy > 0:
            performance_improvement = ((mean_legacy - mean_converted) / mean_legacy) * 100
        else:
            performance_improvement = 0

        # Determine severity (for performance, lower is better)
        if abs(performance_improvement) <= 10:
            severity = "low"
        elif abs(performance_improvement) <= 25:
            severity = "medium"
        elif performance_improvement < -50:  # Significant slowdown
            severity = "high"
        elif performance_improvement < -100:  # Major slowdown
            severity = "critical"
        else:
            severity = "low"  # Performance improvement is good

        metrics.append(DifferenceMetric(
            metric_name="execution_time_performance",
            legacy_value=mean_legacy,
            converted_value=mean_converted,
            absolute_difference=abs(mean_converted - mean_legacy),
            relative_difference=abs(performance_improvement),
            statistical_significance=None,
            confidence_interval=None,
            interpretation=f"Execution time changed by {performance_improvement:.1f}% ({'improved' if performance_improvement > 0 else 'degraded'})",
            severity=severity
        ))

        return metrics

    def _perform_statistical_analysis(self, legacy_outputs: Dict[str, List[Any]],
                                    converted_outputs: Dict[str, List[Any]]) -> Dict[str, Any]:
        """Perform comprehensive statistical analysis"""
        analysis = {}

        # Overall correlation analysis
        try:
            # Position sizes correlation
            legacy_sizes = [s for s in legacy_outputs["position_sizes"] if s > 0 and np.isfinite(s)]
            converted_sizes = [s for s in converted_outputs["position_sizes"] if s > 0 and np.isfinite(s)]

            if len(legacy_sizes) == len(converted_sizes) and len(legacy_sizes) > 1:
                correlation, p_value = stats.pearsonr(legacy_sizes, converted_sizes)
                analysis["position_size_correlation"] = {
                    "correlation": correlation,
                    "p_value": p_value,
                    "interpretation": self._interpret_correlation(correlation)
                }
        except Exception as e:
            analysis["position_size_correlation"] = {"error": str(e)}

        # Distribution comparison (Kolmogorov-Smirnov test)
        try:
            if len(legacy_sizes) > 5 and len(converted_sizes) > 5:
                ks_stat, ks_p_value = stats.ks_2samp(legacy_sizes, converted_sizes)
                analysis["distribution_comparison"] = {
                    "ks_statistic": ks_stat,
                    "p_value": ks_p_value,
                    "interpretation": "Distributions are significantly different" if ks_p_value < 0.05 else "Distributions are similar"
                }
        except Exception as e:
            analysis["distribution_comparison"] = {"error": str(e)}

        # Summary statistics
        analysis["summary_statistics"] = {
            "sample_size": len(legacy_outputs.get("entry_conditions", [])),
            "valid_position_sizes": len(legacy_sizes) if "legacy_sizes" in locals() else 0,
            "entry_signal_rate_legacy": np.mean([1 if e else 0 for e in legacy_outputs["entry_conditions"] if e is not None]),
            "entry_signal_rate_converted": np.mean([1 if e else 0 for e in converted_outputs["entry_conditions"] if e is not None])
        }

        return analysis

    def _interpret_correlation(self, correlation: float) -> str:
        """Interpret correlation coefficient"""
        abs_corr = abs(correlation)

        if abs_corr >= 0.9:
            return "Very strong correlation"
        elif abs_corr >= 0.7:
            return "Strong correlation"
        elif abs_corr >= 0.5:
            return "Moderate correlation"
        elif abs_corr >= 0.3:
            return "Weak correlation"
        else:
            return "Very weak or no correlation"

    def _calculate_severity_breakdown(self, difference_metrics: List[DifferenceMetric]) -> Dict[str, int]:
        """Calculate breakdown of differences by severity"""
        breakdown = {"low": 0, "medium": 0, "high": 0, "critical": 0}

        for metric in difference_metrics:
            if metric.severity in breakdown:
                breakdown[metric.severity] += 1

        return breakdown

    def _generate_recommendations(self, difference_metrics: List[DifferenceMetric],
                                statistical_summary: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on analysis"""
        recommendations = []

        # Check for critical differences
        critical_metrics = [m for m in difference_metrics if m.severity == "critical"]
        if critical_metrics:
            recommendations.append(f"CRITICAL: {len(critical_metrics)} metrics show critical differences. Migration should not proceed without addressing these issues.")
            for metric in critical_metrics:
                recommendations.append(f"  - {metric.metric_name}: {metric.interpretation}")

        # Check for high severity differences
        high_metrics = [m for m in difference_metrics if m.severity == "high"]
        if high_metrics:
            recommendations.append(f"HIGH: {len(high_metrics)} metrics show high differences. Review and consider adjustments.")

        # Check correlation
        pos_size_corr = statistical_summary.get("position_size_correlation", {})
        if "correlation" in pos_size_corr:
            corr = pos_size_corr["correlation"]
            if corr < 0.7:
                recommendations.append(f"Position size correlation is low ({corr:.3f}). Review position sizing component mapping.")

        # Check distribution differences
        dist_comp = statistical_summary.get("distribution_comparison", {})
        if "p_value" in dist_comp and dist_comp["p_value"] < 0.05:
            recommendations.append("Position size distributions are significantly different. Review parameter mappings.")

        # Performance recommendations
        exec_time_metrics = [m for m in difference_metrics if "execution_time" in m.metric_name]
        for metric in exec_time_metrics:
            if metric.severity in ["high", "critical"] and metric.relative_difference > 50:
                recommendations.append("Significant performance degradation detected. Consider optimization.")

        # Overall assessment
        if not critical_metrics and not high_metrics:
            recommendations.append("Analysis shows acceptable differences. Migration appears successful.")

        return recommendations

    def _prepare_visualization_data(self, legacy_outputs: Dict[str, List[Any]],
                                  converted_outputs: Dict[str, List[Any]],
                                  difference_metrics: List[DifferenceMetric]) -> Dict[str, Any]:
        """Prepare data for visualization"""
        viz_data = {}

        # Position size comparison data
        legacy_sizes = [s for s in legacy_outputs["position_sizes"] if s > 0 and np.isfinite(s)]
        converted_sizes = [s for s in converted_outputs["position_sizes"] if s > 0 and np.isfinite(s)]

        if len(legacy_sizes) == len(converted_sizes):
            viz_data["position_size_scatter"] = {
                "legacy": legacy_sizes,
                "converted": converted_sizes,
                "title": "Position Size Comparison",
                "xlabel": "Legacy Strategy Position Size",
                "ylabel": "Converted Strategy Position Size"
            }

        # Difference metrics summary for bar chart
        severity_counts = {}
        for metric in difference_metrics:
            severity_counts[metric.severity] = severity_counts.get(metric.severity, 0) + 1

        viz_data["severity_breakdown"] = {
            "categories": list(severity_counts.keys()),
            "counts": list(severity_counts.values()),
            "title": "Difference Severity Breakdown"
        }

        # Execution time comparison
        legacy_times = [t for t in legacy_outputs["execution_times"] if t >= 0 and np.isfinite(t)]
        converted_times = [t for t in converted_outputs["execution_times"] if t >= 0 and np.isfinite(t)]

        if legacy_times and converted_times:
            viz_data["execution_time_comparison"] = {
                "legacy_mean": np.mean(legacy_times),
                "converted_mean": np.mean(converted_times),
                "legacy_std": np.std(legacy_times),
                "converted_std": np.std(converted_times),
                "title": "Execution Time Comparison (ms)"
            }

        # Metric values for radar chart
        metric_names = [m.metric_name for m in difference_metrics]
        metric_severities = [m.severity for m in difference_metrics]

        viz_data["metrics_radar"] = {
            "metric_names": metric_names,
            "severity_scores": [{"low": 1, "medium": 2, "high": 3, "critical": 4}.get(s, 0) for s in metric_severities],
            "title": "Difference Metrics Overview"
        }

        return viz_data

    def get_analysis_history(self) -> List[DifferenceAnalysisReport]:
        """Get history of all difference analyses"""
        return self.analysis_history.copy()

    def clear_analysis_history(self) -> None:
        """Clear analysis history"""
        self.analysis_history.clear()
        self.logger.info("Difference analysis history cleared")

    def generate_analysis_summary(self) -> Dict[str, Any]:
        """Generate summary of all analyses performed"""
        if not self.analysis_history:
            return {
                "total_analyses": 0,
                "average_metrics_per_analysis": 0,
                "common_critical_issues": [],
                "overall_migration_success_rate": 0
            }

        total_analyses = len(self.analysis_history)
        total_metrics = sum(len(report.difference_metrics) for report in self.analysis_history)
        avg_metrics = total_metrics / total_analyses if total_analyses > 0 else 0

        # Collect critical issues
        all_critical_metrics = []
        for report in self.analysis_history:
            critical_metrics = [m for m in report.difference_metrics if m.severity == "critical"]
            all_critical_metrics.extend([m.metric_name for m in critical_metrics])

        # Count frequency of critical issues
        critical_counts = {}
        for metric_name in all_critical_metrics:
            critical_counts[metric_name] = critical_counts.get(metric_name, 0) + 1

        common_critical = sorted(critical_counts.items(), key=lambda x: x[1], reverse=True)[:5]

        # Calculate success rate (analyses with no critical issues)
        successful_analyses = sum(1 for report in self.analysis_history
                                if not any(m.severity == "critical" for m in report.difference_metrics))
        success_rate = (successful_analyses / total_analyses) * 100 if total_analyses > 0 else 0

        return {
            "total_analyses": total_analyses,
            "average_metrics_per_analysis": avg_metrics,
            "common_critical_issues": [{"issue": issue, "count": count} for issue, count in common_critical],
            "overall_migration_success_rate": success_rate,
            "severity_distribution": {
                "low": sum(report.severity_breakdown.get("low", 0) for report in self.analysis_history),
                "medium": sum(report.severity_breakdown.get("medium", 0) for report in self.analysis_history),
                "high": sum(report.severity_breakdown.get("high", 0) for report in self.analysis_history),
                "critical": sum(report.severity_breakdown.get("critical", 0) for report in self.analysis_history)
            }
        }
