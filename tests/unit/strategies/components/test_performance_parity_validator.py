"""
Unit tests for Performance Parity Validator.
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta

from src.strategies.components.testing.performance_parity_validator import (
    MetricComparison,
    MetricType,
    PerformanceComparisonReport,
    PerformanceParityValidator,
    ToleranceConfig,
    ValidationResult,
)


class TestToleranceConfig:
    """Test ToleranceConfig dataclass."""

    def test_default_values(self):
        """Test default tolerance values."""
        config = ToleranceConfig()

        assert config.total_return_tolerance == 0.02
        assert config.cagr_tolerance == 0.02
        assert config.max_drawdown_tolerance == 0.01
        assert config.volatility_tolerance == 0.05
        assert config.sharpe_ratio_tolerance == 0.1
        assert config.win_rate_tolerance == 0.05
        assert config.statistical_significance_level == 0.05
        assert config.minimum_sample_size == 30
        assert config.minimum_correlation == 0.95
        assert config.trade_count_tolerance == 0.1
        assert config.avg_trade_duration_tolerance == 0.2

    def test_custom_values(self):
        """Test custom tolerance values."""
        config = ToleranceConfig(
            total_return_tolerance=0.05, sharpe_ratio_tolerance=0.2, minimum_correlation=0.90
        )

        assert config.total_return_tolerance == 0.05
        assert config.sharpe_ratio_tolerance == 0.2
        assert config.minimum_correlation == 0.90
        # Other values should remain default
        assert config.cagr_tolerance == 0.02


class TestMetricComparison:
    """Test MetricComparison dataclass."""

    def test_metric_comparison_creation(self):
        """Test creating a metric comparison."""
        comparison = MetricComparison(
            metric_name="Total Return",
            metric_type=MetricType.RETURN,
            legacy_value=0.15,
            new_value=0.16,
            difference=0.01,
            relative_difference=0.01,
            tolerance=0.02,
            result=ValidationResult.PASS,
            p_value=0.5,
            notes="Test metric",
        )

        assert comparison.metric_name == "Total Return"
        assert comparison.metric_type == MetricType.RETURN
        assert comparison.legacy_value == 0.15
        assert comparison.new_value == 0.16
        assert comparison.difference == 0.01
        assert comparison.relative_difference == 0.01
        assert comparison.tolerance == 0.02
        assert comparison.result == ValidationResult.PASS
        assert comparison.p_value == 0.5
        assert comparison.notes == "Test metric"


class TestPerformanceParityValidator:
    """Test PerformanceParityValidator class."""

    @pytest.fixture
    def validator(self):
        """Create a validator instance."""
        return PerformanceParityValidator()

    @pytest.fixture
    def sample_backtest_data(self):
        """Create sample backtest data for testing."""
        dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")

        # Create legacy results with steady growth
        legacy_balance = 10000 * (1 + np.cumsum(np.random.normal(0.001, 0.02, len(dates))))
        legacy_results = pd.DataFrame(
            {
                "timestamp": dates,
                "balance": legacy_balance,
                "trade_pnl": np.random.normal(50, 100, len(dates)),
            }
        )

        # Create new results with similar but slightly different performance
        new_balance = 10000 * (1 + np.cumsum(np.random.normal(0.0012, 0.021, len(dates))))
        new_results = pd.DataFrame(
            {
                "timestamp": dates,
                "balance": new_balance,
                "trade_pnl": np.random.normal(52, 98, len(dates)),
            }
        )

        return legacy_results, new_results

    def test_validator_initialization(self):
        """Test validator initialization."""
        # Default initialization
        validator = PerformanceParityValidator()
        assert isinstance(validator.tolerance_config, ToleranceConfig)

        # Custom tolerance config
        custom_config = ToleranceConfig(total_return_tolerance=0.05)
        validator = PerformanceParityValidator(custom_config)
        assert validator.tolerance_config.total_return_tolerance == 0.05

    def test_validate_input_data_success(self, validator, sample_backtest_data):
        """Test successful input data validation."""
        legacy_results, new_results = sample_backtest_data

        # Should not raise any exception
        validator._validate_input_data(legacy_results, new_results)

    def test_validate_input_data_empty_dataframe(self, validator):
        """Test input validation with empty DataFrame."""
        empty_df = pd.DataFrame()
        valid_df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01", periods=10),
                "balance": np.random.randn(10) + 10000,
            }
        )

        with pytest.raises(ValueError, match="empty"):
            validator._validate_input_data(empty_df, valid_df)

    def test_validate_input_data_missing_columns(self, validator):
        """Test input validation with missing columns."""
        invalid_df = pd.DataFrame({"price": [1, 2, 3]})
        valid_df = pd.DataFrame(
            {"timestamp": pd.date_range("2023-01-01", periods=3), "balance": [10000, 10100, 10200]}
        )

        with pytest.raises(ValueError, match="missing columns"):
            validator._validate_input_data(invalid_df, valid_df)

    def test_validate_input_data_no_temporal_overlap(self, validator):
        """Test input validation with no temporal overlap."""
        df1 = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01", periods=10),
                "balance": np.random.randn(10) + 10000,
            }
        )
        df2 = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=10),
                "balance": np.random.randn(10) + 10000,
            }
        )

        with pytest.raises(ValueError, match="No temporal overlap"):
            validator._validate_input_data(df1, df2)

    def test_create_metric_comparison_absolute_tolerance(self, validator):
        """Test creating metric comparison with absolute tolerance."""
        comparison = validator._create_metric_comparison(
            "Test Metric",
            MetricType.RETURN,
            0.10,  # legacy value
            0.11,  # new value
            0.02,  # tolerance
            use_relative_tolerance=False,
        )

        assert comparison.metric_name == "Test Metric"
        assert comparison.metric_type == MetricType.RETURN
        assert comparison.legacy_value == pytest.approx(0.10)
        assert comparison.new_value == pytest.approx(0.11)
        assert comparison.difference == pytest.approx(0.01)
        assert comparison.relative_difference == pytest.approx(0.01)
        assert comparison.tolerance == pytest.approx(0.02)
        assert comparison.result == ValidationResult.PASS  # Within tolerance

    def test_create_metric_comparison_relative_tolerance(self, validator):
        """Test creating metric comparison with relative tolerance."""
        comparison = validator._create_metric_comparison(
            "Test Metric",
            MetricType.RISK,
            0.10,  # legacy value
            0.12,  # new value (20% increase)
            0.15,  # 15% relative tolerance
            use_relative_tolerance=True,
        )

        assert comparison.relative_difference == pytest.approx(0.2)  # 20% relative difference
        assert comparison.result == ValidationResult.FAIL  # Outside 15% tolerance

    def test_compare_return_metrics(self, validator, sample_backtest_data):
        """Test return metrics comparison."""
        legacy_results, new_results = sample_backtest_data
        report = PerformanceComparisonReport(
            strategy_name="Test",
            comparison_period="test",
            legacy_strategy_id="legacy",
            new_strategy_id="new",
            overall_result=ValidationResult.INCONCLUSIVE,
        )

        validator._compare_return_metrics(legacy_results, new_results, report)

        # Should have added return metrics
        metric_names = [comp.metric_name for comp in report.metric_comparisons]
        assert "Total Return" in metric_names
        assert "CAGR" in metric_names

        # All should be return type
        return_metrics = [
            comp for comp in report.metric_comparisons if comp.metric_type == MetricType.RETURN
        ]
        assert len(return_metrics) >= 2

    def test_cagr_calculation_different_periods(self, validator):
        """Test that CAGR calculation uses separate periods for legacy and new strategies."""
        # Create legacy results with 365 days - 10% annual growth
        legacy_start = pd.Timestamp("2023-01-01")
        legacy_end = pd.Timestamp("2023-12-31")
        legacy_final_balance = 10000 * (1.1 ** (365 / 365))  # 10% growth over 365 days
        legacy_results = pd.DataFrame(
            {"timestamp": [legacy_start, legacy_end], "balance": [10000, legacy_final_balance]}
        )

        # Create new results with 180 days - 10% annual growth
        new_start = pd.Timestamp("2023-07-01")
        new_end = pd.Timestamp("2023-12-28")
        new_final_balance = 10000 * (1.1 ** (180 / 365))  # 10% annualized growth over 180 days
        new_results = pd.DataFrame(
            {"timestamp": [new_start, new_end], "balance": [10000, new_final_balance]}
        )

        report = PerformanceComparisonReport(
            strategy_name="Test",
            comparison_period="test",
            legacy_strategy_id="legacy",
            new_strategy_id="new",
            overall_result=ValidationResult.INCONCLUSIVE,
        )

        validator._compare_return_metrics(legacy_results, new_results, report)

        # Find CAGR comparison
        cagr_comparison = next(
            comp for comp in report.metric_comparisons if comp.metric_name == "CAGR"
        )

        # Both should have ~10% CAGR since both are designed for 10% annual growth
        assert abs(cagr_comparison.legacy_value - 0.10) < 0.01  # ~10% CAGR
        assert abs(cagr_comparison.new_value - 0.10) < 0.01  # ~10% CAGR

        # The difference should be small since both are 10% annualized
        assert abs(cagr_comparison.difference) < 0.01

    def test_compare_risk_metrics(self, validator, sample_backtest_data):
        """Test risk metrics comparison."""
        legacy_results, new_results = sample_backtest_data
        report = PerformanceComparisonReport(
            strategy_name="Test",
            comparison_period="test",
            legacy_strategy_id="legacy",
            new_strategy_id="new",
            overall_result=ValidationResult.INCONCLUSIVE,
        )

        validator._compare_risk_metrics(legacy_results, new_results, report)

        # Should have added risk metrics
        metric_names = [comp.metric_name for comp in report.metric_comparisons]
        assert "Maximum Drawdown" in metric_names
        assert "Annualized Volatility" in metric_names

        # All should be risk type
        risk_metrics = [
            comp for comp in report.metric_comparisons if comp.metric_type == MetricType.RISK
        ]
        assert len(risk_metrics) >= 2

    def test_compare_efficiency_metrics(self, validator, sample_backtest_data):
        """Test efficiency metrics comparison."""
        legacy_results, new_results = sample_backtest_data
        report = PerformanceComparisonReport(
            strategy_name="Test",
            comparison_period="test",
            legacy_strategy_id="legacy",
            new_strategy_id="new",
            overall_result=ValidationResult.INCONCLUSIVE,
        )

        validator._compare_efficiency_metrics(legacy_results, new_results, report)

        # Should have added efficiency metrics
        metric_names = [comp.metric_name for comp in report.metric_comparisons]
        assert "Sharpe Ratio" in metric_names
        assert "Win Rate" in metric_names  # Should be present due to trade_pnl column

        # All should be efficiency type
        efficiency_metrics = [
            comp for comp in report.metric_comparisons if comp.metric_type == MetricType.EFFICIENCY
        ]
        assert len(efficiency_metrics) >= 2

    def test_compare_trade_metrics(self, validator, sample_backtest_data):
        """Test trade metrics comparison."""
        legacy_results, new_results = sample_backtest_data
        report = PerformanceComparisonReport(
            strategy_name="Test",
            comparison_period="test",
            legacy_strategy_id="legacy",
            new_strategy_id="new",
            overall_result=ValidationResult.INCONCLUSIVE,
        )

        validator._compare_trade_metrics(legacy_results, new_results, report)

        # Should have set trade counts
        assert report.trade_count_legacy > 0
        assert report.trade_count_new > 0

        # Should have added trade count comparison
        metric_names = [comp.metric_name for comp in report.metric_comparisons]
        assert "Trade Count" in metric_names

    def test_analyze_equity_curve_correlation(self, validator, sample_backtest_data):
        """Test equity curve correlation analysis."""
        legacy_results, new_results = sample_backtest_data
        report = PerformanceComparisonReport(
            strategy_name="Test",
            comparison_period="test",
            legacy_strategy_id="legacy",
            new_strategy_id="new",
            overall_result=ValidationResult.INCONCLUSIVE,
        )

        validator._analyze_equity_curve_correlation(legacy_results, new_results, report)

        # Should have calculated correlation (can be negative)
        assert -1 <= report.equity_curve_correlation <= 1

        # Should have added correlation metric
        metric_names = [comp.metric_name for comp in report.metric_comparisons]
        assert "Equity Curve Correlation" in metric_names

    def test_determine_overall_result_pass(self, validator):
        """Test determining overall result when all metrics pass."""
        report = PerformanceComparisonReport(
            strategy_name="Test",
            comparison_period="test",
            legacy_strategy_id="legacy",
            new_strategy_id="new",
            overall_result=ValidationResult.INCONCLUSIVE,
        )

        # Add passing metrics
        report.metric_comparisons = [
            MetricComparison(
                metric_name="Test 1",
                metric_type=MetricType.RETURN,
                legacy_value=0.1,
                new_value=0.1,
                difference=0.0,
                relative_difference=0.0,
                tolerance=0.02,
                result=ValidationResult.PASS,
            ),
            MetricComparison(
                metric_name="Test 2",
                metric_type=MetricType.RISK,
                legacy_value=0.05,
                new_value=0.05,
                difference=0.0,
                relative_difference=0.0,
                tolerance=0.01,
                result=ValidationResult.PASS,
            ),
        ]

        validator._determine_overall_result(report)

        assert report.overall_result == ValidationResult.PASS
        assert report.certified == True
        assert report.total_metrics_tested == 2
        assert report.metrics_passed == 2
        assert report.metrics_failed == 0
        assert report.metrics_warning == 0

    def test_determine_overall_result_fail(self, validator):
        """Test determining overall result when metrics fail."""
        report = PerformanceComparisonReport(
            strategy_name="Test",
            comparison_period="test",
            legacy_strategy_id="legacy",
            new_strategy_id="new",
            overall_result=ValidationResult.INCONCLUSIVE,
        )

        # Add failing metric
        report.metric_comparisons = [
            MetricComparison(
                metric_name="Test 1",
                metric_type=MetricType.RETURN,
                legacy_value=0.1,
                new_value=0.2,
                difference=0.1,
                relative_difference=0.1,
                tolerance=0.02,
                result=ValidationResult.FAIL,
            )
        ]

        validator._determine_overall_result(report)

        assert report.overall_result == ValidationResult.FAIL
        assert report.certified == False
        assert report.metrics_failed == 1

    def test_determine_overall_result_warning(self, validator):
        """Test determining overall result with warnings."""
        report = PerformanceComparisonReport(
            strategy_name="Test",
            comparison_period="test",
            legacy_strategy_id="legacy",
            new_strategy_id="new",
            overall_result=ValidationResult.INCONCLUSIVE,
        )

        # Add warning metric
        report.metric_comparisons = [
            MetricComparison(
                metric_name="Test 1",
                metric_type=MetricType.TIMING,
                legacy_value=100,
                new_value=110,
                difference=10,
                relative_difference=0.1,
                tolerance=0.1,
                result=ValidationResult.WARNING,
            )
        ]

        validator._determine_overall_result(report)

        assert report.overall_result == ValidationResult.WARNING
        assert report.certified == False
        assert report.metrics_warning == 1

    def test_full_validation_workflow(self, validator, sample_backtest_data):
        """Test the complete validation workflow."""
        legacy_results, new_results = sample_backtest_data

        report = validator.validate_performance_parity(
            legacy_results, new_results, "Test Strategy", "legacy_test", "new_test", "full_test"
        )

        # Check basic report structure
        assert report.strategy_name == "Test Strategy"
        assert report.legacy_strategy_id == "legacy_test"
        assert report.new_strategy_id == "new_test"
        assert report.comparison_period == "full_test"

        # Should have various metrics
        assert len(report.metric_comparisons) > 0

        # Should have determined overall result
        assert report.overall_result in [
            ValidationResult.PASS,
            ValidationResult.WARNING,
            ValidationResult.FAIL,
        ]

        # Should have counts
        assert report.total_metrics_tested > 0
        assert report.total_metrics_tested == (
            report.metrics_passed + report.metrics_failed + report.metrics_warning
        )

        # Should have correlation (can be negative)
        assert -1 <= report.equity_curve_correlation <= 1

    def test_generate_certification_report(self, validator, sample_backtest_data):
        """Test generating certification report."""
        legacy_results, new_results = sample_backtest_data

        report = validator.validate_performance_parity(
            legacy_results, new_results, "Test Strategy", "legacy_test", "new_test"
        )

        cert_report = validator.generate_certification_report(report)

        # Should be a string
        assert isinstance(cert_report, str)

        # Should contain key information
        assert "PERFORMANCE PARITY VALIDATION REPORT" in cert_report
        assert "Test Strategy" in cert_report
        assert "OVERALL RESULT:" in cert_report
        assert "DETAILED METRIC COMPARISONS:" in cert_report

        # Should contain metric details
        for comparison in report.metric_comparisons:
            assert comparison.metric_name in cert_report


class TestPerformanceComparisonReport:
    """Test PerformanceComparisonReport dataclass."""

    def test_report_creation(self):
        """Test creating a performance comparison report."""
        report = PerformanceComparisonReport(
            strategy_name="Test Strategy",
            comparison_period="2023",
            legacy_strategy_id="legacy_v1",
            new_strategy_id="new_v2",
            overall_result=ValidationResult.PASS,
        )

        assert report.strategy_name == "Test Strategy"
        assert report.comparison_period == "2023"
        assert report.legacy_strategy_id == "legacy_v1"
        assert report.new_strategy_id == "new_v2"
        assert report.overall_result == ValidationResult.PASS

        # Default values
        assert report.metric_comparisons == []
        assert report.equity_curve_correlation == 0.0
        assert report.trade_count_legacy == 0
        assert report.trade_count_new == 0
        assert report.certified == False
        assert report.certification_timestamp is None
