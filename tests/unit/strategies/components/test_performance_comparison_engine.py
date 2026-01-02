"""
Unit tests for Performance Comparison Engine.
"""

from datetime import UTC, datetime
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from src.strategies.components.testing.performance_comparison_engine import (
    ComparisonConfig,
    PerformanceComparisonEngine,
    StrategyComparisonResult,
    quick_strategy_comparison,
    validate_migration_readiness,
)
from src.strategies.components.testing.performance_parity_validator import (
    ToleranceConfig,
    ValidationResult,
)


class TestComparisonConfig:
    """Test ComparisonConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ComparisonConfig()

        assert isinstance(config.tolerance_config, ToleranceConfig)
        assert config.statistical_significance_level == 0.05
        assert config.equivalence_margin == 0.01
        assert config.initial_balance == 10000.0
        assert config.commission_rate == 0.001
        assert config.generate_detailed_report == True
        assert config.export_results == True
        assert config.export_directory is None
        assert config.require_statistical_equivalence == True
        assert config.require_performance_parity == True
        assert config.minimum_correlation_threshold == 0.95

    def test_custom_config(self):
        """Test custom configuration values."""
        custom_tolerance = ToleranceConfig(total_return_tolerance=0.05)

        config = ComparisonConfig(
            tolerance_config=custom_tolerance,
            statistical_significance_level=0.01,
            equivalence_margin=0.02,
            initial_balance=50000.0,
            export_results=False,
        )

        assert config.tolerance_config.total_return_tolerance == 0.05
        assert config.statistical_significance_level == 0.01
        assert config.equivalence_margin == 0.02
        assert config.initial_balance == 50000.0
        assert config.export_results == False


class TestStrategyComparisonResult:
    """Test StrategyComparisonResult dataclass."""

    def test_result_creation(self):
        """Test creating a strategy comparison result."""
        from src.strategies.components.testing.performance_parity_validator import (
            PerformanceComparisonReport,
        )

        parity_report = PerformanceComparisonReport(
            strategy_name="Test",
            comparison_period="2023",
            legacy_strategy_id="legacy",
            new_strategy_id="new",
            overall_result=ValidationResult.PASS,
        )

        result = StrategyComparisonResult(
            comparison_id="test_123",
            timestamp=datetime(2023, 1, 1),
            legacy_strategy_name="LegacyStrategy",
            new_strategy_name="NewStrategy",
            parity_report=parity_report,
            statistical_tests={},
            equivalence_tests=[],
            overall_validation_result=ValidationResult.PASS,
            certification_status="CERTIFIED",
        )

        assert result.comparison_id == "test_123"
        assert result.timestamp == datetime(2023, 1, 1)
        assert result.legacy_strategy_name == "LegacyStrategy"
        assert result.new_strategy_name == "NewStrategy"
        assert result.parity_report == parity_report
        assert result.statistical_tests == {}
        assert result.equivalence_tests == []
        assert result.overall_validation_result == ValidationResult.PASS
        assert result.certification_status == "CERTIFIED"
        assert result.recommendations == []  # Default empty list


class TestPerformanceComparisonEngine:
    """Test PerformanceComparisonEngine class."""

    @pytest.fixture
    def mock_backtest_engine(self):
        """Create a mock backtest engine."""
        engine = Mock()

        # Mock backtest results
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        balance_legacy = 10000 * (1 + np.cumsum(np.random.normal(0.001, 0.02, 100)))
        balance_new = 10000 * (1 + np.cumsum(np.random.normal(0.0012, 0.021, 100)))

        legacy_results = pd.DataFrame(
            {
                "timestamp": dates,
                "balance": balance_legacy,
                "trade_pnl": np.random.normal(50, 100, 100),
            }
        )

        new_results = pd.DataFrame(
            {"timestamp": dates, "balance": balance_new, "trade_pnl": np.random.normal(52, 98, 100)}
        )

        # Configure mock to return different results based on strategy
        def mock_run_backtest(strategy, data, **kwargs):
            if "legacy" in str(strategy.__class__.__name__).lower():
                return legacy_results
            else:
                return new_results

        engine.run_backtest.side_effect = mock_run_backtest
        return engine

    @pytest.fixture
    def mock_strategies(self):
        """Create mock strategies."""
        legacy_strategy = Mock()
        legacy_strategy.__class__.__name__ = "LegacyStrategy"

        new_strategy = Mock()
        new_strategy.__class__.__name__ = "NewStrategy"

        return legacy_strategy, new_strategy

    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data."""
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        return pd.DataFrame(
            {
                "timestamp": dates,
                "open": np.random.uniform(100, 200, 100),
                "high": np.random.uniform(150, 250, 100),
                "low": np.random.uniform(50, 150, 100),
                "close": np.random.uniform(100, 200, 100),
                "volume": np.random.uniform(1000, 10000, 100),
            }
        )

    def test_engine_initialization_default(self):
        """Test engine initialization with defaults."""
        engine = PerformanceComparisonEngine()

        assert isinstance(engine.config, ComparisonConfig)
        assert engine.backtest_engine is None  # Backtest engine is optional and provided by caller
        assert engine.parity_validator is not None
        assert engine.statistical_tests is not None
        assert engine.equivalence_tests is not None

    def test_engine_initialization_custom(self, mock_backtest_engine):
        """Test engine initialization with custom parameters."""
        config = ComparisonConfig(initial_balance=50000.0)

        engine = PerformanceComparisonEngine(config, mock_backtest_engine)

        assert engine.config.initial_balance == 50000.0
        assert engine.backtest_engine == mock_backtest_engine

    def test_run_backtest_success(self, mock_backtest_engine, mock_strategies, sample_market_data):
        """Test successful backtest run."""
        engine = PerformanceComparisonEngine(backtest_engine=mock_backtest_engine)
        legacy_strategy, _ = mock_strategies

        result = engine._run_backtest(legacy_strategy, sample_market_data, "legacy")

        assert isinstance(result, pd.DataFrame)
        assert "balance" in result.columns
        assert "timestamp" in result.columns
        assert len(result) > 0

        # Verify backtest engine was called
        mock_backtest_engine.run_backtest.assert_called_once()

    def test_run_backtest_missing_balance_column(self, mock_strategies, sample_market_data):
        """Test backtest run with missing balance column."""
        # Create mock engine that returns invalid results
        mock_engine = Mock()
        mock_engine.run_backtest.return_value = pd.DataFrame({"price": [1, 2, 3]})

        engine = PerformanceComparisonEngine(backtest_engine=mock_engine)
        legacy_strategy, _ = mock_strategies

        with pytest.raises(ValueError, match="missing 'balance' column"):
            engine._run_backtest(legacy_strategy, sample_market_data, "legacy")

    def test_run_backtest_missing_timestamp_column(self, mock_strategies, sample_market_data):
        """Test backtest run with missing timestamp column."""
        # Create mock engine that returns results without timestamp column
        mock_engine = Mock()
        mock_engine.run_backtest.return_value = pd.DataFrame(
            {"balance": [10000, 10100, 10200], "trade_pnl": [0, 100, 200]}
        )

        engine = PerformanceComparisonEngine(backtest_engine=mock_engine)
        legacy_strategy, _ = mock_strategies

        with pytest.raises(ValueError, match="missing 'timestamp' column"):
            engine._run_backtest(legacy_strategy, sample_market_data, "legacy")

    def test_run_backtest_timestamp_index_success(self, mock_strategies, sample_market_data):
        """Test backtest run with timestamp as index name."""
        # Create mock engine that returns results with timestamp as index
        dates = pd.date_range("2023-01-01", periods=3, freq="D")
        mock_engine = Mock()
        mock_engine.run_backtest.return_value = pd.DataFrame(
            {"balance": [10000, 10100, 10200], "trade_pnl": [0, 100, 200]}, index=dates
        ).rename_axis("timestamp")

        engine = PerformanceComparisonEngine(backtest_engine=mock_engine)
        legacy_strategy, _ = mock_strategies

        result = engine._run_backtest(legacy_strategy, sample_market_data, "legacy")

        assert isinstance(result, pd.DataFrame)
        assert "timestamp" in result.columns
        assert "balance" in result.columns
        assert len(result) == 3
        # Verify the timestamp column contains the expected dates
        assert result["timestamp"].tolist() == dates.tolist()

    def test_run_backtest_index_with_name_but_not_timestamp(
        self, mock_strategies, sample_market_data
    ):
        """Test backtest run with index that has a name but not 'timestamp'."""
        # Create mock engine that returns results with named index (not 'timestamp')
        dates = pd.date_range("2023-01-01", periods=3, freq="D")
        mock_engine = Mock()
        mock_engine.run_backtest.return_value = pd.DataFrame(
            {"balance": [10000, 10100, 10200], "trade_pnl": [0, 100, 200]}, index=dates
        ).rename_axis(
            "date"
        )  # Index name is 'date', not 'timestamp'

        engine = PerformanceComparisonEngine(backtest_engine=mock_engine)
        legacy_strategy, _ = mock_strategies

        with pytest.raises(ValueError, match="missing 'timestamp' column"):
            engine._run_backtest(legacy_strategy, sample_market_data, "legacy")

    def test_run_backtest_range_index_with_name_none(self, mock_strategies, sample_market_data):
        """Test backtest run with RangeIndex that has name=None."""
        # Create mock engine that returns results with RangeIndex (name=None)
        mock_engine = Mock()
        df = pd.DataFrame({"balance": [10000, 10100, 10200], "trade_pnl": [0, 100, 200]})
        # RangeIndex has name=None by default
        assert df.index.name is None
        mock_engine.run_backtest.return_value = df

        engine = PerformanceComparisonEngine(backtest_engine=mock_engine)
        legacy_strategy, _ = mock_strategies

        with pytest.raises(ValueError, match="missing 'timestamp' column"):
            engine._run_backtest(legacy_strategy, sample_market_data, "legacy")

    def test_perform_statistical_analysis(self, mock_backtest_engine):
        """Test statistical analysis performance."""
        engine = PerformanceComparisonEngine(backtest_engine=mock_backtest_engine)

        # Create sample results
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        legacy_results = pd.DataFrame(
            {
                "timestamp": dates,
                "balance": 10000 * (1 + np.cumsum(np.random.normal(0.001, 0.02, 100))),
            }
        )
        new_results = pd.DataFrame(
            {
                "timestamp": dates,
                "balance": 10000 * (1 + np.cumsum(np.random.normal(0.0012, 0.021, 100))),
            }
        )

        results = engine._perform_statistical_analysis(legacy_results, new_results)

        assert isinstance(results, dict)
        assert len(results) > 0

        # Should have various test categories
        expected_categories = ["distribution_equality", "mean_equality", "variance_equality"]

        for category in expected_categories:
            assert category in results

    def test_perform_equivalence_tests(self, mock_backtest_engine):
        """Test equivalence tests performance."""
        engine = PerformanceComparisonEngine(backtest_engine=mock_backtest_engine)

        # Create sample results
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        legacy_results = pd.DataFrame(
            {
                "timestamp": dates,
                "balance": 10000 * (1 + np.cumsum(np.random.normal(0.001, 0.02, 100))),
            }
        )
        new_results = pd.DataFrame(
            {
                "timestamp": dates,
                "balance": 10000 * (1 + np.cumsum(np.random.normal(0.0012, 0.021, 100))),
            }
        )

        results = engine._perform_equivalence_tests(legacy_results, new_results)

        assert isinstance(results, list)
        assert len(results) >= 1

        # Should have TOST result
        tost_results = [r for r in results if "TOST" in r.test_name]
        assert len(tost_results) == 1

    def test_assess_overall_result_pass(self, mock_backtest_engine):
        """Test overall result assessment for passing case."""
        from src.strategies.components.testing.performance_parity_validator import (
            MetricComparison,
            MetricType,
            PerformanceComparisonReport,
        )
        from src.strategies.components.testing.statistical_tests import StatisticalTestResult

        engine = PerformanceComparisonEngine(backtest_engine=mock_backtest_engine)

        # Create a result with passing parity report
        parity_report = PerformanceComparisonReport(
            strategy_name="Test",
            comparison_period="2023",
            legacy_strategy_id="legacy",
            new_strategy_id="new",
            overall_result=ValidationResult.PASS,
            equity_curve_correlation=0.98,
        )

        # Add passing metric comparisons
        parity_report.metric_comparisons = [
            MetricComparison(
                metric_name="Total Return",
                metric_type=MetricType.RETURN,
                legacy_value=0.1,
                new_value=0.1,
                difference=0.0,
                relative_difference=0.0,
                tolerance=0.02,
                result=ValidationResult.PASS,
            )
        ]

        result = StrategyComparisonResult(
            comparison_id="test",
            timestamp=datetime.now(UTC),
            legacy_strategy_name="Legacy",
            new_strategy_name="New",
            parity_report=parity_report,
            statistical_tests={},
            equivalence_tests=[
                StatisticalTestResult(
                    test_name="Two One-Sided Test (TOST) for Equivalence",
                    statistic=1.0,
                    p_value=0.01,
                    reject_null=True,  # Equivalence confirmed
                    interpretation="Equivalent",
                )
            ],
            overall_validation_result=ValidationResult.INCONCLUSIVE,
            certification_status="Pending",
        )

        engine._assess_overall_result(result)

        assert result.overall_validation_result == ValidationResult.PASS
        assert "CERTIFIED" in result.certification_status
        assert len(result.recommendations) > 0
        assert any("proceed with confidence" in rec.lower() for rec in result.recommendations)

    def test_assess_overall_result_fail(self, mock_backtest_engine):
        """Test overall result assessment for failing case."""
        from src.strategies.components.testing.performance_parity_validator import (
            MetricComparison,
            MetricType,
            PerformanceComparisonReport,
        )

        engine = PerformanceComparisonEngine(backtest_engine=mock_backtest_engine)

        # Create a result with failing parity report
        parity_report = PerformanceComparisonReport(
            strategy_name="Test",
            comparison_period="2023",
            legacy_strategy_id="legacy",
            new_strategy_id="new",
            overall_result=ValidationResult.FAIL,
            equity_curve_correlation=0.80,
        )

        # Add failing metric comparison
        parity_report.metric_comparisons = [
            MetricComparison(
                metric_name="Total Return",
                metric_type=MetricType.RETURN,
                legacy_value=0.1,
                new_value=0.2,
                difference=0.1,
                relative_difference=0.1,
                tolerance=0.02,
                result=ValidationResult.FAIL,
            )
        ]

        result = StrategyComparisonResult(
            comparison_id="test",
            timestamp=datetime.now(UTC),
            legacy_strategy_name="Legacy",
            new_strategy_name="New",
            parity_report=parity_report,
            statistical_tests={},
            equivalence_tests=[],
            overall_validation_result=ValidationResult.INCONCLUSIVE,
            certification_status="Pending",
        )

        engine._assess_overall_result(result)

        assert result.overall_validation_result == ValidationResult.FAIL
        assert "NOT CERTIFIED" in result.certification_status
        assert len(result.recommendations) > 0
        assert any("should not proceed" in rec.lower() for rec in result.recommendations)

    @patch("src.strategies.components.testing.performance_comparison_engine.Path")
    def test_export_results(self, mock_path, mock_backtest_engine):
        """Test results export functionality."""
        from src.strategies.components.testing.performance_parity_validator import (
            PerformanceComparisonReport,
        )

        # Mock Path operations
        mock_export_dir = Mock()
        mock_path.return_value = mock_export_dir
        mock_export_dir.mkdir.return_value = None

        config = ComparisonConfig(export_results=True, export_directory="test_exports")
        engine = PerformanceComparisonEngine(config, mock_backtest_engine)

        # Create a sample result
        parity_report = PerformanceComparisonReport(
            strategy_name="Test",
            comparison_period="2023",
            legacy_strategy_id="legacy",
            new_strategy_id="new",
            overall_result=ValidationResult.PASS,
        )

        result = StrategyComparisonResult(
            comparison_id="test_123",
            timestamp=datetime(2023, 1, 1, 12, 0, 0),
            legacy_strategy_name="Legacy",
            new_strategy_name="New",
            parity_report=parity_report,
            statistical_tests={},
            equivalence_tests=[],
            overall_validation_result=ValidationResult.PASS,
            certification_status="CERTIFIED",
        )

        # Should not raise exception
        engine._export_results(result)

        # Verify directory creation was attempted
        mock_export_dir.mkdir.assert_called_once_with(exist_ok=True)

    def test_generate_text_report(self, mock_backtest_engine):
        """Test text report generation."""
        from src.strategies.components.testing.performance_parity_validator import (
            MetricComparison,
            MetricType,
            PerformanceComparisonReport,
        )
        from src.strategies.components.testing.statistical_tests import StatisticalTestResult

        engine = PerformanceComparisonEngine(backtest_engine=mock_backtest_engine)

        # Create a comprehensive result
        parity_report = PerformanceComparisonReport(
            strategy_name="Test Strategy",
            comparison_period="2023",
            legacy_strategy_id="legacy",
            new_strategy_id="new",
            overall_result=ValidationResult.PASS,
            total_metrics_tested=2,
            metrics_passed=2,
            metrics_failed=0,
            metrics_warning=0,
            equity_curve_correlation=0.98,
            certified=True,
        )

        parity_report.metric_comparisons = [
            MetricComparison(
                metric_name="Total Return",
                metric_type=MetricType.RETURN,
                legacy_value=0.1,
                new_value=0.1,
                difference=0.0,
                relative_difference=0.0,
                tolerance=0.02,
                result=ValidationResult.PASS,
            )
        ]

        result = StrategyComparisonResult(
            comparison_id="test_123",
            timestamp=datetime(2023, 1, 1),
            legacy_strategy_name="LegacyStrategy",
            new_strategy_name="NewStrategy",
            parity_report=parity_report,
            statistical_tests={
                "test_category": [
                    StatisticalTestResult(
                        test_name="Test",
                        statistic=1.0,
                        p_value=0.1,
                        reject_null=False,
                        interpretation="Not significant",
                    )
                ]
            },
            equivalence_tests=[
                StatisticalTestResult(
                    test_name="TOST",
                    statistic=1.0,
                    p_value=0.01,
                    reject_null=True,
                    interpretation="Equivalent",
                )
            ],
            overall_validation_result=ValidationResult.PASS,
            certification_status="CERTIFIED",
            recommendations=["Migration approved"],
        )

        report_text = engine.generate_text_report(result)

        assert isinstance(report_text, str)
        assert "COMPREHENSIVE STRATEGY COMPARISON REPORT" in report_text
        assert "test_123" in report_text
        assert "LegacyStrategy" in report_text
        assert "NewStrategy" in report_text
        assert "CERTIFIED" in report_text
        assert "Migration approved" in report_text
        assert "Total Return" in report_text
        assert "STATISTICAL TEST RESULTS" in report_text
        assert "EQUIVALENCE TEST RESULTS" in report_text

    def test_compare_strategies_success(
        self, mock_backtest_engine, mock_strategies, sample_market_data
    ):
        """Test successful strategy comparison."""
        config = ComparisonConfig(export_results=False)  # Disable export for test
        engine = PerformanceComparisonEngine(config, mock_backtest_engine)

        legacy_strategy, new_strategy = mock_strategies

        result = engine.compare_strategies(
            legacy_strategy, new_strategy, sample_market_data, "test_comparison"
        )

        assert isinstance(result, StrategyComparisonResult)
        assert result.comparison_id == "test_comparison"
        assert result.legacy_strategy_name == "LegacyStrategy"
        assert result.new_strategy_name == "NewStrategy"
        assert result.overall_validation_result in [
            ValidationResult.PASS,
            ValidationResult.WARNING,
            ValidationResult.FAIL,
        ]
        assert len(result.recommendations) > 0

        # Verify backtests were run
        assert mock_backtest_engine.run_backtest.call_count == 2

    def test_compare_strategies_failure(self, mock_strategies, sample_market_data):
        """Test strategy comparison with backtest failure."""
        # Create mock engine that raises exception
        mock_engine = Mock()
        mock_engine.run_backtest.side_effect = Exception("Backtest failed")

        config = ComparisonConfig(export_results=False)
        engine = PerformanceComparisonEngine(config, mock_engine)

        legacy_strategy, new_strategy = mock_strategies

        result = engine.compare_strategies(legacy_strategy, new_strategy, sample_market_data)

        assert result.overall_validation_result == ValidationResult.FAIL
        assert "Failed:" in result.certification_status
        assert any("Fix error:" in rec for rec in result.recommendations)


class TestConvenienceFunctions:
    """Test convenience functions."""

    @pytest.fixture
    def mock_strategies(self):
        """Create mock strategies."""
        legacy_strategy = Mock()
        legacy_strategy.__class__.__name__ = "LegacyStrategy"

        new_strategy = Mock()
        new_strategy.__class__.__name__ = "NewStrategy"

        return legacy_strategy, new_strategy

    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data."""
        dates = pd.date_range("2023-01-01", periods=50, freq="D")
        return pd.DataFrame({"timestamp": dates, "close": np.random.uniform(100, 200, 50)})

    @patch(
        "src.strategies.components.testing.performance_comparison_engine.PerformanceComparisonEngine"
    )
    def test_quick_strategy_comparison(
        self, mock_engine_class, mock_strategies, sample_market_data
    ):
        """Test quick strategy comparison function."""
        # Mock the engine and its methods
        mock_engine = Mock()
        mock_result = Mock()
        mock_engine.compare_strategies.return_value = mock_result
        mock_engine_class.return_value = mock_engine

        legacy_strategy, new_strategy = mock_strategies

        result = quick_strategy_comparison(legacy_strategy, new_strategy, sample_market_data)

        assert result == mock_result
        mock_engine_class.assert_called_once()
        mock_engine.compare_strategies.assert_called_once_with(
            legacy_strategy, new_strategy, sample_market_data
        )

    @patch(
        "src.strategies.components.testing.performance_comparison_engine.PerformanceComparisonEngine"
    )
    def test_validate_migration_readiness_ready(
        self, mock_engine_class, mock_strategies, sample_market_data
    ):
        """Test migration readiness validation for ready migration."""
        # Mock successful comparison result
        mock_engine = Mock()
        mock_result = Mock()
        mock_result.overall_validation_result = ValidationResult.PASS
        mock_result.parity_report.metrics_failed = 0
        mock_result.parity_report.metric_comparisons = []
        mock_result.recommendations = ["Migration approved"]
        mock_engine.compare_strategies.return_value = mock_result
        mock_engine_class.return_value = mock_engine

        legacy_strategy, new_strategy = mock_strategies

        is_ready, issues = validate_migration_readiness(
            legacy_strategy, new_strategy, sample_market_data, strict_validation=True
        )

        assert is_ready == True
        assert len(issues) == 0

    @patch(
        "src.strategies.components.testing.performance_comparison_engine.PerformanceComparisonEngine"
    )
    def test_validate_migration_readiness_not_ready(
        self, mock_engine_class, mock_strategies, sample_market_data
    ):
        """Test migration readiness validation for not ready migration."""
        from src.strategies.components.testing.performance_parity_validator import (
            MetricComparison,
            MetricType,
        )

        # Mock failed comparison result
        mock_engine = Mock()
        mock_result = Mock()
        mock_result.overall_validation_result = ValidationResult.FAIL
        mock_result.parity_report.metrics_failed = 1
        mock_result.parity_report.metric_comparisons = [
            MetricComparison(
                metric_name="Total Return",
                metric_type=MetricType.RETURN,
                legacy_value=0.1,
                new_value=0.2,
                difference=0.1,
                relative_difference=0.1,
                tolerance=0.02,
                result=ValidationResult.FAIL,
            )
        ]
        mock_result.recommendations = ["Fix performance issues"]
        mock_engine.compare_strategies.return_value = mock_result
        mock_engine_class.return_value = mock_engine

        legacy_strategy, new_strategy = mock_strategies

        is_ready, issues = validate_migration_readiness(
            legacy_strategy, new_strategy, sample_market_data, strict_validation=True
        )

        assert is_ready == False
        assert len(issues) > 0
        assert any("Fix performance issues" in issue for issue in issues)
        assert any("Total Return" in issue for issue in issues)

    @patch(
        "src.strategies.components.testing.performance_comparison_engine.PerformanceComparisonEngine"
    )
    def test_validate_migration_readiness_lenient(
        self, mock_engine_class, mock_strategies, sample_market_data
    ):
        """Test migration readiness validation with lenient settings."""
        mock_engine = Mock()
        mock_result = Mock()
        mock_result.overall_validation_result = ValidationResult.WARNING
        mock_result.parity_report.metrics_failed = 0
        mock_result.parity_report.metric_comparisons = []
        mock_result.recommendations = []
        mock_engine.compare_strategies.return_value = mock_result
        mock_engine_class.return_value = mock_engine

        legacy_strategy, new_strategy = mock_strategies

        is_ready, issues = validate_migration_readiness(
            legacy_strategy, new_strategy, sample_market_data, strict_validation=False
        )

        assert is_ready == True  # WARNING is acceptable for lenient validation

        # Check that lenient tolerances were used
        call_args = mock_engine_class.call_args[0][0]  # First positional argument (config)
        assert (
            call_args.tolerance_config.total_return_tolerance == 0.05
        )  # Relaxed from default 0.02
        assert call_args.tolerance_config.minimum_correlation == 0.90  # Relaxed from default 0.95
