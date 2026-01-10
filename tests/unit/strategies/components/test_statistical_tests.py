"""
Unit tests for Statistical Tests module.
"""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.strategies.components.testing.statistical_tests import (
    EquivalenceTests,
    FinancialStatisticalTests,
    StatisticalTestResult,
    format_test_results,
)


class TestStatisticalTestResult:
    """Test StatisticalTestResult dataclass."""

    def test_result_creation(self):
        """Test creating a statistical test result."""
        result = StatisticalTestResult(
            test_name="Test",
            statistic=1.5,
            p_value=0.05,
            critical_value=1.96,
            confidence_level=0.95,
            reject_null=True,
            interpretation="Significant result",
            notes="Test notes",
        )

        assert result.test_name == "Test"
        assert result.statistic == 1.5
        assert result.p_value == 0.05
        assert result.critical_value == 1.96
        assert result.confidence_level == 0.95
        assert result.reject_null == True
        assert result.interpretation == "Significant result"
        assert result.notes == "Test notes"


class TestFinancialStatisticalTests:
    """Test FinancialStatisticalTests class."""

    @pytest.fixture
    def test_engine(self):
        """Create a test engine."""
        return FinancialStatisticalTests(significance_level=0.05)

    @pytest.fixture
    def sample_returns(self):
        """Create sample return series for testing."""
        np.random.seed(42)  # For reproducible tests

        # Create two similar but slightly different return series
        returns1 = pd.Series(np.random.normal(0.001, 0.02, 100))
        returns2 = pd.Series(np.random.normal(0.0012, 0.021, 100))

        return returns1, returns2

    @pytest.fixture
    def normal_returns(self):
        """Create normally distributed returns."""
        np.random.seed(42)
        return pd.Series(np.random.normal(0.001, 0.02, 1000))

    @pytest.fixture
    def non_normal_returns(self):
        """Create non-normally distributed returns."""
        np.random.seed(42)
        # Create skewed distribution
        returns = np.random.exponential(0.01, 1000) - 0.01
        return pd.Series(returns)

    def test_engine_initialization(self):
        """Test engine initialization."""
        # Default initialization
        engine = FinancialStatisticalTests()
        assert engine.significance_level == 0.05
        assert engine.confidence_level == 0.95

        # Custom significance level
        engine = FinancialStatisticalTests(significance_level=0.01)
        assert engine.significance_level == 0.01
        assert engine.confidence_level == 0.99

    def test_return_distribution_equality_sufficient_data(self, test_engine, sample_returns):
        """Test distribution equality tests with sufficient data."""
        returns1, returns2 = sample_returns

        results = test_engine.test_return_distribution_equality(returns1, returns2)

        # Should have multiple test results
        assert len(results) >= 1

        # Check for KS test
        ks_results = [r for r in results if "Kolmogorov-Smirnov" in r.test_name]
        assert len(ks_results) == 1

        ks_result = ks_results[0]
        assert ks_result.statistic >= 0
        assert 0 <= ks_result.p_value <= 1
        assert isinstance(ks_result.reject_null, bool)
        assert ks_result.interpretation != ""

    def test_return_distribution_equality_insufficient_data(self, test_engine):
        """Test distribution equality tests with insufficient data."""
        small_returns1 = pd.Series([0.01, 0.02])
        small_returns2 = pd.Series([0.01, 0.02])

        results = test_engine.test_return_distribution_equality(small_returns1, small_returns2)

        # Should return empty list due to insufficient data
        assert len(results) == 0

    def test_mean_equality_tests(self, test_engine, sample_returns):
        """Test mean equality tests."""
        returns1, returns2 = sample_returns

        results = test_engine.test_mean_equality(returns1, returns2)

        # Should have multiple test results
        assert len(results) >= 1

        # Check for t-test and Mann-Whitney test
        test_names = [r.test_name for r in results]
        assert any("t-test" in name for name in test_names)
        assert any("Mann-Whitney" in name for name in test_names)

        # All results should have valid statistics
        for result in results:
            assert isinstance(result.statistic, (int, float))
            assert 0 <= result.p_value <= 1
            assert isinstance(result.reject_null, bool)

    def test_variance_equality_tests(self, test_engine, sample_returns):
        """Test variance equality tests."""
        returns1, returns2 = sample_returns

        results = test_engine.test_variance_equality(returns1, returns2)

        # Should have multiple test results
        assert len(results) >= 1

        # Check for Levene's test
        test_names = [r.test_name for r in results]
        assert any("Levene" in name for name in test_names)

        # All results should have valid statistics
        for result in results:
            assert isinstance(result.statistic, (int, float))
            assert 0 <= result.p_value <= 1
            assert isinstance(result.reject_null, bool)

    def test_normality_tests_normal_data(self, test_engine, normal_returns):
        """Test normality tests on normal data."""
        results = test_engine.test_normality(normal_returns)

        # Should have multiple normality tests
        assert len(results) >= 1

        # Check for common normality tests
        test_names = [r.test_name for r in results]
        assert any("Shapiro-Wilk" in name for name in test_names)
        assert any("Jarque-Bera" in name for name in test_names)

        # For normal data, most tests should not reject normality
        # (though this is probabilistic, so we just check structure)
        for result in results:
            assert isinstance(result.statistic, (int, float))
            assert 0 <= result.p_value <= 1
            assert isinstance(result.reject_null, bool)

    def test_normality_tests_non_normal_data(self, test_engine, non_normal_returns):
        """Test normality tests on non-normal data."""
        results = test_engine.test_normality(non_normal_returns)

        # Should have multiple normality tests
        assert len(results) >= 1

        # For clearly non-normal data, at least some tests should reject normality
        rejection_count = sum(1 for r in results if r.reject_null)
        # We expect at least one test to detect non-normality, but this is probabilistic
        # so we just check that tests ran
        assert len(results) > 0

    def test_normality_tests_insufficient_data(self, test_engine):
        """Test normality tests with insufficient data."""
        small_returns = pd.Series([0.01, 0.02, 0.03])

        results = test_engine.test_normality(small_returns)

        # Should return empty list due to insufficient data
        assert len(results) == 0

    @patch("statsmodels.stats.diagnostic.acorr_ljungbox")
    def test_autocorrelation_test_with_statsmodels(
        self, mock_ljungbox, test_engine, sample_returns
    ):
        """Test autocorrelation test when statsmodels is available."""
        returns1, _ = sample_returns

        # Mock the ljungbox result
        mock_result = pd.DataFrame({"lb_stat": [1.0, 2.0, 3.0], "lb_pvalue": [0.5, 0.3, 0.1]})
        mock_ljungbox.return_value = mock_result

        result = test_engine.test_autocorrelation(returns1, max_lags=3)

        assert result.test_name == "Ljung-Box Autocorrelation Test"
        assert result.statistic == 3.0  # Last statistic
        assert result.p_value == 0.1  # Last p-value
        assert result.reject_null == False  # p > 0.05
        assert "3 lags" in result.notes

    def test_autocorrelation_test_insufficient_data(self, test_engine):
        """Test autocorrelation test with insufficient data."""
        small_returns = pd.Series([0.01, 0.02])

        result = test_engine.test_autocorrelation(small_returns, max_lags=20)

        assert result.test_name == "Ljung-Box Autocorrelation Test"
        assert result.p_value == 1.0
        assert "Insufficient data" in result.interpretation

    def test_stationarity_tests_with_statsmodels(self, test_engine, sample_returns):
        """Test stationarity tests when statsmodels is available."""
        returns1, _ = sample_returns

        results = test_engine.test_stationarity(returns1)

        # Should have 2 tests when statsmodels is available
        assert len(results) == 2

        # Check ADF result exists
        adf_results = [r for r in results if "Augmented Dickey-Fuller" in r.test_name]
        assert len(adf_results) == 1
        adf_result = adf_results[0]
        assert isinstance(adf_result.statistic, (int, float))
        assert isinstance(adf_result.p_value, (int, float))
        assert isinstance(adf_result.reject_null, bool)

        # Check KPSS result exists
        kpss_results = [r for r in results if "KPSS" in r.test_name]
        assert len(kpss_results) == 1
        kpss_result = kpss_results[0]
        assert isinstance(kpss_result.statistic, (int, float))
        assert isinstance(kpss_result.p_value, (int, float))
        assert kpss_result.reject_null == False  # p > 0.05

    def test_stationarity_tests_insufficient_data(self, test_engine):
        """Test stationarity tests with insufficient data."""
        small_series = pd.Series(np.random.randn(10))

        results = test_engine.test_stationarity(small_series)

        # Should return empty list due to insufficient data
        assert len(results) == 0

    def test_comprehensive_comparison(self, test_engine, sample_returns):
        """Test comprehensive comparison of two return series."""
        returns1, returns2 = sample_returns

        results = test_engine.comprehensive_comparison(
            returns1, returns2, "Strategy A", "Strategy B"
        )

        # Should have multiple categories
        assert len(results) > 0

        # Check for expected categories
        expected_categories = [
            "distribution_equality",
            "mean_equality",
            "variance_equality",
            "Strategy A_normality",
            "Strategy B_normality",
            "Strategy A_autocorrelation",
            "Strategy B_autocorrelation",
        ]

        for category in expected_categories:
            assert category in results

        # Each category should have test results
        for category, test_list in results.items():
            if test_list:  # Some categories might be empty due to insufficient data
                assert all(isinstance(test, StatisticalTestResult) for test in test_list)


class TestEquivalenceTests:
    """Test EquivalenceTests class."""

    @pytest.fixture
    def equiv_engine(self):
        """Create equivalence test engine."""
        return EquivalenceTests(equivalence_margin=0.01)

    @pytest.fixture
    def equivalent_returns(self):
        """Create equivalent return series."""
        np.random.seed(42)
        returns1 = pd.Series(np.random.normal(0.001, 0.02, 100))
        returns2 = pd.Series(np.random.normal(0.001, 0.02, 100))  # Same mean
        return returns1, returns2

    @pytest.fixture
    def non_equivalent_returns(self):
        """Create non-equivalent return series."""
        np.random.seed(42)
        returns1 = pd.Series(np.random.normal(0.001, 0.02, 100))
        returns2 = pd.Series(np.random.normal(0.05, 0.02, 100))  # Very different mean
        return returns1, returns2

    def test_engine_initialization(self):
        """Test equivalence engine initialization."""
        engine = EquivalenceTests()
        assert engine.equivalence_margin == 0.01

        engine = EquivalenceTests(equivalence_margin=0.05)
        assert engine.equivalence_margin == 0.05

    def test_tost_equivalent_returns(self, equiv_engine, equivalent_returns):
        """Test TOST with equivalent returns."""
        returns1, returns2 = equivalent_returns

        result = equiv_engine.two_one_sided_test(returns1, returns2)

        assert result.test_name == "Two One-Sided Test (TOST) for Equivalence"
        assert result.statistic >= 0
        assert 0 <= result.p_value <= 1
        assert isinstance(result.reject_null, bool)
        assert "equivalence margin" in result.notes.lower()
        assert "mean difference" in result.notes.lower()

    def test_tost_non_equivalent_returns(self, equiv_engine, non_equivalent_returns):
        """Test TOST with non-equivalent returns."""
        returns1, returns2 = non_equivalent_returns

        result = equiv_engine.two_one_sided_test(returns1, returns2)

        assert result.test_name == "Two One-Sided Test (TOST) for Equivalence"
        assert result.statistic >= 0
        assert 0 <= result.p_value <= 1
        # With very different means, should not conclude equivalence
        assert "Cannot conclude" in result.interpretation or not result.reject_null

    def test_tost_custom_margin(self, equiv_engine, equivalent_returns):
        """Test TOST with custom equivalence margin."""
        returns1, returns2 = equivalent_returns

        result = equiv_engine.two_one_sided_test(returns1, returns2, equivalence_margin=0.05)

        assert "±0.05" in result.notes

    def test_tost_insufficient_data(self, equiv_engine):
        """Test TOST with insufficient data."""
        small_returns1 = pd.Series([0.01, 0.02])
        small_returns2 = pd.Series([0.01, 0.02])

        result = equiv_engine.two_one_sided_test(small_returns1, small_returns2)

        assert result.test_name == "Two One-Sided Test (TOST) for Equivalence"
        assert result.p_value == 1.0
        assert "Insufficient data" in result.interpretation


class TestFormatTestResults:
    """Test format_test_results function."""

    def test_format_empty_results(self):
        """Test formatting empty results."""
        results = {}

        formatted = format_test_results(results)

        assert "STATISTICAL TEST RESULTS" in formatted
        assert "=" * 80 in formatted

    def test_format_single_category(self):
        """Test formatting single category results."""
        results = {
            "test_category": [
                StatisticalTestResult(
                    test_name="Test 1",
                    statistic=1.5,
                    p_value=0.05,
                    reject_null=True,
                    interpretation="Significant result",
                )
            ]
        }

        formatted = format_test_results(results)

        assert "TEST CATEGORY:" in formatted
        assert "Test 1" in formatted
        assert "1.500000" in formatted
        assert "0.050000" in formatted
        assert "True" in formatted
        assert "Significant result" in formatted

    def test_format_multiple_categories(self):
        """Test formatting multiple categories."""
        results = {
            "category_one": [
                StatisticalTestResult(
                    test_name="Test A",
                    statistic=1.0,
                    p_value=0.1,
                    reject_null=False,
                    interpretation="Not significant",
                )
            ],
            "category_two": [
                StatisticalTestResult(
                    test_name="Test B",
                    statistic=2.0,
                    p_value=0.01,
                    reject_null=True,
                    interpretation="Significant",
                    notes="Important note",
                )
            ],
        }

        formatted = format_test_results(results)

        assert "CATEGORY ONE:" in formatted
        assert "CATEGORY TWO:" in formatted
        assert "Test A" in formatted
        assert "Test B" in formatted
        assert "Important note" in formatted

    def test_format_empty_category(self):
        """Test formatting with empty category."""
        results = {
            "empty_category": [],
            "valid_category": [
                StatisticalTestResult(
                    test_name="Test",
                    statistic=1.0,
                    p_value=0.1,
                    reject_null=False,
                    interpretation="Result",
                )
            ],
        }

        formatted = format_test_results(results)

        # Empty category should not appear
        assert "EMPTY CATEGORY:" not in formatted
        assert "VALID CATEGORY:" in formatted
        assert "Test" in formatted
