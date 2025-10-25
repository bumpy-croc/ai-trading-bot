"""
Statistical Testing Utilities for Performance Parity Validation

This module provides specialized statistical tests for comparing trading strategy
performance, including tests specifically designed for financial time series data.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import jarque_bera, normaltest

logger = logging.getLogger(__name__)


@dataclass
class StatisticalTestResult:
    """Result of a statistical test."""

    test_name: str
    statistic: float
    p_value: float
    critical_value: Optional[float] = None
    confidence_level: float = 0.95
    reject_null: bool = False
    interpretation: str = ""
    notes: str = ""


class FinancialStatisticalTests:
    """
    Specialized statistical tests for financial time series and trading performance.
    """

    def __init__(self, significance_level: float = 0.05):
        """
        Initialize statistical tests with significance level.

        Args:
            significance_level: Alpha level for hypothesis testing (default 0.05)
        """
        self.significance_level = significance_level
        self.confidence_level = 1 - significance_level
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def test_return_distribution_equality(
        self, returns1: pd.Series, returns2: pd.Series
    ) -> List[StatisticalTestResult]:
        """
        Test if two return series have the same distribution.

        Args:
            returns1: First return series
            returns2: Second return series

        Returns:
            List of statistical test results
        """
        results = []

        # Clean data
        returns1_clean = returns1.dropna()
        returns2_clean = returns2.dropna()

        if len(returns1_clean) < 10 or len(returns2_clean) < 10:
            self.logger.warning("Insufficient data for distribution tests")
            return results

        # Kolmogorov-Smirnov test
        try:
            ks_stat, ks_p = stats.ks_2samp(returns1_clean, returns2_clean)
            results.append(
                StatisticalTestResult(
                    test_name="Kolmogorov-Smirnov Two-Sample Test",
                    statistic=ks_stat,
                    p_value=ks_p,
                    reject_null=bool(ks_p < self.significance_level),
                    interpretation=(
                        "Distributions are significantly different"
                        if ks_p < self.significance_level
                        else "No significant difference in distributions"
                    ),
                    notes="Tests if two samples come from the same distribution",
                )
            )
        except Exception as e:
            self.logger.warning(f"KS test failed: {e}")

        # Anderson-Darling test
        try:
            ad_stat, ad_critical, ad_p = stats.anderson_ksamp([returns1_clean, returns2_clean])
            results.append(
                StatisticalTestResult(
                    test_name="Anderson-Darling k-Sample Test",
                    statistic=ad_stat,
                    p_value=ad_p,
                    critical_value=ad_critical[2],  # 5% significance level
                    reject_null=bool(ad_p < self.significance_level),
                    interpretation=(
                        "Distributions are significantly different"
                        if ad_p < self.significance_level
                        else "No significant difference in distributions"
                    ),
                    notes="More sensitive to tail differences than KS test",
                )
            )
        except Exception as e:
            self.logger.warning(f"Anderson-Darling test failed: {e}")

        return results

    def test_mean_equality(
        self, returns1: pd.Series, returns2: pd.Series
    ) -> List[StatisticalTestResult]:
        """
        Test if two return series have equal means.

        Args:
            returns1: First return series
            returns2: Second return series

        Returns:
            List of statistical test results
        """
        results = []

        returns1_clean = returns1.dropna()
        returns2_clean = returns2.dropna()

        if len(returns1_clean) < 10 or len(returns2_clean) < 10:
            self.logger.warning("Insufficient data for mean tests")
            return results

        # Welch's t-test (unequal variances)
        try:
            t_stat, t_p = stats.ttest_ind(returns1_clean, returns2_clean, equal_var=False)
            results.append(
                StatisticalTestResult(
                    test_name="Welch's t-test (Unequal Variances)",
                    statistic=t_stat,
                    p_value=t_p,
                    reject_null=bool(t_p < self.significance_level),
                    interpretation=(
                        "Means are significantly different"
                        if t_p < self.significance_level
                        else "No significant difference in means"
                    ),
                    notes="Assumes unequal variances between samples",
                )
            )
        except Exception as e:
            self.logger.warning(f"Welch's t-test failed: {e}")

        # Mann-Whitney U test (non-parametric)
        try:
            mw_stat, mw_p = stats.mannwhitneyu(
                returns1_clean, returns2_clean, alternative="two-sided"
            )
            results.append(
                StatisticalTestResult(
                    test_name="Mann-Whitney U Test",
                    statistic=mw_stat,
                    p_value=mw_p,
                    reject_null=bool(mw_p < self.significance_level),
                    interpretation=(
                        "Medians are significantly different"
                        if mw_p < self.significance_level
                        else "No significant difference in medians"
                    ),
                    notes="Non-parametric test for median equality",
                )
            )
        except Exception as e:
            self.logger.warning(f"Mann-Whitney U test failed: {e}")

        return results

    def test_variance_equality(
        self, returns1: pd.Series, returns2: pd.Series
    ) -> List[StatisticalTestResult]:
        """
        Test if two return series have equal variances.

        Args:
            returns1: First return series
            returns2: Second return series

        Returns:
            List of statistical test results
        """
        results = []

        returns1_clean = returns1.dropna()
        returns2_clean = returns2.dropna()

        if len(returns1_clean) < 10 or len(returns2_clean) < 10:
            self.logger.warning("Insufficient data for variance tests")
            return results

        # Levene's test
        try:
            levene_stat, levene_p = stats.levene(returns1_clean, returns2_clean)
            results.append(
                StatisticalTestResult(
                    test_name="Levene's Test for Equal Variances",
                    statistic=levene_stat,
                    p_value=levene_p,
                    reject_null=bool(levene_p < self.significance_level),
                    interpretation=(
                        "Variances are significantly different"
                        if levene_p < self.significance_level
                        else "No significant difference in variances"
                    ),
                    notes="Robust test for variance equality",
                )
            )
        except Exception as e:
            self.logger.warning(f"Levene's test failed: {e}")

        # Bartlett's test (assumes normality)
        try:
            bartlett_stat, bartlett_p = stats.bartlett(returns1_clean, returns2_clean)
            results.append(
                StatisticalTestResult(
                    test_name="Bartlett's Test for Equal Variances",
                    statistic=bartlett_stat,
                    p_value=bartlett_p,
                    reject_null=bool(bartlett_p < self.significance_level),
                    interpretation=(
                        "Variances are significantly different"
                        if bartlett_p < self.significance_level
                        else "No significant difference in variances"
                    ),
                    notes="Assumes normal distributions; sensitive to non-normality",
                )
            )
        except Exception as e:
            self.logger.warning(f"Bartlett's test failed: {e}")

        return results

    def test_normality(self, returns: pd.Series) -> List[StatisticalTestResult]:
        """
        Test if return series follows normal distribution.

        Args:
            returns: Return series to test

        Returns:
            List of statistical test results
        """
        results = []

        returns_clean = returns.dropna()

        if len(returns_clean) < 20:
            self.logger.warning("Insufficient data for normality tests")
            return results

        # Shapiro-Wilk test
        try:
            if len(returns_clean) <= 5000:  # Shapiro-Wilk has sample size limit
                sw_stat, sw_p = stats.shapiro(returns_clean)
                results.append(
                    StatisticalTestResult(
                        test_name="Shapiro-Wilk Normality Test",
                        statistic=sw_stat,
                        p_value=sw_p,
                        reject_null=bool(sw_p < self.significance_level),
                        interpretation=(
                            "Data is not normally distributed"
                            if sw_p < self.significance_level
                            else "Data appears normally distributed"
                        ),
                        notes="Most powerful normality test for small samples",
                    )
                )
        except Exception as e:
            self.logger.warning(f"Shapiro-Wilk test failed: {e}")

        # Jarque-Bera test
        try:
            jb_stat, jb_p = jarque_bera(returns_clean)
            results.append(
                StatisticalTestResult(
                    test_name="Jarque-Bera Normality Test",
                    statistic=jb_stat,
                    p_value=jb_p,
                    reject_null=bool(jb_p < self.significance_level),
                    interpretation=(
                        "Data is not normally distributed"
                        if jb_p < self.significance_level
                        else "Data appears normally distributed"
                    ),
                    notes="Tests for normality based on skewness and kurtosis",
                )
            )
        except Exception as e:
            self.logger.warning(f"Jarque-Bera test failed: {e}")

        # D'Agostino's normality test
        try:
            da_stat, da_p = normaltest(returns_clean)
            results.append(
                StatisticalTestResult(
                    test_name="D'Agostino's Normality Test",
                    statistic=da_stat,
                    p_value=da_p,
                    reject_null=bool(da_p < self.significance_level),
                    interpretation=(
                        "Data is not normally distributed"
                        if da_p < self.significance_level
                        else "Data appears normally distributed"
                    ),
                    notes="Combines tests for skewness and kurtosis",
                )
            )
        except Exception as e:
            self.logger.warning(f"D'Agostino's test failed: {e}")

        return results

    def test_autocorrelation(self, returns: pd.Series, max_lags: int = 20) -> StatisticalTestResult:
        """
        Test for autocorrelation in return series using Ljung-Box test.

        Args:
            returns: Return series to test
            max_lags: Maximum number of lags to test

        Returns:
            Statistical test result
        """
        returns_clean = returns.dropna()

        if len(returns_clean) < max_lags * 2:
            self.logger.warning("Insufficient data for autocorrelation test")
            return StatisticalTestResult(
                test_name="Ljung-Box Autocorrelation Test",
                statistic=0.0,
                p_value=1.0,
                interpretation="Insufficient data for test",
            )

        try:
            from statsmodels.stats.diagnostic import acorr_ljungbox

            lb_result = acorr_ljungbox(returns_clean, lags=max_lags, return_df=True)

            # Use the test statistic and p-value for the maximum lag
            lb_stat = lb_result["lb_stat"].iloc[-1]
            lb_p = lb_result["lb_pvalue"].iloc[-1]

            return StatisticalTestResult(
                test_name="Ljung-Box Autocorrelation Test",
                statistic=lb_stat,
                p_value=lb_p,
                reject_null=bool(lb_p < self.significance_level),
                interpretation=(
                    "Significant autocorrelation detected"
                    if lb_p < self.significance_level
                    else "No significant autocorrelation"
                ),
                notes=f"Tested up to {max_lags} lags",
            )

        except ImportError:
            self.logger.warning("statsmodels not available for Ljung-Box test")
            return StatisticalTestResult(
                test_name="Ljung-Box Autocorrelation Test",
                statistic=0.0,
                p_value=1.0,
                interpretation="statsmodels required for this test",
            )
        except Exception as e:
            self.logger.warning(f"Ljung-Box test failed: {e}")
            return StatisticalTestResult(
                test_name="Ljung-Box Autocorrelation Test",
                statistic=0.0,
                p_value=1.0,
                interpretation=f"Test failed: {str(e)}",
            )

    def test_stationarity(self, series: pd.Series) -> List[StatisticalTestResult]:
        """
        Test for stationarity in time series.

        Args:
            series: Time series to test

        Returns:
            List of statistical test results
        """
        results = []

        series_clean = series.dropna()

        if len(series_clean) < 50:
            self.logger.warning("Insufficient data for stationarity tests")
            return results

        try:
            from statsmodels.tsa.stattools import adfuller, kpss

            # Augmented Dickey-Fuller test
            adf_result = adfuller(series_clean, autolag="AIC")
            results.append(
                StatisticalTestResult(
                    test_name="Augmented Dickey-Fuller Test",
                    statistic=adf_result[0],
                    p_value=adf_result[1],
                    critical_value=adf_result[4]["5%"],
                    reject_null=bool(adf_result[1] < self.significance_level),
                    interpretation=(
                        "Series is stationary"
                        if adf_result[1] < self.significance_level
                        else "Series is non-stationary (has unit root)"
                    ),
                    notes="Tests null hypothesis of unit root (non-stationarity)",
                )
            )

            # KPSS test
            kpss_result = kpss(series_clean, regression="c")
            results.append(
                StatisticalTestResult(
                    test_name="KPSS Stationarity Test",
                    statistic=kpss_result[0],
                    p_value=kpss_result[1],
                    critical_value=kpss_result[3]["5%"],
                    reject_null=bool(kpss_result[1] < self.significance_level),
                    interpretation=(
                        "Series is non-stationary"
                        if kpss_result[1] < self.significance_level
                        else "Series is stationary"
                    ),
                    notes="Tests null hypothesis of stationarity",
                )
            )

        except ImportError:
            self.logger.warning("statsmodels not available for stationarity tests")
        except Exception as e:
            self.logger.warning(f"Stationarity tests failed: {e}")

        return results

    def comprehensive_comparison(
        self,
        returns1: pd.Series,
        returns2: pd.Series,
        series1_name: str = "Series 1",
        series2_name: str = "Series 2",
    ) -> Dict[str, List[StatisticalTestResult]]:
        """
        Perform comprehensive statistical comparison of two return series.

        Args:
            returns1: First return series
            returns2: Second return series
            series1_name: Name for first series
            series2_name: Name for second series

        Returns:
            Dictionary of test categories and their results
        """
        self.logger.info(f"Performing comprehensive comparison: {series1_name} vs {series2_name}")

        results = {
            "distribution_equality": self.test_return_distribution_equality(returns1, returns2),
            "mean_equality": self.test_mean_equality(returns1, returns2),
            "variance_equality": self.test_variance_equality(returns1, returns2),
            f"{series1_name}_normality": self.test_normality(returns1),
            f"{series2_name}_normality": self.test_normality(returns2),
            f"{series1_name}_autocorrelation": [self.test_autocorrelation(returns1)],
            f"{series2_name}_autocorrelation": [self.test_autocorrelation(returns2)],
        }

        # Add stationarity tests if data is sufficient
        if len(returns1.dropna()) >= 50:
            results[f"{series1_name}_stationarity"] = self.test_stationarity(returns1)
        if len(returns2.dropna()) >= 50:
            results[f"{series2_name}_stationarity"] = self.test_stationarity(returns2)

        return results


class EquivalenceTests:
    """
    Specialized equivalence tests for determining if two strategies are
    practically equivalent rather than just statistically different.
    """

    def __init__(self, equivalence_margin: float = 0.01):
        """
        Initialize equivalence tests.

        Args:
            equivalence_margin: Margin for practical equivalence (e.g., 1% difference)
        """
        self.equivalence_margin = equivalence_margin
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def two_one_sided_test(
        self, returns1: pd.Series, returns2: pd.Series, equivalence_margin: Optional[float] = None
    ) -> StatisticalTestResult:
        """
        Perform Two One-Sided Test (TOST) for equivalence of means.

        Args:
            returns1: First return series
            returns2: Second return series
            equivalence_margin: Margin for equivalence (uses instance default if None)

        Returns:
            Statistical test result
        """
        margin = equivalence_margin or self.equivalence_margin

        returns1_clean = returns1.dropna()
        returns2_clean = returns2.dropna()

        if len(returns1_clean) < 10 or len(returns2_clean) < 10:
            return StatisticalTestResult(
                test_name="Two One-Sided Test (TOST) for Equivalence",
                statistic=0.0,
                p_value=1.0,
                interpretation="Insufficient data for equivalence test",
            )

        try:
            # Calculate means and pooled standard error
            mean1 = returns1_clean.mean()
            mean2 = returns2_clean.mean()
            mean_diff = mean1 - mean2

            # Pooled standard error
            n1, n2 = len(returns1_clean), len(returns2_clean)
            var1, var2 = returns1_clean.var(ddof=1), returns2_clean.var(ddof=1)
            pooled_se = np.sqrt(var1 / n1 + var2 / n2)

            # Two one-sided t-tests
            t1 = (mean_diff - margin) / pooled_se  # Test if diff > -margin
            t2 = (mean_diff + margin) / pooled_se  # Test if diff < +margin

            # Degrees of freedom (Welch's formula)
            df = (var1 / n1 + var2 / n2) ** 2 / (
                (var1 / n1) ** 2 / (n1 - 1) + (var2 / n2) ** 2 / (n2 - 1)
            )

            # P-values for one-sided tests
            p1 = stats.t.cdf(t1, df)  # P(T < t1)
            p2 = 1 - stats.t.cdf(t2, df)  # P(T > t2)

            # TOST p-value is the maximum of the two
            tost_p = max(p1, p2)

            return StatisticalTestResult(
                test_name="Two One-Sided Test (TOST) for Equivalence",
                statistic=max(abs(t1), abs(t2)),
                p_value=tost_p,
                reject_null=bool(tost_p < 0.05),  # Reject null of non-equivalence
                interpretation=(
                    f"Means are practically equivalent (within ±{margin:.4f})"
                    if tost_p < 0.05
                    else f"Cannot conclude practical equivalence (within ±{margin:.4f})"
                ),
                notes=f"Equivalence margin: ±{margin:.4f}, Mean difference: {mean_diff:.6f}",
            )

        except Exception as e:
            self.logger.warning(f"TOST failed: {e}")
            return StatisticalTestResult(
                test_name="Two One-Sided Test (TOST) for Equivalence",
                statistic=0.0,
                p_value=1.0,
                interpretation=f"Test failed: {str(e)}",
            )


def format_test_results(results: Dict[str, List[StatisticalTestResult]]) -> str:
    """
    Format statistical test results into a readable report.

    Args:
        results: Dictionary of test categories and results

    Returns:
        Formatted string report
    """
    lines = ["=" * 80, "STATISTICAL TEST RESULTS", "=" * 80]

    for category, test_list in results.items():
        if not test_list:
            continue

        lines.extend(["", f"{category.upper().replace('_', ' ')}:", "-" * 40])

        for test in test_list:
            lines.extend(
                [
                    f"Test: {test.test_name}",
                    f"  Statistic: {test.statistic:.6f}",
                    f"  P-value: {test.p_value:.6f}",
                    f"  Reject Null: {test.reject_null}",
                    f"  Interpretation: {test.interpretation}",
                    f"  Notes: {test.notes}" if test.notes else "",
                    "",
                ]
            )

    return "\n".join(lines)
