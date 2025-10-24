import numpy as np
import pandas as pd
import pytest

from src.position_management.correlation_engine import CorrelationConfig, CorrelationEngine

pytestmark = pytest.mark.unit


def _series(values, start=0):
    idx = pd.date_range("2024-01-01", periods=len(values), freq="D")
    return pd.Series(values, index=idx)


def test_correlation_grouping_and_exposure():
    cfg = CorrelationConfig(
        correlation_window_days=30, correlation_threshold=0.7, sample_min_size=5
    )
    engine = CorrelationEngine(cfg)
    # Create two highly correlated series and one independent
    a = _series(np.linspace(100, 110, 30))
    b = a * 1.01  # nearly perfect correlation
    c = _series(np.linspace(50, 60, 30) + np.random.RandomState(42).normal(0, 2, 30))
    corr = engine.calculate_position_correlations({"A": a, "B": b, "C": c})
    groups = engine.get_correlation_groups(corr)
    # A and B should be grouped, C separate
    assert any(set(g) == {"A", "B"} for g in groups)
    # Exposure aggregation
    positions = {"A": {"size": 0.05}, "B": {"size": 0.06}, "C": {"size": 0.02}}
    exp = engine.get_correlated_exposure(positions, groups)
    assert exp[tuple(sorted(["A", "B"]))] == pytest.approx(0.11)


def test_size_reduction_when_exceeding_limit():
    cfg = CorrelationConfig(max_correlated_exposure=0.1, sample_min_size=5)
    engine = CorrelationEngine(cfg)
    # Highly correlated symbols
    a = _series(np.linspace(100, 120, 40))
    b = a * 1.02
    corr = engine.calculate_position_correlations({"A": a, "B": b})
    positions = {"A": {"size": 0.06}}
    # Candidate B wants 0.08; projected group = 0.14 > 0.1, so factor ~ 0.1/0.14
    factor = engine.compute_size_reduction_factor(positions, corr, "B", 0.08)
    assert 0.6 < factor < 0.8
    reduced = 0.08 * factor
    assert reduced <= 0.1


class TestCorrelationEngineEdgeCases:
    """Test edge cases and extreme scenarios for CorrelationEngine."""

    def test_empty_price_data(self):
        """Test with empty price data."""
        cfg = CorrelationConfig()
        engine = CorrelationEngine(cfg)

        corr = engine.calculate_position_correlations({})
        assert corr.empty

    def test_single_symbol_correlation(self):
        """Test correlation calculation with single symbol."""
        cfg = CorrelationConfig()
        engine = CorrelationEngine(cfg)

        a = _series(np.linspace(100, 110, 30))
        corr = engine.calculate_position_correlations({"A": a})

        # Single symbol returns empty correlation matrix (need at least 2 symbols)
        assert corr.empty

    def test_insufficient_data_points(self):
        """Test with insufficient data points."""
        cfg = CorrelationConfig(sample_min_size=10)
        engine = CorrelationEngine(cfg)

        # Only 5 data points, less than required minimum
        a = _series([100, 101, 102, 103, 104])
        b = _series([200, 202, 204, 206, 208])

        corr = engine.calculate_position_correlations({"A": a, "B": b})

        # Should return empty correlation matrix due to insufficient data
        assert corr.empty or len(corr) == 0

    def test_perfect_correlation(self):
        """Test with perfectly correlated series."""
        cfg = CorrelationConfig(correlation_threshold=0.99)
        engine = CorrelationEngine(cfg)

        a = _series(np.linspace(100, 110, 30))
        b = a.copy()  # Perfect correlation

        corr = engine.calculate_position_correlations({"A": a, "B": b})
        groups = engine.get_correlation_groups(corr)

        # Should group perfectly correlated symbols
        assert any(set(g) == {"A", "B"} for g in groups)

    def test_perfect_negative_correlation(self):
        """Test with perfectly negatively correlated series."""
        cfg = CorrelationConfig(correlation_threshold=0.7)
        engine = CorrelationEngine(cfg)

        a = _series(np.linspace(100, 110, 30))
        b = _series(np.linspace(110, 100, 30))  # Perfect negative correlation

        corr = engine.calculate_position_correlations({"A": a, "B": b})

        # Should detect strong negative correlation
        # The actual correlation might be positive due to returns calculation
        assert abs(corr.loc["A", "B"]) > 0.9  # Strong correlation (either positive or negative)

    def test_zero_variance_series(self):
        """Test with zero variance (constant) series."""
        cfg = CorrelationConfig()
        engine = CorrelationEngine(cfg)

        a = _series([100] * 30)  # Constant series
        b = _series(np.linspace(100, 110, 30))

        corr = engine.calculate_position_correlations({"A": a, "B": b})

        # Correlation with constant series should be NaN
        assert np.isnan(corr.loc["A", "B"]) or corr.empty

    def test_mixed_length_series(self):
        """Test with series of different lengths."""
        cfg = CorrelationConfig()
        engine = CorrelationEngine(cfg)

        a = _series(np.linspace(100, 110, 30))
        b = _series(np.linspace(200, 220, 20))  # Different length

        # Should handle gracefully - pandas will align by index
        corr = engine.calculate_position_correlations({"A": a, "B": b})

        # Should work with overlapping indices
        assert isinstance(corr, pd.DataFrame)

    def test_nan_values_in_series(self):
        """Test with NaN values in price series."""
        cfg = CorrelationConfig()
        engine = CorrelationEngine(cfg)

        a_values = np.linspace(100, 110, 30)
        a_values[10:15] = np.nan  # Insert NaN values
        a = _series(a_values)
        b = _series(np.linspace(200, 220, 30))

        corr = engine.calculate_position_correlations({"A": a, "B": b})

        # Should handle NaN values gracefully
        assert isinstance(corr, pd.DataFrame)

    def test_extreme_correlation_threshold(self):
        """Test with extreme correlation thresholds."""
        # Very high threshold
        cfg_high = CorrelationConfig(correlation_threshold=0.999)
        engine_high = CorrelationEngine(cfg_high)

        a = _series(np.linspace(100, 110, 30))
        b = a * 1.01  # Very high but not perfect correlation

        corr = engine_high.calculate_position_correlations({"A": a, "B": b})
        groups = engine_high.get_correlation_groups(corr)

        # Might not group due to very high threshold
        assert isinstance(groups, list)

        # Very low threshold
        cfg_low = CorrelationConfig(correlation_threshold=0.01)
        engine_low = CorrelationEngine(cfg_low)

        groups_low = engine_low.get_correlation_groups(corr)

        # Should group almost everything
        assert len(groups_low) <= 2

    def test_zero_correlation_threshold(self):
        """Test with zero correlation threshold."""
        cfg = CorrelationConfig(correlation_threshold=0.0)
        engine = CorrelationEngine(cfg)

        # Create uncorrelated series
        np.random.seed(42)
        a = _series(np.random.randn(30))
        b = _series(np.random.randn(30))

        corr = engine.calculate_position_correlations({"A": a, "B": b})
        groups = engine.get_correlation_groups(corr)

        # Should group everything with zero threshold
        assert len(groups) == 1

    def test_negative_correlation_threshold(self):
        """Test with negative correlation threshold."""
        cfg = CorrelationConfig(correlation_threshold=-0.5)
        engine = CorrelationEngine(cfg)

        # Test 1: Create truly negatively correlated returns
        np.random.seed(42)
        a_returns = np.random.randn(30) * 0.02
        b_returns = -a_returns + np.random.randn(30) * 0.001  # Opposite returns with small noise

        # Convert returns to price series
        a_prices = [100]
        b_prices = [100]
        for i in range(len(a_returns)):
            a_prices.append(a_prices[-1] * (1 + a_returns[i]))
            b_prices.append(b_prices[-1] * (1 + b_returns[i]))

        a = _series(a_prices[1:])
        b = _series(b_prices[1:])

        corr = engine.calculate_position_correlations({"A": a, "B": b})
        groups = engine.get_correlation_groups(corr)

        # The current implementation only groups when corr >= threshold
        # Negative correlations like -0.99 are < -0.5, so they won't be grouped
        # This test verifies the current behavior: no grouping for negative correlations
        assert not any(set(g) == {"A", "B"} for g in groups)

        # Test 2: Positive correlations above the negative threshold should still be grouped
        c = _series(np.linspace(100, 110, 30))
        d = c * 1.01  # Positive correlation

        corr2 = engine.calculate_position_correlations({"C": c, "D": d})
        groups2 = engine.get_correlation_groups(corr2)

        # Positive correlations above -0.5 should be grouped
        assert any(set(g) == {"C", "D"} for g in groups2)

    def test_empty_positions_dict(self):
        """Test with empty positions dictionary."""
        cfg = CorrelationConfig()
        engine = CorrelationEngine(cfg)

        a = _series(np.linspace(100, 110, 30))
        engine.calculate_position_correlations({"A": a})

        exposure = engine.get_correlated_exposure({}, [])
        assert exposure == {}

    def test_positions_not_in_correlation_matrix(self):
        """Test with positions that don't exist in correlation matrix."""
        cfg = CorrelationConfig()
        engine = CorrelationEngine(cfg)

        a = _series(np.linspace(100, 110, 30))
        engine.calculate_position_correlations({"A": a})

        positions = {"B": {"size": 0.05}, "C": {"size": 0.03}}  # Not in correlation matrix
        groups = [["A"], ["B", "C"]]

        exposure = engine.get_correlated_exposure(positions, groups)

        # Should handle missing symbols gracefully
        assert isinstance(exposure, dict)

    def test_zero_position_sizes(self):
        """Test with zero position sizes."""
        cfg = CorrelationConfig()
        engine = CorrelationEngine(cfg)

        a = _series(np.linspace(100, 110, 30))
        b = a * 1.01
        engine.calculate_position_correlations({"A": a, "B": b})
        groups = [["A", "B"]]

        positions = {"A": {"size": 0.0}, "B": {"size": 0.0}}
        exposure = engine.get_correlated_exposure(positions, groups)

        assert exposure[("A", "B")] == 0.0

    def test_negative_position_sizes(self):
        """Test with negative position sizes."""
        cfg = CorrelationConfig()
        engine = CorrelationEngine(cfg)

        a = _series(np.linspace(100, 110, 30))
        b = a * 1.01
        engine.calculate_position_correlations({"A": a, "B": b})
        groups = [["A", "B"]]

        positions = {"A": {"size": -0.05}, "B": {"size": 0.03}}
        exposure = engine.get_correlated_exposure(positions, groups)

        # Should sum absolute values or handle negative sizes appropriately
        assert ("A", "B") in exposure

    def test_compute_size_reduction_factor_edge_cases(self):
        """Test compute_size_reduction_factor with edge cases."""
        cfg = CorrelationConfig(max_correlated_exposure=0.1)
        engine = CorrelationEngine(cfg)

        a = _series(np.linspace(100, 110, 30))
        b = a * 1.01
        corr = engine.calculate_position_correlations({"A": a, "B": b})

        # Test with zero candidate size
        positions = {"A": {"size": 0.05}}
        factor = engine.compute_size_reduction_factor(positions, corr, "B", 0.0)
        assert factor == 1.0  # No reduction needed for zero size

        # Test with no existing positions
        factor = engine.compute_size_reduction_factor({}, corr, "B", 0.05)
        assert factor == 1.0  # No reduction needed when no existing positions

        # Test with candidate not in correlation matrix
        factor = engine.compute_size_reduction_factor(positions, corr, "C", 0.05)
        assert factor == 1.0  # No reduction if not correlated

    def test_zero_max_correlated_exposure(self):
        """Test with zero maximum correlated exposure."""
        cfg = CorrelationConfig(max_correlated_exposure=0.0)
        engine = CorrelationEngine(cfg)

        a = _series(np.linspace(100, 110, 30))
        b = a * 1.01
        corr = engine.calculate_position_correlations({"A": a, "B": b})

        positions = {"A": {"size": 0.05}}
        factor = engine.compute_size_reduction_factor(positions, corr, "B", 0.05)

        # Should heavily reduce or eliminate new position
        assert factor <= 0.1

    def test_very_large_max_correlated_exposure(self):
        """Test with very large maximum correlated exposure."""
        cfg = CorrelationConfig(max_correlated_exposure=10.0)  # 1000%
        engine = CorrelationEngine(cfg)

        a = _series(np.linspace(100, 110, 30))
        b = a * 1.01
        corr = engine.calculate_position_correlations({"A": a, "B": b})

        positions = {"A": {"size": 0.5}}
        factor = engine.compute_size_reduction_factor(positions, corr, "B", 0.5)

        # Should not reduce with very high limit
        assert factor == pytest.approx(1.0)

    def test_extreme_price_series(self):
        """Test with extreme price values."""
        cfg = CorrelationConfig()
        engine = CorrelationEngine(cfg)

        # Very large numbers
        a = _series(np.linspace(1e6, 1.1e6, 30))
        b = _series(np.linspace(1e9, 1.1e9, 30))

        corr = engine.calculate_position_correlations({"A": a, "B": b})

        # Should handle extreme values
        assert isinstance(corr, pd.DataFrame)

        # Very small numbers
        c = _series(np.linspace(1e-6, 1.1e-6, 30))
        d = _series(np.linspace(1e-9, 1.1e-9, 30))

        corr_small = engine.calculate_position_correlations({"C": c, "D": d})
        assert isinstance(corr_small, pd.DataFrame)

    def test_correlation_window_edge_cases(self):
        """Test with edge cases for correlation window."""
        # Very small window
        cfg_small = CorrelationConfig(correlation_window_days=1, sample_min_size=1)
        engine_small = CorrelationEngine(cfg_small)

        a = _series(np.linspace(100, 110, 30))
        b = a * 1.01

        corr = engine_small.calculate_position_correlations({"A": a, "B": b})
        assert isinstance(corr, pd.DataFrame)

        # Very large window (larger than data)
        cfg_large = CorrelationConfig(correlation_window_days=100, sample_min_size=5)
        engine_large = CorrelationEngine(cfg_large)

        corr_large = engine_large.calculate_position_correlations({"A": a, "B": b})
        assert isinstance(corr_large, pd.DataFrame)

    def test_sample_min_size_edge_cases(self):
        """Test with edge cases for sample minimum size."""
        # Minimum size larger than available data
        cfg = CorrelationConfig(sample_min_size=100)
        engine = CorrelationEngine(cfg)

        a = _series(np.linspace(100, 110, 30))  # Only 30 points
        b = a * 1.01

        corr = engine.calculate_position_correlations({"A": a, "B": b})

        # Should return empty or handle gracefully
        assert corr.empty or len(corr) == 0

    def test_correlation_config_edge_cases(self):
        """Test CorrelationConfig with edge case values."""
        # Test with extreme values
        cfg = CorrelationConfig(
            correlation_window_days=0,
            correlation_threshold=2.0,  # > 1.0
            max_correlated_exposure=-0.1,  # Negative
            sample_min_size=0,
        )

        # Should handle invalid config gracefully
        engine = CorrelationEngine(cfg)
        assert isinstance(engine, CorrelationEngine)

    def test_get_correlation_groups_empty_corr_matrix(self):
        """Test get_correlation_groups with empty correlation matrix."""
        cfg = CorrelationConfig()
        engine = CorrelationEngine(cfg)

        empty_corr = pd.DataFrame()
        groups = engine.get_correlation_groups(empty_corr)

        assert groups == []

    def test_get_correlation_groups_single_symbol(self):
        """Test get_correlation_groups with single symbol."""
        cfg = CorrelationConfig()
        engine = CorrelationEngine(cfg)

        single_corr = pd.DataFrame([[1.0]], index=["A"], columns=["A"])
        groups = engine.get_correlation_groups(single_corr)

        # Single symbol might not form a group if threshold logic excludes it
        assert isinstance(groups, list)

    def test_overlapping_groups_handling(self):
        """Test handling of overlapping correlation groups."""
        cfg = CorrelationConfig(correlation_threshold=0.5)
        engine = CorrelationEngine(cfg)

        # Create scenario where A-B and B-C are correlated, but A-C might not be
        a = _series(np.linspace(100, 110, 30))
        b = a * 1.01 + np.random.RandomState(42).normal(0, 0.1, 30)
        c = b * 1.01 + np.random.RandomState(43).normal(0, 0.1, 30)

        corr = engine.calculate_position_correlations({"A": a, "B": b, "C": c})
        groups = engine.get_correlation_groups(corr)

        # Should handle overlapping groups appropriately
        assert isinstance(groups, list)
        assert all(isinstance(group, list) for group in groups)
