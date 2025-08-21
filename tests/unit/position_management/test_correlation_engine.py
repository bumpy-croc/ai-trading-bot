import pandas as pd
import numpy as np
import pytest

from src.position_management.correlation_engine import CorrelationEngine, CorrelationConfig

pytestmark = pytest.mark.unit


def _series(values, start=0):
	idx = pd.date_range("2024-01-01", periods=len(values), freq="D")
	return pd.Series(values, index=idx)


def test_correlation_grouping_and_exposure():
	cfg = CorrelationConfig(correlation_window_days=30, correlation_threshold=0.7, sample_min_size=5)
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