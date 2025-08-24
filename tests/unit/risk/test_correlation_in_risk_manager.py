import numpy as np
import pandas as pd
import pytest

from src.position_management.correlation_engine import CorrelationConfig, CorrelationEngine
from src.risk.risk_manager import RiskManager, RiskParameters

pytestmark = pytest.mark.unit


def test_risk_manager_applies_correlation_reduction():
	# Setup
	params = RiskParameters(
		base_risk_per_trade=0.1,  # high to make effect visible
		max_risk_per_trade=0.2,
		max_position_size=0.5,
		max_daily_risk=1.0,
	)
	rm = RiskManager(parameters=params)
	# Pretend an existing correlated position
	rm.positions = {"A": {"size": 0.06, "entry_price": 100.0, "side": "long"}}

	# Dataframe with candidate symbol series
	idx = pd.date_range("2024-01-01", periods=50, freq="D")
	prices_a = pd.Series(np.linspace(100, 120, 50), index=idx)
	prices_b = prices_a * 1.01
	df = pd.DataFrame({"close": prices_b.values}, index=idx)

	# Correlation engine and matrix
	cfg = CorrelationConfig(max_correlated_exposure=0.1, sample_min_size=10)
	engine = CorrelationEngine(cfg)
	corr = engine.calculate_position_correlations({"A": prices_a, "B": prices_b})

	# Baseline fraction before reduction (fixed_fraction 0.08 ~8%)
	overrides = {"position_sizer": "fixed_fraction", "base_fraction": 0.08}
	fraction = rm.calculate_position_fraction(
		df=df,
		index=len(df) - 1,
		balance=10_000,
		price=float(df["close"].iloc[-1]),
		strategy_overrides=overrides,
		correlation_ctx={
			"engine": engine,
			"candidate_symbol": "B",
			"corr_matrix": corr,
		},
	)
	# Projected exposure would be 0.06 + 0.08 = 0.14 > 0.1, expect reduction below 0.1
	assert fraction <= 0.1
	assert fraction > 0.05