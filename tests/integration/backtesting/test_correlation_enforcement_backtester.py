from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from src.backtesting.engine import Backtester
from src.risk.risk_manager import RiskParameters
from src.strategies.base import BaseStrategy

pytestmark = pytest.mark.integration


class DummyStrategy(BaseStrategy):
	def __init__(self):
		super().__init__(name="DummyBacktest")
		self.take_profit_pct = 0.04

	def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
		return df

	def check_entry_conditions(self, df: pd.DataFrame, index: int) -> bool:
		# Always signal entry when not in position
		return True

	def check_exit_conditions(self, df: pd.DataFrame, index: int, entry_price: float) -> bool:
		return False

	def calculate_position_size(self, df: pd.DataFrame, index: int, balance: float) -> float:
		return 0.0

	def calculate_stop_loss(self, df: pd.DataFrame, index: int, price: float, side: str = "long") -> float:
		return price * 0.98

	def get_parameters(self) -> dict:
		return {}

	def get_risk_overrides(self) -> dict:
		# Base fraction 8%, correlated cap 10%
		return {"position_sizer": "fixed_fraction", "base_fraction": 0.08, "correlation_control": {"max_correlated_exposure": 0.1}}


def _df(prices):
	idx = pd.date_range("2024-01-01", periods=len(prices), freq="H")
	return pd.DataFrame({"open": prices, "high": prices, "low": prices, "close": prices, "volume": np.ones(len(prices))}, index=idx)


def test_backtester_correlation_reduces_size(monkeypatch):
	# Prepare data
	candidate_prices = np.linspace(100, 120, 50)
	df = _df(candidate_prices)

	# Provider mock returns candidate df for candidate symbol, slightly scaled for correlated symbol
	provider = Mock()
	def _hist(sym, timeframe, start, end):
		if sym == "ETHUSDT":
			return df
		# Correlated series
		return _df(candidate_prices * (1.01 if sym != "ETHUSDT" else 1.0))
	provider.get_historical_data.side_effect = _hist

	# Risk params without daily cap interference
	risk_params = RiskParameters(base_risk_per_trade=0.2, max_risk_per_trade=0.2, max_position_size=0.5, max_daily_risk=1.0)

	# Backtester
	bt = Backtester(strategy=DummyStrategy(), data_provider=provider, risk_parameters=risk_params)
	bt.balance = 10_000

	# Seed an existing correlated open position in risk manager
	bt.risk_manager.positions = {"BTCUSDT": {"size": 0.06, "entry_price": 110.0, "side": "long"}}
	# Also expose a list of positions with symbols so correlation_ctx includes BTCUSDT
	bt.positions = [SimpleNamespace(symbol="BTCUSDT")]

	# Wrap calculate_position_fraction to capture the sized fraction
	original_calc = bt.risk_manager.calculate_position_fraction
	captures = {}
	def _wrap(*args, **kwargs):
		res = original_calc(*args, **kwargs)
		captures.setdefault("fractions", []).append(res)
		captures["last_ctx"] = kwargs.get("correlation_ctx")
		return res
	monkeypatch.setattr(bt.risk_manager, "calculate_position_fraction", _wrap)

	# Run minimal backtest
	start = df.index[0]
	end = df.index[-1]
	bt.run(symbol="ETHUSDT", timeframe="1h", start=start, end=end)

	# Verify correlation_ctx was provided and fraction reduced
	assert captures.get("last_ctx") is not None
	frac_used = max(captures.get("fractions", [0.0]))
	# With existing 0.06 and base 0.08, expected reduced fraction ~ 0.08 * (0.1/(0.06+0.08)) ~ 0.057
	assert 0.045 <= frac_used <= 0.07