"""Backtest results must auto-report the effective (resolved) sizing.

Part of the GH #1021 visibility fix: the sizing limits a run actually
enforced appear in the results payload, so a clamped or defaulted
``max_position_size`` can never silently invalidate a study.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import pytest

from src.engines.backtest.engine import Backtester
from src.risk.risk_manager import RiskParameters
from src.strategies.ml_basic import create_ml_basic_strategy

pytestmark = pytest.mark.fast


def _make_backtester(mock_data_provider, **kwargs) -> Backtester:
    return Backtester(
        strategy=create_ml_basic_strategy(),
        data_provider=mock_data_provider,
        initial_balance=10_000,
        log_to_database=False,
        **kwargs,
    )


class TestEffectiveSizingReport:
    def test_final_results_include_effective_sizing(self, mock_data_provider):
        backtester = _make_backtester(
            mock_data_provider,
            risk_parameters=RiskParameters(
                max_position_size=0.25,
                base_risk_per_trade=0.02,
                max_risk_per_trade=0.03,
            ),
        )

        results = backtester.run(
            symbol="BTCUSDT",
            timeframe="1h",
            start=datetime(2024, 1, 1, tzinfo=UTC),
            end=datetime(2024, 1, 2, tzinfo=UTC),
        )

        assert results["effective_sizing"] == {
            "max_position_size": 0.25,
            "base_risk_per_trade": 0.02,
            "max_risk_per_trade": 0.03,
        }

    def test_empty_data_results_include_effective_sizing(self, mock_data_provider):
        mock_data_provider.get_historical_data.return_value = pd.DataFrame()
        backtester = _make_backtester(
            mock_data_provider,
            risk_parameters=RiskParameters(max_position_size=0.17),
        )

        results = backtester.run(
            symbol="BTCUSDT",
            timeframe="1h",
            start=datetime(2024, 1, 1, tzinfo=UTC),
            end=datetime(2024, 1, 2, tzinfo=UTC),
        )

        assert results["total_trades"] == 0
        assert results["effective_sizing"]["max_position_size"] == 0.17

    def test_default_parameters_report_their_resolved_values(self, mock_data_provider):
        """No explicit risk parameters: the report shows what the defaults
        resolved to — that is exactly the visibility #1021 asked for."""
        backtester = _make_backtester(mock_data_provider)

        results = backtester.run(
            symbol="BTCUSDT",
            timeframe="1h",
            start=datetime(2024, 1, 1, tzinfo=UTC),
            end=datetime(2024, 1, 2, tzinfo=UTC),
        )

        defaults = RiskParameters()
        assert results["effective_sizing"] == {
            "max_position_size": defaults.max_position_size,
            "base_risk_per_trade": defaults.base_risk_per_trade,
            "max_risk_per_trade": defaults.max_risk_per_trade,
        }
