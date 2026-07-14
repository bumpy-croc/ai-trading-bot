"""Clamp-binding visibility in the shared position-sizing seam (GH #1021).

When a strategy requests a ``max_fraction`` above the risk manager's
``max_position_size`` cap, the clamp used to bind silently — a study could
run at half its intended sizing with no trace. The seam
(``PortfolioRiskManager._parse_position_sizing_params``) now emits a
structured WARNING naming the requested value, the effective cap, and the
clamped result. Both engines flow through this seam: component strategies
via ``CoreRiskAdapter.calculate_position_size`` (backtest + live) and the
live short path via ``entry_coordinator``.

The clamp VALUE semantics are unchanged — only visibility is added.
"""

from __future__ import annotations

import logging
from unittest.mock import Mock

import pandas as pd
import pytest

from src.risk.risk_manager import PortfolioRiskManager, RiskParameters
from src.strategies.components.risk_adapter import CoreRiskAdapter

pytestmark = pytest.mark.fast


@pytest.fixture
def df() -> pd.DataFrame:
    closes = [100.0 + i for i in range(30)]
    return pd.DataFrame(
        {
            "open": closes,
            "high": [c + 1 for c in closes],
            "low": [c - 1 for c in closes],
            "close": closes,
            "volume": [1000.0] * len(closes),
        },
        index=pd.date_range("2026-01-01", periods=len(closes), freq="1h"),
    )


@pytest.fixture
def manager() -> PortfolioRiskManager:
    return PortfolioRiskManager(RiskParameters(max_position_size=0.2))


def _fraction(manager, df, overrides):
    return manager.calculate_position_fraction(
        df=df,
        index=len(df) - 1,
        balance=10_000.0,
        price=float(df["close"].iloc[-1]),
        strategy_overrides=overrides,
    )


def _clamp_warnings(caplog):
    return [
        record
        for record in caplog.records
        if record.levelno == logging.WARNING
        and "max_fraction" in record.getMessage()
        and "clamp" in record.getMessage()
    ]


class TestClampBindingWarning:
    def test_binding_clamp_emits_warning_with_values(self, manager, df, caplog):
        with caplog.at_level(logging.WARNING):
            _fraction(manager, df, {"max_fraction": 0.5, "base_fraction": 0.1})

        warnings = _clamp_warnings(caplog)
        assert len(warnings) == 1
        message = warnings[0].getMessage()
        assert "0.5" in message  # requested
        assert "0.2" in message  # effective cap == clamped result

    def test_repeated_identical_clamp_warns_once(self, manager, df, caplog):
        overrides = {"max_fraction": 0.5, "base_fraction": 0.1}
        with caplog.at_level(logging.WARNING):
            _fraction(manager, df, overrides)
            _fraction(manager, df, overrides)
            _fraction(manager, df, overrides)

        assert len(_clamp_warnings(caplog)) == 1

    def test_new_requested_value_warns_again(self, manager, df, caplog):
        with caplog.at_level(logging.WARNING):
            _fraction(manager, df, {"max_fraction": 0.5, "base_fraction": 0.1})
            _fraction(manager, df, {"max_fraction": 0.6, "base_fraction": 0.1})

        assert len(_clamp_warnings(caplog)) == 2

    def test_request_within_cap_does_not_warn(self, manager, df, caplog):
        with caplog.at_level(logging.WARNING):
            _fraction(manager, df, {"max_fraction": 0.1, "base_fraction": 0.05})

        assert _clamp_warnings(caplog) == []

    def test_request_equal_to_cap_does_not_warn(self, manager, df, caplog):
        with caplog.at_level(logging.WARNING):
            _fraction(manager, df, {"max_fraction": 0.2, "base_fraction": 0.05})

        assert _clamp_warnings(caplog) == []

    def test_no_max_fraction_request_does_not_warn(self, manager, df, caplog):
        with caplog.at_level(logging.WARNING):
            _fraction(manager, df, {"base_fraction": 0.1})

        assert _clamp_warnings(caplog) == []

    def test_clamped_sizing_value_is_unchanged_by_the_warning(self, manager, df):
        """Visibility only: the returned fraction equals the pre-change clamp
        semantics — min(requested, cap) then downstream risk limits."""
        with_warning = _fraction(manager, df, {"max_fraction": 0.5, "base_fraction": 0.5})
        quiet_manager = PortfolioRiskManager(RiskParameters(max_position_size=0.2))
        baseline = _fraction(quiet_manager, df, {"max_fraction": 0.2, "base_fraction": 0.5})
        assert with_warning == pytest.approx(baseline)


class TestSharedSeamCoverage:
    def test_component_adapter_path_emits_the_same_warning(self, manager, df, caplog):
        """CoreRiskAdapter (the component-strategy sizing path used by BOTH
        engines) flows through the same seam and inherits the warning."""
        adapter = CoreRiskAdapter(core_manager=manager)
        adapter.set_strategy_overrides({"max_fraction": 0.5, "base_fraction": 0.1})

        with caplog.at_level(logging.WARNING):
            adapter.calculate_position_size(
                Mock(name="signal"),
                balance=10_000.0,
                df=df,
                index=len(df) - 1,
                price=float(df["close"].iloc[-1]),
            )

        assert len(_clamp_warnings(caplog)) == 1
