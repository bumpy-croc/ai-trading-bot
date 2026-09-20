"""Time-exit policy resolution is logged at boot and matches backtest/shared (#1083).

DEFAULT_MAX_HOLDING_HOURS is a fallback *inside* a strategy-supplied
``time_exits`` config, not a universal cap. These tests pin that a strategy
without ``time_exits`` resolves to no policy on every path (live, backtest,
shared builder) and that the resolved policy is visible in the boot log.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from src.engines.backtest.engine import Backtester
from src.engines.live.trading_engine import LiveTradingEngine
from src.engines.shared.risk_configuration import build_time_exit_policy

pytestmark = pytest.mark.fast


class _Strategy:
    def __init__(self, overrides):
        self._overrides = overrides

    def get_risk_overrides(self):
        return self._overrides


def _live_stub(strategy):
    stub = SimpleNamespace(strategy=strategy, risk_manager=None)
    stub._log_resolved_time_exit_policy = lambda: LiveTradingEngine._log_resolved_time_exit_policy(
        stub
    )
    return stub


def test_no_time_exits_resolves_to_no_policy_on_every_path(caplog):
    strategy = _Strategy(None)

    live = _live_stub(strategy)
    with caplog.at_level(logging.INFO, logger="src.engines.live.trading_engine"):
        LiveTradingEngine._init_time_exit_policy(live, None)

    backtest = SimpleNamespace(strategy=strategy, risk_manager=None)

    assert live.time_exit_policy is None
    assert Backtester._build_time_exit_policy(backtest) is None
    assert build_time_exit_policy(strategy) is None
    assert any("Time-exit policy: NONE" in r.getMessage() for r in caplog.records)


def test_configured_time_exits_are_logged_and_match_backtest(caplog):
    strategy = _Strategy({"time_exits": {"max_holding_hours": 48}})

    live = _live_stub(strategy)
    with caplog.at_level(logging.INFO, logger="src.engines.live.trading_engine"):
        LiveTradingEngine._init_time_exit_policy(live, None)

    backtest = SimpleNamespace(strategy=strategy, risk_manager=None)

    assert live.time_exit_policy.max_holding_hours == 48
    assert Backtester._build_time_exit_policy(backtest).max_holding_hours == 48
    assert any("max_holding_hours=48" in r.getMessage() for r in caplog.records)
