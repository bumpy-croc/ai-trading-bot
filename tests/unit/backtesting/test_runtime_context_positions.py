"""Regression tests for #756: backtest RuntimeContext must expose open positions.

``Backtester._build_runtime_context`` passed the shared ``PositionSide`` enum
into ``ComponentPosition``, whose validation expects the strings
``"long"``/``"short"`` — construction raised ``ValueError`` on every candle,
was swallowed, and component strategies always saw ``current_positions=None``
while live strategies saw the real position (parity gap).
"""

import logging
from datetime import UTC, datetime
from unittest.mock import Mock

import pytest

from src.engines.backtest.engine import Backtester
from src.engines.backtest.models import ActiveTrade
from src.engines.shared.models import PositionSide
from src.strategies.ml_basic import create_ml_basic_strategy

pytestmark = [pytest.mark.unit, pytest.mark.fast]


@pytest.fixture
def backtester(mock_data_provider):
    return Backtester(
        strategy=create_ml_basic_strategy(),
        data_provider=mock_data_provider,
        initial_balance=10_000,
        log_to_database=False,
    )


def _open_position(backtester: Backtester, side: PositionSide) -> None:
    backtester.position_tracker.open_position(
        ActiveTrade(
            symbol="BTCUSDT",
            side=side,
            entry_price=100.0,
            entry_time=datetime(2024, 1, 1, tzinfo=UTC),
            size=0.1,
            entry_balance=10_000.0,
        )
    )


class TestRuntimeContextPositions:
    def test_open_long_position_visible_to_strategies(self, backtester, caplog):
        """The active trade appears in RuntimeContext with a string side."""
        _open_position(backtester, PositionSide.LONG)

        with caplog.at_level(logging.WARNING):
            ctx = backtester._build_runtime_context(
                balance=10_000.0, current_price=110.0, current_time=datetime.now(UTC)
            )

        assert ctx.current_positions is not None
        assert len(ctx.current_positions) == 1
        position = ctx.current_positions[0]
        assert position.symbol == "BTCUSDT"
        assert position.side == "long"
        assert position.entry_price == 100.0
        assert position.current_price == 110.0
        # The old bug logged a translation warning every run
        assert "Failed to translate active trade" not in caplog.text

    def test_open_short_position_visible_to_strategies(self, backtester):
        _open_position(backtester, PositionSide.SHORT)

        ctx = backtester._build_runtime_context(
            balance=10_000.0, current_price=90.0, current_time=datetime.now(UTC)
        )

        assert ctx.current_positions is not None
        assert ctx.current_positions[0].side == "short"

    def test_no_position_yields_none(self, backtester):
        ctx = backtester._build_runtime_context(
            balance=10_000.0, current_price=100.0, current_time=datetime.now(UTC)
        )

        assert ctx.current_positions is None

    def test_position_size_uses_notional(self, backtester):
        """Size is the notional (fraction × balance), matching live's quantity
        semantics for the runtime context."""
        _open_position(backtester, PositionSide.LONG)

        ctx = backtester._build_runtime_context(
            balance=10_000.0, current_price=110.0, current_time=datetime.now(UTC)
        )

        assert ctx.current_positions is not None
        assert ctx.current_positions[0].size == pytest.approx(0.1 * 10_000.0)


class TestRuntimeContextTranslationFailure:
    def test_unexpected_failure_warns_once_then_debug(self, backtester, caplog):
        """The defensive warn-once path still guards unexpected failures."""
        _open_position(backtester, PositionSide.LONG)
        # Corrupt the trade so translation genuinely fails
        backtester.position_tracker.current_trade.entry_price = Mock(side_effect=TypeError("boom"))
        backtester.position_tracker.current_trade.symbol = None  # ComponentPosition rejects

        with caplog.at_level(logging.WARNING):
            ctx1 = backtester._build_runtime_context(
                balance=10_000.0, current_price=110.0, current_time=datetime.now(UTC)
            )
            ctx2 = backtester._build_runtime_context(
                balance=10_000.0, current_price=110.0, current_time=datetime.now(UTC)
            )

        assert ctx1.current_positions is None
        assert ctx2.current_positions is None
        assert caplog.text.count("Failed to translate active trade") == 1
