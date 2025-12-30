"""Unit tests for the shared execution model."""

from datetime import datetime

import pytest

from src.data_providers.exchange_interface import OrderSide, OrderType
from src.engines.shared.execution.execution_model import ExecutionModel
from src.engines.shared.execution.fill_policy import default_fill_policy
from src.engines.shared.execution.market_snapshot import MarketSnapshot
from src.engines.shared.execution.order_intent import OrderIntent

BASE_TIME = datetime(2025, 1, 1)


def _snapshot(
    high: float, low: float, last: float = 100.0, close: float = 100.0
) -> MarketSnapshot:
    """Build a MarketSnapshot for execution model tests."""
    return MarketSnapshot(
        symbol="BTCUSDT",
        timestamp=BASE_TIME,
        last_price=last,
        high=high,
        low=low,
        close=close,
        volume=0.0,
    )


def test_take_profit_limit_fills_at_limit_price_long() -> None:
    """Take-profit limit orders fill at the limit price when crossed."""
    model = ExecutionModel(default_fill_policy())
    snapshot = _snapshot(high=110.0, low=95.0)
    order_intent = OrderIntent(
        symbol="BTCUSDT",
        side=OrderSide.SELL,
        order_type=OrderType.TAKE_PROFIT,
        quantity=1.0,
        limit_price=105.0,
    )

    decision = model.decide_fill(order_intent, snapshot)

    assert decision.should_fill is True
    assert decision.fill_price == pytest.approx(105.0)
    assert decision.liquidity == "maker"


def test_take_profit_limit_fills_at_limit_price_short() -> None:
    """Short take-profit limit orders fill at the limit price when crossed."""
    model = ExecutionModel(default_fill_policy())
    snapshot = _snapshot(high=105.0, low=80.0)
    order_intent = OrderIntent(
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        order_type=OrderType.TAKE_PROFIT,
        quantity=1.0,
        limit_price=85.0,
    )

    decision = model.decide_fill(order_intent, snapshot)

    assert decision.should_fill is True
    assert decision.fill_price == pytest.approx(85.0)
    assert decision.liquidity == "maker"


def test_limit_order_no_price_improvement_under_conservative_policy() -> None:
    """Conservative OHLC policy does not allow price improvement."""
    model = ExecutionModel(default_fill_policy())
    snapshot = _snapshot(high=110.0, low=90.0)
    order_intent = OrderIntent(
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=1.0,
        limit_price=95.0,
    )

    decision = model.decide_fill(order_intent, snapshot)

    assert decision.should_fill is True
    assert decision.fill_price == pytest.approx(95.0)


def test_stop_order_fills_with_taker_liquidity() -> None:
    """Stop orders fill at the stop price with taker liquidity."""
    model = ExecutionModel(default_fill_policy())
    snapshot = _snapshot(high=105.0, low=90.0)
    order_intent = OrderIntent(
        symbol="BTCUSDT",
        side=OrderSide.SELL,
        order_type=OrderType.STOP_LOSS,
        quantity=1.0,
        stop_price=95.0,
    )

    decision = model.decide_fill(order_intent, snapshot)

    assert decision.should_fill is True
    assert decision.fill_price == pytest.approx(95.0)
    assert decision.liquidity == "taker"
