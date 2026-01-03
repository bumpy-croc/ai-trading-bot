"""OHLC-based fill model for execution simulation."""

from __future__ import annotations

import math

from src.data_providers.exchange_interface import OrderSide
from src.engines.shared.execution.execution_decision import ExecutionDecision
from src.engines.shared.execution.fill_policy import FillPolicy
from src.engines.shared.execution.market_snapshot import MarketSnapshot
from src.engines.shared.execution.order_intent import OrderIntent

ZERO_PRICE = 0.0
ZERO_QUANTITY = 0.0


class OhlcFillModel:
    """Decides fills using OHLC data and conservative assumptions."""

    def decide_fill(
        self,
        order_intent: OrderIntent,
        snapshot: MarketSnapshot,
        policy: FillPolicy,
    ) -> ExecutionDecision:
        """Decide whether an order fills using OHLC data."""
        # Validate quantity is positive and finite to prevent NaN propagation
        if (
            order_intent.quantity <= ZERO_QUANTITY
            or not math.isfinite(order_intent.quantity)
        ):
            return ExecutionDecision.no_fill("quantity must be positive and finite")

        if order_intent.is_market_order():
            return self._fill_market(order_intent, snapshot)

        if order_intent.is_limit_order():
            return self._fill_limit(order_intent, snapshot, policy)

        if order_intent.is_stop_order():
            return self._fill_stop(order_intent, snapshot)

        return ExecutionDecision.no_fill("unsupported order type")

    def _fill_market(
        self,
        order_intent: OrderIntent,
        snapshot: MarketSnapshot,
    ) -> ExecutionDecision:
        """Fill a market order at the last known price."""
        market_price = self._select_market_price(snapshot)
        if market_price <= ZERO_PRICE:
            return ExecutionDecision.no_fill("market price unavailable")

        return ExecutionDecision(
            should_fill=True,
            fill_price=market_price,
            filled_quantity=order_intent.quantity,
            liquidity="taker",
            reason="market order",
        )

    def _fill_limit(
        self,
        order_intent: OrderIntent,
        snapshot: MarketSnapshot,
        policy: FillPolicy,
    ) -> ExecutionDecision:
        """Fill a limit-style order when the bar crosses its limit price."""
        if order_intent.limit_price is None:
            return ExecutionDecision.no_fill("limit price missing")

        if not math.isfinite(order_intent.limit_price):
            return ExecutionDecision.no_fill("limit price must be finite")

        if order_intent.limit_price <= ZERO_PRICE:
            return ExecutionDecision.no_fill("limit price must be positive")

        if not self._limit_crossed(order_intent, snapshot, order_intent.limit_price):
            return ExecutionDecision.no_fill("limit price not crossed")

        fill_price = self._limit_fill_price(order_intent, snapshot, policy)
        return ExecutionDecision(
            should_fill=True,
            fill_price=fill_price,
            filled_quantity=order_intent.quantity,
            liquidity="maker",
            reason="limit order filled",
        )

    def _fill_stop(
        self,
        order_intent: OrderIntent,
        snapshot: MarketSnapshot,
    ) -> ExecutionDecision:
        """Fill a stop-loss style order when the stop price is triggered.

        Stop orders fill at adverse prices to model gap-through scenarios:
        - SELL stop (long exit): fills at candle low (worst case for longs)
        - BUY stop (short cover): fills at candle high (worst case for shorts)
        """
        if order_intent.stop_price is None:
            return ExecutionDecision.no_fill("stop price missing")

        if not math.isfinite(order_intent.stop_price):
            return ExecutionDecision.no_fill("stop price must be finite")

        if order_intent.stop_price <= ZERO_PRICE:
            return ExecutionDecision.no_fill("stop price must be positive")

        if not self._stop_triggered(order_intent, snapshot, order_intent.stop_price):
            return ExecutionDecision.no_fill("stop price not triggered")

        # Use adverse fill price for gap-through scenarios.
        fill_price = self._stop_adverse_fill_price(order_intent, snapshot)

        return ExecutionDecision(
            should_fill=True,
            fill_price=fill_price,
            filled_quantity=order_intent.quantity,
            liquidity="taker",
            reason="stop order triggered",
        )

    def _select_market_price(self, snapshot: MarketSnapshot) -> float:
        """Select the base market price from the snapshot."""
        if snapshot.last_price > ZERO_PRICE:
            return snapshot.last_price
        return snapshot.close

    def _limit_crossed(
        self,
        order_intent: OrderIntent,
        snapshot: MarketSnapshot,
        limit_price: float,
    ) -> bool:
        """Return True when the bar crosses the limit price."""
        if order_intent.side == OrderSide.BUY:
            return snapshot.low <= limit_price
        return snapshot.high >= limit_price

    def _stop_triggered(
        self,
        order_intent: OrderIntent,
        snapshot: MarketSnapshot,
        stop_price: float,
    ) -> bool:
        """Return True when the bar crosses the stop price."""
        if order_intent.side == OrderSide.BUY:
            return snapshot.high >= stop_price
        return snapshot.low <= stop_price

    def _stop_adverse_fill_price(
        self,
        order_intent: OrderIntent,
        snapshot: MarketSnapshot,
    ) -> float:
        """Return the adverse fill price for a triggered stop order.

        Models gap-through scenarios where market moves past the stop price:
        - SELL stop (long exit): fills at candle low (worst case for longs)
        - BUY stop (short cover): fills at candle high (worst case for shorts)
        """
        if order_intent.side == OrderSide.BUY:
            # Short cover - worst case is buying at the high
            return snapshot.high
        # Long exit - worst case is selling at the low
        return snapshot.low

    def _limit_fill_price(
        self,
        order_intent: OrderIntent,
        snapshot: MarketSnapshot,
        policy: FillPolicy,
    ) -> float:
        """Determine the fill price for a crossed limit order."""
        if order_intent.limit_price is None:
            return snapshot.close

        if not policy.allow_price_improvement:
            return order_intent.limit_price

        if order_intent.side == OrderSide.BUY:
            return min(order_intent.limit_price, snapshot.low)
        return max(order_intent.limit_price, snapshot.high)
