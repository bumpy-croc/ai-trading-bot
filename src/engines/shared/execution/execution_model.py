"""Execution model coordinator for selecting fill models."""

from __future__ import annotations

from src.engines.shared.execution.execution_decision import ExecutionDecision
from src.engines.shared.execution.fill_policy import FillPolicy
from src.engines.shared.execution.market_snapshot import MarketSnapshot
from src.engines.shared.execution.ohlc_fill_model import OhlcFillModel
from src.engines.shared.execution.order_intent import OrderIntent

SUPPORTED_FIDELITIES = ("ohlc",)


class ExecutionModel:
    """Coordinates fill decisions based on the configured policy."""

    def __init__(self, policy: FillPolicy) -> None:
        """Initialize the execution model with a fill policy."""
        self.policy = policy
        self._ohlc_model = OhlcFillModel()

    def decide_fill(
        self,
        order_intent: OrderIntent,
        snapshot: MarketSnapshot,
    ) -> ExecutionDecision:
        """Return the fill decision for an order intent."""
        decision = self._select_model().decide_fill(order_intent, snapshot, self.policy)
        if self.policy.fidelity in SUPPORTED_FIDELITIES:
            return decision

        return ExecutionDecision(
            should_fill=decision.should_fill,
            fill_price=decision.fill_price,
            filled_quantity=decision.filled_quantity,
            liquidity=decision.liquidity,
            reason=f"{decision.reason}; fallback to ohlc model",
        )

    def _select_model(self) -> OhlcFillModel:
        """Select the fill model based on the current policy."""
        return self._ohlc_model
