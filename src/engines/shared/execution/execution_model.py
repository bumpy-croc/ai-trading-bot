"""Execution model coordinator for selecting fill models."""

from __future__ import annotations

import logging
from typing import Protocol

from src.engines.shared.execution.execution_decision import ExecutionDecision
from src.engines.shared.execution.fill_policy import FillPolicy
from src.engines.shared.execution.market_snapshot import MarketSnapshot
from src.engines.shared.execution.ohlc_fill_model import OhlcFillModel
from src.engines.shared.execution.order_intent import OrderIntent

logger = logging.getLogger(__name__)

SUPPORTED_FIDELITIES = ("ohlc",)


class FillModelProtocol(Protocol):
    """Protocol for fill models that determine order execution prices.

    Implementations decide whether an order should fill and at what price
    based on the order intent, market snapshot, and policy configuration.
    """

    def decide_fill(
        self,
        order_intent: OrderIntent,
        snapshot: MarketSnapshot,
        policy: FillPolicy,
    ) -> ExecutionDecision:
        """Decide whether an order fills and at what price.

        Args:
            order_intent: The order to evaluate.
            snapshot: Current market data.
            policy: Fill policy configuration.

        Returns:
            ExecutionDecision with fill determination.
        """
        ...


class ExecutionModel:
    """Coordinates fill decisions based on the configured policy.

    This class selects the appropriate fill model based on the policy's
    fidelity level and delegates fill decisions to that model.
    """

    def __init__(
        self,
        policy: FillPolicy,
        fill_model: FillModelProtocol | None = None,
    ) -> None:
        """Initialize the execution model with a fill policy.

        Args:
            policy: Fill policy configuration.
            fill_model: Optional custom fill model (defaults to OhlcFillModel).
        """
        self.policy = policy
        self._fill_model: FillModelProtocol = fill_model or OhlcFillModel()
        self._warned_unsupported_fidelity = False

    def decide_fill(
        self,
        order_intent: OrderIntent,
        snapshot: MarketSnapshot,
    ) -> ExecutionDecision:
        """Return the fill decision for an order intent.

        Args:
            order_intent: The order to evaluate.
            snapshot: Current market data.

        Returns:
            ExecutionDecision with fill determination.
        """
        if (
            self.policy.fidelity not in SUPPORTED_FIDELITIES
            and not self._warned_unsupported_fidelity
        ):
            logger.warning(
                "Unsupported fidelity '%s' requested; using OHLC fill model. "
                "Supported fidelities: %s",
                self.policy.fidelity,
                SUPPORTED_FIDELITIES,
            )
            self._warned_unsupported_fidelity = True

        return self._fill_model.decide_fill(order_intent, snapshot, self.policy)
