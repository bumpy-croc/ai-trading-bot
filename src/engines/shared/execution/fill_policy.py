"""Execution fill policies for simulated execution models."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

logger = logging.getLogger(__name__)

FidelityLevel = Literal["ohlc", "quote", "order_book"]

DEFAULT_FILL_POLICY_NAME = "ohlc_conservative"
DEFAULT_FIDELITY: FidelityLevel = "ohlc"


@dataclass(frozen=True)
class FillPolicy:
    """Describes the execution model fidelity and price improvement rules.

    The default policy is "ohlc_conservative": limit orders fill at the
    limit price when a bar crosses the level, price improvement is not
    allowed without quote data, and stop orders are modeled as stop-market.
    """

    name: str
    fidelity: FidelityLevel
    allow_price_improvement: bool = False

    def uses_quotes(self) -> bool:
        """Return True when the policy expects quote or order-book data."""
        return self.fidelity in ("quote", "order_book")


DEFAULT_FILL_POLICY = FillPolicy(
    name=DEFAULT_FILL_POLICY_NAME,
    fidelity=DEFAULT_FIDELITY,
    allow_price_improvement=False,
)

FILL_POLICIES: dict[str, FillPolicy] = {
    DEFAULT_FILL_POLICY_NAME: DEFAULT_FILL_POLICY,
}


def default_fill_policy() -> FillPolicy:
    """Return the default fill policy instance."""
    return DEFAULT_FILL_POLICY


def resolve_fill_policy(name: str | None) -> FillPolicy:
    """Resolve a fill policy by name, defaulting when unknown.

    Args:
        name: Policy name to resolve (case-insensitive).

    Returns:
        The matching FillPolicy, or DEFAULT_FILL_POLICY if not found.
    """
    if not name:
        return DEFAULT_FILL_POLICY

    normalized = name.strip().lower()
    if normalized not in FILL_POLICIES:
        logger.warning(
            "Unknown fill policy '%s'; using default '%s'. " "Available policies: %s",
            name,
            DEFAULT_FILL_POLICY_NAME,
            list(FILL_POLICIES.keys()),
        )
        return DEFAULT_FILL_POLICY

    return FILL_POLICIES[normalized]
