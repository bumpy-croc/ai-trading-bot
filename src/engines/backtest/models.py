from __future__ import annotations

from dataclasses import dataclass

from src.engines.shared.models import BasePosition, BaseTrade

# Backtest Trade is identical to BaseTrade - just use it directly
Trade = BaseTrade


@dataclass
class ActiveTrade(BasePosition):
    """Represents an active trading position during backtest iteration.

    Extends BasePosition with the average cost of a position built from
    several fills. Exit information is tracked separately when converting to a
    completed Trade.

    ``entry_price`` stays the first fill and drives every trigger (partial-exit
    targets, trailing activation, breakeven). ``cost_basis_price`` is the
    quantity-weighted average cost and drives P&L, fees and mark-to-market.

    Note: component_notional removed - compute on-demand as:
        notional = current_size * balance

    Attributes:
        avg_cost_price: Average cost per unit once a scale-in has been blended
            in; ``None`` until then.
    """

    avg_cost_price: float | None = None

    @property
    def cost_basis_price(self) -> float:
        """Average cost per unit used for P&L; the entry price before any scale-in."""
        return self.entry_price if self.avg_cost_price is None else self.avg_cost_price
