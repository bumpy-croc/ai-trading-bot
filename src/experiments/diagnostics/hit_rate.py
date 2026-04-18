"""Direction-conditional accuracy record for a single forward horizon."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class HitRate:
    """Direction-conditional accuracy at a specific forward horizon.

    ``buy_accuracy`` is ``P(forward_return > 0 | BUY)`` over ``buy_samples``
    bars; ``sell_accuracy`` is ``P(forward_return < 0 | SELL)``. Both are
    floats in [0, 1]; both are zero when their sample count is zero.
    """

    horizon: int
    buy_samples: int
    buy_accuracy: float
    sell_samples: int
    sell_accuracy: float

    def to_dict(self) -> dict[str, Any]:
        """Return a plain-dict serialization suitable for JSON output."""
        return {
            "horizon": self.horizon,
            "buy_samples": self.buy_samples,
            "buy_accuracy": self.buy_accuracy,
            "sell_samples": self.sell_samples,
            "sell_accuracy": self.sell_accuracy,
        }


__all__ = ["HitRate"]
