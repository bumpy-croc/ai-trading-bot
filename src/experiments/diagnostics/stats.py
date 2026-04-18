"""Summary statistics dataclass for a numeric series."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class DistributionStats:
    """Summary statistics for a numeric series.

    Produced by :meth:`from_series` over the ``predicted_return`` /
    ``confidence`` streams gathered during a signal-quality diagnostic walk.
    The ``positive_fraction`` field is especially useful for catching a
    model that always returns a fallback sentinel — a healthy ML model's
    predicted-return distribution straddles zero (fraction roughly 0.5);
    a broken one is 0.0 or 1.0.
    """

    n: int
    mean: float
    std: float
    min: float
    max: float
    positive_fraction: float

    @classmethod
    def from_series(cls, values: list[float]) -> DistributionStats:
        """Summarize ``values`` as count / mean / sample-std / min / max / pos-frac."""
        if not values:
            return cls(n=0, mean=0.0, std=0.0, min=0.0, max=0.0, positive_fraction=0.0)
        n = len(values)
        mean = sum(values) / n
        # Guard n=1 (std is 0 by definition). Use sample std for n≥2.
        if n == 1:
            std = 0.0
        else:
            var = sum((v - mean) ** 2 for v in values) / (n - 1)
            std = math.sqrt(var)
        pos = sum(1 for v in values if v > 0) / n
        return cls(
            n=n,
            mean=float(mean),
            std=float(std),
            min=float(min(values)),
            max=float(max(values)),
            positive_fraction=float(pos),
        )

    def to_dict(self) -> dict[str, float | int]:
        """Return a plain-dict serialization suitable for JSON output."""
        return {
            "n": self.n,
            "mean": self.mean,
            "std": self.std,
            "min": self.min,
            "max": self.max,
            "positive_fraction": self.positive_fraction,
        }


__all__ = ["DistributionStats"]
