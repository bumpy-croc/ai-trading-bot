"""Training-set-derived distribution statistics for confidence scoring.

Harness-wide rule (TARGET-REDESIGN tournament preregistration §2/§4): every
entrant's raw output converts to signal strength/confidence via statistics
of its OWN training-set target distribution (percentile/z-score), computed
once on the training split and never updated using eval-window data --
never a hardcoded constant. The `confidence = |return| x 12` class of
formula (``ml_signal_generator.py``'s ``CONFIDENCE_MULTIPLIER``) is
prohibited everywhere in this tournament.

This module is the literal FreqAI ``&*_std``/``&*_mean``-equivalent pattern
the Board directive names for entrant (d) (smoothed forward return): a
frozen percentile table built once at training time on the training-set
distribution of ``|predicted_smoothed_return|``, persisted into model
metadata, and consulted at inference time via linear interpolation.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FrozenDistribution:
    """A percentile table computed once on a training split.

    ``values`` must be sorted strictly ascending; ``percentiles[i]`` is the
    percentage (0-100) of the training-set distribution at or below
    ``values[i]``.
    """

    values: tuple[float, ...]
    percentiles: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.values) != len(self.percentiles):
            raise ValueError(
                f"values and percentiles must be the same length, got "
                f"{len(self.values)} and {len(self.percentiles)}"
            )
        if len(self.values) < 2:
            raise ValueError("distribution needs at least 2 points to interpolate")
        if list(self.values) != sorted(self.values):
            raise ValueError("values must be sorted ascending")

    @classmethod
    def from_samples(cls, samples: Sequence[float], num_points: int = 101) -> FrozenDistribution:
        """Build a frozen percentile table from raw training-set samples.

        Computes ``num_points`` evenly-spaced percentiles (default 101 ->
        0, 1, ..., 100) of ``samples``. Call this ONCE at training time and
        persist the result (``to_metadata()``) into model metadata -- never
        recompute it on eval-window data (that would leak eval-window
        information into the "training-set-only" confidence mapping the
        harness-wide rule requires).

        Args:
            samples: Raw training-set values (e.g. |predicted_smoothed_return|
                over the training split). Non-finite values are dropped.
            num_points: Number of percentile grid points (default 101).

        Returns:
            A FrozenDistribution ready for percentile_rank_confidence().

        Raises:
            ValueError: No finite samples were provided.
        """
        arr = np.asarray(samples, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            raise ValueError("samples must contain at least one finite value")

        pct_grid = np.linspace(0.0, 100.0, num_points)
        raw_values = np.percentile(arr, pct_grid)

        # np.percentile can produce a non-strictly-increasing sequence when
        # the underlying distribution has repeated/constant regions (e.g.
        # many identical predictions) -- dedupe while preserving
        # monotonicity so interpolation stays well-defined.
        dedup_values: list[float] = []
        dedup_pcts: list[float] = []
        last_value: float | None = None
        for value, pct in zip(raw_values, pct_grid, strict=True):
            value_f = float(value)
            if last_value is None or value_f > last_value:
                dedup_values.append(value_f)
                dedup_pcts.append(float(pct))
                last_value = value_f

        if len(dedup_values) < 2:
            # Fully degenerate (every sample identical) -- fabricate a
            # 2-point table so interpolation still works; any input maps
            # near the single observed value's percentile.
            base = float(arr[0])
            dedup_values = [base, base + 1e-12]
            dedup_pcts = [0.0, 100.0]

        return cls(values=tuple(dedup_values), percentiles=tuple(dedup_pcts))

    def to_metadata(self) -> dict[str, list[float]]:
        """Serialize to a JSON-safe dict for model metadata.json."""
        return {"values": list(self.values), "percentiles": list(self.percentiles)}

    @classmethod
    def from_metadata(cls, data: dict) -> FrozenDistribution:
        """Deserialize from a model metadata.json dict (see to_metadata())."""
        return cls(values=tuple(data["values"]), percentiles=tuple(data["percentiles"]))


def percentile_rank_confidence(value: float, distribution: FrozenDistribution) -> float:
    """Map a raw value to a [0, 1] confidence via a frozen percentile table.

    This is the harness-wide fix for the prohibited ``confidence = |x| * C``
    formula: instead of an arbitrary constant, confidence is where
    ``value`` falls within its OWN training-set distribution, expressed as
    a fraction (percentile / 100). Values outside the training-set range
    clamp to the nearest edge rather than extrapolating.

    Args:
        value: Raw magnitude to score (callers pass e.g.
            ``abs(predicted_return)`` for entrant (d)).
        distribution: A FrozenDistribution built at training time.

    Returns:
        Confidence in [0.0, 1.0]. NaN input returns 0.0 (matches the
        existing "invalid prediction -> zero confidence" convention used
        throughout the signal-generator layer).
    """
    if not math.isfinite(value):
        return 0.0

    values = distribution.values
    percentiles = distribution.percentiles

    if value <= values[0]:
        pct = percentiles[0]
    elif value >= values[-1]:
        pct = percentiles[-1]
    else:
        pct = float(np.interp(value, values, percentiles))

    return max(0.0, min(1.0, pct / 100.0))


__all__ = ["FrozenDistribution", "percentile_rank_confidence"]
