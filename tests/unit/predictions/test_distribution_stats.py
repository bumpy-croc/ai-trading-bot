"""Unit tests for training-set-derived distribution statistics.

Harness-wide rule (TARGET-REDESIGN tournament preregistration §2d/§4): raw
model output converts to confidence via statistics of its OWN training-set
target distribution (percentile/z-score), computed once on the training
split and never updated using eval-window data -- never a hardcoded
constant. This is the direct fix for the prohibited `confidence = |x| * 12`
formula (src/strategies/components/ml_signal_generator.py's
CONFIDENCE_MULTIPLIER).
"""

import numpy as np
import pytest

from src.prediction.distribution_stats import (
    FrozenDistribution,
    percentile_rank_confidence,
)


class TestFrozenDistributionFromSamples:
    def test_builds_monotonic_table(self):
        samples = np.linspace(0.0, 1.0, 1000)
        dist = FrozenDistribution.from_samples(samples)
        assert list(dist.values) == sorted(dist.values)
        assert list(dist.percentiles) == sorted(dist.percentiles)

    def test_median_sample_lands_near_50th_percentile(self):
        samples = np.linspace(0.0, 100.0, 10_000)
        dist = FrozenDistribution.from_samples(samples)
        confidence = percentile_rank_confidence(50.0, dist)
        assert confidence == pytest.approx(0.5, abs=0.02)

    def test_rejects_empty_samples(self):
        with pytest.raises(ValueError):
            FrozenDistribution.from_samples([])

    def test_rejects_all_nan_samples(self):
        with pytest.raises(ValueError):
            FrozenDistribution.from_samples([float("nan"), float("nan")])

    def test_constant_distribution_does_not_crash(self):
        # A degenerate distribution (every training sample identical) must
        # still produce a usable table, not divide-by-zero or crash.
        dist = FrozenDistribution.from_samples([5.0] * 100)
        confidence = percentile_rank_confidence(5.0, dist)
        assert 0.0 <= confidence <= 1.0

    def test_metadata_round_trip(self):
        samples = np.linspace(0.0, 1.0, 500)
        dist = FrozenDistribution.from_samples(samples)
        restored = FrozenDistribution.from_metadata(dist.to_metadata())
        assert restored.values == dist.values
        assert restored.percentiles == dist.percentiles


class TestFrozenDistributionValidation:
    def test_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError, match="length"):
            FrozenDistribution(values=(1.0, 2.0), percentiles=(0.0,))

    def test_rejects_too_few_points(self):
        with pytest.raises(ValueError):
            FrozenDistribution(values=(1.0,), percentiles=(0.0,))

    def test_rejects_unsorted_values(self):
        with pytest.raises(ValueError, match="sorted"):
            FrozenDistribution(values=(2.0, 1.0), percentiles=(0.0, 100.0))


class TestPercentileRankConfidence:
    @pytest.fixture
    def dist(self):
        # Hand-computable: values 0..100 map directly to percentile 0..100.
        return FrozenDistribution(
            values=tuple(float(v) for v in range(0, 101, 10)),
            percentiles=tuple(float(v) for v in range(0, 101, 10)),
        )

    def test_exact_grid_point(self, dist):
        assert percentile_rank_confidence(50.0, dist) == pytest.approx(0.5)

    def test_interpolates_between_grid_points(self, dist):
        # Halfway between values=40 (pct 40) and values=50 (pct 50) -> pct 45.
        assert percentile_rank_confidence(45.0, dist) == pytest.approx(0.45)

    def test_below_minimum_clamps_to_lowest_percentile(self, dist):
        assert percentile_rank_confidence(-999.0, dist) == pytest.approx(0.0)

    def test_above_maximum_clamps_to_highest_percentile(self, dist):
        assert percentile_rank_confidence(999.0, dist) == pytest.approx(1.0)

    def test_nan_value_returns_zero_confidence(self, dist):
        assert percentile_rank_confidence(float("nan"), dist) == 0.0

    def test_inf_value_returns_zero_confidence(self, dist):
        # An infinite model output is a numerical bug, not a legitimate
        # extreme prediction -- treated the same as NaN (0.0 confidence,
        # degrade to HOLD upstream) rather than clamped to max confidence.
        assert percentile_rank_confidence(float("inf"), dist) == 0.0

    def test_result_always_in_unit_interval(self, dist):
        for value in [-1e9, -1.0, 0.0, 25.0, 50.0, 75.0, 100.0, 1e9]:
            confidence = percentile_rank_confidence(value, dist)
            assert 0.0 <= confidence <= 1.0

    def test_no_hardcoded_multiplier_used(self, dist):
        """Doubling the input must NOT double the confidence -- this is the
        entire point of using a frozen percentile table instead of the
        prohibited `confidence = |x| * constant` formula.

        A linear-multiplier formula always gives confidence(2x) == 2 *
        confidence(x); the percentile-rank formula need not (and on this
        fixture's linear grid it happens to coincide -- so assert the
        property on a genuinely nonlinear distribution instead)."""
        skewed = FrozenDistribution(
            values=(0.0, 1.0, 2.0, 100.0),
            percentiles=(0.0, 90.0, 95.0, 100.0),
        )
        c_small = percentile_rank_confidence(1.0, skewed)
        c_double = percentile_rank_confidence(2.0, skewed)
        assert c_double != pytest.approx(2 * c_small)
