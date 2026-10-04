"""Tests for src/trading/precision (quantize/snap/conform/format helpers)."""

import math
import random
from decimal import Decimal

import pytest

from src.trading.precision import (
    conform_to_step,
    format_price,
    format_quantity,
    quantize_to_step,
    snap_to_step,
)

pytestmark = [pytest.mark.unit, pytest.mark.fast]


@pytest.mark.parametrize(
    "value,step,max_decimals",
    [
        (0.00030000000000000003, 0.0001, 4),  # the exact 51077 artifact
        (0.004000000000000001, 0.0001, 4),
        (1.2300000000001, 0.01, 2),
        (0.123456789, 0.001, 3),
        (5.0000000001, 1.0, 1),  # integer-ish step: value already clean
    ],
)
def test_quantize_strips_float_artifacts(value, step, max_decimals):
    """Result carries no more decimals than step_size implies (avoids Binance 51077)."""
    result = quantize_to_step(value, step)
    exponent = Decimal(str(result)).as_tuple().exponent
    assert isinstance(exponent, int)
    assert exponent >= -max_decimals


def test_quantize_preserves_value_within_step_precision():
    """The quantize is a value-preserving clamp, not a re-round of the lot multiple."""
    assert quantize_to_step(0.004000000000000001, 0.0001) == pytest.approx(0.004)
    assert quantize_to_step(1.23, 0.01) == pytest.approx(1.23)


def test_quantize_noop_for_nonpositive_step():
    """step_size <= 0 returns the value unchanged (no precision info to apply)."""
    assert quantize_to_step(1.2345, 0) == 1.2345
    assert quantize_to_step(1.2345, -1.0) == 1.2345


# --- snap_to_step / conform_to_step / format_price (#789) ---------------------------


@pytest.mark.parametrize(
    "value,step,expected",
    [
        (1648.82, 0.01, 1648.82),  # round(1648.82 / 0.01) * 0.01 == 1648.8200000000001
        (1648.8249, 0.01, 1648.82),
        (1648.826, 0.01, 1648.83),
        (0.0040000000000001, 0.0001, 0.004),  # 40 * 0.0001 == 0.004000000000000001
        (0.00030000000000000003, 0.0001, 0.0003),
        (123456.789, 0.001, 123456.789),
        (0.0000123456, 0.00000001, 0.00001235),  # tiny-price coin: repr is scientific
        (7.0, 1.0, 7.0),
        (7.4, 1.0, 7.0),
        (7.5, 1.0, 8.0),
        (1648.82, 0.05, 1648.8),  # step is not a power of ten
    ],
)
def test_snap_to_step_nearest_is_exact(value, step, expected):
    result = snap_to_step(value, step)
    assert result == expected
    assert Decimal(str(result)).as_tuple().exponent >= -_decimals(step)


@pytest.mark.parametrize(
    "value,step,expected",
    [
        (1648.829, 0.01, 1648.82),
        (0.0049999, 0.0001, 0.0049),
        (0.00399999999999999, 0.0001, 0.004),  # stored a hair low: not a whole step down
        (0.3, 0.1, 0.3),  # 0.3 / 0.1 == 2.9999999999999996 in float
        (1.1, 0.1, 1.1),
        (0.00005, 0.0001, 0.0),
    ],
)
def test_snap_to_step_floor_never_rounds_up(value, step, expected):
    assert snap_to_step(value, step, "floor") == expected


def test_snap_to_step_floor_never_exceeds_input():
    rng = random.Random(789)
    for _ in range(2000):
        step = rng.choice([0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 0.05])
        value = rng.uniform(step, 5000)
        snapped = snap_to_step(value, step, "floor")
        assert snapped <= value + 1e-9 * step
        assert value - snapped < step * (1 + 1e-6)


def test_snap_to_step_random_values_serialize_exactly_on_step():
    """Adversarial floats snapped to a step always serialize with no excess decimals."""
    rng = random.Random(1648)
    for _ in range(3000):
        step = rng.choice([1e-8, 1e-5, 1e-4, 1e-3, 0.01, 0.1, 1.0])
        value = rng.uniform(0, 100000) * rng.choice([1, 0.001, 1e-6])
        mode = rng.choice(["nearest", "floor"])
        text = format_quantity(snap_to_step(value, step, mode), step)
        assert "e" not in text.lower()
        decimals = _decimals(step)
        assert len(text.partition(".")[2]) == decimals
        units = Decimal(text) / Decimal(str(step))
        assert units == units.to_integral_value()


@pytest.mark.parametrize("step", [0, -0.01, float("nan"), float("inf")])
def test_snap_to_step_passes_through_unusable_step(step):
    assert snap_to_step(1.2345, step) == 1.2345
    assert snap_to_step(1.2345, step, "floor") == 1.2345


def test_snap_to_step_passes_through_non_finite_value():
    assert math.isnan(snap_to_step(float("nan"), 0.01))
    assert snap_to_step(float("inf"), 0.01) == float("inf")


def test_conform_strips_noise_from_on_step_value():
    assert conform_to_step(1648.8200000000001, 0.01) == (1648.82, True)
    assert conform_to_step(0.004000000000000001, 0.0001) == (0.004, True)


def test_conform_leaves_off_step_value_unchanged():
    """A genuinely off-step size is never silently moved -- the exchange should reject it."""
    assert conform_to_step(0.0049, 0.001) == (0.0049, False)
    assert conform_to_step(1648.826, 0.01) == (1648.826, False)


@pytest.mark.parametrize("step", [None, 0, -1.0, float("nan")])
def test_conform_passes_through_without_usable_step(step):
    assert conform_to_step(1.2345, step) == (1.2345, True)


def test_format_price_is_fixed_point_never_scientific():
    assert format_price(0.00001234, 0.00000001) == "0.00001234"
    assert format_price(1648.82, 0.01) == "1648.82"
    assert format_price(50000.0, 0.01) == "50000.00"
    assert "e" not in format_price(0.00009).lower()


def _decimals(step: float) -> int:
    return max(0, -int(Decimal(str(step)).as_tuple().exponent))
