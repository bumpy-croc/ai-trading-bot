"""Tests for src/trading/precision.quantize_to_step."""

from decimal import Decimal

import pytest

from src.trading.precision import quantize_to_step

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
