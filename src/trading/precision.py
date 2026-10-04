"""Exchange-bound numeric precision helpers shared by the exchange + execution layers.

Every quantity, price and stopPrice sent to an exchange must be produced by
`snap_to_step` / `quantize_to_step` and serialized by `format_quantity` /
`format_price` -- never by ``round(x / step) * step`` in float math.
"""

import math
from decimal import ROUND_FLOOR, ROUND_HALF_UP, Decimal
from typing import Literal

# A quantity that is mathematically an exact step multiple but stored a hair low
# (float noise) must not be floored a whole step down.
_FLOOR_NOISE_TOLERANCE = Decimal("1e-9")

# Max distance from a whole number of steps still treated as float noise.
_ON_STEP_ABS_TOLERANCE = Decimal("1e-6")
_ON_STEP_REL_TOLERANCE = Decimal("1e-12")


def _decimal_places(reference: float) -> int:
    """Number of decimal places implied by ``reference``'s shortest decimal string.

    Shared by `quantize_to_step` (rounds to this precision) and `format_quantity`
    (formats to this precision) so both derive "how many decimals" the same way.
    """
    # .exponent is a negative int for any finite Decimal (str of a real value);
    # the isinstance guard defaults to 0 decimals for the impossible non-finite case.
    exponent = Decimal(str(reference)).as_tuple().exponent
    return max(0, -exponent) if isinstance(exponent, int) else 0


def _is_usable_step(step_size: float | None) -> bool:
    return step_size is not None and step_size > 0 and math.isfinite(step_size)


def quantize_to_step(value: float, step_size: float) -> float:
    """Clamp ``value`` to the decimal precision implied by ``step_size``.

    After rounding a quantity to a LOT_SIZE step via float multiplication
    (``round(value / step) * step``), the result can carry float artifacts like
    ``0.004000000000000001`` that exceed an asset's max precision, so Binance
    rejects the order with code 51077. This strips them by rounding to the number
    of decimal places ``step_size`` implies. A no-op for already-clean values;
    returns ``value`` unchanged when ``step_size`` is non-positive.
    """
    if step_size <= 0:
        return value
    return round(value, _decimal_places(step_size))


def snap_to_step(
    value: float, step_size: float, mode: Literal["nearest", "floor"] = "nearest"
) -> float:
    """Snap ``value`` to a multiple of ``step_size`` using exact decimal arithmetic.

    The shared replacement for ``round(value / step) * step`` and
    ``math.floor(value / step + 1e-9) * step``. Division and multiplication run
    in `Decimal` on the shortest decimal strings, so the result is the exact
    decimal multiple: ``snap_to_step(1648.82, 0.01) == 1648.82`` where the float
    form gives ``1648.8200000000001``. ``mode="floor"`` never rounds up (with a
    tolerance of 1e-9 steps, or 1e-12 relative at large ratios, for exact multiples
    stored a hair low). Returns ``value``
    unchanged for a non-positive or non-finite step, or a non-finite value.
    """
    if not _is_usable_step(step_size) or not math.isfinite(value):
        return value
    decimal_step = Decimal(str(step_size))
    ratio = Decimal(str(value)) / decimal_step
    if mode == "floor":
        tolerance = max(_FLOOR_NOISE_TOLERANCE, abs(ratio) * _ON_STEP_REL_TOLERANCE)
        units = (ratio + tolerance).to_integral_value(rounding=ROUND_FLOOR)
    else:
        units = ratio.to_integral_value(rounding=ROUND_HALF_UP)
    return float(units * decimal_step)


def conform_to_step(value: float, step_size: float | None) -> tuple[float, bool]:
    """Boundary-guard helper: strip float noise from ``value`` if it is on-step.

    Returns ``(conformed, on_step)``. A value within float noise of a step
    multiple comes back as the exact multiple with ``on_step=True``. A value
    that is genuinely off-step comes back UNCHANGED with ``on_step=False``:
    silently moving an order's size is worse than letting the exchange reject
    it, so the caller decides how loudly to complain. With no usable step the
    value passes through as ``(value, True)``.
    """
    if step_size is None or not _is_usable_step(step_size) or not math.isfinite(value):
        return value, True
    ratio = Decimal(str(value)) / Decimal(str(step_size))
    distance = abs(ratio - ratio.to_integral_value(rounding=ROUND_HALF_UP))
    tolerance = max(_ON_STEP_ABS_TOLERANCE, abs(ratio) * _ON_STEP_REL_TOLERANCE)
    if distance > tolerance:
        return value, False
    return snap_to_step(value, step_size), True


def format_quantity(value: float, step_size: float | None = None) -> str:
    """Format an already-quantized exchange amount as a plain fixed-point string.

    python-binance urlencodes order params, and Python's default float-to-str
    conversion switches to scientific notation below 1e-4 (``str(0.00009) ==
    "9e-05"``), which Binance rejects with -1100 ("illegal characters"). This
    formats using the decimal count implied by ``step_size`` when given
    (matching the precision the value was already quantized to via
    `quantize_to_step`), or the value's own shortest decimal representation
    otherwise. Never guesses a fixed decimal count -- that would either
    truncate a fine-grained amount or reintroduce float noise (e.g.
    ``f"{0.1:.20f}" == "0.10000000000000000555"``).
    """
    decimals = _decimal_places(step_size) if step_size and step_size > 0 else _decimal_places(value)
    return f"{value:.{decimals}f}"


def format_price(value: float, tick_size: float | None = None) -> str:
    """Format an already-quantized price/stopPrice as a plain fixed-point string.

    Same contract as `format_quantity`, named for the PRICE_FILTER ``tickSize``
    side of the exchange filters.
    """
    return format_quantity(value, tick_size)
