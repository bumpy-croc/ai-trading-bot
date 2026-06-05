"""Order-quantity precision helpers shared by the exchange + execution layers."""

from decimal import Decimal


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
    # .exponent is a negative int for any finite Decimal (str of a real step);
    # the isinstance guard defaults to 0 decimals for the impossible non-finite case.
    exponent = Decimal(str(step_size)).as_tuple().exponent
    decimals = max(0, -exponent) if isinstance(exponent, int) else 0
    return round(value, decimals)
