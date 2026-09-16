"""Order-quantity precision helpers shared by the exchange + execution layers."""

from decimal import Decimal


def _decimal_places(reference: float) -> int:
    """Number of decimal places implied by ``reference``'s shortest decimal string.

    Shared by `quantize_to_step` (rounds to this precision) and `format_quantity`
    (formats to this precision) so both derive "how many decimals" the same way.
    """
    # .exponent is a negative int for any finite Decimal (str of a real value);
    # the isinstance guard defaults to 0 decimals for the impossible non-finite case.
    exponent = Decimal(str(reference)).as_tuple().exponent
    return max(0, -exponent) if isinstance(exponent, int) else 0


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
