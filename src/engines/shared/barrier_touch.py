"""Pure OHLC-based stop-loss/take-profit barrier-touch detection.

Single source of truth for "did this bar's high/low cross a stop-loss or
take-profit level" — extracted from
``ExitHandler.check_exit_conditions`` so both live/backtest position exits
and the training pipeline's triple-barrier label simulator
(``src/ml/training_pipeline/labels.py``) agree bar-for-bar on what counts as
a barrier touch. Reusing this function (instead of hand-rolling the
comparison a second time) is the explicit instruction in the TARGET-REDESIGN
tournament preregistration (§2c, §8): triple-barrier labels must reuse
``src/engines/shared/`` fill logic, not reimplement it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

_VALID_SIDES = frozenset({"long", "short"})


@dataclass(frozen=True)
class BarrierTouchResult:
    """Whether a bar's high/low crossed the stop-loss/take-profit levels.

    Attributes:
        hit_stop_loss: True if the stop-loss level was crossed this bar.
        hit_take_profit: True if the take-profit level was crossed this bar.
        stop_loss_exit_price: Worst-case (gap-through) fill price if
            ``hit_stop_loss`` — candle low for longs, candle high for shorts.
            Defined even when ``hit_stop_loss`` is False (equals the
            relevant candle extreme), so callers never branch on None.
        take_profit_exit_price: Fill price if ``hit_take_profit`` — exactly
            the take-profit level (no gap-through assumption), matching
            ExitHandler's existing convention. Defined even when
            ``hit_take_profit`` is False.

    Both flags are reported independently; resolving same-bar ambiguity
    (both barriers touched in one candle) is a caller concern. The existing
    money-path convention (``ExitHandler.check_exit_conditions``) resolves
    it stop-loss-first (conservative); label-generation callers should do
    the same for consistency with what the exam harness would have actually
    executed.
    """

    hit_stop_loss: bool
    hit_take_profit: bool
    stop_loss_exit_price: float
    take_profit_exit_price: float


def check_barrier_touch(
    side: str,
    candle_high: float,
    candle_low: float,
    stop_loss_price: float | None,
    take_profit_price: float | None,
) -> BarrierTouchResult:
    """Check whether a bar's high/low crossed the stop-loss/take-profit levels.

    Args:
        side: ``"long"`` or ``"short"``.
        candle_high: Bar high (caller is responsible for any NaN fallback —
            this function assumes finite inputs, matching ExitHandler's
            pre-validated candle_high/candle_low).
        candle_low: Bar low.
        stop_loss_price: Stop-loss level, or None if no stop-loss is set.
        take_profit_price: Take-profit level, or None if no take-profit is set.

    Returns:
        BarrierTouchResult with both hit flags and exit prices.

    Raises:
        ValueError: If ``side`` is not "long" or "short".
    """
    if side not in _VALID_SIDES:
        raise ValueError(f"side must be 'long' or 'short', got {side!r}")

    is_long = side == "long"

    hit_stop_loss = False
    stop_loss_exit_price = candle_low if is_long else candle_high
    if stop_loss_price is not None and math.isfinite(stop_loss_price):
        if is_long:
            hit_stop_loss = candle_low <= stop_loss_price
        else:
            hit_stop_loss = candle_high >= stop_loss_price

    hit_take_profit = False
    take_profit_exit_price = take_profit_price if take_profit_price is not None else candle_high
    if take_profit_price is not None and math.isfinite(take_profit_price):
        if is_long:
            hit_take_profit = candle_high >= take_profit_price
        else:
            hit_take_profit = candle_low <= take_profit_price
        take_profit_exit_price = take_profit_price

    return BarrierTouchResult(
        hit_stop_loss=hit_stop_loss,
        hit_take_profit=hit_take_profit,
        stop_loss_exit_price=stop_loss_exit_price,
        take_profit_exit_price=take_profit_exit_price,
    )


__all__ = ["BarrierTouchResult", "check_barrier_touch"]
