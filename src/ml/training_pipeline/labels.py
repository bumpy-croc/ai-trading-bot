"""Alternative training-target label generation.

Implements the label definitions for the TARGET-REDESIGN tournament entrants
(see docs/research/experiments/2026-07-10_target-redesign-tournament-prereg.md
§2/§8): binary fixed-horizon direction classification (b), triple-barrier
ternary classification (c), and smoothed forward return (d). Meta-labeling
(a) lives in ``meta_labels.py`` since it depends on running a primary signal
generator forward, not just transforming the close-price series.

Every function here reads FORWARD bars by construction (the whole point of a
training target). Each returns a ``LabelResult`` with an explicit
``horizon_bars`` (the forward window each label consumes) and a
``valid_mask`` marking which rows have a *fully realized* label — the
trailing rows of any series cannot have a valid label because there is no
future data to compute them from, and must never be silently zero-filled
(that would be a lookahead-adjacent correctness bug: a "0"/"down" label that
actually means "unknown" is indistinguishable from a real one downstream).

``horizon_bars`` is exposed so the fold-runner's purge/embargo logic (per
the preregistration §3) knows exactly how many forward bars each entrant's
label reads, per entrant, without having to infer it from label code.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.engines.shared.barrier_touch import check_barrier_touch

_REQUIRED_TRIPLE_BARRIER_COLUMNS = ("close", "high", "low")


@dataclass(frozen=True)
class LabelResult:
    """A generated label series plus its forward-lookahead metadata.

    Attributes:
        values: Label array, same length as the input series. Entries where
            ``valid_mask`` is False are placeholders (0 for integer labels,
            NaN for float labels) and must be excluded before training —
            they do not mean "label is 0", they mean "label is unknown".
        valid_mask: True where the label is fully realized (had enough
            forward data to resolve).
        horizon_bars: Number of forward bars this label type consumes at
            each row (fixed for binary_direction/smoothed_return; the
            maximum possible for triple_barrier, whose actual per-row
            resolution can be earlier).
    """

    values: np.ndarray
    valid_mask: np.ndarray
    horizon_bars: int


def _validate_horizon(horizon: int) -> None:
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")


def binary_direction_labels(close: pd.Series, horizon: int = 1) -> LabelResult:
    """Binary fixed-horizon direction label: ``1 if close[t+H] > close[t] else 0``.

    Entrant (b) per the preregistration §2b. ``H=1`` isolates "does a
    classification loss/output type alone help" from "does horizon help"
    (same horizon as the incumbent next-bar regression target).

    Args:
        close: Close price series (chronological, no gaps assumed).
        horizon: Forward horizon in bars, H >= 1.

    Returns:
        LabelResult with int64 values in {0, 1}.
    """
    _validate_horizon(horizon)
    close_arr = close.to_numpy(dtype=np.float64)
    n = len(close_arr)

    values = np.zeros(n, dtype=np.int64)
    valid_mask = np.zeros(n, dtype=bool)

    n_valid = n - horizon
    if n_valid > 0:
        future = close_arr[horizon:]
        current = close_arr[:n_valid]
        values[:n_valid] = (future > current).astype(np.int64)
        valid_mask[:n_valid] = True

    return LabelResult(values=values, valid_mask=valid_mask, horizon_bars=horizon)


def smoothed_forward_return_labels(close: pd.Series, horizon: int) -> LabelResult:
    """Smoothed forward return: mean price ratio over the next N bars (FreqAI convention).

    Entrant (d) per the preregistration §2d (FreqAI convention):
    ``y = mean(close[t+1..t+N]) / close[t] - 1`` -- the mean of the next N
    closes expressed as a return relative to ``close[t]`` (not the mean of N
    separately-computed bar-over-bar returns, though the two are related).

    Args:
        close: Close price series.
        horizon: Forward window N in bars, N >= 1.

    Returns:
        LabelResult with float64 values; invalid rows are NaN.
    """
    _validate_horizon(horizon)
    close_arr = close.to_numpy(dtype=np.float64)
    n = len(close_arr)

    values = np.full(n, np.nan, dtype=np.float64)
    valid_mask = np.zeros(n, dtype=bool)

    n_valid = n - horizon
    if n_valid > 0:
        # sum(close[t+1 .. t+horizon]) via a prefix-sum, so the whole
        # computation is vectorized rather than looping per row.
        prefix_sum = np.concatenate(([0.0], np.cumsum(close_arr)))
        idx = np.arange(n_valid)
        forward_sum = prefix_sum[idx + 1 + horizon] - prefix_sum[idx + 1]
        mean_future_close = forward_sum / horizon

        # close_arr[idx] (the entry price, close[t]) is the divisor -- a
        # zero/negative/non-finite entry price must not silently produce
        # inf/nan written into the label array as if it were a valid,
        # fully-realized row. Guard BEFORE dividing (not divide-then-
        # discard) so a corrupted entry price never raises a numpy
        # RuntimeWarning either. Only the divisor is checked here; a
        # legitimate (if unusual) zero/negative FUTURE close is not an
        # error -- that is real market data, not a corrupted read.
        divisor = close_arr[idx]
        safe_divisor = np.isfinite(divisor) & (divisor > 0)

        computed = np.full(n_valid, np.nan, dtype=np.float64)
        computed[safe_divisor] = mean_future_close[safe_divisor] / divisor[safe_divisor] - 1.0
        values[idx] = computed
        valid_mask[idx] = safe_divisor

    return LabelResult(values=values, valid_mask=valid_mask, horizon_bars=horizon)


def triple_barrier_labels(
    df: pd.DataFrame,
    take_profit_pct: float,
    stop_loss_pct: float,
    max_holding_bars: int,
) -> LabelResult:
    """Triple-barrier ternary label using intrabar high/low fill logic.

    Entrant (c) per the preregistration §2c: for each bar, simulate forward
    with an upper barrier ``+take_profit_pct`` and lower barrier
    ``-stop_loss_pct`` (both anchored to ``close[t]``, applied symmetrically
    — the label describes the price path, not a directional bet) and a
    vertical time barrier at ``max_holding_bars``. Label = {+1, -1, 0} for
    whichever barrier is hit first.

    Reuses ``src.engines.shared.barrier_touch.check_barrier_touch`` — the
    same intrabar high/low comparison ``ExitHandler`` uses for live/backtest
    stop-loss/take-profit detection — per the preregistration's explicit
    instruction not to hand-roll this a second time (§8). Same-bar
    ambiguity (both barriers touched in one candle) resolves stop-loss-first,
    matching that money-path convention.

    Args:
        df: DataFrame with "close", "high", "low" columns (chronological).
        take_profit_pct: Upper barrier distance as a fraction (e.g. 0.04).
        stop_loss_pct: Lower barrier distance as a fraction (e.g. 0.05).
        max_holding_bars: Vertical (time) barrier in bars.

    Returns:
        LabelResult with int64 values in {-1, 0, 1}. ``horizon_bars`` is
        ``max_holding_bars`` (the maximum a row could consume; individual
        rows may resolve earlier via a barrier touch).

    Raises:
        ValueError: Missing required columns, or non-positive parameters.
    """
    missing = [c for c in _REQUIRED_TRIPLE_BARRIER_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"triple_barrier_labels requires columns {_REQUIRED_TRIPLE_BARRIER_COLUMNS}, "
            f"missing: {missing}"
        )
    if take_profit_pct <= 0:
        raise ValueError(f"take_profit_pct must be > 0, got {take_profit_pct}")
    if stop_loss_pct <= 0:
        raise ValueError(f"stop_loss_pct must be > 0, got {stop_loss_pct}")
    if max_holding_bars < 1:
        raise ValueError(f"max_holding_bars must be >= 1, got {max_holding_bars}")

    close = df["close"].to_numpy(dtype=np.float64)
    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    n = len(close)

    values = np.zeros(n, dtype=np.int64)
    valid_mask = np.zeros(n, dtype=bool)

    for t in range(n):
        entry_price = close[t]
        stop_loss_price = entry_price * (1.0 - stop_loss_pct)
        take_profit_price = entry_price * (1.0 + take_profit_pct)
        max_bar = min(t + max_holding_bars, n - 1)

        resolved = False
        for bar in range(t + 1, max_bar + 1):
            touch = check_barrier_touch(
                side="long",
                candle_high=high[bar],
                candle_low=low[bar],
                stop_loss_price=stop_loss_price,
                take_profit_price=take_profit_price,
            )
            if touch.hit_stop_loss:
                values[t] = -1
                valid_mask[t] = True
                resolved = True
                break
            if touch.hit_take_profit:
                values[t] = 1
                valid_mask[t] = True
                resolved = True
                break

        if not resolved:
            reached_full_vertical_barrier = max_bar == t + max_holding_bars
            if reached_full_vertical_barrier:
                values[t] = 0
                valid_mask[t] = True
            # else: truncated at series end without resolution -- leave
            # invalid (valid_mask[t] stays False). This is NOT the same as
            # "0" -- it means we don't know what would have happened.

    return LabelResult(values=values, valid_mask=valid_mask, horizon_bars=max_holding_bars)


def target_horizon_bars(
    target_type: str,
    *,
    horizon: int | None = None,
    max_holding_bars: int | None = None,
) -> int:
    """Return the forward horizon (in bars) a target_type's label consumes.

    Lets the fold-runner's purge/embargo logic (preregistration §3) look up
    an entrant's lookahead window without materializing labels.

    Args:
        target_type: One of "binary_direction", "smoothed_return",
            "triple_barrier".
        horizon: Required for "binary_direction"/"smoothed_return".
        max_holding_bars: Required for "triple_barrier".

    Returns:
        The forward horizon in bars.

    Raises:
        ValueError: Unknown target_type, or the required param is missing.
    """
    if target_type in ("binary_direction", "smoothed_return"):
        if horizon is None:
            raise ValueError(f"target_type={target_type!r} requires 'horizon'")
        return horizon
    if target_type == "triple_barrier":
        if max_holding_bars is None:
            raise ValueError(f"target_type={target_type!r} requires 'max_holding_bars'")
        return max_holding_bars
    raise ValueError(f"Unknown target_type: {target_type!r}")


__all__ = [
    "LabelResult",
    "binary_direction_labels",
    "smoothed_forward_return_labels",
    "target_horizon_bars",
    "triple_barrier_labels",
]
