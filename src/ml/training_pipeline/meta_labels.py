"""Meta-labeling entrant (a) scaffolding.

Preregistration §2a: label = 1 if simulating the primary signal's fired
trade through the exam-harness exit geometry closes net-profitable after
fees, else 0. Feature set: 48-bar trailing realized volatility of
bar-over-bar returns, rolling hit-rate of the primary signal over its
trailing 20 fired signals, session/time-of-day bucket (cyclical encoding),
``EnhancedRegimeDetector``'s regime label (reused, not reimplemented), and
the primary model's own ``predicted_return`` magnitude as ONE feature among
the above -- never the sole feature (the richer-feature-set requirement is
the whole point of entrant (a): #912 already falsified the primary signal's
own magnitude as a standalone confidence channel).

Separate from ``labels.py`` because meta-labeling needs the primary
signal's actual fire points (a live ``SignalGenerator`` run forward over the
corpus), not just a transform of the close-price series.

Causality note on ``rolling_hit_rate_20`` specifically: a prior fire's
profitability label is only "known" once that fire's own trade has
resolved (up to ``max_holding_bars`` -- 336 by default, 14 days -- after it
fired), not at the moment it fired. With dense fires this resolution can
land well after a LATER fire's own index, so the hit-rate feature only
includes prior fires whose resolution bar (``TradeResolution.exit_index``)
is at-or-before the current fire's index -- never a prior fire that hasn't
resolved yet as of "now."
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd

from src.engines.shared.barrier_touch import check_barrier_touch
from src.engines.shared.cost_calculator import CostCalculator
from src.performance.metrics import Side, pnl_percent
from src.regime.enhanced_detector import EnhancedRegimeDetector
from src.strategies.components.signal_generator import SignalDirection

if TYPE_CHECKING:
    from src.strategies.components.signal_generator import SignalGenerator

# Feature-set constants, per preregistration §2a's exact spec.
REALIZED_VOL_WINDOW_BARS = 48
HIT_RATE_LOOKBACK_FIRES = 20
_HOURS_PER_DAY = 24.0

_VALID_DIRECTIONS = frozenset({1, -1})


@dataclass(frozen=True)
class PrimarySignalRecord:
    """A single non-HOLD fire from the primary signal generator.

    Attributes:
        index: Bar index in the DataFrame the signal fired at.
        direction: 1 for BUY (long), -1 for SELL (short). Never 0 -- only
            fired (non-HOLD) bars are recorded.
        predicted_return: The primary signal's own predicted_return at the
            fire point (from Signal.metadata["predicted_return"]), used as
            ONE meta-label feature among several, per the preregistration's
            explicit "never the sole feature" requirement.
    """

    index: int
    direction: int
    predicted_return: float


def run_primary_signal_forward(
    signal_generator: SignalGenerator,
    df: pd.DataFrame,
    start_index: int | None = None,
) -> list[PrimarySignalRecord]:
    """Run a primary SignalGenerator forward over df, recording every fire.

    This is preregistration §2a/§8 step (i): "running the CURRENT incumbent
    signal generator forward over the training corpus to find its fire
    points." Reuses the SignalGenerator interface directly (generate_signal)
    rather than reimplementing prediction logic.

    Args:
        signal_generator: The primary (incumbent) SignalGenerator instance.
        df: OHLCV DataFrame to run forward over.
        start_index: First index to evaluate. Defaults to
            ``signal_generator.warmup_period``.

    Returns:
        A list of PrimarySignalRecord, one per non-HOLD bar, in order.
    """
    start = start_index if start_index is not None else signal_generator.warmup_period
    records: list[PrimarySignalRecord] = []
    for index in range(start, len(df)):
        signal = signal_generator.generate_signal(df, index)
        if signal.direction == SignalDirection.HOLD:
            continue
        direction = 1 if signal.direction == SignalDirection.BUY else -1
        predicted_return = float(signal.metadata.get("predicted_return", 0.0))
        records.append(
            PrimarySignalRecord(index=index, direction=direction, predicted_return=predicted_return)
        )
    return records


@dataclass(frozen=True)
class TradeResolution:
    """The outcome of a simulated fired trade, AND when it became knowable.

    Attributes:
        profitable: True if net-profitable after fees, False if not.
        exit_index: The bar index where the outcome was determined --
            either the bar a barrier was touched on, or the vertical/time
            exit bar. Callers building features from PRIOR fires' outcomes
            (e.g. a rolling hit-rate) must not treat a fire's label as
            "known" before this bar -- doing so leaks the fire's own
            forward price path into an earlier bar's feature. See
            ``build_meta_label_features``'s ``rolling_hit_rate_20``.
    """

    profitable: bool
    exit_index: int


def resolve_fired_trade(
    df: pd.DataFrame,
    fire_index: int,
    direction: int,
    take_profit_pct: float,
    stop_loss_pct: float,
    max_holding_bars: int,
    cost_calculator: CostCalculator | None = None,
) -> TradeResolution | None:
    """Simulate a fired trade through the exam harness's exit geometry.

    Reuses ``check_barrier_touch`` (the same intrabar high/low logic
    ``ExitHandler`` uses) directionally: barriers are placed above/below
    entry according to the fired side (long: TP above/SL below; short: TP
    below/SL above), scanning forward until the first barrier touch or the
    vertical (time) barrier at ``max_holding_bars``. Net profitability
    subtracts a round-trip fee estimate (``2 * cost_calculator.fee_rate``,
    slippage not modeled -- a documented approximation for a *label*, not a
    real execution).

    Args:
        df: DataFrame with "close", "high", "low" columns.
        fire_index: Bar index the primary signal fired at (the entry bar).
        direction: 1 for a fired long, -1 for a fired short.
        take_profit_pct: Take-profit distance as a fraction.
        stop_loss_pct: Stop-loss distance as a fraction.
        max_holding_bars: Vertical (time) barrier in bars.
        cost_calculator: Fee model; defaults to CostCalculator()'s defaults.

    Returns:
        A TradeResolution (profitable + exit_index), or None if the trade
        is unresolved (truncated at series end with no forward data) --
        callers must drop unresolved fire points, never treat None as an
        unprofitable outcome.

    Raises:
        ValueError: direction is not 1 or -1.
    """
    if direction not in _VALID_DIRECTIONS:
        raise ValueError(f"direction must be 1 (long) or -1 (short), got {direction}")

    cost_calculator = cost_calculator or CostCalculator()
    side = "long" if direction == 1 else "short"

    close = df["close"].to_numpy(dtype=np.float64)
    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    n = len(close)

    entry_price = close[fire_index]
    # CODE.md#L133: validate entry_price > 0 before any P&L or stop-loss
    # calculation -- a zero/negative/non-finite close at fire_index (bad
    # tick, gap-fill, corrupted data) must raise loudly here, not silently
    # produce NaN barrier levels and mislabel the row downstream.
    if not math.isfinite(entry_price) or entry_price <= 0:
        raise ValueError(
            f"entry_price must be positive and finite, got {entry_price} at "
            f"fire_index={fire_index}"
        )
    if direction == 1:
        stop_loss_price = entry_price * (1.0 - stop_loss_pct)
        take_profit_price = entry_price * (1.0 + take_profit_pct)
    else:
        stop_loss_price = entry_price * (1.0 + stop_loss_pct)
        take_profit_price = entry_price * (1.0 - take_profit_pct)

    max_bar = min(fire_index + max_holding_bars, n - 1)
    exit_price: float | None = None
    exit_index: int | None = None
    for bar in range(fire_index + 1, max_bar + 1):
        touch = check_barrier_touch(
            side=side,
            candle_high=high[bar],
            candle_low=low[bar],
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price,
        )
        if touch.hit_stop_loss:
            exit_price = touch.stop_loss_exit_price
            exit_index = bar
            break
        if touch.hit_take_profit:
            exit_price = touch.take_profit_exit_price
            exit_index = bar
            break

    if exit_price is None:
        reached_full_vertical_barrier = max_bar == fire_index + max_holding_bars
        if not reached_full_vertical_barrier:
            return None  # truncated at series end -- unresolved
        exit_price = close[max_bar]
        exit_index = max_bar

    # Reuse the shared P&L primitive (CODE.md#L331: "never duplicate
    # financial logic") instead of hand-rolling the calculation -- also
    # validates exit_price is positive/finite, since it derives from OHLC
    # data (barrier touch price / vertical exit close) just like entry_price.
    side_enum = Side.LONG if direction == 1 else Side.SHORT
    raw_pct_return = pnl_percent(entry_price, exit_price, side_enum, fraction=1.0)
    round_trip_cost_pct = 2.0 * cost_calculator.fee_rate
    profitable = bool((raw_pct_return - round_trip_cost_pct) > 0.0)
    # cast: exit_index is always set on every path that reaches here (either
    # inside the touch loop before `break`, or the vertical-exit branch above).
    return TradeResolution(profitable=profitable, exit_index=cast(int, exit_index))


def simulate_fired_trade_profitability(
    df: pd.DataFrame,
    fire_index: int,
    direction: int,
    take_profit_pct: float,
    stop_loss_pct: float,
    max_holding_bars: int,
    cost_calculator: CostCalculator | None = None,
) -> bool | None:
    """Simulate a fired trade, returning only the profitable/not outcome.

    Thin wrapper over ``resolve_fired_trade`` for callers that only need
    the boolean outcome. Callers building rolling features over PRIOR
    fires' outcomes (e.g. ``build_meta_label_features``'s
    ``rolling_hit_rate_20``) need the resolution BAR too (when the outcome
    became knowable, not just what it was) to stay causal -- use
    ``resolve_fired_trade`` directly for that.

    Returns:
        True if net-profitable after fees, False if not, None if the trade
        is unresolved (truncated at series end) -- never treat None as False.

    Raises:
        ValueError: direction is not 1 or -1.
    """
    resolution = resolve_fired_trade(
        df,
        fire_index,
        direction,
        take_profit_pct,
        stop_loss_pct,
        max_holding_bars,
        cost_calculator,
    )
    return resolution.profitable if resolution is not None else None


def _realized_volatility_series(close: pd.Series, window: int) -> pd.Series:
    """48-bar trailing realized volatility of bar-over-bar returns.

    A plain ``rolling(window).std()`` -- causal by construction (each row
    only reads its own trailing window), so a lookup at any fire index
    never sees data from bars after it.
    """
    returns = close.pct_change()
    return returns.rolling(window=window).std()


def _session_cyclical_encoding(timestamp: pd.Timestamp) -> tuple[float, float]:
    """Cyclical (sin, cos) encoding of hour-of-day, per §2a's spec."""
    hour_fraction = timestamp.hour / _HOURS_PER_DAY
    angle = 2.0 * math.pi * hour_fraction
    return math.sin(angle), math.cos(angle)


def build_meta_label_features(
    df: pd.DataFrame,
    fired_signals: Sequence[PrimarySignalRecord],
    resolutions: Sequence[TradeResolution],
    regime_detector: EnhancedRegimeDetector | None = None,
    realized_vol_window: int = REALIZED_VOL_WINDOW_BARS,
    hit_rate_lookback: int = HIT_RATE_LOOKBACK_FIRES,
) -> pd.DataFrame:
    """Build the meta-labeling feature set, per preregistration §2a.

    One row per fired signal. Every feature is causal: it reads only bars
    at-or-before the fire index. ``rolling_hit_rate_20`` additionally only
    uses PRIOR fires whose trade had actually RESOLVED
    (``resolution.exit_index <= fire.index``) by the current fire's index --
    a prior fire that hasn't resolved yet as of "now" is excluded, because
    its label depends on bars strictly after the current fire (see the
    module docstring's causality note).

    Args:
        df: OHLCV DataFrame the fires were recorded against.
        fired_signals: PrimarySignalRecord list from run_primary_signal_forward,
            in chronological order.
        resolutions: TradeResolution per fire (same order/length as
            fired_signals) -- typically resolve_fired_trade()'s output with
            unresolved (None) fires already filtered out by the caller.
        regime_detector: Reused EnhancedRegimeDetector instance; a fresh one
            is created if not supplied.
        realized_vol_window: Trailing bars for realized volatility (default 48).
        hit_rate_lookback: Trailing eligible PRIOR fires for rolling hit-rate
            (default 20).

    Returns:
        DataFrame with one row per fire: index, realized_vol_48,
        rolling_hit_rate_20, session_sin, session_cos, regime_trend,
        regime_volatility, regime_confidence, predicted_return_magnitude,
        label.

    Raises:
        ValueError: len(fired_signals) != len(resolutions).
    """
    if len(fired_signals) != len(resolutions):
        raise ValueError(
            f"fired_signals and resolutions must have the same length, "
            f"got {len(fired_signals)} and {len(resolutions)}"
        )

    detector = regime_detector or EnhancedRegimeDetector()
    realized_vol = _realized_volatility_series(df["close"], realized_vol_window)

    rows: list[dict[str, Any]] = []
    for position, fire in enumerate(fired_signals):
        # Only prior fires that had RESOLVED by this fire's own index are
        # "known" information at this point in time -- a prior fire whose
        # trade is still open (exit_index > fire.index) would leak that
        # fire's own future price path into this feature.
        eligible_prior = [
            resolutions[i].profitable
            for i in range(position)
            if resolutions[i].exit_index <= fire.index
        ]
        if eligible_prior:
            recent_prior = eligible_prior[-hit_rate_lookback:]
            rolling_hit_rate = float(np.mean(recent_prior))
        else:
            rolling_hit_rate = float("nan")

        timestamp = df.index[fire.index]
        session_sin, session_cos = _session_cyclical_encoding(pd.Timestamp(timestamp))

        regime = detector.detect_regime(df, fire.index)

        rows.append(
            {
                "index": fire.index,
                "realized_vol_48": float(realized_vol.iloc[fire.index]),
                "rolling_hit_rate_20": rolling_hit_rate,
                "session_sin": session_sin,
                "session_cos": session_cos,
                "regime_trend": regime.trend.value,
                "regime_volatility": regime.volatility.value,
                "regime_confidence": float(regime.confidence),
                "predicted_return_magnitude": abs(fire.predicted_return),
                "label": 1 if resolutions[position].profitable else 0,
            }
        )

    return pd.DataFrame(rows)


__all__ = [
    "HIT_RATE_LOOKBACK_FIRES",
    "REALIZED_VOL_WINDOW_BARS",
    "PrimarySignalRecord",
    "TradeResolution",
    "build_meta_label_features",
    "resolve_fired_trade",
    "run_primary_signal_forward",
    "simulate_fired_trade_profitability",
]
