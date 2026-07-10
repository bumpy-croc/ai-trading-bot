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
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from src.engines.shared.barrier_touch import check_barrier_touch
from src.engines.shared.cost_calculator import CostCalculator
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


def simulate_fired_trade_profitability(
    df: pd.DataFrame,
    fire_index: int,
    direction: int,
    take_profit_pct: float,
    stop_loss_pct: float,
    max_holding_bars: int,
    cost_calculator: CostCalculator | None = None,
) -> bool | None:
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
        True if net-profitable after fees, False if not, None if the trade
        is unresolved (truncated at series end with no forward data) --
        callers must drop unresolved fire points, never treat None as False.

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
    if direction == 1:
        stop_loss_price = entry_price * (1.0 - stop_loss_pct)
        take_profit_price = entry_price * (1.0 + take_profit_pct)
    else:
        stop_loss_price = entry_price * (1.0 + stop_loss_pct)
        take_profit_price = entry_price * (1.0 - take_profit_pct)

    max_bar = min(fire_index + max_holding_bars, n - 1)
    exit_price: float | None = None
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
            break
        if touch.hit_take_profit:
            exit_price = touch.take_profit_exit_price
            break

    if exit_price is None:
        reached_full_vertical_barrier = max_bar == fire_index + max_holding_bars
        if not reached_full_vertical_barrier:
            return None  # truncated at series end -- unresolved
        exit_price = close[max_bar]

    raw_pct_return = direction * (exit_price - entry_price) / entry_price
    round_trip_cost_pct = 2.0 * cost_calculator.fee_rate
    return bool((raw_pct_return - round_trip_cost_pct) > 0.0)


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
    labels: Sequence[bool],
    regime_detector: EnhancedRegimeDetector | None = None,
    realized_vol_window: int = REALIZED_VOL_WINDOW_BARS,
    hit_rate_lookback: int = HIT_RATE_LOOKBACK_FIRES,
) -> pd.DataFrame:
    """Build the meta-labeling feature set, per preregistration §2a.

    One row per fired signal. Every feature is causal: it reads only bars
    at-or-before the fire index, and (for rolling_hit_rate_20) only STRICTLY
    PRIOR fires' labels -- never the fire's own label or any later fire.

    Args:
        df: OHLCV DataFrame the fires were recorded against.
        fired_signals: PrimarySignalRecord list from run_primary_signal_forward,
            in chronological order.
        labels: Resolved profitability label per fire (same order/length as
            fired_signals) -- typically simulate_fired_trade_profitability()'s
            output with unresolved (None) fires already filtered out by the
            caller.
        regime_detector: Reused EnhancedRegimeDetector instance; a fresh one
            is created if not supplied.
        realized_vol_window: Trailing bars for realized volatility (default 48).
        hit_rate_lookback: Trailing PRIOR fires for rolling hit-rate (default 20).

    Returns:
        DataFrame with one row per fire: index, realized_vol_48,
        rolling_hit_rate_20, session_sin, session_cos, regime_trend,
        regime_volatility, regime_confidence, predicted_return_magnitude,
        label.

    Raises:
        ValueError: len(fired_signals) != len(labels).
    """
    if len(fired_signals) != len(labels):
        raise ValueError(
            f"fired_signals and labels must have the same length, "
            f"got {len(fired_signals)} and {len(labels)}"
        )

    detector = regime_detector or EnhancedRegimeDetector()
    realized_vol = _realized_volatility_series(df["close"], realized_vol_window)

    label_ints = [1 if bool(label) else 0 for label in labels]

    rows: list[dict[str, Any]] = []
    for position, fire in enumerate(fired_signals):
        prior_labels = label_ints[:position]
        if prior_labels:
            recent_prior = prior_labels[-hit_rate_lookback:]
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
                "label": label_ints[position],
            }
        )

    return pd.DataFrame(rows)


__all__ = [
    "HIT_RATE_LOOKBACK_FIRES",
    "REALIZED_VOL_WINDOW_BARS",
    "PrimarySignalRecord",
    "build_meta_label_features",
    "run_primary_signal_forward",
    "simulate_fired_trade_profitability",
]
