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

import hashlib
import json
import logging
import math
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
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

logger = logging.getLogger(__name__)

# Feature-set constants, per preregistration §2a's exact spec.
REALIZED_VOL_WINDOW_BARS = 48
HIT_RATE_LOOKBACK_FIRES = 20
_HOURS_PER_DAY = 24.0

# At the measured ~200 bars/sec steady-state inference rate (#955), 1000
# bars is a checkpoint roughly every 5 seconds -- cheap enough to be the
# default whenever a checkpoint path is supplied.
FIRE_GENERATION_CHECKPOINT_EVERY_BARS = 1000

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


def fire_generation_corpus_fingerprint(
    signal_generator: SignalGenerator,
    df: pd.DataFrame,
    start_index: int | None = None,
) -> str:
    """Content fingerprint identifying one (corpus, start, generator) run.

    Guards checkpoint resumes: two equal-length but different corpora (a
    shifted window, another symbol) must never resume-splice into one
    training set, so the fingerprint hashes the frame's index endpoints,
    the full close column, the resolved start index, and the generator's
    identity string. Best-effort on generator identity: the name does not
    capture model WEIGHTS, so callers must not reuse a checkpoint across
    primary-model retrains (the pipeline deletes its checkpoint after each
    successful run for exactly this reason).

    Raises:
        ValueError: df is empty (nothing to fingerprint or checkpoint).
    """
    if len(df) == 0:
        raise ValueError("cannot fingerprint an empty corpus")
    start = start_index if start_index is not None else signal_generator.warmup_period
    generator_identity = getattr(signal_generator, "name", type(signal_generator).__name__)
    close_digest = hashlib.sha256(df["close"].to_numpy(dtype=np.float64).tobytes()).hexdigest()
    payload = "|".join(
        [
            str(df.index[0]),
            str(df.index[-1]),
            str(len(df)),
            str(start),
            close_digest,
            str(generator_identity),
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _checkpoint_rejection(checkpoint_path: Path, reason: str) -> ValueError:
    """Uniform rejection policy: raise, never guess, never overwrite.

    A checkpoint that cannot be positively identified as THIS run's own
    (corpus fingerprint mismatch, unparseable JSON, missing keys) is never
    silently ignored or overwritten -- overwriting would destroy either
    another run's progress or, if the caller mis-pointed checkpoint_path,
    an unrelated file. The operator resolves it: delete the file or point
    at a different path to start fresh.
    """
    return ValueError(
        f"checkpoint at {checkpoint_path} {reason} -- refusing to resume from or "
        f"overwrite it; delete the file or pass a different checkpoint_path to "
        f"start fresh"
    )


def _load_fire_generation_checkpoint(
    checkpoint_path: Path, fingerprint: str
) -> tuple[list[PrimarySignalRecord], int] | None:
    """Load a prior run's progress, or None if no checkpoint exists yet.

    Returns:
        (records so far, first index still to evaluate).

    Raises:
        ValueError: the checkpoint belongs to a different (corpus, start,
            generator) fingerprint, or is corrupt/incomplete -- see
            ``_checkpoint_rejection`` for the policy.
    """
    if not checkpoint_path.exists():
        return None
    try:
        state = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        recorded_fingerprint = state["fingerprint"]
        last_completed_index = int(state["last_completed_index"])
        records = [
            PrimarySignalRecord(
                index=int(record["index"]),
                direction=int(record["direction"]),
                predicted_return=float(record["predicted_return"]),
            )
            for record in state["records"]
        ]
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise _checkpoint_rejection(
            checkpoint_path, f"is corrupt or not a fire-generation checkpoint ({exc})"
        ) from exc
    if recorded_fingerprint != fingerprint:
        raise _checkpoint_rejection(
            checkpoint_path,
            "was recorded for a different corpus, start index, or signal generator "
            f"(fingerprint {recorded_fingerprint!r} != this run's {fingerprint!r})",
        )
    return records, last_completed_index + 1


def _write_fire_generation_checkpoint(
    checkpoint_path: Path,
    fingerprint: str,
    start: int,
    total_bars: int,
    last_completed_index: int,
    records: Sequence[PrimarySignalRecord],
) -> None:
    """Persist progress crash-safely: write a temp file, then atomically rename.

    ``start``/``total_bars`` are stored for human debugging only -- resume
    validation compares the fingerprint (which already covers both).
    """
    state = {
        "fingerprint": fingerprint,
        "start_index": start,
        "total_bars": total_bars,
        "last_completed_index": last_completed_index,
        "records": [
            {
                "index": record.index,
                "direction": record.direction,
                "predicted_return": record.predicted_return,
            }
            for record in records
        ],
    }
    tmp_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(state), encoding="utf-8")
    os.replace(tmp_path, checkpoint_path)


def run_primary_signal_forward(
    signal_generator: SignalGenerator,
    df: pd.DataFrame,
    start_index: int | None = None,
    *,
    checkpoint_path: str | Path | None = None,
    checkpoint_every_bars: int = FIRE_GENERATION_CHECKPOINT_EVERY_BARS,
) -> list[PrimarySignalRecord]:
    """Run a primary SignalGenerator forward over df, recording every fire.

    This is preregistration §2a/§8 step (i): "running the CURRENT incumbent
    signal generator forward over the training corpus to find its fire
    points." Reuses the SignalGenerator interface directly (generate_signal)
    rather than reimplementing prediction logic.

    Long corpora (~47k bars at ~200 bars/sec) run for the better part of an
    hour, and task-lifetime caps killed uncheckpointed passes twice during
    the #933 tournament (#955 defect 2) -- so progress can optionally be
    checkpointed: every ``checkpoint_every_bars`` completed bars, the fires
    so far and the last completed bar index are written to
    ``checkpoint_path`` (temp file + atomic rename). A rerun pointed at the
    same checkpoint resumes after the last completed bar instead of
    reprocessing from the start.

    Args:
        signal_generator: The primary (incumbent) SignalGenerator instance.
        df: OHLCV DataFrame to run forward over.
        start_index: First index to evaluate. Defaults to
            ``signal_generator.warmup_period``.
        checkpoint_path: Optional JSON file to persist/resume progress
            through. The file is left in place after a completed run (a
            rerun then returns its fires without reprocessing); the caller
            owns cleanup. An existing file that cannot be positively
            identified as this exact run's checkpoint (corpus/start/
            generator fingerprint mismatch, corrupt or incomplete JSON) is
            rejected with ValueError -- never resumed from, never
            overwritten (see ``_checkpoint_rejection``).
        checkpoint_every_bars: Completed bars between checkpoint writes.
            Only meaningful with ``checkpoint_path``.

    Returns:
        A list of PrimarySignalRecord, one per non-HOLD bar, in order.

    Raises:
        ValueError: ``checkpoint_every_bars`` < 1, or the checkpoint file
            was rejected (see ``_load_fire_generation_checkpoint``).
    """
    start = start_index if start_index is not None else signal_generator.warmup_period
    records: list[PrimarySignalRecord] = []
    resume_from = start

    checkpoint = Path(checkpoint_path) if checkpoint_path is not None else None
    fingerprint = ""
    if checkpoint is not None:
        if checkpoint_every_bars < 1:
            raise ValueError(f"checkpoint_every_bars must be >= 1, got {checkpoint_every_bars}")
        if len(df) == 0:
            checkpoint = None  # nothing to evaluate or checkpoint
        else:
            fingerprint = fire_generation_corpus_fingerprint(signal_generator, df, start)
            loaded = _load_fire_generation_checkpoint(checkpoint, fingerprint)
            if loaded is not None:
                records, resume_from = loaded
                logger.info(
                    "fire generation resuming from bar %d of %d (%d fires loaded from %s)",
                    resume_from,
                    len(df),
                    len(records),
                    checkpoint,
                )

    bars_since_checkpoint = 0
    for index in range(resume_from, len(df)):
        signal = signal_generator.generate_signal(df, index)
        if signal.direction != SignalDirection.HOLD:
            direction = 1 if signal.direction == SignalDirection.BUY else -1
            predicted_return = float(signal.metadata.get("predicted_return", 0.0))
            records.append(
                PrimarySignalRecord(
                    index=index, direction=direction, predicted_return=predicted_return
                )
            )
        bars_since_checkpoint += 1
        if checkpoint is not None and bars_since_checkpoint >= checkpoint_every_bars:
            _write_fire_generation_checkpoint(
                checkpoint, fingerprint, start, len(df), index, records
            )
            bars_since_checkpoint = 0

    if checkpoint is not None and resume_from < len(df):
        # Final checkpoint marks the run complete -- a rerun against the same
        # file returns these fires without re-evaluating any bar.
        _write_fire_generation_checkpoint(
            checkpoint, fingerprint, start, len(df), len(df) - 1, records
        )
        logger.info(
            "fire generation complete: %d fires over bars %d..%d, checkpoint finalized at %s",
            len(records),
            start,
            len(df) - 1,
            checkpoint,
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


def resolution_ordered_hit_rate(resolved: Sequence[tuple[int, int, bool]], lookback: int) -> float:
    """Rolling hit-rate over the last ``lookback`` RESOLVED fires.

    Single source of truth for ``rolling_hit_rate_20``, used identically by
    training (``build_meta_label_features``) and the exam-time consumer
    (``MetaLabelExamSignalGenerator._rolling_hit_rate``) so the two windowing
    conventions can never drift apart again.

    A fire's profitability only becomes "known" at its RESOLUTION bar
    (``exit_index``), not its fire bar -- and fires can resolve
    out-of-fire-order (a long-held early fire commonly resolves after a
    later, quickly-resolved one; see the module docstring's causality
    note). The online exam path naturally builds its history in
    ``(exit_index, fire_index)`` order: bars are processed in strictly
    increasing index order, and within a single resolution bar, multiple
    fires resolve in their original registration (fire) order. Training
    reconstructs the SAME ordering from a fully-resolved offline corpus by
    sorting explicitly here -- this function always re-sorts before
    windowing, so callers may pass an already-ordered or unordered
    sequence.

    Args:
        resolved: ``(exit_index, fire_index, profitable)`` tuples for every
            fire resolved so far, in any order.
        lookback: Number of most-recent (by the ordering above) resolved
            fires to average over.

    Returns:
        Mean of the last ``lookback`` profitability outcomes, or NaN if
        ``resolved`` is empty.
    """
    if not resolved:
        return float("nan")
    ordered = sorted(resolved, key=lambda item: (item[0], item[1]))
    recent = [profitable for _, _, profitable in ordered[-lookback:]]
    return float(np.mean(recent))


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
            is created if not supplied. If ``df`` already carries the regime
            annotation columns (``regime_label`` et al., from
            ``RegimeDetector.annotate``), it is used as-is with no further
            annotation pass.
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

    # detect_regime re-annotates the ENTIRE frame on every call when the
    # regime columns are absent -- per-fire that is O(n_fires * n_bars) and
    # made real (~46k-fire x ~47k-bar) corpora infeasible (#955). Annotate
    # once here; detect_regime sees the columns and skips its own pass.
    regime_df = df if "regime_label" in df.columns else detector.base_detector.annotate(df.copy())

    rows: list[dict[str, Any]] = []
    for position, fire in enumerate(fired_signals):
        # Only prior fires that had RESOLVED by this fire's own index are
        # "known" information at this point in time -- a prior fire whose
        # trade is still open (exit_index > fire.index) would leak that
        # fire's own future price path into this feature. The eligible set
        # is windowed by resolution_ordered_hit_rate in (exit_index,
        # fire_index) order -- NOT this loop's fire-order iteration -- to
        # match MetaLabelExamSignalGenerator's online windowing exactly.
        eligible_prior = [
            (resolutions[i].exit_index, fired_signals[i].index, resolutions[i].profitable)
            for i in range(position)
            if resolutions[i].exit_index <= fire.index
        ]
        rolling_hit_rate = resolution_ordered_hit_rate(eligible_prior, hit_rate_lookback)

        timestamp = df.index[fire.index]
        session_sin, session_cos = _session_cyclical_encoding(pd.Timestamp(timestamp))

        regime = detector.detect_regime(regime_df, fire.index)

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


# Fixed feature order for the sklearn LogisticRegression classifier --
# single source of truth for BOTH training (encode_meta_label_features_for_
# training) and inference (encode_meta_label_feature_row) so the two can
# never drift out of sync. Also documents the ONNX graph's expected input
# column order (sklearn_onnx_export.py has no column names of its own).
META_LABEL_FEATURE_ORDER: tuple[str, ...] = (
    "realized_vol_48",
    "rolling_hit_rate_20",
    "session_sin",
    "session_cos",
    "regime_trend_encoded",
    "regime_volatility_encoded",
    "regime_confidence",
    "predicted_return_magnitude",
)

# Fixed numeric codes for the string regime enums (TrendLabel/VolLabel) --
# sklearn needs numeric input. -1/0/1 for trend mirrors down/range/up
# ordering; 0/1 for volatility is a plain low/high indicator.
_TREND_ENCODING: dict[str, float] = {"trend_down": -1.0, "range": 0.0, "trend_up": 1.0}
_VOLATILITY_ENCODING: dict[str, float] = {"low_vol": 0.0, "high_vol": 1.0}

# rolling_hit_rate_20 is NaN for a fire with no eligible prior resolved
# fires yet (see build_meta_label_features docstring) -- fill with a
# neutral prior (neither a good nor bad track record) rather than 0.0
# (which would read as "100% of trailing fires unprofitable", actively
# misleading the classifier for the earliest fires in any training corpus).
_NEUTRAL_HIT_RATE_PRIOR = 0.5


def encode_meta_label_feature_row(
    *,
    realized_vol_48: float,
    rolling_hit_rate_20: float,
    session_sin: float,
    session_cos: float,
    regime_trend: str,
    regime_volatility: str,
    regime_confidence: float,
    predicted_return_magnitude: float,
) -> np.ndarray:
    """Encode ONE meta-label feature row into META_LABEL_FEATURE_ORDER.

    The inference-time counterpart to
    ``encode_meta_label_features_for_training`` -- used by the exam
    harness's stateful ``MetaLabelExamSignalGenerator``, which computes
    features incrementally (one fire at a time) rather than as a batch
    DataFrame. Must encode identically to the training-time batch encoder
    for the same logical values (verified directly by a regression test).

    Args:
        realized_vol_48: 48-bar trailing realized volatility.
        rolling_hit_rate_20: Rolling hit-rate of prior RESOLVED fires, or
            NaN if none are eligible yet (filled with a neutral 0.5 prior).
        session_sin: sin(2*pi*hour/24) cyclical encoding.
        session_cos: cos(2*pi*hour/24) cyclical encoding.
        regime_trend: EnhancedRegimeDetector's TrendLabel.value string.
        regime_volatility: EnhancedRegimeDetector's VolLabel.value string.
        regime_confidence: EnhancedRegimeDetector's regime confidence, [0,1].
        predicted_return_magnitude: abs() of the primary signal's own
            predicted_return at the fire point.

    Returns:
        float32 array of shape (len(META_LABEL_FEATURE_ORDER),).

    Raises:
        KeyError: regime_trend/regime_volatility is not a recognized enum value.
    """
    hit_rate = (
        _NEUTRAL_HIT_RATE_PRIOR
        if rolling_hit_rate_20 is None or math.isnan(rolling_hit_rate_20)
        else float(rolling_hit_rate_20)
    )
    values = {
        "realized_vol_48": float(realized_vol_48),
        "rolling_hit_rate_20": hit_rate,
        "session_sin": float(session_sin),
        "session_cos": float(session_cos),
        "regime_trend_encoded": _TREND_ENCODING[regime_trend],
        "regime_volatility_encoded": _VOLATILITY_ENCODING[regime_volatility],
        "regime_confidence": float(regime_confidence),
        "predicted_return_magnitude": float(predicted_return_magnitude),
    }
    return np.array([values[name] for name in META_LABEL_FEATURE_ORDER], dtype=np.float32)


def encode_meta_label_features_for_training(
    features_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode build_meta_label_features()'s output into (X, y) training arrays.

    One row per fire, columns in META_LABEL_FEATURE_ORDER -- ready for
    ``sklearn.linear_model.LogisticRegression.fit(X, y)``.

    Args:
        features_df: Output of ``build_meta_label_features``.

    Returns:
        (X, y): X is float32 shape (n_fires, len(META_LABEL_FEATURE_ORDER)),
        y is the "label" column, shape (n_fires,).

    Raises:
        ValueError: features_df is empty.
    """
    if features_df.empty:
        raise ValueError("features_df is empty -- no fired signals to train on")

    rows = [
        encode_meta_label_feature_row(
            realized_vol_48=row["realized_vol_48"],
            rolling_hit_rate_20=row["rolling_hit_rate_20"],
            session_sin=row["session_sin"],
            session_cos=row["session_cos"],
            regime_trend=row["regime_trend"],
            regime_volatility=row["regime_volatility"],
            regime_confidence=row["regime_confidence"],
            predicted_return_magnitude=row["predicted_return_magnitude"],
        )
        for _, row in features_df.iterrows()
    ]
    X = np.stack(rows).astype(np.float32)
    y = features_df["label"].to_numpy(dtype=np.float32)
    return X, y


__all__ = [
    "FIRE_GENERATION_CHECKPOINT_EVERY_BARS",
    "HIT_RATE_LOOKBACK_FIRES",
    "META_LABEL_FEATURE_ORDER",
    "REALIZED_VOL_WINDOW_BARS",
    "PrimarySignalRecord",
    "TradeResolution",
    "build_meta_label_features",
    "encode_meta_label_feature_row",
    "encode_meta_label_features_for_training",
    "fire_generation_corpus_fingerprint",
    "resolution_ordered_hit_rate",
    "resolve_fired_trade",
    "run_primary_signal_forward",
    "simulate_fired_trade_profitability",
]
