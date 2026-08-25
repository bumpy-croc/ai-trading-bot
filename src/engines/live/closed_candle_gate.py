"""Closed-candle gating for live signal decisions (parity plan P1.0, decision D1).

Backtest is inherently closed-bar: the backtest engine iterates completed
candles, so strategy signals never see a mutating bar. Live, by contrast,
rewrites the kline-buffer tail on every WebSocket tick and the trading loop
decides on ``df.iloc[-1]`` — a partially formed candle
(docs/refactor/backtest_live_parity_plan.md, §2 divergence #1 and D1; PM
ratified option (a)).

When the ``closed_candle_gating`` feature flag is ON, the trading loop
evaluates the strategy signal exactly once per newly closed bar, at that bar's
index. Because the ML feature window is sliced *exclusive* of the evaluated
index (``ml_signal_generator._get_ml_prediction``) and the signal's reference
price is ``df["close"].iloc[index]`` (the ``predicted_return`` denominator),
evaluating at the last closed bar's index reproduces the backtest decision at
that bar exactly: closed-bars-only feature window, reference price frozen to
the bar's final close. The forming-bar flip-rate study
(docs/research/experiments/2026-07-06_forming-bar-fliprate.md) measured that
the floating reference price is the flip mechanism — 43.2% of decisions at
minute 5 disagree with the closed-bar decision.

Hard protective paths — stop-loss, trailing, exit checks, reconciliation,
account monitoring — stay tick-driven and UNGATED (PM condition on D1: never
delay protection). Policy refresh (trailing-stop/partial-exit/dynamic-risk
config hydration) and strategy-signal exits ride on the decision and therefore
move to bar cadence when the flag is ON — which is what backtest does, so that
shift is itself a parity gain rather than a protection loss.

Seam rationale (loop-level index gating rather than a truncated closed-only
frame):

- evaluating at the last closed bar's index with the forming bar present at a
  later row is exactly the shape backtest uses (``process_candle(df, i)``
  mid-frame with future rows present — strategies must already be causal
  w.r.t. index or backtest itself would be look-ahead biased), and is
  input-equivalent to truncating the frame at the closed bar;
- one feature-pipeline pass per tick — a separate closed-only decision frame
  would force a second indicator/ML feature computation on the hot path;
- protective paths keep the very same frame/index/price they receive today.

Liveness: a gate that stops evaluating is silent by construction — the only
symptom is the *absence* of a decision, which looks exactly like a quiet
market (#1094/#1095: four days of silent halt whose lesson was that absence of
events is not evidence of health). ``stall_observation`` therefore exposes a
positive assertion that the gate is still deciding, wired into
``LatchedConditionMonitor`` alongside the other latched conditions.

Thread-safety: a ``ClosedCandleGate`` instance is owned by the trading loop
and used only on the trading-loop thread (single reader/writer), so it takes
no locks. The closed-bar frontier it consumes is captured atomically with the
frame by ``KlineBuffer.snapshot()`` — see ``_closed_frontier``.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.engines.live.kline_buffer import timeframe_to_ms

logger = logging.getLogger(__name__)

# A gate that has not evaluated for this many bar intervals is treated as
# stalled. Three intervals tolerates a missed close plus a resync without
# crying wolf, while still catching a permanent jam inside one timeframe's
# worth of trading on any realistic interval.
STALL_INTERVALS = 3
# Floor for the stall threshold when the timeframe is unknown or very short,
# so a 1m timeframe cannot page on a few seconds of ordinary quiet.
MIN_STALL_SECONDS = 900.0


@dataclass(frozen=True)
class GateDecision:
    """Resolved decision-bar view for one trading-loop tick.

    Attributes:
        evaluate: Whether the loop should run signal evaluation this tick.
        index: Position in the frame to evaluate at (tail when gating is off).
        bar_time: Open time of the decision bar (None only for an empty frame).
        bar_closed: Whether the decision bar is known to be closed.
    """

    evaluate: bool
    index: int
    bar_time: Any
    bar_closed: bool


@dataclass(frozen=True)
class LoopSignalDecision:
    """One trading-loop tick's resolved signal decision.

    Attributes:
        decision: Decision to feed the exit path (cached closed-bar decision
            between closes when gating is ON), or None.
        entry_index: Frame index the entry pipeline must use.
        allow_entries: Whether the entry pipeline may run this tick.
        commit_bar: Bar whose evaluation should be marked consumed once entry
            execution has had its attempt, or None when nothing was evaluated.
    """

    decision: Any
    entry_index: int
    allow_entries: bool
    commit_bar: Any = None


@dataclass(frozen=True)
class StallObservation:
    """Liveness of the gate, for the latched-condition monitor.

    Attributes:
        stalled: True when gating is ON and no bar has been evaluated for
            longer than the stall threshold.
        elapsed_seconds: Seconds since the last evaluation (or since the gate
            was constructed, when none has happened yet).
        reason: Operator-facing description, or None when healthy.
    """

    stalled: bool
    elapsed_seconds: float
    reason: str | None = None


class ClosedCandleGate:
    """Once-per-closed-bar signal evaluation gate for the live trading loop.

    Disabled (flag OFF, the shipped default): ``resolve`` always targets the
    tail bar and always evaluates — today's decision behavior, unchanged.

    Not *byte*-identical, and the difference is worth naming: ``bar_closed`` is
    still computed so ``stamp_decision_signal`` can write three keys
    (``decision_bar_open_time``, ``decision_bar_close_time``,
    ``decision_bar_closed``) into ``Signal.metadata`` on every tick, in both
    modes, so the A/B has attribution. ``Strategy._extract_indicators`` fans
    all metadata keys into the indicator snapshot, so those keys do reach
    persisted rows. No behavior depends on them; the claim is inertness, not
    byte-identity.

    Enabled: ``resolve`` targets the newest provably closed bar and evaluates
    only when that bar is newer than the last one marked evaluated. The
    monotonic bar-time guard makes evaluation idempotent across loop ticks,
    WebSocket reconnects, and REST backfills (a rewound frame can never
    re-trigger an old bar).
    """

    def __init__(
        self,
        enabled: bool,
        timeframe: str | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Resolve the flag once at construction (never per tick — disk I/O).

        Args:
            enabled: Resolved ``closed_candle_gating`` feature-flag value.
            timeframe: Candle timeframe, used only to size the stall threshold.
            clock: Monotonic seconds source; injectable for tests.
        """
        self.enabled = enabled
        self.timeframe = timeframe
        self._clock = clock
        self._last_evaluated_bar: Any = None
        self._last_evaluation_at: float = clock()
        interval_ms = timeframe_to_ms(timeframe) if timeframe else None
        self._stall_threshold_seconds = max(
            MIN_STALL_SECONDS,
            (interval_ms / 1000.0) * STALL_INTERVALS if interval_ms else 0.0,
        )
        # Latched log state: each condition warns once on entry and once on
        # recovery, so a persistent degradation cannot flood the loop log and
        # a transient one still leaves both edges in the record.
        self._warned_no_frontier = False
        self._warned_incomparable = False

    def configure_timeframe(self, timeframe: str | None) -> None:
        """Adopt the session's timeframe, sizing the stall threshold.

        The engine does not know its timeframe at construction; the trading
        loop does. Resets the liveness baseline so a long-idle engine does not
        report stalled the instant it starts trading.
        """
        self.timeframe = timeframe
        interval_ms = timeframe_to_ms(timeframe) if timeframe else None
        self._stall_threshold_seconds = max(
            MIN_STALL_SECONDS,
            (interval_ms / 1000.0) * STALL_INTERVALS if interval_ms else 0.0,
        )
        self._last_evaluation_at = self._clock()

    def resolve(self, df: pd.DataFrame, buffer_frontier: Any = None) -> GateDecision:
        """Resolve the decision bar for this tick.

        Args:
            df: The loop's prepared frame (tail = forming/tick bar).
            buffer_frontier: The closed-bar frontier captured atomically with
                the frame by ``KlineBuffer.snapshot()``, or None when the frame
                came from REST. Frames without it still get closed-bar evidence
                from their own shape (a bar with a successor row is closed by
                construction).

        Returns:
            GateDecision for this tick. With gating disabled, always
            ``evaluate=True`` on the tail.
        """
        if len(df) == 0:
            return GateDecision(evaluate=False, index=0, bar_time=None, bar_closed=False)

        tail_pos = len(df) - 1
        tail_ts = df.index[tail_pos]
        frontier = self._closed_frontier(df, buffer_frontier)

        if not self.enabled:
            bar_closed = frontier is not None and self._le(tail_ts, frontier)
            return GateDecision(
                evaluate=True, index=tail_pos, bar_time=tail_ts, bar_closed=bar_closed
            )

        self._check_frontier_evidence(buffer_frontier)

        if frontier is None:
            # No bar in this frame is provably closed — nothing to decide on.
            return GateDecision(evaluate=False, index=tail_pos, bar_time=tail_ts, bar_closed=False)

        # Newest frame position at or below the frontier (the frontier bar
        # itself may have been dropped by essential-column dropna upstream).
        try:
            pos = int(df.index.searchsorted(frontier, side="right")) - 1
        except TypeError:
            # Same mixed-tz hazard _closed_frontier guards; fail closed rather
            # than escaping into the loop's generic handler, where it would
            # count toward consecutive_errors and shutdown.
            self._warn_incomparable(frontier)
            return GateDecision(evaluate=False, index=tail_pos, bar_time=tail_ts, bar_closed=False)
        if pos < 0:
            return GateDecision(evaluate=False, index=tail_pos, bar_time=tail_ts, bar_closed=False)

        bar_time = df.index[pos]
        already_evaluated = self._last_evaluated_bar is not None and self._le(
            bar_time, self._last_evaluated_bar
        )
        return GateDecision(
            evaluate=not already_evaluated, index=pos, bar_time=bar_time, bar_closed=True
        )

    def mark_evaluated(self, bar_time: Any) -> None:
        """Record that ``bar_time`` received its one signal evaluation.

        Called by the loop only AFTER entry execution has had its attempt, so
        an exception mid-execution leaves the bar unconsumed and the next tick
        retries — matching the ungated path, which retried on every tick.
        """
        self._last_evaluated_bar = bar_time
        self._last_evaluation_at = self._clock()

    def reset(self) -> None:
        """Forget the evaluated-bar high-water mark (strategy hot-swap).

        A swapped-in strategy must be allowed to decide the current bar; the
        retired strategy's evaluation must not block it.
        """
        self._last_evaluated_bar = None
        self._last_evaluation_at = self._clock()

    def stall_observation(self) -> StallObservation:
        """Liveness assertion: is the gate still evaluating bars?

        Only meaningful with gating ON — with the flag OFF every tick evaluates
        and the concept does not apply, so the gate reports healthy.
        """
        elapsed = max(0.0, self._clock() - self._last_evaluation_at)
        if not self.enabled or elapsed <= self._stall_threshold_seconds:
            return StallObservation(stalled=False, elapsed_seconds=elapsed)
        return StallObservation(
            stalled=True,
            elapsed_seconds=elapsed,
            reason=(
                f"closed-candle gating has not evaluated a bar for {int(elapsed)}s "
                f"(threshold {int(self._stall_threshold_seconds)}s, timeframe "
                f"{self.timeframe or 'unknown'}); signal decisions are frozen while "
                "protective paths keep running"
            ),
        )

    @staticmethod
    def _le(left: Any, right: Any) -> bool:
        """``left <= right``, False when the two are not comparable.

        Timestamps from a degraded REST fallback can be tz-naive while the
        buffer's are tz-aware. Comparing them raises, and an unguarded raise
        here escapes into the loop's generic handler and counts toward
        ``consecutive_errors``. False means "cannot prove closed/already
        evaluated", which fails closed in both call sites.
        """
        try:
            return bool(left <= right)
        except TypeError:
            return False

    def _check_frontier_evidence(self, buffer_frontier: Any) -> None:
        """Warn (latched) while gating runs without WebSocket closure evidence.

        Without a frontier the only evidence is frame shape — a bar with a
        successor row. That is always *sound* (never certifies a forming bar),
        but against a provider whose REST tail is already a closed bar it is
        one bar stale, which would make gating quietly worse for parity than
        leaving it off. Deliberately NOT promoted by wall-clock: after a REST
        frame's tail bar has passed its close time, the row still holds the
        partial snapshot captured when it was fetched, so a clock-based
        promotion would certify incomplete data as final — the precise defect
        this whole change exists to remove.
        """
        if buffer_frontier is None:
            if not self._warned_no_frontier:
                self._warned_no_frontier = True
                logger.warning(
                    "Closed-candle gating active without a WebSocket closed-bar frontier "
                    "(REST-sourced frame); falling back to frame-shape evidence, which can "
                    "lag one bar. Parity is degraded until the WebSocket buffer recovers."
                )
        elif self._warned_no_frontier:
            self._warned_no_frontier = False
            logger.info("Closed-candle gating: WebSocket closed-bar frontier restored")

    def _warn_incomparable(self, value: Any) -> None:
        """Warn (latched) that timestamps could not be compared.

        Downgrades the parity guarantee for as long as it holds, so this is a
        WARNING rather than the DEBUG it would deserve as a mere curiosity.
        """
        if not self._warned_incomparable:
            self._warned_incomparable = True
            logger.warning(
                "Closed-candle gate: frame index is not comparable to the closed-bar "
                "frontier %r (mixed tz-awareness after a degraded REST fallback?). "
                "Gating is failing closed — no bar will be evaluated until this clears.",
                value,
            )

    def _closed_frontier(self, df: pd.DataFrame, buffer_frontier: Any) -> Any:
        """Newest bar time in ``df`` that is provably closed, or None.

        Evidence, merged by max:
        - the second-to-last row is closed by construction (a successor row
          exists) — works for REST-sourced frames with no ``x`` flags;
        - the buffer frontier is authoritative (Binance kline ``x: true``),
          clamped to the frame's tail: a frontier at or beyond the tail
          proves the tail itself closed. The clamp is only sound because the
          frontier is captured in the same lock acquisition as the frame
          (``KlineBuffer.snapshot``); read separately, the tail could close
          between the two reads and the clamp would certify a row the frame
          still holds mid-formation.
        """
        candidates = []
        if len(df) >= 2:
            candidates.append(df.index[-2])
        if buffer_frontier is not None:
            try:
                candidates.append(min(buffer_frontier, df.index[-1]))
            except TypeError:
                self._warn_incomparable(buffer_frontier)
        if not candidates:
            return None
        try:
            return max(candidates)
        except TypeError:
            self._warn_incomparable(buffer_frontier)
            return None


def stamp_decision_signal(
    decision: Any, *, bar_time: Any, bar_closed: bool, timeframe: str | None
) -> None:
    """Stamp Signal.metadata with the decision bar's identity — in BOTH modes.

    Lets the staging(ON)/prod(OFF) A/B and the parity ledger attribute every
    decision to the bar it was made on (plan §P1.0 observability). Keys:
    ``decision_bar_open_time``, ``decision_bar_close_time`` (open + interval;
    None when the timeframe is unknown), ``decision_bar_closed``.

    Never raises: this runs on the flag-OFF path too, where any exception would
    count toward ``consecutive_errors`` and break the inertness guarantee that
    makes merging with the flag off safe.
    """
    try:
        signal = getattr(decision, "signal", None) if decision is not None else None
        metadata = getattr(signal, "metadata", None)
        if not isinstance(metadata, dict) or bar_time is None:
            return

        open_ts = pd.Timestamp(bar_time)
        interval_ms = timeframe_to_ms(timeframe) if timeframe else None
        metadata["decision_bar_open_time"] = open_ts.isoformat()
        metadata["decision_bar_close_time"] = (
            (open_ts + pd.Timedelta(milliseconds=interval_ms)).isoformat() if interval_ms else None
        )
        metadata["decision_bar_closed"] = bool(bar_closed)
    except Exception as e:  # pragma: no cover - defensive
        logger.debug("Decision-bar stamping skipped: %s", e)
