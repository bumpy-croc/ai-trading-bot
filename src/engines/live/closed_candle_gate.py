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

The forming bar stays fully visible to every protective path — stop-loss,
trailing, exit checks, reconciliation, account monitoring remain tick-driven
and ungated (PM condition on D1: never delay protection).

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

Thread-safety: a ``ClosedCandleGate`` instance is owned by the trading loop
and used only on the trading-loop thread (single reader/writer), so it takes
no locks. The authoritative closed-bar frontier is read from
``KlineBuffer.last_closed_bar_time``, which locks internally.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.engines.live.kline_buffer import timeframe_to_ms

logger = logging.getLogger(__name__)


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


class ClosedCandleGate:
    """Once-per-closed-bar signal evaluation gate for the live trading loop.

    Disabled (flag OFF, the shipped default): ``resolve`` always targets the
    tail bar and always evaluates — today's behavior, byte-identical. The
    ``bar_closed`` field is still computed so Signal.metadata stamping works
    in both modes (A/B attribution).

    Enabled: ``resolve`` targets the newest provably closed bar and evaluates
    only when that bar is newer than the last one marked evaluated. The
    monotonic bar-time guard makes evaluation idempotent across loop ticks,
    WebSocket reconnects, and REST backfills (a rewound frame can never
    re-trigger an old bar).
    """

    def __init__(self, enabled: bool) -> None:
        """Resolve the flag once at construction (never per tick — disk I/O).

        Args:
            enabled: Resolved ``closed_candle_gating`` feature-flag value.
        """
        self.enabled = enabled
        self._last_evaluated_bar: Any = None

    def resolve(self, df: pd.DataFrame, buffer_frontier: Any = None) -> GateDecision:
        """Resolve the decision bar for this tick.

        Args:
            df: The loop's prepared frame (tail = forming/tick bar).
            buffer_frontier: ``KlineBuffer.last_closed_bar_time`` when a WS
                buffer is active, else None. Frames sourced from REST still
                get closed-bar evidence from their own shape (a bar with a
                successor row is closed by construction).

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
            bar_closed = frontier is not None and tail_ts <= frontier
            return GateDecision(
                evaluate=True, index=tail_pos, bar_time=tail_ts, bar_closed=bar_closed
            )

        if frontier is None:
            # No bar in this frame is provably closed — nothing to decide on.
            return GateDecision(evaluate=False, index=tail_pos, bar_time=tail_ts, bar_closed=False)

        # Newest frame position at or below the frontier (the frontier bar
        # itself may have been dropped by essential-column dropna upstream).
        pos = int(df.index.searchsorted(frontier, side="right")) - 1
        if pos < 0:
            return GateDecision(evaluate=False, index=tail_pos, bar_time=tail_ts, bar_closed=False)

        bar_time = df.index[pos]
        already_evaluated = (
            self._last_evaluated_bar is not None and bar_time <= self._last_evaluated_bar
        )
        return GateDecision(
            evaluate=not already_evaluated, index=pos, bar_time=bar_time, bar_closed=True
        )

    def mark_evaluated(self, bar_time: Any) -> None:
        """Record that ``bar_time`` received its one signal evaluation."""
        self._last_evaluated_bar = bar_time

    @staticmethod
    def _closed_frontier(df: pd.DataFrame, buffer_frontier: Any) -> Any:
        """Newest bar time in ``df`` that is provably closed, or None.

        Evidence, merged by max:
        - the second-to-last row is closed by construction (a successor row
          exists) — works for REST-sourced frames with no ``x`` flags;
        - the buffer frontier is authoritative (Binance kline ``x: true``),
          clamped to the frame's tail: a frontier at or beyond the tail
          proves the tail itself closed.
        """
        candidates = []
        if len(df) >= 2:
            candidates.append(df.index[-2])
        if buffer_frontier is not None:
            try:
                candidates.append(min(buffer_frontier, df.index[-1]))
            except TypeError:
                # Incomparable timestamp types (mixed tz-awareness after a
                # degraded REST fallback) — fall back to frame-shape evidence.
                logger.debug(
                    "Closed-candle gate: buffer frontier %r not comparable to frame index",
                    buffer_frontier,
                )
        return max(candidates) if candidates else None


def stamp_decision_signal(
    decision: Any, *, bar_time: Any, bar_closed: bool, timeframe: str | None
) -> None:
    """Stamp Signal.metadata with the decision bar's identity — in BOTH modes.

    Lets the staging(ON)/prod(OFF) A/B and the parity ledger attribute every
    decision to the bar it was made on (plan §P1.0 observability). Keys:
    ``decision_bar_open_time``, ``decision_bar_close_time`` (open + interval;
    None when the timeframe is unknown), ``decision_bar_closed``.

    No-op when the decision, its signal, or its metadata dict is absent.
    """
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
