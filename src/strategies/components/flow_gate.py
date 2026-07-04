"""ETF net-flow entry gate (#803).

Blocks NEW LONG entries while US spot ETF net flows are in a sustained-outflow
regime (5-day net-flow z-score below a configurable threshold). Implemented as a
``SignalGenerator`` decorator so it composes with any base generator and applies
identically in the backtest and live engines (both run the strategy's
``process_candle`` → the wrapped generator), with no per-engine wiring.

The gate is rule-based and works today (it reads flows from the
``ETFFlowProvider`` directly). It is independent of the ETF-flow *feature
extractor*, which feeds a model and is inert until a compatible model is
retrained.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pandas as pd

from src.config.constants import (
    DEFAULT_ETF_FLOW_LONG_BLOCK_ZSCORE,
    DEFAULT_ETF_FLOW_ZSCORE_LONG_WINDOW,
    DEFAULT_ETF_FLOW_ZSCORE_SHORT_WINDOW,
)
from src.config.feature_flags import is_enabled
from src.data_providers.etf_flow_provider import (
    BTC_FLOW_COL,
    ETFFlowProvider,
    compute_flow_features,
)
from src.strategies.components.signal_generator import Signal, SignalDirection, SignalGenerator

if TYPE_CHECKING:
    from src.strategies.components.regime_context import RegimeContext

logger = logging.getLogger(__name__)

# Wide fetch window (once) so per-candle feature computation is in-memory.
_HISTORY_START = datetime(2020, 1, 1, tzinfo=UTC)
_HISTORY_END = datetime(2035, 1, 1, tzinfo=UTC)


class ETFFlowGate:
    """Decides whether to block a new long given ETF net-flow conditions."""

    def __init__(
        self,
        provider: ETFFlowProvider | None = None,
        *,
        block_zscore: float = DEFAULT_ETF_FLOW_LONG_BLOCK_ZSCORE,
        short_window: int = DEFAULT_ETF_FLOW_ZSCORE_SHORT_WINDOW,
        long_window: int = DEFAULT_ETF_FLOW_ZSCORE_LONG_WINDOW,
        asset_col: str = BTC_FLOW_COL,
    ) -> None:
        self._provider = provider if provider is not None else ETFFlowProvider()
        self._block_zscore = float(block_zscore)
        self._short_window = short_window
        self._long_window = long_window
        self._asset_col = asset_col
        self._flows: pd.DataFrame | None = None

    def _ensure_flows(self, as_of: datetime) -> None:
        # Fetch the whole available series once, then compute features in-memory.
        if self._flows is None:
            self._flows = self._provider.get_flows(_HISTORY_START, _HISTORY_END)

    def should_block_long(self, as_of: datetime) -> tuple[bool, str | None]:
        """Return ``(block, reason)`` for a proposed long entry at ``as_of``.

        Unknown flow (insufficient history / no data) does NOT block — the gate
        only acts on a confirmed outflow regime, so missing data degrades to
        "allow" and logs rather than halting all longs.
        """
        self._ensure_flows(as_of)
        feats = compute_flow_features(
            self._flows,
            as_of,
            asset_col=self._asset_col,
            short_window=self._short_window,
            long_window=self._long_window,
        )
        z5 = feats.get("netflow_zscore_5d")
        if z5 is None:
            return False, None
        if z5 < self._block_zscore:
            return True, f"etf_flow_outflow_z5_{z5:.2f}"
        return False, None


class FlowGatedSignalGenerator(SignalGenerator):
    """Wraps a signal generator and vetoes BUY signals during ETF outflows.

    SELL/HOLD signals pass through unchanged (the gate only blocks *new longs*,
    matching #803). If the frame has no usable timestamp the signal passes
    through (the gate cannot evaluate flow without a date).
    """

    def __init__(self, base: SignalGenerator, gate: ETFFlowGate | None = None) -> None:
        super().__init__(name=f"flow_gated({getattr(base, 'name', 'signal')})")
        self._base = base
        self._gate = gate if gate is not None else ETFFlowGate()

    def generate_signal(
        self, df: pd.DataFrame, index: int, regime: RegimeContext | None = None
    ) -> Signal:
        signal = self._base.generate_signal(df, index, regime)
        if signal.direction != SignalDirection.BUY:
            return signal
        as_of = _timestamp_at(df, index)
        if as_of is None:
            return signal
        block, reason = self._gate.should_block_long(as_of)
        if not block:
            return signal
        metadata = dict(signal.metadata)
        metadata["flow_gate_blocked"] = reason
        metadata["flow_gate_original_direction"] = signal.direction.value
        logger.debug("ETF flow gate blocked long at %s: %s", as_of, reason)
        return Signal(
            direction=SignalDirection.HOLD,
            strength=0.0,
            confidence=signal.confidence,
            metadata=metadata,
        )

    def get_confidence(self, df: pd.DataFrame, index: int) -> float:
        return self._base.get_confidence(df, index)

    @property
    def warmup_period(self) -> int:
        return getattr(self._base, "warmup_period", 0)

    def get_feature_generators(self):
        getter = getattr(self._base, "get_feature_generators", None)
        return getter() if callable(getter) else []


def _timestamp_at(df: pd.DataFrame, index: int) -> datetime | None:
    """Best-effort UTC timestamp for candle ``index`` (None if unavailable)."""
    try:
        if isinstance(df.index, pd.DatetimeIndex):
            ts = df.index[index]
        elif "timestamp" in df.columns:
            ts = pd.Timestamp(df["timestamp"].iloc[index])
        else:
            return None
    except (IndexError, KeyError, ValueError):
        return None
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.to_pydatetime()


FEATURE_FLAG = "enable_etf_flow_gate"


def maybe_wrap_with_flow_gate(
    signal_generator: SignalGenerator, gate: ETFFlowGate | None = None
) -> SignalGenerator:
    """Wrap ``signal_generator`` with the flow gate iff the feature flag is on.

    Resolves the flag ONCE (at strategy-build time) so there is no per-candle
    feature-flag I/O; a flag change requires a restart, which rebuilds the
    strategy. Returns the generator unchanged when the flag is off (zero cost).
    """
    if is_enabled(FEATURE_FLAG, default=False):
        return FlowGatedSignalGenerator(signal_generator, gate)
    return signal_generator


__all__ = ["ETFFlowGate", "FlowGatedSignalGenerator", "maybe_wrap_with_flow_gate"]
