"""Shadow observability for config-suppressed short entries (GH #1020).

When a long-only deployment (``allow_shorts=False``) withholds a short
ENTRY at signal generation, the live engine records a DB-durable
"would-have-entered-short" ``system_events`` row so the counterfactual the
config was ratified on stays measurable post-ship (risk-review condition C6
on proposal 2026-07-12-01).

Rows are bounded by the same episode-dedup pattern as the short-guard
rejection events (#1016, ``execution_engine.py``): within one contiguous
suppression episode per symbol, write the first suppression and every
``SHORT_SUPPRESSION_EMIT_EVERY_N``-th one, plus an episode-end summary
carrying the TRUE total after an inactivity gap.

Live-path only: this monitor is constructed exclusively by
``LiveEntryCoordinator`` — the backtest engine never touches it, so
backtests write no events. Entirely fault-isolated: bookkeeping or DB
failures can never affect the trading decision, which was already made
upstream at signal generation.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from src.database.models import EventType

logger = logging.getLogger(__name__)

# Suppressions separated by more than this quiet gap belong to different
# episodes — same rationale as the short-guard value (#1016): live decision
# cycles run seconds-to-minutes apart, and two hours spans any configured
# candle interval without merging distinct episodes.
SHORT_SUPPRESSION_EPISODE_GAP_SECONDS = 2 * 3600.0
# Within an episode, write the first suppression and every Nth after it so a
# long SELL streak stays queryable without one system_events row per cycle.
SHORT_SUPPRESSION_EMIT_EVERY_N = 10

# Signal-metadata keys copied onto the event row. A whitelist keeps the JSON
# payload bounded and serializable regardless of what generators stuff into
# metadata.
_SIGNAL_METADATA_WHITELIST = (
    "prediction",
    "current_price",
    "predicted_return",
    "long_entry_threshold",
    "short_entry_threshold",
    "engine_model_name",
    "model_type",
    "model_timeframe",
    "trading_symbol",
    "model_symbol",
)


@dataclass
class _SuppressionEpisode:
    """One contiguous run of suppressed short entries for a symbol."""

    started_at: datetime
    last_suppressed_at: datetime
    last_suppressed_monotonic: float
    suppression_count: int


class ShortSuppressionMonitor:
    """Episode-deduped writer of would-have-entered-short system events."""

    def __init__(self) -> None:
        # The lock guards the episode dict only; DB writes happen outside it
        # so a slow insert cannot serialize suppressions on other symbols.
        self._lock = threading.Lock()
        self._episodes: dict[str, _SuppressionEpisode] = {}
        # Injectable clock for deterministic episode-gap tests.
        self._monotonic: Callable[[], float] = time.monotonic

    def record_suppression(
        self,
        symbol: str,
        *,
        db_manager: Any,
        session_id: int | None,
        price: float,
        position_size_notional: float,
        signal_strength: float,
        signal_confidence: float,
        signal_metadata: Mapping[str, Any] | None,
    ) -> None:
        """Record one suppressed short entry; write a bounded event row.

        Fault-isolated: any failure degrades to a warning log. Callers pass
        ``db_manager``/``session_id`` at call time (they are wired on the
        engine after construction); without both, bookkeeping still runs but
        nothing is written.
        """
        try:
            now_monotonic = self._monotonic()
            now_utc = datetime.now(UTC)
            ended: _SuppressionEpisode | None = None
            with self._lock:
                episode = self._episodes.get(symbol)
                if (
                    episode is not None
                    and now_monotonic - episode.last_suppressed_monotonic
                    > SHORT_SUPPRESSION_EPISODE_GAP_SECONDS
                ):
                    ended = episode
                    episode = None
                if episode is None:
                    episode = _SuppressionEpisode(
                        started_at=now_utc,
                        last_suppressed_at=now_utc,
                        last_suppressed_monotonic=now_monotonic,
                        suppression_count=1,
                    )
                    self._episodes[symbol] = episode
                    emit = True
                else:
                    episode.suppression_count += 1
                    episode.last_suppressed_at = now_utc
                    episode.last_suppressed_monotonic = now_monotonic
                    emit = episode.suppression_count % SHORT_SUPPRESSION_EMIT_EVERY_N == 0
                suppression_count = episode.suppression_count
                episode_started_at = episode.started_at

            if ended is not None:
                self._write_episode_end(symbol, ended, db_manager=db_manager, session_id=session_id)
            if not emit:
                return

            details: dict[str, Any] = {
                "symbol": symbol,
                "side": "short",
                "reason": "allow_shorts_false",
                "price": float(price),
                "position_size_notional": float(position_size_notional),
                "signal": {
                    "strength": float(signal_strength),
                    "confidence": float(signal_confidence),
                    **_whitelisted_signal_metadata(signal_metadata),
                },
                "episode": {
                    "started_at": episode_started_at.isoformat(),
                    "suppression_count": suppression_count,
                },
            }
            message = (
                f"Short entry suppressed by long-only config for {symbol} "
                f"(would have entered short): suppression {suppression_count} "
                f"of episode started {episode_started_at.isoformat()}"
            )
            self._write_event(
                db_manager,
                session_id,
                message=message,
                error_code="WOULD_ENTER_SHORT",
                details=details,
            )
        except Exception as e:
            logger.warning("Failed to record short suppression for %s: %s", symbol, e)

    def _write_episode_end(
        self,
        symbol: str,
        episode: _SuppressionEpisode,
        *,
        db_manager: Any,
        session_id: int | None,
    ) -> None:
        """Write the episode summary row carrying the TRUE suppression total.

        Per-suppression rows are sampled (every Nth), so counterfactual
        counts come from ``suppressions_total`` here.
        """
        duration_s = max(0.0, (episode.last_suppressed_at - episode.started_at).total_seconds())
        details = {
            "symbol": symbol,
            "end_reason": "inactivity_gap",
            "suppressions_total": episode.suppression_count,
            "episode_started_at": episode.started_at.isoformat(),
            "episode_last_suppression_at": episode.last_suppressed_at.isoformat(),
            "episode_duration_seconds": duration_s,
        }
        message = (
            f"Short-suppression episode ended for {symbol} (inactivity_gap): "
            f"{episode.suppression_count} suppression(s) over {duration_s / 60.0:.1f} min"
        )
        self._write_event(
            db_manager,
            session_id,
            message=message,
            error_code="SHORT_SUPPRESSION_EPISODE_END",
            details=details,
        )

    @staticmethod
    def _write_event(
        db_manager: Any,
        session_id: int | None,
        *,
        message: str,
        error_code: str,
        details: dict[str, Any],
    ) -> None:
        """Write one system_events row; silent no-op without an active session."""
        if not (db_manager and session_id):
            return
        db_manager.log_event(
            event_type=EventType.SHORT_ENTRY_SUPPRESSED,
            message=message,
            severity="info",
            component="entry_coordinator",
            details=details,
            session_id=session_id,
            error_code=error_code,
        )


def _whitelisted_signal_metadata(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    """Copy JSON-safe whitelisted keys from signal metadata; drop the rest."""
    if not metadata:
        return {}
    safe: dict[str, Any] = {}
    for key in _SIGNAL_METADATA_WHITELIST:
        if key not in metadata:
            continue
        value = metadata[key]
        if value is None or isinstance(value, str | bool):
            safe[key] = value
        elif isinstance(value, int | float):
            safe[key] = float(value)
    return safe


__all__ = [
    "SHORT_SUPPRESSION_EMIT_EVERY_N",
    "SHORT_SUPPRESSION_EPISODE_GAP_SECONDS",
    "ShortSuppressionMonitor",
]
