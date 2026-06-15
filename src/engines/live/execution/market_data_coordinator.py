"""Live market-data & context acquisition (#486).

Owns the live engine's per-candle read path: fetching the latest frame
(``get_latest_data`` — WS-cache vs REST, resync handling), enriching it with
sentiment (``add_sentiment_data``), the strategy-context readiness gate
(``is_context_ready``), and building the correlation-sizing context
(``build_correlation_context``). These were moved verbatim out of
``LiveTradingEngine`` (a mechanical ``self.`` -> ``state.`` rewrite against an
engine backref); the engine keeps thin delegating wrappers.

This is the read path — no order placement or balance mutation. It runs on the
trading-loop thread (and ``build_correlation_context`` is also invoked by the
``StrategyRuntimeCoordinator`` via the engine wrapper). The coordinator holds no
state of its own; everything is read through the ``state`` backref at call time,
so behaviour and freshness/resync semantics are unchanged. ``is_context_ready``
defers freshness to the engine's ``_is_data_fresh`` via ``state``.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Protocol, cast

import pandas as pd

from src.data_providers.binance_provider import WebSocketState
from src.data_providers.sentiment_provider import SentimentDataProvider

if TYPE_CHECKING:
    from src.engines.live.execution.position_tracker import LivePositionTracker
    from src.risk.risk_manager import RiskManager

logger = logging.getLogger(__name__)


class LiveMarketDataEngineState(Protocol):
    """Engine state the market-data coordinator reads at call time.

    Read path only; ``get_latest_data`` writes ``last_data_update``. Accessed
    dynamically through this backref because providers/buffers/session are wired
    during construction and ``start()``.
    """

    enable_live_trading: bool
    timeframe: str | None
    last_data_update: datetime | None
    strategy: Any
    data_provider: Any
    sentiment_provider: Any
    correlation_engine: Any
    live_position_tracker: LivePositionTracker
    risk_manager: RiskManager
    _active_symbol: str | None
    _ws_kline_provider: Any
    _kline_buffer: Any

    # Stays on the engine; invoked via this backref so subclass/test overrides
    # on the engine still apply.
    def _is_data_fresh(self, df: pd.DataFrame) -> bool: ...


class LiveMarketDataCoordinator:
    """Owns the live engine's per-candle market-data + context read path."""

    def __init__(self, engine_state: LiveMarketDataEngineState) -> None:
        """Bind to the engine's live state (see protocol for the surface)."""
        self._state = engine_state

    def is_context_ready(self, df: pd.DataFrame) -> tuple[bool, str]:
        """Check if the current frame has enough context for strategy-driven decisions.

        Returns (ready, reason_if_not_ready).
        """
        state = self._state
        try:
            rows = len(df)
            # Required rows from ML sequence length (for ML strategies only)
            try:
                seq_len = int(getattr(state.strategy, "sequence_length", 0) or 0)
            except Exception:
                seq_len = 0
            # Do not assume a large indicator window by default; strategies can opt-in via attribute
            try:
                max_window_attr = getattr(state.strategy, "max_indicator_window", 0)
                max_window = int(max_window_attr or 0)
            except Exception:
                max_window = 0
            min_needed_base = max(seq_len, max_window)
            min_needed = (min_needed_base + 1) if min_needed_base > 0 else 2

            if rows < min_needed:
                return False, f"insufficient_rows:{rows}<min_needed:{min_needed}"

            # Current index must have valid essentials
            idx = rows - 1
            essentials = ["open", "high", "low", "close", "volume"]
            for col in essentials:
                try:
                    if pd.isna(df.iloc[idx][col]):
                        return False, f"nan_in_essentials:{col}"
                except Exception:
                    return False, f"missing_essential:{col}"

            # Strategy-specific readiness: prediction availability for ML strategies
            if seq_len > 0:
                if "onnx_pred" in df.columns:
                    try:
                        if pd.isna(df["onnx_pred"].iloc[idx]):
                            return False, "prediction_unavailable_at_current_index"
                    except Exception:
                        return False, "prediction_column_access_error"

            # Data freshness check
            if not state._is_data_fresh(df):
                return False, "stale_data"

            return True, ""
        except Exception as e:
            logger.debug("Context readiness check failed: %s", e)
            return False, "readiness_check_error"

    def get_latest_data(self, symbol: str, timeframe: str) -> pd.DataFrame | None:
        """Fetch latest market data — from WS cache or REST.

        During RESYNCING in live mode, returns None to freeze trading.
        Paper mode falls back to REST immediately during RESYNCING.
        """
        state = self._state
        try:
            # During resync in live mode, return None to trigger skip-cycle.
            # Paper mode falls back to REST immediately (no exchange-side SL).
            if (
                state.enable_live_trading
                and state._ws_kline_provider
                and getattr(state._ws_kline_provider, "_kline_ws_state", None)
                == WebSocketState.RESYNCING
            ):
                logger.info("WebSocket resyncing — skipping data fetch")
                return None

            # If kline buffer detected a gap, trigger REST resync
            if state._kline_buffer and state._kline_buffer.needs_resync:
                logger.info("KlineBuffer gap detected — resyncing from REST")
                state._kline_buffer.resync_from_rest(
                    state.data_provider,
                    state._active_symbol or symbol,
                    state.timeframe or timeframe,
                )

            # Use WS cache if available and healthy
            if (
                state._kline_buffer
                and state._kline_buffer.is_fresh
                and state._ws_kline_provider
                and getattr(state._ws_kline_provider, "ws_healthy", False)
            ):
                return state._kline_buffer.get_dataframe()

            # Fallback to REST (existing behavior)
            df = state.data_provider.get_live_data(symbol, timeframe, limit=500)
            state.last_data_update = datetime.now(UTC)
            return df
        except Exception as e:
            logger.error("Failed to fetch market data: %s", e, exc_info=True)
            return None

    def add_sentiment_data(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Add sentiment data to price data.

        Backfills the entire buffer with historical sentiment first
        (parity with backtest's full-history merge at
        src/engines/backtest/engine.py:957-964), then layers the live
        snapshot on top of recent candles. Without the historical pass,
        bars older than 4 hours in the live buffer carried 0.0 sentiment
        while backtest had populated values — so ML strategies that
        consume a sequence_length window saw materially different inputs
        between the two engines.
        """
        state = self._state
        # The trading loop calls this only when sentiment_provider is set.
        sentiment_provider = cast(SentimentDataProvider, state.sentiment_provider)
        try:
            # Step 1: backfill historical sentiment over the buffer if the
            # provider supports it. Mirrors backtest's `_merge_sentiment_data`
            # join + ffill so older candles carry real sentiment values.
            if hasattr(sentiment_provider, "get_historical_sentiment") and not df.empty:
                try:
                    start = df.index.min().to_pydatetime()
                    end = df.index.max().to_pydatetime()
                    sentiment_df = sentiment_provider.get_historical_sentiment(symbol, start, end)
                    if sentiment_df is not None and not sentiment_df.empty:
                        # Aggregate when the provider supports it. Fall back
                        # to "1h" when state.timeframe is not yet set (e.g.
                        # warmup paths) so the join shape is well-defined
                        # rather than producing NaN-padded raw rows that
                        # silently diverge from backtest.
                        if hasattr(sentiment_provider, "aggregate_sentiment"):
                            sentiment_df = sentiment_provider.aggregate_sentiment(
                                sentiment_df, window=state.timeframe or "1h"
                            )
                        # Restrict the merge to the sentiment namespace and
                        # drop any pre-existing sentiment columns from the
                        # local df so we never produce ``sentiment_score_x``/
                        # ``_y`` collisions when the buffer was already
                        # enriched on a prior call (e.g. retained
                        # ``_kline_buffer``). Filtering BOTH sides on the
                        # ``sentiment*`` prefix means: (a) OHLCV / indicator
                        # columns on ``df`` survive even if a future
                        # provider's frame happens to expose a same-named
                        # column like ``volume`` or ``close``; (b) the join
                        # never raises pandas' default-overlap error.
                        sentiment_only_cols = [
                            c for c in sentiment_df.columns if c.startswith("sentiment_")
                        ]
                        if not sentiment_only_cols:
                            # Provider returned no ``sentiment_*`` columns
                            # at all (e.g. misconfigured aggregator that
                            # emits only OHLCV-style columns). Skip the
                            # merge entirely rather than letting the
                            # provider's residual column names leak into
                            # ``collision_cols`` and silently strip OHLCV
                            # from ``df``.
                            logger.debug(
                                "Historical sentiment provider returned no "
                                "sentiment_* columns for %s — skipping merge",
                                symbol,
                            )
                        else:
                            sentiment_df = sentiment_df[sentiment_only_cols]
                            collision_cols = [c for c in df.columns if c in sentiment_df.columns]
                            if collision_cols:
                                df = df.drop(columns=collision_cols)
                            df = df.join(sentiment_df, how="left")
                            if "sentiment_score" in df.columns:
                                df["sentiment_score"] = df["sentiment_score"].ffill().fillna(0)
                except Exception as e:
                    logger.warning(
                        "Historical sentiment backfill failed for %s: %s — "
                        "continuing with live-only sentiment which may differ from backtest",
                        symbol,
                        e,
                    )

            # Step 2: overlay the latest real-time sentiment snapshot on
            # the most recent 4 hours of candles. The historical backfill
            # has already populated older rows with the right values.
            # The 4h window matches the live sentiment provider's
            # freshness contract — bars older than that rely on the
            # historical backfill above.
            if hasattr(sentiment_provider, "get_live_sentiment"):
                live_sentiment = sentiment_provider.get_live_sentiment()
                if live_sentiment and not df.empty:
                    # 4h: live sentiment freshness window (see step-2 comment).
                    recent_mask = df.index >= (df.index.max() - pd.Timedelta(hours=4))
                    for feature, value in live_sentiment.items():
                        if feature not in df.columns:
                            df[feature] = 0.0
                        df.loc[recent_mask, feature] = value
                    if "sentiment_freshness" not in df.columns:
                        df["sentiment_freshness"] = 0
                    df.loc[recent_mask, "sentiment_freshness"] = 1
                    logger.debug("Applied live sentiment to %s recent candles", recent_mask.sum())
            else:
                logger.debug("Using historical sentiment data only (no live provider)")

        except Exception as e:
            logger.error("Failed to add sentiment data: %s", e, exc_info=True)

        return df

    def build_correlation_context(
        self, symbol: str, df: pd.DataFrame, overrides: dict | None
    ) -> dict | None:
        """
        Build correlation context dict for risk manager sizing, including corr matrix and optional exposure override.
        Returns None if correlation engine is unavailable or an error occurs.
        """
        state = self._state
        try:
            if state.correlation_engine is None:
                return None
            # Build price series for candidate + currently open symbols
            symbols_to_check = set([symbol]) | set(
                p.symbol for p in state.live_position_tracker.positions.values()
            )
            price_series: dict[str, pd.Series] = {}
            end_ts = df.index[-1] if len(df) > 0 else None
            start_ts = (
                end_ts - pd.Timedelta(days=state.risk_manager.params.correlation_window_days)
                if end_ts is not None
                else None
            )
            if symbol:
                try:
                    price_series[str(symbol)] = df["close"].copy()
                except Exception as e:
                    logger.warning(
                        "Failed to seed candidate price series for %s from current "
                        "frame: %s — correlation sizing degrades to history-only.",
                        symbol,
                        e,
                    )
            for sym in symbols_to_check:
                s = str(sym)
                if s in price_series:
                    continue
                try:
                    if start_ts is not None and end_ts is not None:
                        # Use the strategy's actual trading timeframe instead of hardcoding "1h"
                        trading_timeframe = state.timeframe or "1h"  # Fallback to "1h" if not set
                        hist = state.data_provider.get_historical_data(
                            s,
                            timeframe=trading_timeframe,
                            start=start_ts.to_pydatetime(),
                            end=end_ts.to_pydatetime(),
                        )
                        if not hist.empty and "close" in hist:
                            price_series[s] = hist["close"]
                except Exception as e:
                    # Best-effort per-symbol history fetch inside a per-candle
                    # loop; transient provider errors are expected, so log at
                    # debug to avoid flooding while keeping the trace visible.
                    logger.debug(
                        "Skipping %s in correlation matrix — history fetch failed: %s",
                        s,
                        e,
                    )
                    continue
            corr_matrix = state.correlation_engine.calculate_position_correlations(price_series)
            return {
                "engine": state.correlation_engine,
                "candidate_symbol": symbol,
                "corr_matrix": corr_matrix,
                "max_exposure_override": (
                    overrides.get("correlation_control", {}).get("max_correlated_exposure")
                    if overrides
                    else None
                ),
            }
        except Exception:
            return None
