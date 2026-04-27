"""Live engine sentiment merge parity test.

Backtest backfills historical sentiment over the full dataframe via
``SentimentDataProvider.get_historical_sentiment`` + ``aggregate_sentiment`` +
``ffill().fillna(0)`` (src/engines/backtest/engine.py:957-964). Live used to
ONLY apply ``get_live_sentiment`` to the most recent 4 hours of the buffer,
leaving older bars at 0.0. ML strategies that consume a sequence_length
window (commonly 60-240 hours) saw materially different inputs in live vs
backtest.

The fix in ``LiveTradingEngine._add_sentiment_data`` adds a historical
backfill pass before the live overlay, so older bars carry real sentiment
values just like backtest.
"""

from __future__ import annotations

from datetime import UTC
from unittest.mock import Mock

import pandas as pd
import pytest

from src.engines.live.trading_engine import LiveTradingEngine
from tests.mocks import MockDatabaseManager


@pytest.fixture(autouse=True)
def mock_database_manager(monkeypatch):
    original_init = MockDatabaseManager.__init__

    def patched_init(self, database_url=None):
        original_init(self, database_url)
        self._fallback_balance = 10_000.0

    monkeypatch.setattr(MockDatabaseManager, "__init__", patched_init)
    monkeypatch.setattr("src.engines.live.trading_engine.DatabaseManager", MockDatabaseManager)


def _make_buffer_df(periods: int = 24) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=periods, freq="1h", tz=UTC)
    return pd.DataFrame(
        {
            "open": [100.0] * periods,
            "high": [101.0] * periods,
            "low": [99.0] * periods,
            "close": [100.5] * periods,
            "volume": [1000.0] * periods,
        },
        index=idx,
    )


def _make_engine(sentiment_provider) -> LiveTradingEngine:
    strategy = Mock()
    strategy.get_risk_overrides.return_value = None
    strategy.name = "test"
    data_provider = Mock()
    data_provider.get_current_price.return_value = 100.0
    engine = LiveTradingEngine(
        strategy=strategy,
        data_provider=data_provider,
        sentiment_provider=sentiment_provider,
        initial_balance=10_000.0,
        enable_live_trading=False,
        log_trades=False,
        fee_rate=0.0,
        slippage_rate=0.0,
    )
    engine.timeframe = "1h"
    return engine


@pytest.mark.fast
class TestLiveSentimentMergeParity:
    """Live must backfill historical sentiment so older bars match backtest."""

    def test_historical_sentiment_populates_full_buffer(self) -> None:
        df = _make_buffer_df(periods=24)
        sentiment_idx = pd.date_range("2024-01-01", periods=24, freq="1h", tz=UTC)
        historical = pd.DataFrame({"sentiment_score": [0.5] * 24}, index=sentiment_idx)

        provider = Mock()
        provider.get_historical_sentiment.return_value = historical
        provider.aggregate_sentiment.return_value = historical
        provider.get_live_sentiment.return_value = {"sentiment_score": 0.9}

        engine = _make_engine(provider)
        out = engine._add_sentiment_data(df, "TEST")

        assert "sentiment_score" in out.columns
        # Every bar must have a sentiment value (no zero defaults left over
        # from the old recent-only path).
        assert (out["sentiment_score"] != 0.0).all()
        # Recent 4 hours overlay with the live snapshot value.
        recent_mask = out.index >= (out.index.max() - pd.Timedelta(hours=4))
        assert (out.loc[recent_mask, "sentiment_score"].sub(0.9).abs() < 1e-9).all()
        # Older bars keep the historical value.
        older_mask = ~recent_mask
        assert (out.loc[older_mask, "sentiment_score"].sub(0.5).abs() < 1e-9).all()

    def test_no_historical_provider_falls_back_to_live_only(self) -> None:
        """Provider without get_historical_sentiment must not crash; older
        bars stay zero (legacy behaviour) — the fix is non-destructive."""
        df = _make_buffer_df(periods=24)
        provider = Mock(spec=["get_live_sentiment"])
        provider.get_live_sentiment.return_value = {"sentiment_score": 0.9}

        engine = _make_engine(provider)
        out = engine._add_sentiment_data(df, "TEST")

        assert "sentiment_score" in out.columns
        recent_mask = out.index >= (out.index.max() - pd.Timedelta(hours=4))
        assert (out.loc[recent_mask, "sentiment_score"].sub(0.9).abs() < 1e-9).all()

    def test_historical_failure_does_not_crash_live_path(self) -> None:
        """If historical fetch raises, live overlay still applies — graceful
        degradation, no full crash."""
        df = _make_buffer_df(periods=24)
        provider = Mock()
        provider.get_historical_sentiment.side_effect = RuntimeError("upstream down")
        provider.get_live_sentiment.return_value = {"sentiment_score": 0.9}

        engine = _make_engine(provider)
        out = engine._add_sentiment_data(df, "TEST")

        assert "sentiment_score" in out.columns
        recent_mask = out.index >= (out.index.max() - pd.Timedelta(hours=4))
        assert (out.loc[recent_mask, "sentiment_score"].sub(0.9).abs() < 1e-9).all()

    def test_empty_df_handled_safely(self) -> None:
        provider = Mock()
        provider.get_historical_sentiment.return_value = pd.DataFrame()
        provider.get_live_sentiment.return_value = {"sentiment_score": 0.9}

        engine = _make_engine(provider)
        out = engine._add_sentiment_data(pd.DataFrame(), "TEST")
        assert isinstance(out, pd.DataFrame)
