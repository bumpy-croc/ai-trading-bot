"""Backtest must emit ``sentiment_freshness`` for parity with live.

Live's ``_add_sentiment_data`` (src/engines/live/trading_engine.py:2097-2098)
sets ``sentiment_freshness=0`` across the buffer and flips it to 1 only for
the most recent 4 hours. Backtest historically never created the column,
so feature schemas listing it (src/tech/adapters/row_extractors.py:35)
saw NaN/missing on backtest and a populated 0/1 flag on live.

The fix in ``Backtester._merge_sentiment_data`` always emits the column
with value 0 (historical sentiment is by definition stale). ML feature
shapes now line up; the runtime distribution may still differ by design,
but the column is guaranteed present on both engines.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import Mock

import pandas as pd
import pytest

from src.engines.backtest.engine import Backtester


def _make_engine(sentiment_provider) -> Backtester:
    strategy = Mock()
    strategy.name = "test"
    strategy.get_risk_overrides.return_value = None
    strategy.calculate_indicators = Mock(return_value=None)
    data_provider = Mock()
    return Backtester(
        strategy=strategy,
        data_provider=data_provider,
        sentiment_provider=sentiment_provider,
        initial_balance=10_000.0,
    )


def _make_price_df(periods: int = 24) -> pd.DataFrame:
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


@pytest.mark.fast
class TestBacktestSentimentFreshnessParity:
    """Backtest's sentiment merge must emit a sentiment_freshness column."""

    def test_freshness_column_emitted_when_provider_returns_data(self) -> None:
        df = _make_price_df(periods=12)
        sentiment_idx = pd.date_range("2024-01-01", periods=12, freq="1h", tz=UTC)
        historical = pd.DataFrame({"sentiment_score": [0.5] * 12}, index=sentiment_idx)

        provider = Mock()
        provider.get_historical_sentiment.return_value = historical
        provider.aggregate_sentiment.return_value = historical

        engine = _make_engine(provider)
        out = engine._merge_sentiment_data(
            df,
            "TEST",
            "1h",
            datetime(2024, 1, 1, tzinfo=UTC),
            datetime(2024, 1, 1, 12, tzinfo=UTC),
        )

        assert "sentiment_freshness" in out.columns, (
            "backtest must emit sentiment_freshness so feature schemas line "
            "up with live (row_extractors.py expects this column)"
        )
        # Historical sentiment is always treated as stale.
        assert (out["sentiment_freshness"] == 0).all()

    def test_freshness_column_emitted_when_provider_returns_empty(self) -> None:
        """Even when the provider returns no rows, the column must still
        be present so downstream feature builders never see a missing
        column on backtest while live populates it."""
        df = _make_price_df(periods=6)

        provider = Mock()
        provider.get_historical_sentiment.return_value = pd.DataFrame()

        engine = _make_engine(provider)
        out = engine._merge_sentiment_data(
            df,
            "TEST",
            "1h",
            datetime(2024, 1, 1, tzinfo=UTC),
            datetime(2024, 1, 1, 6, tzinfo=UTC),
        )

        assert "sentiment_freshness" in out.columns
        assert (out["sentiment_freshness"] == 0).all()

    def test_freshness_column_not_clobbered_when_already_present(self) -> None:
        """If a future flow pre-populates the freshness column (e.g. a
        custom warmup step), the merge must not overwrite it."""
        df = _make_price_df(periods=6)
        df["sentiment_freshness"] = 1  # caller-set marker

        provider = Mock()
        provider.get_historical_sentiment.return_value = pd.DataFrame()

        engine = _make_engine(provider)
        out = engine._merge_sentiment_data(
            df,
            "TEST",
            "1h",
            datetime(2024, 1, 1, tzinfo=UTC),
            datetime(2024, 1, 1, 6, tzinfo=UTC),
        )

        # Pre-existing values preserved.
        assert (out["sentiment_freshness"] == 1).all()

    def test_no_provider_returns_df_unchanged(self) -> None:
        """When the engine has no sentiment provider configured, the merge
        is a no-op and the column is NOT introduced — matching the
        existing contract that callers control whether sentiment is
        modeled at all."""
        df = _make_price_df(periods=6)

        engine = _make_engine(sentiment_provider=None)
        out = engine._merge_sentiment_data(
            df,
            "TEST",
            "1h",
            datetime(2024, 1, 1, tzinfo=UTC),
            datetime(2024, 1, 1, 6, tzinfo=UTC),
        )

        assert "sentiment_freshness" not in out.columns
