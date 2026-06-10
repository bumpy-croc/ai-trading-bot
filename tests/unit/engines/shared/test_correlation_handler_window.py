"""Regression tests for #759: correlation peer-symbol history windows.

The no-window fallback called ``get_historical_data(sym, timeframe=...)``
without the **required** ``start`` argument — every call raised ``TypeError``,
was swallowed, and the peer symbol silently dropped out of correlation
exposure. Peers are now fetched with the default window when the configured
window computation fails, and skipped **loudly** when no time window is
derivable at all (non-datetime index — fabricating a wall-clock window there
would be lookahead in a backtest).
"""

import logging
from unittest.mock import Mock, create_autospec

import pandas as pd
import pytest

from src.config.constants import DEFAULT_CORRELATION_WINDOW_DAYS
from src.engines.shared.correlation_handler import CorrelationHandler

pytestmark = [pytest.mark.unit, pytest.mark.fast]


def _datetime_df(periods: int = 10) -> pd.DataFrame:
    """Primary-symbol frame with a datetime index (windows derivable)."""
    idx = pd.date_range("2024-01-10", periods=periods, freq="1h")
    return pd.DataFrame({"close": [100.0 + i for i in range(periods)]}, index=idx)


def _int_index_df(periods: int = 10) -> pd.DataFrame:
    """Primary-symbol frame with an integer index (no derivable window)."""
    return pd.DataFrame({"close": [100.0 + i for i in range(periods)]})


def _peer_history() -> pd.DataFrame:
    """History frame the mocked provider returns for peer symbols."""
    idx = pd.date_range("2024-01-01", periods=10, freq="1h")
    return pd.DataFrame({"close": [50.0 + i for i in range(10)]}, index=idx)


def _make_handler(correlation_window_days=30) -> tuple[CorrelationHandler, Mock]:
    """Handler with autospecced collaborators and a stubbed history provider."""
    from src.data_providers.data_provider import DataProvider
    from src.position_management.correlation_engine import CorrelationEngine
    from src.risk.risk_manager import RiskManager, RiskParameters

    risk_manager = create_autospec(RiskManager, instance=True)
    # params is created in __init__, invisible to autospec — attach a real one
    risk_manager.params = RiskParameters()
    risk_manager.params.correlation_window_days = correlation_window_days
    data_provider = create_autospec(DataProvider, instance=True)
    data_provider.get_historical_data.return_value = _peer_history()
    handler = CorrelationHandler(
        correlation_engine=create_autospec(CorrelationEngine, instance=True),
        risk_manager=risk_manager,
        data_provider=data_provider,
        strategy=None,
    )
    return handler, data_provider


class TestCorrelationWindowFallback:
    def test_configured_window_passes_start_and_end(self):
        """Happy path: peer fetched with the configured window bounds."""
        handler, provider = _make_handler(correlation_window_days=7)
        df = _datetime_df()

        series = handler._build_price_series(
            symbol="BTCUSDT",
            timeframe="1h",
            df=df,
            index=5,
            positions_snapshot={"ETHUSDT": {"size": 0.1}},
        )

        assert "ETHUSDT" in series
        kwargs = provider.get_historical_data.call_args.kwargs
        end = pd.Timestamp(kwargs["end"])
        start = pd.Timestamp(kwargs["start"])
        assert end == df.index[5]
        assert (end - start) == pd.Timedelta(days=7)

    def test_failed_window_computation_falls_back_to_default(self):
        """Regression: a broken correlation_window_days no longer drops the
        peer — the default window is used and the provider IS called with a
        valid start."""
        handler, provider = _make_handler(correlation_window_days=object())
        df = _datetime_df()

        series = handler._build_price_series(
            symbol="BTCUSDT",
            timeframe="1h",
            df=df,
            index=5,
            positions_snapshot={"ETHUSDT": {"size": 0.1}},
        )

        assert "ETHUSDT" in series
        kwargs = provider.get_historical_data.call_args.kwargs
        end = pd.Timestamp(kwargs["end"])
        start = pd.Timestamp(kwargs["start"])
        assert (end - start) == pd.Timedelta(days=DEFAULT_CORRELATION_WINDOW_DAYS)

    def test_non_datetime_index_skips_peers_loudly(self, caplog):
        """No derivable window (integer index): peers are skipped with a
        WARNING — never fetched with a fabricated wall-clock window."""
        handler, provider = _make_handler()
        df = _int_index_df()

        with caplog.at_level(logging.WARNING):
            series = handler._build_price_series(
                symbol="BTCUSDT",
                timeframe="1h",
                df=df,
                index=5,
                positions_snapshot={"ETHUSDT": {"size": 0.1}},
            )

        assert "BTCUSDT" in series
        assert "ETHUSDT" not in series
        provider.get_historical_data.assert_not_called()
        assert "Correlation control degraded" in caplog.text
        assert "ETHUSDT" in caplog.text

    def test_no_peers_no_warning(self, caplog):
        """Without peer positions there is nothing to fetch and no noise."""
        handler, provider = _make_handler()
        df = _int_index_df()

        with caplog.at_level(logging.WARNING):
            series = handler._build_price_series(
                symbol="BTCUSDT",
                timeframe="1h",
                df=df,
                index=5,
                positions_snapshot={"BTCUSDT": {"size": 0.1}},
            )

        assert "BTCUSDT" in series
        provider.get_historical_data.assert_not_called()
        assert "Correlation control degraded" not in caplog.text

    def test_provider_failure_still_skips_with_warning(self, caplog):
        """A genuinely failing provider degrades loudly per peer."""
        handler, provider = _make_handler()
        provider.get_historical_data.side_effect = ConnectionError("api down")
        df = _datetime_df()

        with caplog.at_level(logging.WARNING):
            series = handler._build_price_series(
                symbol="BTCUSDT",
                timeframe="1h",
                df=df,
                index=5,
                positions_snapshot={"ETHUSDT": {"size": 0.1}},
            )

        assert "ETHUSDT" not in series
        assert "Failed to fetch correlation history" in caplog.text
