"""Shared fixtures for live-engine unit tests."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def make_live_engine():
    """Factory for a LiveTradingEngine with mocked DB/exchange boundaries.

    The DB mock resumes the session balance and serves the account_history
    session peak the drawdown guard seeds from; everything else stays real so
    tests exercise the genuine wiring between the loop, the guards, and the
    entry/exit coordinators.
    """

    def _make(
        *,
        db_peak: float | None = 100.0,
        resumed_balance: float = 100.0,
        session_id: int = 20,
        **engine_kwargs,
    ):
        from src.engines.live.trading_engine import LiveTradingEngine

        db = MagicMock()
        db.get_active_session_id.return_value = session_id
        db.get_current_balance.return_value = resumed_balance
        db.get_session_peak_balance.return_value = db_peak
        with (
            patch("src.engines.live.trading_engine.DatabaseManager", return_value=db),
            patch("src.engines.live.trading_engine.get_config", return_value={}),
            patch(
                "src.engines.live.trading_engine._create_exchange_provider",
                return_value=(MagicMock(), "mock"),
            ),
        ):
            # enable_live_trading=True so _resume_balance_from_snapshot runs
            # (balance recovery is live-mode only); the exchange is a mock and
            # these tests never reach order execution.
            engine = LiveTradingEngine(
                strategy=MagicMock(),
                data_provider=MagicMock(),
                initial_balance=100.0,
                enable_live_trading=True,
                resume_from_last_balance=True,
                **engine_kwargs,
            )
        # The guard seeds lazily from the session peak; provide the id that
        # start() would have resolved.
        engine.trading_session_id = session_id
        return engine

    return _make
