"""Tests for transient-DB-error handling in the live trading engine.

Regression coverage for the 2026-05-19 incident: a multi-hour Railway
internal-DNS outage made Postgres unresolvable, every trading-loop iteration
raised ``OperationalError``, the consecutive-error counter hit its limit, and
the engine shut itself down — staying silently offline for ~12 days.

Covers:
- ``LiveTradingEngine._is_transient_db_error`` classification (transient vs
  permanent vs unrelated), including exception-chain unwrapping.
- Loop behaviour: transient DB errors are ridden out (not counted toward
  ``max_consecutive_errors``); permanent/unrelated errors still trigger
  shutdown; a prolonged outage escalates to close-only mode.
"""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest
from sqlalchemy.exc import InterfaceError, OperationalError

from src.data_providers.data_provider import DataProvider
from src.engines.live.trading_engine import LiveTradingEngine

is_transient = LiveTradingEngine._is_transient_db_error


def _op_error(message: str) -> OperationalError:
    """Build a SQLAlchemy OperationalError(statement, params, orig)."""
    return OperationalError("SELECT 1", {}, Exception(message))


def _make_engine() -> LiveTradingEngine:
    """A paper-mode engine with the database manager patched out."""
    from src.strategies.ml_basic import create_ml_basic_strategy

    with patch("src.engines.live.trading_engine.DatabaseManager"):
        return LiveTradingEngine(
            strategy=create_ml_basic_strategy(),
            data_provider=Mock(spec=DataProvider),
            initial_balance=10000,
            enable_live_trading=False,
        )


# --------------------------------------------------------------------------- #
# Classifier: transient connectivity errors -> ride out
# --------------------------------------------------------------------------- #


@pytest.mark.fast
@pytest.mark.parametrize(
    "exc",
    [
        _op_error('could not translate host name "postgres.railway.internal" to address'),
        _op_error("server closed the connection unexpectedly"),
        _op_error("terminating connection due to administrator command"),
        InterfaceError("SELECT 1", {}, Exception("connection already closed")),
        Exception(
            'could not translate host name "postgres.railway.internal" to address: '
            "Temporary failure in name resolution"
        ),
        Exception("connection refused"),
        Exception("could not connect to server"),
    ],
)
def test_transient_db_errors_detected(exc: BaseException) -> None:
    assert is_transient(exc) is True


# --------------------------------------------------------------------------- #
# Classifier: permanent faults -> fail fast (NOT transient)
# --------------------------------------------------------------------------- #


@pytest.mark.fast
@pytest.mark.parametrize(
    "exc",
    [
        _op_error('password authentication failed for user "trading_bot"'),
        _op_error('role "trading_bot" does not exist'),
        _op_error('database "ai_trading_bot" does not exist'),
        _op_error("permission denied for table trades"),
        Exception("FATAL: password authentication failed"),
    ],
)
def test_permanent_db_errors_not_transient(exc: BaseException) -> None:
    # Permanent misconfig is a psycopg2 OperationalError too, but must fail fast
    # rather than retry forever.
    assert is_transient(exc) is False


# --------------------------------------------------------------------------- #
# Classifier: unrelated errors and chain unwrapping
# --------------------------------------------------------------------------- #


@pytest.mark.fast
@pytest.mark.parametrize(
    "exc",
    [
        ValueError("invalid position size"),
        KeyError("close"),
        RuntimeError("strategy produced an invalid signal"),
        Exception("some unrelated application failure"),
    ],
)
def test_non_transient_errors_not_flagged(exc: BaseException) -> None:
    assert is_transient(exc) is False


@pytest.mark.fast
def test_wrapped_transient_error_is_unwrapped() -> None:
    inner = _op_error("could not connect to server")
    outer = RuntimeError("trading loop iteration failed")
    outer.__cause__ = inner
    assert is_transient(outer) is True


@pytest.mark.fast
def test_context_chained_transient_error_is_unwrapped() -> None:
    inner = Exception("temporary failure in name resolution")
    outer = RuntimeError("wrapper")
    outer.__context__ = inner
    assert is_transient(outer) is True


@pytest.mark.fast
def test_cyclic_cause_chain_terminates() -> None:
    a = Exception("a")
    b = Exception("b")
    a.__cause__ = b
    b.__cause__ = a
    assert is_transient(a) is False


# --------------------------------------------------------------------------- #
# Loop behaviour
# --------------------------------------------------------------------------- #


@pytest.mark.fast
def test_transient_db_error_in_loop_is_ridden_out() -> None:
    """Transient DB errors each iteration must NOT count toward shutdown."""
    engine = _make_engine()
    engine.max_consecutive_errors = 1  # would stop after a single *counted* error
    engine.is_running = True
    engine._sleep_with_interrupt = Mock()  # skip the real backoff sleep
    engine._get_latest_data = Mock(side_effect=_op_error('could not translate host name "db"'))

    engine._trading_loop("BTCUSDT", "1h", max_steps=3)

    # Rode out all three iterations without tripping the shutdown counter.
    assert engine._get_latest_data.call_count == 3
    assert engine.consecutive_errors == 0
    assert engine.db_unreachable_since is not None


@pytest.mark.fast
def test_non_transient_error_in_loop_still_shuts_down() -> None:
    """A non-transient error must still increment the counter and stop."""
    engine = _make_engine()
    engine.max_consecutive_errors = 1
    engine.is_running = True
    engine._sleep_with_interrupt = Mock()
    engine._get_latest_data = Mock(side_effect=ValueError("boom"))

    engine._trading_loop("BTCUSDT", "1h", max_steps=5)

    # Shut down after the first counted error — did not run all five iterations.
    assert engine._get_latest_data.call_count == 1
    assert engine.is_running is False


@pytest.mark.fast
def test_prolonged_db_outage_enters_close_only() -> None:
    """After the close-only threshold, the loop suspends new entries."""
    import time

    engine = _make_engine()
    engine.max_consecutive_errors = 100
    engine.is_running = True
    engine._sleep_with_interrupt = Mock()
    engine._get_latest_data = Mock(side_effect=_op_error("connection refused"))
    # Pretend the outage began well beyond the close-only threshold.
    engine.db_unreachable_since = time.monotonic() - 10_000

    engine._trading_loop("BTCUSDT", "1h", max_steps=1)

    assert engine._close_only_mode is True
