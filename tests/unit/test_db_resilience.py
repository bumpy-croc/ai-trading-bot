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
- An abnormal loop death (unhandled crash or error exhaustion) is recorded so
  ``start()`` exits non-zero for an orchestrator restart instead of exiting 0
  and leaving the bot silently dead (#630).
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
    # A clean max_steps stop is not a crash — start() would exit 0 here.
    assert engine._loop_crashed is False


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
    # Error exhaustion is an abnormal stop: start() must exit non-zero (#630).
    assert engine._loop_crashed is True


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


# --------------------------------------------------------------------------- #
# Loop-death -> non-zero exit so the orchestrator restarts the process (#630)
# --------------------------------------------------------------------------- #


@pytest.mark.fast
def test_run_trading_loop_records_unhandled_crash() -> None:
    """A crash escaping the loop is caught and recorded, not silently swallowed."""
    engine = _make_engine()
    engine._trading_loop = Mock(side_effect=RuntimeError("loop blew up"))

    # The thread target must not propagate — it records the crash instead.
    engine._run_trading_loop("BTCUSDT", "1h")

    assert engine._loop_crashed is True


@pytest.mark.fast
def test_exit_if_loop_crashed_exits_nonzero_when_opted_in() -> None:
    """An abnormal loop death makes start() exit 1 so Railway ON_FAILURE restarts."""
    engine = _make_engine()
    engine._loop_crashed = True
    engine.main_thread = None  # nothing to join in this unit

    with pytest.raises(SystemExit) as exc_info:
        engine._exit_if_loop_crashed(exit_on_crash=True)

    assert exc_info.value.code == 1


@pytest.mark.fast
def test_exit_if_loop_crashed_is_noop_after_clean_stop() -> None:
    """A clean stop leaves the process to exit 0 (no SystemExit raised)."""
    engine = _make_engine()
    engine._loop_crashed = False

    # Must return normally — start() then exits 0 as before.
    assert engine._exit_if_loop_crashed(exit_on_crash=True) is None


@pytest.mark.fast
def test_exit_if_loop_crashed_returns_when_not_opted_in() -> None:
    """A crash must NOT kill library callers (e.g. the migration baseline tool)
    that drive start() and read results afterward — only the runner opts in."""
    engine = _make_engine()
    engine._loop_crashed = True
    engine.main_thread = None

    # exit_on_crash defaults False: returns instead of calling sys.exit().
    assert engine._exit_if_loop_crashed(exit_on_crash=False) is None


@pytest.mark.fast
def test_start_exits_nonzero_on_loop_crash_when_opted_in() -> None:
    """End-to-end wiring: start(exit_on_crash=True) routes the loop through the
    crash-catching wrapper and exits 1 when it dies (locks the thread target)."""
    engine = _make_engine()
    engine.resume_from_last_balance = False
    engine._start_websocket_streams = Mock()  # no real WS in this unit
    engine._trading_loop = Mock(side_effect=RuntimeError("loop blew up"))

    with pytest.raises(SystemExit) as exc_info:
        engine.start("BTCUSDT", "1h", exit_on_crash=True)

    assert exc_info.value.code == 1
    assert engine._loop_crashed is True


@pytest.mark.fast
def test_start_returns_on_loop_crash_when_not_opted_in() -> None:
    """End-to-end wiring: without opt-in, start() returns even if the loop crashes."""
    engine = _make_engine()
    engine.resume_from_last_balance = False
    engine._start_websocket_streams = Mock()
    engine._trading_loop = Mock(side_effect=RuntimeError("loop blew up"))

    engine.start("BTCUSDT", "1h")  # exit_on_crash defaults False

    # Recorded the crash but did not exit the process.
    assert engine._loop_crashed is True


# --------------------------------------------------------------------------- #
# WS health watchdog — the main loop restarts a dead monitor thread (#631)
# --------------------------------------------------------------------------- #


def _dead_thread() -> Mock:
    t = Mock()
    t.is_alive.return_value = False
    return t


@pytest.mark.fast
def test_watchdog_respawns_dead_ws_thread() -> None:
    engine = _make_engine()
    engine.is_running = True
    engine._ws_kline_active = True
    engine._ws_kline_provider = Mock()  # a kline stream exists → monitor is supervised (#662)
    engine._ws_health_thread = _dead_thread()
    engine._start_ws_health_monitor = Mock()

    engine._ensure_ws_health_monitor_alive()

    engine._start_ws_health_monitor.assert_called_once()


@pytest.mark.fast
def test_watchdog_noop_when_thread_alive() -> None:
    engine = _make_engine()
    engine.is_running = True
    engine._ws_kline_active = True
    engine._ws_kline_provider = (
        Mock()
    )  # kline stream exists → guard passes, reach alive-check (#662)
    alive = Mock()
    alive.is_alive.return_value = True
    engine._ws_health_thread = alive
    engine._start_ws_health_monitor = Mock()

    engine._ensure_ws_health_monitor_alive()

    engine._start_ws_health_monitor.assert_not_called()


@pytest.mark.fast
def test_watchdog_noop_when_no_stream_configured() -> None:
    """No WS stream is being watched → nothing to supervise (pure REST mode)."""
    engine = _make_engine()
    engine.is_running = True
    engine._ws_kline_active = False
    engine._user_data_processor = None
    engine._ws_health_thread = _dead_thread()
    engine._start_ws_health_monitor = Mock()

    engine._ensure_ws_health_monitor_alive()

    engine._start_ws_health_monitor.assert_not_called()


@pytest.mark.fast
def test_watchdog_noop_when_stopping() -> None:
    """A shutting-down engine must not respawn the monitor."""
    engine = _make_engine()
    engine.is_running = True
    engine._ws_kline_active = True
    engine._ws_kline_provider = (
        Mock()
    )  # kline stream exists → guard passes, reach stopping-check (#662)
    engine.stop_event.set()
    engine._ws_health_thread = _dead_thread()
    engine._start_ws_health_monitor = Mock()

    engine._ensure_ws_health_monitor_alive()

    engine._start_ws_health_monitor.assert_not_called()


# --------------------------------------------------------------------------- #
# Deferred stop-loss fill exits — non-blocking, off the OrderTracker thread (#631)
# --------------------------------------------------------------------------- #


def _seed_sl_position(engine, order_id: str = "pos1", sl_order_id: str = "sl1"):
    """Give the engine a position whose stop-loss is `sl_order_id`."""
    pos = Mock()
    pos.order_id = order_id
    pos.stop_loss_order_id = sl_order_id
    engine.live_position_tracker = Mock()
    engine.live_position_tracker.positions = {order_id: pos}
    engine.live_position_tracker.get_position = Mock(
        side_effect=lambda oid: pos if oid == order_id else None
    )
    return pos


@pytest.mark.fast
def test_handle_order_fill_enqueues_sl_exit_instead_of_running_it() -> None:
    """A stop-loss fill is queued for the loop, NOT executed on the poll thread."""
    engine = _make_engine()
    _seed_sl_position(engine, order_id="pos1", sl_order_id="sl1")
    engine._execute_exit = Mock()

    engine._handle_order_fill("sl1", "BTCUSDT", 0.5, 100.0)

    assert engine._pending_fill_exits.get_nowait() == ("pos1", 100.0)
    engine._execute_exit.assert_not_called()  # deferred, not run inline


@pytest.mark.fast
def test_handle_order_fill_entry_does_not_enqueue() -> None:
    """A non-stop-loss (entry) fill enqueues nothing."""
    engine = _make_engine()
    _seed_sl_position(engine, order_id="pos1", sl_order_id="sl1")
    engine._execute_exit = Mock()

    engine._handle_order_fill("some-entry-order", "BTCUSDT", 0.5, 100.0)

    assert engine._pending_fill_exits.empty()
    engine._execute_exit.assert_not_called()


@pytest.mark.fast
def test_drain_executes_pending_exit() -> None:
    engine = _make_engine()
    _seed_sl_position(engine, order_id="pos1", sl_order_id="sl1")
    engine._execute_exit = Mock()
    engine._pending_fill_exits.put(("pos1", 100.0))

    engine._drain_pending_fill_exits()

    engine._execute_exit.assert_called_once()
    kwargs = engine._execute_exit.call_args.kwargs
    assert kwargs["skip_live_close"] is True
    assert kwargs["reason"] == "stop_loss"
    assert engine._pending_fill_exits.empty()


@pytest.mark.fast
def test_drain_skips_already_closed_position() -> None:
    """If the position is already gone, the drain no-ops without error."""
    engine = _make_engine()
    engine.live_position_tracker = Mock()
    engine.live_position_tracker.get_position = Mock(return_value=None)
    engine._execute_exit = Mock()
    engine._pending_fill_exits.put(("gone", 100.0))

    engine._drain_pending_fill_exits()

    engine._execute_exit.assert_not_called()
    assert engine._pending_fill_exits.empty()


@pytest.mark.fast
def test_drain_failure_is_isolated_and_logged(caplog) -> None:
    """One failing exit is logged CRITICAL and does NOT abort the rest of the drain."""
    engine = _make_engine()
    engine.live_position_tracker = Mock()
    engine.live_position_tracker.get_position = Mock(return_value=Mock())
    engine._execute_exit = Mock(side_effect=[RuntimeError("db down"), None])
    engine._pending_fill_exits.put(("p1", 100.0))
    engine._pending_fill_exits.put(("p2", 101.0))

    with caplog.at_level("CRITICAL"):
        engine._drain_pending_fill_exits()

    assert engine._execute_exit.call_count == 2  # did not abort after the first failure
    assert "draining deferred stop-loss exit" in caplog.text
    assert engine._pending_fill_exits.empty()


# --------------------------------------------------------------------------- #
# Disconnect handlers stay exception-safe once REST calls can time out (#631)
# --------------------------------------------------------------------------- #


@pytest.mark.fast
def test_kline_disconnect_survives_reconnect_timeout() -> None:
    """A raising kline reconnect (e.g. REST timeout) must not propagate (#631).

    Post-#662 _handle_kline_disconnect is state-free — dropping to REST is owned by
    _check_kline_health (which already did so before calling here). This pins only
    the exception-safety guarantee: a timing-out reconnect can't crash the WS
    health thread.
    """
    engine = _make_engine()
    engine._kline_buffer = None  # skip REST resync
    provider = Mock()
    provider.reconnect_kline.side_effect = TimeoutError("rest socket hung")
    engine._ws_kline_provider = provider

    engine._handle_kline_disconnect()  # must not propagate

    provider.reconnect_kline.assert_called_once()
    provider.mark_kline_degraded.assert_not_called()  # not this function's job anymore


@pytest.mark.fast
def test_user_stream_disconnect_survives_resync_timeout() -> None:
    """A REST resync timeout mid-recovery must not abort the handler (#631)."""
    engine = _make_engine()
    engine.enable_live_trading = True
    exchange = Mock()
    exchange.reconnect_user.return_value = False  # force the degraded fallback
    engine.exchange_interface = exchange
    tracker = Mock()
    tracker.poll_once.side_effect = TimeoutError("rest socket hung")  # step-4 resync
    engine.order_tracker = tracker
    engine._user_data_processor = None
    engine._periodic_reconciler = None

    with patch("src.engines.live.user_data_processor.UserDataProcessor"):
        engine._handle_user_stream_disconnect()  # must not propagate

    exchange.mark_user_degraded.assert_called_once()
