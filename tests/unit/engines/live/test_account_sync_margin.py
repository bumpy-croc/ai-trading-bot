"""Unit tests for margin balance reconciliation (prevention layer 2).

Margin mode used to skip balance sync entirely, so realized losses that were
never booked to the tracked balance drifted from true equity unnoticed (a ~15%
gap was observed in production). These cover the reconcile that uses the
exchange's account-level net equity (`get_account_equity`) and only corrects
while flat.
"""

import logging
from datetime import UTC, datetime
from unittest.mock import Mock, patch

import pytest

from src.config.constants import DEFAULT_STARTUP_EQUITY_MAX_RETRIES
from src.database.models import EventType
from src.engines.live.account_sync import AccountSynchronizer

pytestmark = pytest.mark.unit


@patch("src.data_providers.binance_provider.Client")
@patch("src.data_providers.binance_provider.get_config")
def test_get_account_equity_margin_uses_total_net_asset(mock_config, mock_client_class):
    """Margin equity = totalNetAssetOfBtc (net of liabilities) valued in USDT."""
    from src.data_providers.binance_provider import BinanceProvider

    mock_config.return_value = Mock(get_required=Mock(return_value="fake_key"))
    mock_client = Mock()
    mock_client_class.return_value = mock_client
    mock_client.get_margin_account.return_value = {"totalNetAssetOfBtc": "0.00125"}
    mock_client.get_symbol_ticker.return_value = {"price": "67000.0"}

    provider = BinanceProvider()
    provider._use_margin = True

    assert provider.get_account_equity() == pytest.approx(0.00125 * 67000.0)


@patch("src.data_providers.binance_provider.Client")
@patch("src.data_providers.binance_provider.get_config")
def test_get_account_equity_spot_uses_usdt_total(mock_config, mock_client_class):
    from src.data_providers.binance_provider import BinanceProvider

    mock_config.return_value = Mock(get_required=Mock(return_value="fake_key"))
    mock_client_class.return_value = Mock()

    provider = BinanceProvider()
    provider._use_margin = False
    provider.get_balance = Mock(return_value=Mock(total=84.5))

    assert provider.get_account_equity() == pytest.approx(84.5)


def _make_sync(equity, db_balance, usdt_total):
    exchange = Mock()
    exchange.get_account_equity.return_value = equity
    exchange.get_balance.return_value = Mock(total=usdt_total) if usdt_total is not None else None
    db = Mock()
    db.get_current_balance.return_value = db_balance
    sync = AccountSynchronizer(exchange=exchange, db_manager=db, session_id=1, use_margin=True)
    return sync, db


def test_margin_equity_corrects_when_flat_and_divergent():
    """Flat (equity == USDT cash) + >1% divergence -> correct down to true equity."""
    sync, db = _make_sync(equity=84.22, db_balance=99.89, usdt_total=84.22)

    res = sync._sync_margin_equity()

    assert res["corrected"] is True
    assert res["new_balance"] == pytest.approx(84.22)
    assert res["old_balance"] == pytest.approx(99.89)
    db.update_balance.assert_called_once()


def test_margin_equity_no_correct_within_threshold():
    """Flat but divergence under the 1% threshold -> no correction."""
    sync, db = _make_sync(equity=99.50, db_balance=99.89, usdt_total=99.50)

    res = sync._sync_margin_equity()

    assert res["corrected"] is False
    db.update_balance.assert_not_called()


def test_margin_equity_no_correct_when_position_held():
    """Divergent AND a position is held (equity != USDT) -> warn, do NOT correct.

    Guards against an open/unreconciled position folding its value into cash.
    """
    sync, db = _make_sync(equity=84.22, db_balance=99.89, usdt_total=65.41)

    res = sync._sync_margin_equity()

    assert res.get("corrected") is not True
    assert res["reason"] == "position held"
    db.update_balance.assert_not_called()


@patch("src.engines.live.account_sync.time.sleep")
def test_margin_equity_no_sync_when_equity_unavailable(mock_sleep):
    """Equity unreadable -> no sync, no correction (fail safe)."""
    sync, db = _make_sync(equity=None, db_balance=99.89, usdt_total=99.89)

    res = sync._sync_margin_equity()

    assert res["synced"] is False
    db.update_balance.assert_not_called()


@patch("src.engines.live.account_sync.time.sleep")
def test_margin_equity_no_correct_when_equity_zero_or_negative(mock_sleep):
    """Non-positive equity is treated as unavailable -> no correction.

    A fixed 0.0 (not None) on every attempt gets the same bounded startup
    retry as None before the skip-and-alert path takes over (#1226 review:
    the retry predicate must match the caller's "unavailable" check).
    """
    sync, db = _make_sync(equity=0.0, db_balance=99.89, usdt_total=99.89)

    res = sync._sync_margin_equity()

    assert res["synced"] is False
    db.update_balance.assert_not_called()
    assert sync.exchange.get_account_equity.call_count == 3


def test_margin_equity_correction_records_audit_and_system_event():
    """A book-down MUST leave both an audit row and a warning+ system event.

    Regression guard: in prod on 2026-06-03 a margin_equity_sync_correction booked
    the tracked balance from $99.89 down to true equity $84.14 (-15.8%, the single
    largest capital event of the fortnight) yet wrote ZERO reconciliation_audit_events
    and ZERO warning+ system_events rows, so monitoring never saw it.
    """
    sync, db = _make_sync(equity=84.14, db_balance=99.89, usdt_total=84.14)

    res = sync._sync_margin_equity()

    assert res["corrected"] is True
    db.update_balance.assert_called_once()

    # (a) immutable reconciliation audit row with before/after values
    db.log_audit_event.assert_called_once()
    audit = db.log_audit_event.call_args.kwargs
    assert audit["session_id"] == 1
    assert audit["entity_type"] == "balance"
    assert audit["field"] == "total_balance"
    assert audit["severity"] == "CRITICAL"  # >=5% book-down is material
    assert float(audit["old_value"]) == pytest.approx(99.89)
    assert float(audit["new_value"]) == pytest.approx(84.14)
    assert audit["reason"]  # non-empty human-readable explanation

    # (b) operator-facing system event at warning+ severity for alerting
    db.log_event.assert_called_once()
    event = db.log_event.call_args.kwargs
    assert event["event_type"] == EventType.BALANCE_ADJUSTMENT
    assert event["severity"] == "critical"  # warning+ (critical for >=5%)
    assert event["session_id"] == 1


def test_margin_equity_small_correction_is_high_warning():
    """A >1% but <5% book-down -> HIGH audit + 'warning' system event (not critical).

    Mirrors the second unaudited prod event (2026-06-05, ~ -1.6%).
    """
    # 84.14 -> 82.77 is -1.63%: above the 1% correction gate, below the 5% CRITICAL line.
    sync, db = _make_sync(equity=82.77, db_balance=84.14, usdt_total=82.77)

    res = sync._sync_margin_equity()

    assert res["corrected"] is True
    assert db.log_audit_event.call_args.kwargs["severity"] == "HIGH"
    assert db.log_event.call_args.kwargs["severity"] == "warning"


def test_margin_equity_no_audit_or_event_when_within_threshold():
    """No correction -> no audit row and no system event are emitted."""
    sync, db = _make_sync(equity=99.50, db_balance=99.89, usdt_total=99.50)

    sync._sync_margin_equity()

    db.update_balance.assert_not_called()
    db.log_audit_event.assert_not_called()
    db.log_event.assert_not_called()


def test_margin_equity_audit_logging_failure_does_not_break_correction():
    """Observability writes are best-effort: a failure must neither raise nor
    unwind the already-persisted balance correction, and the two writes are
    independently guarded (an audit failure still attempts the system event)."""
    sync, db = _make_sync(equity=84.14, db_balance=99.89, usdt_total=84.14)
    db.log_audit_event.side_effect = RuntimeError("audit table unavailable")

    res = sync._sync_margin_equity()  # must not raise

    assert res["corrected"] is True
    db.update_balance.assert_called_once()
    db.log_event.assert_called_once()  # attempted despite the audit failure


def test_margin_equity_no_audit_when_balance_update_fails():
    """If the balance write itself fails (update_balance -> False), do NOT emit an
    audit/alert claiming a correction that never persisted."""
    sync, db = _make_sync(equity=84.14, db_balance=99.89, usdt_total=84.14)
    db.update_balance.return_value = False

    res = sync._sync_margin_equity()

    assert res["synced"] is False
    assert res["corrected"] is False
    db.log_audit_event.assert_not_called()
    db.log_event.assert_not_called()


def test_margin_equity_correction_audited_during_startup_session_handoff():
    """Startup edge (found in Codex review): the synchronizer's own session_id is
    still None during the INITIAL sync — trading_engine assigns it only AFTER
    sync_account_data() returns — but the DB manager already has a current session, so
    update_balance persists via its _current_session_id fallback. The audit + system
    event MUST still fire, bound to that resolved session, not be silently skipped.
    """
    sync, db = _make_sync(equity=84.14, db_balance=99.89, usdt_total=84.14)
    sync.session_id = None  # not yet assigned during the initial sync
    db._current_session_id = 7  # update_balance / get_current_balance fall back to this
    db.update_balance.return_value = True

    res = sync._sync_margin_equity()

    assert res["corrected"] is True
    db.log_audit_event.assert_called_once()
    assert db.log_audit_event.call_args.kwargs["session_id"] == 7
    db.log_event.assert_called_once()
    assert db.log_event.call_args.kwargs["session_id"] == 7


def test_margin_equity_system_event_failure_does_not_break_correction():
    """The system-event (log_event) write is best-effort and guarded independently of
    the audit write: a log_event failure must neither raise into the sync loop nor flip
    the already-persisted correction. Companion to the log_audit_event-failure test —
    pins the SECOND of the two independently wrapped observability writes so a future
    refactor of that try/except can't let a system_events fault escape into
    sync_account_data() and report success=False after a real book-down.
    """
    sync, db = _make_sync(equity=84.14, db_balance=99.89, usdt_total=84.14)
    db.log_event.side_effect = RuntimeError("system_events table unavailable")

    res = sync._sync_margin_equity()  # must not raise

    assert res["corrected"] is True
    db.update_balance.assert_called_once()
    db.log_audit_event.assert_called_once()  # audit still written despite the event failure


def test_margin_equity_exactly_five_percent_divergence_is_critical():
    """The CRITICAL boundary is inclusive (>=): an exactly-5.0% book-down is
    CRITICAL/critical, not HIGH/warning. 100.00 -> 95.00 makes diff_pct land exactly on
    DEFAULT_RECONCILIATION_BALANCE_THRESHOLD_PCT * 100 (verified float-exact), so a
    >= -> > regression would silently demote boundary book-downs and this test catches it.
    """
    sync, db = _make_sync(equity=95.0, db_balance=100.0, usdt_total=95.0)

    res = sync._sync_margin_equity()

    assert res["corrected"] is True
    assert db.log_audit_event.call_args.kwargs["severity"] == "CRITICAL"
    assert db.log_event.call_args.kwargs["severity"] == "critical"


# ---------------------------------------------------------------------------
# GH #659: cold-boot startup silent-skip regression tests
#
# Prod diagnosis: after a deploy, the startup account-sync's margin-equity
# correction silently no-op'd — tracked balance stayed at ~$99.89 while true
# equity was ~$84.18 (15% divergence) — with NO log line at all, because
# get_account_equity() returned None only at cold boot (a warm call moments
# later returned the correct value) and `_sync_margin_equity`'s
# `if equity is None or equity <= 0: return` was completely silent.
# ---------------------------------------------------------------------------


@patch("src.data_providers.binance_provider.Client")
@patch("src.data_providers.binance_provider.get_config")
def test_get_account_equity_logs_when_client_not_initialized(
    mock_config, mock_client_class, caplog
):
    """get_account_equity() must log WHY it's returning None, not return silently."""
    from src.data_providers.binance_provider import BinanceProvider

    mock_config.return_value = Mock(get_required=Mock(return_value="fake_key"))
    mock_client_class.return_value = Mock()

    provider = BinanceProvider()
    provider._client = None  # simulate the client not being ready yet

    with caplog.at_level(logging.WARNING):
        result = provider.get_account_equity()

    assert result is None
    assert any(
        "cannot read equity" in record.message and "client not initialized" in record.message
        for record in caplog.records
    )


@patch("src.data_providers.binance_provider.Client")
@patch("src.data_providers.binance_provider.get_config")
def test_get_account_equity_logs_when_binance_unavailable(mock_config, mock_client_class, caplog):
    """The other silent branch (python-binance not installed) must also log."""
    from src.data_providers.binance_provider import BinanceProvider

    mock_config.return_value = Mock(get_required=Mock(return_value="fake_key"))
    mock_client_class.return_value = Mock()

    provider = BinanceProvider()

    with (
        patch("src.data_providers.binance_provider.BINANCE_AVAILABLE", False),
        caplog.at_level(logging.WARNING),
    ):
        result = provider.get_account_equity()

    assert result is None
    assert any(
        "cannot read equity" in record.message and "not installed" in record.message
        for record in caplog.records
    )


@patch("src.engines.live.account_sync.time.sleep")
def test_margin_equity_startup_skip_logs_and_arms_pending_retry(mock_sleep, caplog):
    """A cold-boot skip must log a warning, emit a system event, and arm the
    one-shot early retry so the next sync isn't throttled for
    DEFAULT_ACCOUNT_SYNC_MIN_INTERVAL_MINUTES."""
    sync, db = _make_sync(equity=None, db_balance=99.89, usdt_total=99.89)
    assert sync.last_sync_time is None  # fresh synchronizer == cold boot

    with caplog.at_level(logging.WARNING):
        res = sync._sync_margin_equity()

    assert res["synced"] is False
    assert any("Margin equity correction skipped" in r.message for r in caplog.records)
    assert any("startup" in r.message for r in caplog.records)

    db.log_event.assert_called_once()
    event = db.log_event.call_args.kwargs
    assert event["event_type"] == EventType.WARNING
    assert event["severity"] == "warning"
    assert event["session_id"] == 1
    assert event["error_code"] == "MARGIN_EQUITY_SKIPPED"

    assert sync._pending_startup_equity_retry is True
    # DEFAULT_STARTUP_EQUITY_MAX_RETRIES is the TOTAL attempt count, so there
    # is one sleep BETWEEN each pair of attempts -- one fewer sleep than
    # attempts.
    assert mock_sleep.call_count == DEFAULT_STARTUP_EQUITY_MAX_RETRIES - 1


@patch("src.engines.live.account_sync.time.sleep")
def test_margin_equity_periodic_skip_does_not_arm_pending_retry(mock_sleep):
    """A WARM (periodic) skip is a real API failure, not a readiness race — it
    must still log/alert but must NOT force an early bypass of the sync
    throttle (that would spin on an unavailable endpoint), and must NOT get
    the startup inline retry (a single read only)."""
    sync, db = _make_sync(equity=None, db_balance=99.89, usdt_total=99.89)
    sync.last_sync_time = datetime.now(UTC)

    res = sync._sync_margin_equity()

    assert res["synced"] is False
    db.log_event.assert_called_once()
    assert sync._pending_startup_equity_retry is False
    # Discriminates the warm path from the startup path: no inline retry loop
    # (and therefore no sleep) when this isn't a cold-boot sync.
    assert sync.exchange.get_account_equity.call_count == 1
    mock_sleep.assert_not_called()


@patch("src.engines.live.account_sync.time.sleep")
def test_margin_equity_startup_retry_recovers_from_transient_none(mock_sleep):
    """Discriminates the retry fix: a transient None on the first read, then a
    real value on a later attempt, must still apply the correction — this is
    exactly the cold-boot race the bug report describes (mechanism verified
    correct when called warm)."""
    exchange = Mock()
    exchange.get_account_equity.side_effect = [None, None, 84.14]
    exchange.get_balance.return_value = Mock(total=84.14)
    db = Mock()
    db.get_current_balance.return_value = 99.89
    db.update_balance.return_value = True
    sync = AccountSynchronizer(exchange=exchange, db_manager=db, session_id=1, use_margin=True)

    res = sync._sync_margin_equity()

    assert res["corrected"] is True
    assert res["new_balance"] == pytest.approx(84.14)
    db.update_balance.assert_called_once()
    assert exchange.get_account_equity.call_count == 3
    assert mock_sleep.call_count == 2
    # The correction succeeded, so no early-retry needed on the next sync.
    assert sync._pending_startup_equity_retry is False


def test_sync_account_data_bypasses_throttle_after_startup_equity_skip():
    """Part 3: a startup skip must self-heal on the very next sync attempt
    rather than waiting out DEFAULT_ACCOUNT_SYNC_MIN_INTERVAL_MINUTES."""
    exchange = Mock()
    exchange.sync_account_data.return_value = {
        "sync_successful": True,
        "balances": [],
        "positions": [],
        "open_orders": [],
    }
    exchange.get_account_equity.return_value = 84.14
    exchange.get_balance.return_value = Mock(total=84.14)
    db = Mock()
    db.get_current_balance.return_value = 99.89
    db.update_balance.return_value = True
    sync = AccountSynchronizer(exchange=exchange, db_manager=db, session_id=1, use_margin=True)

    # Simulate a prior sync that just happened (so the throttle would normally
    # block an immediate re-sync) and arm the pending flag as the startup skip
    # would have.
    sync.last_sync_time = datetime.now(UTC)
    sync._pending_startup_equity_retry = True

    result = sync.sync_account_data(force=False)

    assert result.success is True
    assert result.data["balance_sync"]["corrected"] is True
    # One-shot: consumed after use.
    assert sync._pending_startup_equity_retry is False


# ---------------------------------------------------------------------------
# PR #1226 review: exception path must not bypass the alert/retry/self-heal
# path (P1), and the retry predicate must match the caller's "unavailable"
# check (P2, zero-equity case covered above).
# ---------------------------------------------------------------------------


@patch("src.engines.live.account_sync.time.sleep")
def test_margin_equity_startup_read_exception_retries_then_recovers(mock_sleep):
    """A cold-boot read that RAISES (a BinanceAPIException on a not-yet-ready
    client, an AttributeError on a half-initialized one -- both #659 shapes)
    must retry exactly like a None read, not escape to a separate early
    return that skips the alert/retry/self-heal path."""
    exchange = Mock()
    exchange.get_account_equity.side_effect = [RuntimeError("client not ready"), 84.14]
    exchange.get_balance.return_value = Mock(total=84.14)
    db = Mock()
    db.get_current_balance.return_value = 99.89
    db.update_balance.return_value = True
    sync = AccountSynchronizer(exchange=exchange, db_manager=db, session_id=1, use_margin=True)

    res = sync._sync_margin_equity()

    assert res["corrected"] is True
    assert exchange.get_account_equity.call_count == 2
    assert mock_sleep.call_count == 1


@patch("src.engines.live.account_sync.time.sleep")
def test_margin_equity_startup_read_exception_exhausted_still_alerts_and_arms(mock_sleep, caplog):
    """If every startup attempt raises, the skip must still log a warning,
    emit the system event, and arm the throttle-bypass retry -- not vanish
    silently through the outer except block's own early return (the exact
    #659 recurrence this review finding described)."""
    exchange = Mock()
    exchange.get_account_equity.side_effect = RuntimeError("client not ready")
    exchange.get_balance.return_value = Mock(total=99.89)
    db = Mock()
    db.get_current_balance.return_value = 99.89
    sync = AccountSynchronizer(exchange=exchange, db_manager=db, session_id=1, use_margin=True)

    with caplog.at_level(logging.WARNING):
        res = sync._sync_margin_equity()

    assert res["synced"] is False
    assert any("Margin equity correction skipped" in r.message for r in caplog.records)
    db.log_event.assert_called_once()
    assert db.log_event.call_args.kwargs["error_code"] == "MARGIN_EQUITY_SKIPPED"
    assert sync._pending_startup_equity_retry is True
    assert exchange.get_account_equity.call_count == 3


def test_margin_equity_outer_backstop_routes_through_skip_handling():
    """Belt-and-suspenders: even if something upstream of _read_account_equity
    still raised unexpectedly, _sync_margin_equity's own except block must
    route through the normal skip-handling (warning + system event + arming),
    not return early on its own."""
    sync, db = _make_sync(equity=None, db_balance=99.89, usdt_total=99.89)
    sync._read_account_equity = Mock(side_effect=RuntimeError("unexpected"))

    res = sync._sync_margin_equity()

    assert res["synced"] is False
    assert res["reason"] == "equity unavailable"
    db.log_event.assert_called_once()
    assert sync._pending_startup_equity_retry is True


# ---------------------------------------------------------------------------
# PR #1226 review, P2: the one-shot retry flag must not be consumed before an
# equity read is even attempted.
# ---------------------------------------------------------------------------


def test_pending_retry_flag_survives_exchange_sync_failure_before_margin_read():
    """exchange.sync_account_data() reporting sync_successful=False must not
    consume the one-shot flag before a margin-equity read is ever attempted
    -- that failure is most likely right after cold boot, exactly when the
    flag is armed."""
    exchange = Mock()
    exchange.sync_account_data.return_value = {"sync_successful": False, "error": "boom"}
    db = Mock()
    sync = AccountSynchronizer(exchange=exchange, db_manager=db, session_id=1, use_margin=True)
    sync.last_sync_time = datetime.now(UTC)  # pretend a previous sync happened
    sync._pending_startup_equity_retry = True

    result = sync.sync_account_data(force=False)

    assert result.success is False
    assert sync._pending_startup_equity_retry is True
    exchange.get_account_equity.assert_not_called()


def test_pending_retry_flag_survives_exception_before_margin_read():
    """The same guarantee when exchange.sync_account_data() itself raises."""
    exchange = Mock()
    exchange.sync_account_data.side_effect = RuntimeError("boom")
    db = Mock()
    sync = AccountSynchronizer(exchange=exchange, db_manager=db, session_id=1, use_margin=True)
    sync._pending_startup_equity_retry = True

    result = sync.sync_account_data(force=False)

    assert result.success is False
    assert sync._pending_startup_equity_retry is True


@patch("src.engines.live.account_sync.time.sleep")
def test_margin_equity_zero_equity_retries_at_startup(mock_sleep):
    """#1226 P2: a cold-boot 0.0 (not None) must get the same bounded retry
    as None -- BinanceProvider.get_account_equity() legitimately returns 0.0,
    not None, when a nested margin field is missing on a half-initialized
    response."""
    exchange = Mock()
    exchange.get_account_equity.side_effect = [0.0, 0.0, 84.14]
    exchange.get_balance.return_value = Mock(total=84.14)
    db = Mock()
    db.get_current_balance.return_value = 99.89
    db.update_balance.return_value = True
    sync = AccountSynchronizer(exchange=exchange, db_manager=db, session_id=1, use_margin=True)

    res = sync._sync_margin_equity()

    assert res["corrected"] is True
    assert exchange.get_account_equity.call_count == 3
    assert mock_sleep.call_count == 2
    assert [c.args[0] for c in mock_sleep.call_args_list] == [1.0, 2.0]


@patch("src.engines.live.account_sync.time.sleep")
def test_margin_equity_nan_retries_and_never_reaches_update_balance(mock_sleep):
    """#1226 re-review (architecture-reviewer): a non-finite equity (NaN, from a
    malformed/non-numeric exchange field) must retry exactly like None/0.0 --
    not slip through both predicates (neither `> 0` nor `<= 0` is True for NaN)
    and reach update_balance(nan, ...)."""
    exchange = Mock()
    exchange.get_account_equity.side_effect = [float("nan"), float("nan"), 84.14]
    exchange.get_balance.return_value = Mock(total=84.14)
    db = Mock()
    db.get_current_balance.return_value = 99.89
    db.update_balance.return_value = True
    sync = AccountSynchronizer(exchange=exchange, db_manager=db, session_id=1, use_margin=True)

    res = sync._sync_margin_equity()

    assert res["corrected"] is True
    assert exchange.get_account_equity.call_count == 3
    # NaN must never be the value passed to update_balance.
    for call in db.update_balance.call_args_list:
        assert call.args[0] == pytest.approx(84.14)
