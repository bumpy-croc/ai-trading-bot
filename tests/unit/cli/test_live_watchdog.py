"""Tests for `atb live-control watchdog` — the external write-gap dead-man's-switch (#689)."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from cli.commands import live_watchdog as wd
from cli.commands.live_watchdog import (
    DbWatchdogStore,
    EnvResult,
    Latch,
    Liveness,
    check_environment,
    find_gaps,
    page_is_due,
    watchdog_main,
)
from src.database.models import (
    AccountHistory,
    Base,
    EventType,
    SystemEvent,
    TradeSource,
    TradingSession,
)
from src.engines.live.monitoring.latched_condition_monitor import (
    CONDITIONS,
    ENTRIES_ENABLED_CODE,
    LatchedConditionMonitor,
)
from src.infrastructure.paging_schedule import (
    PAGE_SCHEDULE_SECONDS,
    PAGE_STEADY_INTERVAL_SECONDS,
    repage_offset,
)

pytestmark = [pytest.mark.unit, pytest.mark.fast]

NOW = datetime(2026, 10, 4, 12, 0, tzinfo=UTC)
THRESHOLD = timedelta(minutes=120)
HOUR = timedelta(hours=1)


def _fresh() -> Liveness:
    return Liveness(NOW - timedelta(minutes=30), NOW - timedelta(minutes=20), "session 7 (live)")


def _stale(hours: float = 5.0) -> Liveness:
    old = NOW - timedelta(hours=hours)
    return Liveness(old, old, "session 7 (live)")


class FakeStore:
    """In-memory `WatchdogStore` that records what the watchdog writes."""

    def __init__(self, reading: Liveness, latch: Latch | None = None) -> None:
        self.reading = reading
        self.latch = latch or Latch()
        self.gap_rows: list[tuple[str, bool]] = []
        self.cleared_rows: list[str] = []
        self.fail_writes = False

    def read_liveness(self) -> Liveness:
        return self.reading

    def read_latch(self) -> Latch:
        return self.latch

    def record_gap(self, message: str, *, delivered: bool) -> None:
        if self.fail_writes:
            raise RuntimeError("db write failed")
        self.gap_rows.append((message, delivered))

    def record_cleared(self, message: str) -> None:
        if self.fail_writes:
            raise RuntimeError("db write failed")
        self.cleared_rows.append(message)


def _check(store: FakeStore, notify=None, *, now: datetime = NOW, webhook: bool = True):
    notify = notify or MagicMock(return_value=True)
    result = check_environment(
        "production",
        store,
        now=now,
        threshold=THRESHOLD,
        notify=notify,
        webhook_configured=webhook,
    )
    return result, notify


class TestFindGaps:
    def test_fresh_streams_report_no_gap(self):
        assert find_gaps(_fresh(), NOW, THRESHOLD) == []

    def test_stale_account_history_alone_is_a_gap(self):
        reading = Liveness(NOW - timedelta(hours=5), NOW - timedelta(minutes=10), None)

        gaps = find_gaps(reading, NOW, THRESHOLD)

        assert len(gaps) == 1
        assert gaps[0].startswith("account_history last written 5h 0m ago")

    def test_stale_loop_state_row_alone_is_a_gap(self):
        reading = Liveness(NOW - timedelta(minutes=10), NOW - timedelta(hours=3), None)

        gaps = find_gaps(reading, NOW, THRESHOLD)

        assert len(gaps) == 1
        assert gaps[0].startswith("loop-state row last written 3h 0m ago")

    def test_both_streams_stale_reports_both(self):
        assert len(find_gaps(_stale(), NOW, THRESHOLD)) == 2

    def test_age_exactly_at_threshold_is_not_a_gap(self):
        reading = Liveness(NOW - THRESHOLD, NOW - THRESHOLD, None)

        assert find_gaps(reading, NOW, THRESHOLD) == []

    def test_never_written_streams_are_skipped(self):
        assert find_gaps(Liveness(None, None, None), NOW, THRESHOLD) == []

    def test_unseen_loop_state_does_not_hide_stale_account_history(self):
        reading = Liveness(NOW - timedelta(hours=5), None, None)

        assert len(find_gaps(reading, NOW, THRESHOLD)) == 1

    def test_naive_database_timestamps_are_read_as_utc(self):
        naive_old = (NOW - timedelta(hours=5)).replace(tzinfo=None)

        gaps = find_gaps(Liveness(naive_old, naive_old, None), NOW, THRESHOLD)

        assert len(gaps) == 2


class TestPageIsDue:
    def test_first_announcement_is_immediate(self):
        assert page_is_due(Latch(), NOW, webhook_configured=True) is True

    @pytest.mark.parametrize(
        ("announcements", "offset_seconds", "due"),
        [
            (1, PAGE_SCHEDULE_SECONDS[0] - 1, False),
            (1, PAGE_SCHEDULE_SECONDS[0], True),
            (2, PAGE_SCHEDULE_SECONDS[1] - 1, False),
            (2, PAGE_SCHEDULE_SECONDS[1], True),
            (3, PAGE_SCHEDULE_SECONDS[2] - 1, False),
            (3, PAGE_SCHEDULE_SECONDS[2], True),
            (4, PAGE_SCHEDULE_SECONDS[2] + PAGE_STEADY_INTERVAL_SECONDS - 1, False),
            (4, PAGE_SCHEDULE_SECONDS[2] + PAGE_STEADY_INTERVAL_SECONDS, True),
            (5, PAGE_SCHEDULE_SECONDS[2] + 2 * PAGE_STEADY_INTERVAL_SECONDS - 1, False),
            (5, PAGE_SCHEDULE_SECONDS[2] + 2 * PAGE_STEADY_INTERVAL_SECONDS, True),
        ],
    )
    def test_follows_the_escalating_schedule(self, announcements, offset_seconds, due):
        latch = Latch(
            started_at=NOW - timedelta(seconds=offset_seconds), announcements=announcements
        )

        assert page_is_due(latch, NOW, webhook_configured=True) is due

    def test_undelivered_page_is_retried_when_a_webhook_is_configured(self):
        latch = Latch(started_at=NOW - timedelta(minutes=10), announcements=1, last_delivered=False)

        assert page_is_due(latch, NOW, webhook_configured=True) is True

    def test_undelivered_page_is_not_retried_without_a_webhook(self):
        latch = Latch(started_at=NOW - timedelta(minutes=10), announcements=1, last_delivered=False)

        assert page_is_due(latch, NOW, webhook_configured=False) is False


class TestRepageOffset:
    def test_matches_the_in_process_monitors_re_page_delays(self):
        monitor = LatchedConditionMonitor.__new__(LatchedConditionMonitor)

        for pages_sent in range(1, 8):
            expected = repage_offset(pages_sent) - repage_offset(pages_sent - 1)
            assert monitor._next_page_delay(pages_sent) == expected

    def test_rejects_a_negative_index(self):
        with pytest.raises(ValueError):
            repage_offset(-1)


class TestLoopStateCodes:
    def test_pinned_to_the_codes_the_monitor_writes(self):
        written = {ENTRIES_ENABLED_CODE} | {c.latched_code for c in CONDITIONS}

        assert set(wd.LOOP_STATE_CODES) == written


class TestCheckEnvironment:
    def test_healthy_environment_writes_and_pages_nothing(self, capsys):
        store = FakeStore(_fresh())

        result, notify = _check(store)

        assert result == EnvResult(gap_open=False)
        notify.assert_not_called()
        assert store.gap_rows == []
        assert store.cleared_rows == []
        assert "OK" in capsys.readouterr().out

    def test_new_gap_pages_immediately_and_records_a_critical_row(self):
        store = FakeStore(_stale())

        result, notify = _check(store)

        assert result == EnvResult(gap_open=True, paged=True)
        notify.assert_called_once()
        message, delivered = store.gap_rows[0]
        assert delivered is True
        assert "HEARTBEAT GAP [production]" in message
        assert "account_history last written 5h 0m ago" in message
        assert "session 7 (live)" in message

    def test_already_announced_gap_stays_quiet_until_the_next_scheduled_page(self):
        latch = Latch(started_at=NOW - timedelta(minutes=30), announcements=1)
        store = FakeStore(_stale(), latch)

        result, notify = _check(store)

        assert result == EnvResult(gap_open=True, paged=False)
        notify.assert_not_called()
        assert store.gap_rows == []

    def test_gap_is_re_announced_when_the_schedule_comes_due(self):
        latch = Latch(started_at=NOW - HOUR, announcements=1)
        store = FakeStore(_stale(), latch)

        result, notify = _check(store)

        assert result.paged is True
        notify.assert_called_once()
        assert len(store.gap_rows) == 1

    def test_pages_past_the_first_day_say_still_down(self):
        latch = Latch(started_at=NOW - timedelta(days=3, hours=1), announcements=4)
        store = FakeStore(_stale(hours=80), latch)

        _check(store)

        assert store.gap_rows[0][0].startswith("STILL DOWN — DAY 3: ")

    def test_first_day_pages_carry_no_still_down_prefix(self):
        store = FakeStore(_stale())

        _check(store)

        assert store.gap_rows[0][0].startswith("HEARTBEAT GAP")

    def test_failed_delivery_is_recorded_as_undelivered(self):
        store = FakeStore(_stale())
        notify = MagicMock(return_value=False)

        result, _ = _check(store, notify)

        assert result == EnvResult(gap_open=True, paged=False)
        assert store.gap_rows[0][1] is False

    def test_row_write_failure_does_not_suppress_the_page_or_raise(self, capsys):
        store = FakeStore(_stale())
        store.fail_writes = True

        result, notify = _check(store)

        notify.assert_called_once()
        assert result == EnvResult(gap_open=True, paged=True)
        assert "could not write the HEARTBEAT_GAP" in capsys.readouterr().out

    def test_resumed_writes_clear_the_latch_without_paging(self):
        latch = Latch(started_at=NOW - timedelta(hours=6), announcements=3)
        store = FakeStore(_fresh(), latch)

        result, notify = _check(store)

        assert result == EnvResult(gap_open=False)
        notify.assert_not_called()
        assert len(store.cleared_rows) == 1
        assert "cleared after 6h 0m" in store.cleared_rows[0]

    def test_clear_row_failure_does_not_raise(self):
        store = FakeStore(_fresh(), Latch(started_at=NOW - HOUR, announcements=1))
        store.fail_writes = True

        result, _ = _check(store)

        assert result.gap_open is False

    def test_clean_stop_context_is_named_in_the_page(self):
        old = NOW - timedelta(hours=5)
        store = FakeStore(Liveness(old, old, None))

        _check(store)

        assert "no active session (clean stop?)" in store.gap_rows[0][0]


class TestWebhookIsNotPostedUnderADatabaseSession:
    def test_page_is_sent_after_the_reads_and_before_the_row_write(self):
        order: list[str] = []
        store = FakeStore(_stale())
        store.read_liveness = lambda: order.append("read") or _stale()  # type: ignore[method-assign]
        original_record = store.record_gap
        store.record_gap = lambda m, *, delivered: (  # type: ignore[method-assign]
            order.append("write"),
            original_record(m, delivered=delivered),
        )
        notify = MagicMock(side_effect=lambda _m: order.append("page") or True)

        _check(store, notify)

        assert order == ["read", "page", "write"]


@pytest.fixture
def sqlite_db():
    """A DatabaseManager stand-in whose reads run against in-memory SQLite."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(
        engine, tables=[AccountHistory.__table__, SystemEvent.__table__, TradingSession.__table__]
    )
    factory = sessionmaker(bind=engine)

    db = MagicMock()

    @contextmanager
    def get_session():
        session = factory()
        try:
            yield session
        finally:
            session.close()

    db.get_session = get_session
    db.factory = factory
    return db


def _naive(moment: datetime) -> datetime:
    return moment.replace(tzinfo=None)


def _add(db, *rows) -> None:
    with db.factory() as session:
        session.add_all(rows)
        session.commit()


def _event(code: str, when: datetime, *, delivered: bool = False) -> SystemEvent:
    return SystemEvent(
        timestamp=_naive(when),
        event_type=EventType.WARNING,
        message=code,
        error_code=code,
        alert_sent=delivered,
    )


def _account_row(when: datetime) -> AccountHistory:
    return AccountHistory(timestamp=_naive(when), balance=100, equity=100)


class TestDbWatchdogStore:
    def test_liveness_reads_the_newest_row_of_each_stream(self, sqlite_db):
        _add(
            sqlite_db,
            _account_row(NOW - timedelta(hours=9)),
            _account_row(NOW - timedelta(hours=3)),
            _event("ENTRIES_ENABLED", NOW - timedelta(hours=8)),
            _event("CLOSE_ONLY_LATCHED", NOW - timedelta(hours=2)),
            _event("ORDER_REJECTED", NOW - timedelta(minutes=1)),
        )

        reading = DbWatchdogStore(sqlite_db).read_liveness()

        assert reading.last_account_history == _naive(NOW - timedelta(hours=3))
        assert reading.last_loop_state == _naive(NOW - timedelta(hours=2))

    def test_liveness_is_none_for_a_database_that_never_ran(self, sqlite_db):
        reading = DbWatchdogStore(sqlite_db).read_liveness()

        assert reading == Liveness(None, None, None)

    def test_liveness_names_the_active_session(self, sqlite_db):
        _add(
            sqlite_db,
            TradingSession(
                id=7,
                session_name="prod-run",
                start_time=_naive(NOW - timedelta(days=2)),
                is_active=True,
                mode=TradeSource.LIVE,
                initial_balance=100,
                strategy_name="hyper_growth",
                symbol="BTCUSDT",
                timeframe="1h",
            ),
        )

        reading = DbWatchdogStore(sqlite_db).read_liveness()

        assert reading.active_session == "session 7 (LIVE) 'prod-run'"

    def test_latch_is_empty_without_watchdog_rows(self, sqlite_db):
        assert DbWatchdogStore(sqlite_db).read_latch() == Latch()

    def test_latch_counts_gap_rows_since_the_last_clear(self, sqlite_db):
        _add(
            sqlite_db,
            _event("HEARTBEAT_GAP", NOW - timedelta(days=5), delivered=True),
            _event("HEARTBEAT_GAP_CLEARED", NOW - timedelta(days=4)),
            _event("HEARTBEAT_GAP", NOW - timedelta(hours=6), delivered=True),
            _event("HEARTBEAT_GAP", NOW - timedelta(hours=5), delivered=False),
        )

        latch = DbWatchdogStore(sqlite_db).read_latch()

        assert latch.started_at == _naive(NOW - timedelta(hours=6))
        assert latch.announcements == 2
        assert latch.last_delivered is False

    def test_latch_is_closed_once_the_newest_watchdog_row_is_a_clear(self, sqlite_db):
        _add(
            sqlite_db,
            _event("HEARTBEAT_GAP", NOW - timedelta(hours=6), delivered=True),
            _event("HEARTBEAT_GAP_CLEARED", NOW - timedelta(hours=1)),
        )

        assert DbWatchdogStore(sqlite_db).read_latch() == Latch()

    def test_record_gap_writes_a_critical_alert_event_with_the_real_outcome(self):
        db = MagicMock()

        DbWatchdogStore(db).record_gap("gap", delivered=True)

        kwargs = db.log_event.call_args.kwargs
        assert kwargs["event_type"] == EventType.ALERT
        assert kwargs["severity"] == "critical"
        assert kwargs["component"] == "watchdog"
        assert kwargs["error_code"] == "HEARTBEAT_GAP"
        assert kwargs["alert_sent"] is True
        assert kwargs["alert_method"] == "webhook"

    def test_record_gap_with_failed_delivery_records_no_alert_method(self):
        db = MagicMock()

        DbWatchdogStore(db).record_gap("gap", delivered=False)

        kwargs = db.log_event.call_args.kwargs
        assert kwargs["alert_sent"] is False
        assert kwargs["alert_method"] is None

    def test_record_cleared_writes_an_info_row_that_pages_nobody(self):
        db = MagicMock()

        DbWatchdogStore(db).record_cleared("resumed")

        kwargs = db.log_event.call_args.kwargs
        assert kwargs["severity"] == "info"
        assert kwargs["error_code"] == "HEARTBEAT_GAP_CLEARED"
        assert "alert_sent" not in kwargs


def _ns(env: str = "all", threshold_minutes: int | None = None) -> argparse.Namespace:
    return argparse.Namespace(env=env, threshold_minutes=threshold_minutes)


@pytest.fixture
def clean_env(monkeypatch):
    for var in (
        "RAILWAY_STAGING_DATABASE_URL",
        "RAILWAY_PRODUCTION_DATABASE_URL",
        "DATABASE_URL",
        "ALERT_WEBHOOK_URL",
        "HEARTBEAT_THRESHOLD_MINUTES",
    ):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def patched_check(clean_env):
    """Route `watchdog_main` through a fake store; yields the check's call log."""
    calls: list[dict] = []
    outcome = {"result": EnvResult(gap_open=False)}

    def fake_check(env, store, **kwargs):
        calls.append({"env": env, **kwargs})
        return outcome["result"]

    with (
        patch.object(wd, "DatabaseManager") as manager_cls,
        patch.object(wd, "check_environment", side_effect=fake_check),
    ):
        manager_cls.return_value.engine = MagicMock()
        yield calls, outcome, manager_cls


class TestWatchdogMain:
    def test_no_configured_targets_is_a_clean_no_op(self, patched_check, capsys):
        calls, _, manager_cls = patched_check

        assert watchdog_main(_ns()) == 0

        manager_cls.assert_not_called()
        assert calls == []
        assert "nothing to check" in capsys.readouterr().out

    def test_all_checks_every_deployed_env_that_has_a_url(self, patched_check, monkeypatch):
        calls, _, _ = patched_check
        monkeypatch.setenv("RAILWAY_STAGING_DATABASE_URL", "postgresql://staging/db")
        monkeypatch.setenv("RAILWAY_PRODUCTION_DATABASE_URL", "postgresql://prod/db")

        assert watchdog_main(_ns()) == 0

        assert [c["env"] for c in calls] == ["production", "staging"]

    def test_all_skips_an_env_without_a_url(self, patched_check, monkeypatch):
        calls, _, _ = patched_check
        monkeypatch.setenv("RAILWAY_PRODUCTION_DATABASE_URL", "postgresql://prod/db")

        watchdog_main(_ns())

        assert [c["env"] for c in calls] == ["production"]

    def test_explicit_env_without_a_url_is_an_error(self, patched_check, capsys):
        _, _, manager_cls = patched_check

        assert watchdog_main(_ns(env="production")) == 1

        manager_cls.assert_not_called()
        assert "RAILWAY_PRODUCTION_DATABASE_URL" in capsys.readouterr().out

    def test_open_gap_exits_non_zero(self, patched_check, monkeypatch):
        _, outcome, _ = patched_check
        outcome["result"] = EnvResult(gap_open=True, paged=True)
        monkeypatch.setenv("RAILWAY_PRODUCTION_DATABASE_URL", "postgresql://prod/db")

        assert watchdog_main(_ns()) == 1

    def test_one_env_gap_fails_the_run_even_when_the_other_is_healthy(
        self, patched_check, monkeypatch
    ):
        monkeypatch.setenv("RAILWAY_STAGING_DATABASE_URL", "postgresql://staging/db")
        monkeypatch.setenv("RAILWAY_PRODUCTION_DATABASE_URL", "postgresql://prod/db")
        results = iter([EnvResult(gap_open=True), EnvResult(gap_open=False)])
        with patch.object(wd, "check_environment", side_effect=lambda *a, **kw: next(results)):
            assert watchdog_main(_ns()) == 1

    def test_unreachable_database_exits_non_zero_and_checks_the_other_env(
        self, clean_env, monkeypatch, capsys
    ):
        monkeypatch.setenv("RAILWAY_STAGING_DATABASE_URL", "postgresql://staging/db")
        monkeypatch.setenv("RAILWAY_PRODUCTION_DATABASE_URL", "postgresql://prod/db")
        managers = [ConnectionError("prod unreachable"), MagicMock()]
        with (
            patch.object(wd, "DatabaseManager", side_effect=managers) as manager_cls,
            patch.object(wd, "check_environment", return_value=EnvResult(gap_open=False)) as check,
        ):
            result = watchdog_main(_ns())

        assert result == 1
        assert manager_cls.call_count == 2
        assert check.call_count == 1
        assert "prod unreachable" in capsys.readouterr().out

    def test_engine_pool_is_disposed_after_the_check(self, patched_check, monkeypatch):
        _, _, manager_cls = patched_check
        monkeypatch.setenv("RAILWAY_PRODUCTION_DATABASE_URL", "postgresql://prod/db")

        watchdog_main(_ns())

        manager_cls.return_value.engine.dispose.assert_called_once()

    def test_threshold_defaults_to_two_hours(self, patched_check, monkeypatch):
        calls, _, _ = patched_check
        monkeypatch.setenv("RAILWAY_PRODUCTION_DATABASE_URL", "postgresql://prod/db")

        watchdog_main(_ns())

        assert calls[0]["threshold"] == timedelta(minutes=120)

    def test_threshold_comes_from_the_environment(self, patched_check, monkeypatch):
        calls, _, _ = patched_check
        monkeypatch.setenv("RAILWAY_PRODUCTION_DATABASE_URL", "postgresql://prod/db")
        monkeypatch.setenv("HEARTBEAT_THRESHOLD_MINUTES", "45")

        watchdog_main(_ns())

        assert calls[0]["threshold"] == timedelta(minutes=45)

    def test_cli_threshold_overrides_the_environment(self, patched_check, monkeypatch):
        calls, _, _ = patched_check
        monkeypatch.setenv("RAILWAY_PRODUCTION_DATABASE_URL", "postgresql://prod/db")
        monkeypatch.setenv("HEARTBEAT_THRESHOLD_MINUTES", "45")

        watchdog_main(_ns(threshold_minutes=30))

        assert calls[0]["threshold"] == timedelta(minutes=30)

    @pytest.mark.parametrize("raw", ["abc", "0", "-5"])
    def test_invalid_environment_threshold_falls_back_to_the_default(
        self, patched_check, monkeypatch, raw
    ):
        calls, _, _ = patched_check
        monkeypatch.setenv("RAILWAY_PRODUCTION_DATABASE_URL", "postgresql://prod/db")
        monkeypatch.setenv("HEARTBEAT_THRESHOLD_MINUTES", raw)

        watchdog_main(_ns())

        assert calls[0]["threshold"] == timedelta(minutes=120)

    def test_webhook_configuration_is_read_from_the_environment(self, patched_check, monkeypatch):
        calls, _, _ = patched_check
        monkeypatch.setenv("RAILWAY_PRODUCTION_DATABASE_URL", "postgresql://prod/db")
        monkeypatch.setenv("ALERT_WEBHOOK_URL", "https://hooks.example/x")

        watchdog_main(_ns())

        assert calls[0]["webhook_configured"] is True


class TestCliWiring:
    @staticmethod
    def _build_parser() -> argparse.ArgumentParser:
        from cli.commands.live import register

        parser = argparse.ArgumentParser()
        register(parser.add_subparsers(dest="cmd"))
        return parser

    def test_live_control_watchdog_parses(self):
        ns = self._build_parser().parse_args(
            ["live-control", "watchdog", "--env", "staging", "--threshold-minutes", "90"]
        )

        assert ns.control_cmd == "watchdog"
        assert ns.env == "staging"
        assert ns.threshold_minutes == 90

    def test_watchdog_defaults_to_every_deployed_environment(self):
        ns = self._build_parser().parse_args(["live-control", "watchdog"])

        assert ns.env == "all"
        assert ns.threshold_minutes is None

    def test_non_positive_threshold_is_rejected(self):
        with pytest.raises(SystemExit):
            self._build_parser().parse_args(
                ["live-control", "watchdog", "--threshold-minutes", "0"]
            )

    def test_control_dispatches_to_the_watchdog(self):
        from cli.commands import live

        ns = argparse.Namespace(control_cmd="watchdog", env="all", threshold_minutes=None)
        with patch.object(live, "watchdog_main", return_value=1) as main:
            assert live._control(ns) == 1

        main.assert_called_once_with(ns)
