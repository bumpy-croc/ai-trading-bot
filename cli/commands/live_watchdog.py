"""External write-gap watchdog: `atb live-control watchdog` (#689).

A dead engine cannot report its own death, so this runs OUTSIDE the engine
process (a scheduled CI job) and judges liveness from what the engine writes to
the database:

- ``account_history`` snapshots (hourly while the engine runs), and
- the loop-state row ``LatchedConditionMonitor`` writes every hour
  (``ENTRIES_ENABLED`` or one of the ``*_LATCHED`` codes).

Either stream going quiet for longer than the threshold raises ``HEARTBEAT_GAP``:
a critical ``system_events`` row plus a webhook page. Audit rows are
event-driven — a quiet audit table is the healthy state — so they are not a
liveness signal.

Alerts follow the latched-condition pattern: the gap pages immediately, then
re-pages on the shared escalating schedule (``paging_schedule``) for as long as
it holds, then writes ``HEARTBEAT_GAP_CLEARED`` once writes resume. The job is
stateless between runs, so the latch is reconstructed from the watchdog's own
``system_events`` rows.

The only writes are the watchdog's own ``HEARTBEAT_GAP`` / ``HEARTBEAT_GAP_CLEARED``
rows. It connects with a bare engine rather than ``DatabaseManager`` so it never runs
``create_all`` against production from whatever branch the scheduler checked out.

The command exits non-zero whenever a gap is open or a check could not be
completed, so the scheduler's own failure notification is a second channel
beside the webhook.

Judged from the GLOBAL newest rows rather than session state, on purpose: a
crashed engine leaves its session ``is_active`` with frozen snapshots while a
clean stop clears ``is_active`` — in both cases the writes simply stop.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Protocol

from sqlalchemy import create_engine, func, insert
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

from cli.commands.live_halt import ENV_DB_URL_VARS, _line, _send_webhook_alert
from src.database.models import AccountHistory, EventType, SystemEvent, TradingSession
from src.infrastructure.paging_schedule import format_elapsed, repage_offset

GAP_CODE = "HEARTBEAT_GAP"
CLEARED_CODE = "HEARTBEAT_GAP_CLEARED"
WATCHDOG_COMPONENT = "watchdog"
# Both streams are written hourly; two missed hours rules out jitter and one
# slow iteration without waiting out a long outage.
DEFAULT_THRESHOLD_MINUTES = 120
THRESHOLD_ENV_VAR = "HEARTBEAT_THRESHOLD_MINUTES"
# Bound on connecting and on any single query, so an unreachable or wedged
# database fails the job instead of hanging it until the CI timeout.
DB_TIMEOUT_SECONDS = 15
# ``--env all`` checks the deployed environments only; development is opt-in.
DEPLOYED_ENVS = ("production", "staging")
# error_codes ``LatchedConditionMonitor`` writes once per hour while the loop runs.
# tests/unit/cli/test_live_watchdog.py pins this to the monitor's own constants.
LOOP_STATE_CODES = (
    "ENTRIES_ENABLED",
    "CLOSE_ONLY_LATCHED",
    "SYSTEM_HALT_LATCHED",
    "CLOSED_CANDLE_GATE_STALLED",
    "ENTRY_PAUSE_LATCHED",
)


@dataclass(frozen=True)
class Liveness:
    """Newest write of each liveness stream; ``None`` means never written."""

    last_account_history: datetime | None
    last_loop_state: datetime | None
    active_session: str | None


@dataclass(frozen=True)
class Latch:
    """What the watchdog itself announced for the gap currently open."""

    started_at: datetime | None = None
    announcements: int = 0
    last_delivered: bool = True


class WatchdogStore(Protocol):
    """Database access the watchdog needs; faked in unit tests."""

    def read_liveness(self) -> Liveness: ...

    def read_latch(self) -> Latch: ...

    def record_gap(self, message: str, *, delivered: bool) -> None: ...

    def record_cleared(self, message: str) -> None: ...


@dataclass(frozen=True)
class EnvResult:
    """Outcome of checking one environment."""

    gap_open: bool
    paged: bool = False


class DbWatchdogStore:
    """`WatchdogStore` over the target environment's database.

    Talks to the database through a bare engine, not ``DatabaseManager``: that
    class runs ``create_all`` on construction, which this unattended job (running
    the default branch's models against production) must never do. Writes name
    their columns explicitly for the same reason — a model column production has
    not migrated yet cannot break the insert.
    """

    def __init__(self, engine: Engine) -> None:
        self._engine = engine
        self._sessions = sessionmaker(bind=engine)

    def read_liveness(self) -> Liveness:
        """Newest ``account_history`` and loop-state timestamps, plus session context."""
        with self._sessions() as session:
            last_account = session.query(func.max(AccountHistory.timestamp)).scalar()
            last_loop = (
                session.query(func.max(SystemEvent.timestamp))
                .filter(SystemEvent.error_code.in_(LOOP_STATE_CODES))
                .scalar()
            )
            active = (
                session.query(TradingSession.id, TradingSession.mode, TradingSession.session_name)
                .filter(TradingSession.is_active.is_(True))
                .order_by(TradingSession.start_time.desc())
                .first()
            )
        context = None
        if active is not None:
            mode = getattr(active.mode, "value", active.mode)
            context = f"session {active.id} ({mode}) '{active.session_name}'"
        return Liveness(last_account, last_loop, context)

    def read_latch(self) -> Latch:
        """Reconstruct the open gap from watchdog rows written since the last clear."""
        with self._sessions() as session:
            cleared_at = (
                session.query(func.max(SystemEvent.timestamp))
                .filter(SystemEvent.error_code == CLEARED_CODE)
                .scalar()
            )
            query = session.query(SystemEvent.timestamp, SystemEvent.alert_sent).filter(
                SystemEvent.error_code == GAP_CODE
            )
            if cleared_at is not None:
                query = query.filter(SystemEvent.timestamp > cleared_at)
            rows = query.order_by(SystemEvent.timestamp.asc(), SystemEvent.id.asc()).all()
        if not rows:
            return Latch()
        return Latch(
            started_at=rows[0].timestamp,
            announcements=len(rows),
            last_delivered=bool(rows[-1].alert_sent),
        )

    def record_gap(self, message: str, *, delivered: bool) -> None:
        """Write the critical ``HEARTBEAT_GAP`` row, recording the real page outcome."""
        self._insert_event(
            EventType.ALERT,
            message,
            severity="critical",
            error_code=GAP_CODE,
            alert_sent=delivered,
            alert_method="webhook" if delivered else None,
        )

    def record_cleared(self, message: str) -> None:
        """Write the durable resolution row (no page: a resolution is for monitors)."""
        self._insert_event(
            EventType.WARNING,
            message,
            severity="info",
            error_code=CLEARED_CODE,
            alert_sent=False,
            alert_method=None,
        )

    def _insert_event(
        self,
        event_type: EventType,
        message: str,
        *,
        severity: str,
        error_code: str,
        alert_sent: bool,
        alert_method: str | None,
    ) -> None:
        """Insert one ``system_events`` row naming exactly the columns it sets."""
        statement = insert(SystemEvent.__table__).values(
            timestamp=datetime.now(UTC),
            event_type=event_type,
            severity=severity,
            message=message,
            component=WATCHDOG_COMPONENT,
            error_code=error_code,
            alert_sent=alert_sent,
            alert_method=alert_method,
        )
        with self._engine.begin() as connection:
            connection.execute(statement)


def connect(url: str) -> Engine:
    """A timeout-bounded engine for ``url`` (no schema creation, no migration)."""
    connect_args: dict[str, object] = {}
    if url.startswith("postgresql"):
        connect_args = {
            "connect_timeout": DB_TIMEOUT_SECONDS,
            "options": f"-c statement_timeout={DB_TIMEOUT_SECONDS * 1000}",
        }
    return create_engine(url, pool_pre_ping=True, connect_args=connect_args)


def _as_aware(moment: datetime) -> datetime:
    """The engine stores naive UTC datetimes; compare them as UTC."""
    return moment if moment.tzinfo is not None else moment.replace(tzinfo=UTC)


def find_gaps(reading: Liveness, now: datetime, threshold: timedelta) -> list[str]:
    """Describe every liveness stream that has been quiet longer than ``threshold``.

    A stream that has never been written is skipped: a fresh database, or an
    environment still on code that predates the loop-state rows, is not an outage.
    """
    gaps: list[str] = []
    streams = (
        ("account_history", reading.last_account_history),
        ("loop-state row", reading.last_loop_state),
    )
    for name, last_write in streams:
        if last_write is None:
            continue
        age = now - _as_aware(last_write)
        if age > threshold:
            gaps.append(
                f"{name} last written {format_elapsed(age.total_seconds())} ago "
                f"({_as_aware(last_write).strftime('%Y-%m-%dT%H:%M:%SZ')})"
            )
    return gaps


def page_is_due(latch: Latch, now: datetime, *, webhook_configured: bool) -> bool:
    """True when the open gap should be announced on this run.

    The first announcement is immediate; later ones follow the shared escalating
    schedule measured from the first announcement. An undelivered page is retried
    on the next run — but only when a webhook is configured, so a missing channel
    cannot write a row every run.
    """
    if latch.announcements == 0 or latch.started_at is None:
        return True
    if webhook_configured and not latch.last_delivered:
        return True
    elapsed = (now - _as_aware(latch.started_at)).total_seconds()
    return elapsed >= repage_offset(latch.announcements - 1)


def _gap_message(
    env: str, gaps: list[str], reading: Liveness, latch: Latch, now: datetime, threshold: timedelta
) -> str:
    """The operator page: what is stale, since when, and what it implies."""
    held_seconds = (now - _as_aware(latch.started_at)).total_seconds() if latch.started_at else 0.0
    days = int(held_seconds // 86400)
    # A day-4 page that reads like the day-0 one is easy to scroll past twice.
    prefix = f"STILL DOWN — DAY {days}: " if days >= 1 else ""
    context = reading.active_session or "no active session (clean stop?)"
    held = f" Announced {format_elapsed(held_seconds)} ago." if latch.started_at else ""
    return (
        f"{prefix}HEARTBEAT GAP [{env}]: {'; '.join(gaps)} "
        f"(threshold {format_elapsed(threshold.total_seconds())}).{held} {context}. "
        "The engine may be dead or wedged with positions unmanaged — check the service "
        "logs. Re-announced on an escalating schedule until writes resume."
    )


def check_environment(
    env: str,
    store: WatchdogStore,
    *,
    now: datetime,
    threshold: timedelta,
    notify: Callable[[str], bool],
    webhook_configured: bool,
) -> EnvResult:
    """Judge one environment, page / clear as the latch requires, return its state.

    The webhook is posted with no database session or lock held, and a failed
    row write never suppresses the page that already went out.
    """
    reading = store.read_liveness()
    latch = store.read_latch()
    gaps = find_gaps(reading, now, threshold)

    if not gaps:
        if latch.started_at is not None:
            held = format_elapsed((now - _as_aware(latch.started_at)).total_seconds())
            message = f"Heartbeat gap [{env}] cleared after {held} — engine writes have resumed."
            _line(message)
            _record(lambda: store.record_cleared(message), "HEARTBEAT_GAP_CLEARED")
        else:
            _line(f"[{env}] OK — {_freshness(reading, now)}")
        return EnvResult(gap_open=False)

    message = _gap_message(env, gaps, reading, latch, now, threshold)
    if not page_is_due(latch, now, webhook_configured=webhook_configured):
        _line(f"[{env}] GAP (already announced) — {'; '.join(gaps)}")
        return EnvResult(gap_open=True)

    _line(f"[{env}] GAP — {message}")
    delivered = notify(message)
    if not delivered:
        _line(
            f"WARNING: [{env}] no alert webhook delivery (ALERT_WEBHOOK_URL unset or POST "
            "failed) — this job's failure is the only notification"
        )
    _record(lambda: store.record_gap(message, delivered=delivered), "HEARTBEAT_GAP")
    return EnvResult(gap_open=True, paged=delivered)


def _freshness(reading: Liveness, now: datetime) -> str:
    """One-line summary of how recent each stream is."""
    parts = []
    for name, last_write in (
        ("account_history", reading.last_account_history),
        ("loop-state row", reading.last_loop_state),
    ):
        if last_write is None:
            parts.append(f"no {name} yet")
        else:
            age = (now - _as_aware(last_write)).total_seconds()
            parts.append(f"{name} {format_elapsed(age)} ago")
    return ", ".join(parts)


def _record(write: Callable[[], None], code: str) -> None:
    """Run a row write; the page has already gone out, so only report a failure."""
    try:
        write()
    except Exception as e:
        _line(f"WARNING: could not write the {code} system_events row: {e}")


def _targets(env: str) -> list[str]:
    """Environments to check for ``--env``."""
    return list(DEPLOYED_ENVS) if env == "all" else [env]


def _threshold_minutes(value: int | None) -> int:
    """CLI value, else ``$HEARTBEAT_THRESHOLD_MINUTES``, else the default."""
    if value is not None:
        return value
    raw = os.environ.get(THRESHOLD_ENV_VAR)
    if not raw:
        return DEFAULT_THRESHOLD_MINUTES
    try:
        parsed = int(raw)
    except ValueError:
        return DEFAULT_THRESHOLD_MINUTES
    return parsed if parsed > 0 else DEFAULT_THRESHOLD_MINUTES


def watchdog_main(ns: argparse.Namespace) -> int:
    """Check each target environment; exit 1 on an open gap or an unverifiable check."""
    threshold = timedelta(minutes=_threshold_minutes(getattr(ns, "threshold_minutes", None)))
    webhook_configured = bool(os.environ.get("ALERT_WEBHOOK_URL"))
    checked = 0
    failures = 0
    for env in _targets(ns.env):
        url = os.environ.get(ENV_DB_URL_VARS[env])
        if not url:
            if ns.env == "all":
                continue  # not every deployment has every secret configured
            _line(f"ERROR: {ENV_DB_URL_VARS[env]} is not set — cannot reach the {env} database.")
            return 1
        checked += 1
        try:
            engine = connect(url)
            try:
                result = check_environment(
                    env,
                    DbWatchdogStore(engine),
                    now=datetime.now(UTC),
                    threshold=threshold,
                    notify=_send_webhook_alert,
                    webhook_configured=webhook_configured,
                )
            finally:
                engine.dispose()
        except Exception as e:
            # Could not confirm liveness — that is itself alert-worthy.
            _line(f"ERROR: [{env}] failed to check ({type(e).__name__}: {e})")
            failures += 1
            continue
        failures += int(result.gap_open)

    if checked == 0:
        _line("WARNING: no target database URLs configured — nothing to check.")
        return 0
    return 1 if failures else 0


def _positive_int(value: str) -> int:
    """argparse type for a strictly positive integer."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register the watchdog's flags on ``parser`` (shared by the atb subcommand and ``-m``)."""
    parser.add_argument(
        "--env",
        default="all",
        choices=("all", *ENV_DB_URL_VARS),
        help="Environment to check; 'all' checks every deployed environment with a configured URL",
    )
    parser.add_argument(
        "--threshold-minutes",
        type=_positive_int,
        default=None,
        help=(
            f"Quiet period that counts as a gap (default ${THRESHOLD_ENV_VAR} or "
            f"{DEFAULT_THRESHOLD_MINUTES})"
        ),
    )


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``python -m cli.commands.live_watchdog`` (the scheduler's job)."""
    parser = argparse.ArgumentParser(description="Engine write-gap dead-man's-switch")
    add_arguments(parser)
    return watchdog_main(parser.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
