#!/usr/bin/env python3
"""Heartbeat staleness monitor for the live trading bots.

Connects (read-only) to one or more environment databases and verifies that the
bot is still writing ``account_history`` snapshots. Exits non-zero if a
heartbeat is stale, so a scheduled CI job (``.github/workflows/heartbeat-monitor.yml``)
fails and notifies maintainers.

Why this exists: on 2026-05-19 both the staging and production bots died
(Railway internal-DNS outage made Postgres unresolvable) and then stayed
*silently* down for ~12 days. The ``account_history`` snapshot (written every
~30 min while running) is the canonical liveness signal.

Liveness is judged from the GLOBAL latest ``account_history`` snapshot rather
than from session state, on purpose: a crashed bot leaves its session
``is_active=True`` with a frozen snapshot, while a clean ``stop()`` sets
``is_active=False`` — in BOTH cases the snapshot simply stops advancing, so the
global max timestamp is the reliable signal. (Filtering on ``is_active`` would
either miss clean-stop deaths or false-positive forever on stale crashed-active
rows.) The active session, if any, is reported only as context.

Read-only: issues only SELECTs against a read-only session. Safe for production.

Configuration (environment variables):
  RAILWAY_STAGING_DATABASE_URL      staging DB connection string (optional)
  RAILWAY_PRODUCTION_DATABASE_URL   production DB connection string (optional)
  HEARTBEAT_THRESHOLD_MINUTES       staleness threshold, default 120

Exit codes: 0 = healthy or nothing to check, 1 = stale heartbeat or check error.
"""
from __future__ import annotations

import os
import sys
from datetime import UTC, datetime

try:
    import psycopg2
except ImportError:  # pragma: no cover
    print("ERROR: psycopg2 is required (pip install psycopg2-binary)", file=sys.stderr)
    sys.exit(1)

DEFAULT_THRESHOLD_MINUTES = 120

# Environment variable holding the connection string -> friendly label.
TARGETS = {
    "RAILWAY_STAGING_DATABASE_URL": "staging",
    "RAILWAY_PRODUCTION_DATABASE_URL": "production",
}


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _as_aware(ts: datetime | None) -> datetime | None:
    """Treat naive timestamps as UTC (the bot stores naive UTC datetimes)."""
    if ts is None:
        return None
    if ts.tzinfo is None:
        return ts.replace(tzinfo=UTC)
    return ts


def check_target(label: str, url: str, threshold_minutes: int) -> list[str]:
    """Return a list of staleness problems for one environment (empty == healthy)."""
    problems: list[str] = []
    conn = psycopg2.connect(url, connect_timeout=15)
    try:
        # Enforce read-only at the connection level — defense in depth for prod.
        conn.set_session(readonly=True, autocommit=True)
        with conn.cursor() as cur:
            cur.execute("SELECT max(timestamp) FROM account_history;")
            row = cur.fetchone()
            last_hb = row[0] if row else None
            cur.execute(
                "SELECT id, mode, session_name FROM trading_sessions "
                "WHERE is_active IS TRUE ORDER BY start_time DESC LIMIT 1;"
            )
            active = cur.fetchone()
    finally:
        conn.close()

    ctx = (
        f"active session {active[0]} ({active[1]}) '{active[2]}'" if active else "no active session"
    )
    last_hb = _as_aware(last_hb)
    if last_hb is None:
        # Never run here (fresh/unused DB) — not an outage.
        print(f"[{label}] no account_history yet — skipping ({ctx})")
        return problems

    age_h = (_utcnow() - last_hb).total_seconds() / 3600.0
    if age_h * 60.0 > threshold_minutes:
        detail = (
            f"last account_history {age_h:.1f}h ago ({last_hb.isoformat()}), "
            f"{ctx} (threshold {threshold_minutes}m)"
        )
        print(f"[{label}] STALE — {detail}")
        problems.append(f"{label} heartbeat stale: {detail}")
    else:
        print(f"[{label}] OK — last heartbeat {age_h:.1f}h ago, {ctx}")
    return problems


def main() -> int:
    threshold = int(os.environ.get("HEARTBEAT_THRESHOLD_MINUTES", DEFAULT_THRESHOLD_MINUTES))
    checked = 0
    all_problems: list[str] = []
    errors: list[str] = []
    for env_var, label in TARGETS.items():
        url = os.environ.get(env_var)
        if not url:
            continue
        checked += 1
        try:
            all_problems.extend(check_target(label, url, threshold))
        except Exception as e:  # noqa: BLE001 - any failure to verify liveness is alert-worthy
            errors.append(f"{label}: failed to check ({type(e).__name__}: {e})")

    if checked == 0:
        # No targets configured (e.g. secrets not set). Don't fail recurring CI;
        # surface a clear warning and exit cleanly.
        print(
            "WARNING: no target database URLs configured "
            "(set RAILWAY_STAGING_DATABASE_URL / RAILWAY_PRODUCTION_DATABASE_URL). "
            "Nothing to check.",
            file=sys.stderr,
        )
        return 0

    for err in errors:
        print(f"ERROR: {err}", file=sys.stderr)

    if all_problems:
        print("\nHEARTBEAT ALERT — stale live-bot heartbeat(s) detected:", file=sys.stderr)
        for p in all_problems:
            print(f"  - {p}", file=sys.stderr)
        return 1

    # A connection/query failure means we could not confirm liveness — treat as alert.
    if errors:
        return 1

    print(f"\nAll heartbeats healthy (threshold {threshold}m).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
