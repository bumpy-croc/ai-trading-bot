#!/usr/bin/env python3
"""Automated backup of the PostgreSQL database to local storage.

This script is intended to be run by a scheduler/cron (e.g. Railway Scheduled Jobs)
Every execution performs the following steps:
1. Parse `DATABASE_URL` from environment/config (same resolver as the app).
2. Generate a compressed custom-format pg_dump archive.
3. Save the dump to local storage under `backups/<db_name>/YYYY/mm/dd/backup-<timestamp>.dump`.
4. Optionally prune backups older than the configured retention period.

Environment Variables (or CLI flags override):
• DATABASE_URL             – PostgreSQL URL
• BACKUP_DIR               – Local backup directory (default: ./backups)
• BACKUP_RETENTION_DAYS    – How long to keep backups (default 7)
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PostgreSQL → Local backup")
    parser.add_argument(
        "--backup-dir", default=os.getenv("BACKUP_DIR", "./backups"), help="Local backup directory"
    )
    parser.add_argument(
        "--retention",
        type=int,
        default=int(os.getenv("BACKUP_RETENTION_DAYS", 7)),
        help="Retention in days",
    )
    return parser.parse_args()


def _get_db_params(db_url: str):
    """Return (dbname, user, host, port, password)."""
    parsed = urlparse(db_url)
    return (
        parsed.path.lstrip("/"),  # dbname
        parsed.username,
        parsed.hostname,
        str(parsed.port or 5432),
        parsed.password,
    )


# ---------------------------------------------------------------------------
# Main backup routine
# ---------------------------------------------------------------------------


def perform_backup(backup_dir: str, retention_days: int) -> None:
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        print("❌ DATABASE_URL not set", file=sys.stderr)
        sys.exit(1)

    dbname, user, host, port, password = _get_db_params(db_url)

    timestamp = _dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    backup_path = Path(backup_dir) / dbname / _dt.datetime.utcnow().strftime("%Y/%m/%d")
    dump_filename = f"backup-{timestamp}.dump"
    dump_path = backup_path / dump_filename

    # Create backup directory
    backup_path.mkdir(parents=True, exist_ok=True)
    print(f"📦 Creating dump: {dump_path}")

    env = os.environ.copy()
    env["PGPASSWORD"] = password or ""  # pg_dump reads password from env

    # Run pg_dump in custom format, highest compression
    cmd = [
        "pg_dump",
        f"--dbname={db_url}",
        "-Fc",  # custom format (compressed, safe for restore)
        "-Z",
        "9",  # maximum compression
        "-f",
        str(dump_path),
    ]
    try:
        subprocess.run(cmd, check=True, env=env, capture_output=True)
    except subprocess.CalledProcessError as exc:
        print(f"❌ pg_dump failed: {exc.stderr.decode()}", file=sys.stderr)
        sys.exit(1)

    print("✅ Backup created successfully")

    # Retention policy – delete files older than retention_days
    if retention_days > 0:
        cutoff = _dt.datetime.utcnow() - _dt.timedelta(days=retention_days)
        print(f"🧹 Deleting backups older than {retention_days} days (before {cutoff.date()})")

        db_backup_dir = Path(backup_dir) / dbname
        if db_backup_dir.exists():
            for backup_file in db_backup_dir.rglob("backup-*.dump"):
                try:
                    file_time = _dt.datetime.fromtimestamp(backup_file.stat().st_mtime)
                    if file_time < cutoff:
                        print(f"   • Removing {backup_file}")
                        backup_file.unlink()
                except OSError as e:
                    print(f"   • Warning: Could not check/remove {backup_file}: {e}")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args_ns = _parse_args()
    perform_backup(args_ns.backup_dir, args_ns.retention)
