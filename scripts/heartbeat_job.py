#!/usr/bin/env python3
"""Heartbeat job that logs a SystemEvent every N minutes.

Run this script via cron/scheduler (e.g. Railway Scheduled Jobs) every 5 minutes
or as needed.  The monitoring system can then alert if no heartbeat rows have
been inserted for a threshold duration, signalling that the trading engine or
scheduler is down.

Environment variables:
• DATABASE_URL – PostgreSQL URL (inherited from existing config system)
• HEARTBEAT_COMPONENT – Optional component name (default: 'scheduler')
"""

from __future__ import annotations

import os
import sys
from datetime import datetime

# Ensure project root is on path
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from database.manager import DatabaseManager  # type: ignore
from database.models import EventType  # type: ignore


def main() -> None:
    component = os.getenv("HEARTBEAT_COMPONENT", "scheduler")

    try:
        db = DatabaseManager()
    except Exception as exc:
        print(f"❌ Failed to init DB manager: {exc}", file=sys.stderr)
        sys.exit(1)

    db.log_event(
        event_type=EventType.TEST,
        message="Heartbeat",
        severity="info",
        component=component,
        details={"timestamp": datetime.utcnow().isoformat()},
    )
    print("✅ Heartbeat logged")


if __name__ == "__main__":
    main()
