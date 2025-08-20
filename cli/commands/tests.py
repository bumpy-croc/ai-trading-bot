from __future__ import annotations

import argparse

from cli.core.forward import forward_to_module_main


def _handle_db(ns: argparse.Namespace) -> int:
    return forward_to_module_main("scripts.test_database", ns.args or [])


def _handle_download(ns: argparse.Namespace) -> int:
    return forward_to_module_main("scripts.test_download", ns.args or [])


def _handle_secrets(ns: argparse.Namespace) -> int:
    return forward_to_module_main("scripts.test_secrets_access", ns.args or [])


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("tests", help="One-off test/diagnostic scripts")
    sub = p.add_subparsers(dest="tests_cmd", required=True)

    p_db = sub.add_parser("db", help="Database test script")
    p_db.add_argument("args", nargs=argparse.REMAINDER)
    p_db.set_defaults(func=_handle_db)

    p_dl = sub.add_parser("download", help="Download test script")
    p_dl.add_argument("args", nargs=argparse.REMAINDER)
    p_dl.set_defaults(func=_handle_download)

    p_sec = sub.add_parser("secrets", help="Secrets access test script")
    p_sec.add_argument("args", nargs=argparse.REMAINDER)
    p_sec.set_defaults(func=_handle_secrets)

    # Heartbeat job
    def _handle_heartbeat(ns: argparse.Namespace) -> int:
        import os
        import sys
        from datetime import datetime

        from database.manager import DatabaseManager
        from database.models import EventType

        component = os.getenv("HEARTBEAT_COMPONENT", "scheduler")
        try:
            db = DatabaseManager()
        except Exception as exc:
            print(f"❌ Failed to init DB manager: {exc}", file=sys.stderr)
            return 1
        db.log_event(
            event_type=EventType.TEST,
            message="Heartbeat",
            severity="info",
            component=component,
            details={"timestamp": datetime.utcnow().isoformat()},
        )
        print("✅ Heartbeat logged")
        return 0

    p_hb = sub.add_parser("heartbeat", help="Log a heartbeat SystemEvent")
    p_hb.set_defaults(func=_handle_heartbeat)
