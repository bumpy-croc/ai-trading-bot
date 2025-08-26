from __future__ import annotations

import argparse

from cli.commands.test_commands import (
    test_database_main,
    test_download_main,
    test_secrets_access_main,
    heartbeat_main,
)


def _handle_db(ns: argparse.Namespace) -> int:
    return test_database_main(ns)


def _handle_download(ns: argparse.Namespace) -> int:
    return test_download_main(ns)


def _handle_secrets(ns: argparse.Namespace) -> int:
    return test_secrets_access_main(ns)


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
        return heartbeat_main(ns)

    p_hb = sub.add_parser("heartbeat", help="Log a heartbeat SystemEvent")
    p_hb.set_defaults(func=_handle_heartbeat)
