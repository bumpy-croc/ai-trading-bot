from __future__ import annotations

import argparse

from cli.core.forward import forward_to_module_main


def _handle(ns: argparse.Namespace) -> int:
    tail = ns.args or []
    return forward_to_module_main("scripts.run_live_trading", tail)


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("live", help="Run live trading (proxies to scripts.run_live_trading)")
    p.add_argument("args", nargs=argparse.REMAINDER, help="Arguments passed through to runner")
    p.set_defaults(func=_handle)


