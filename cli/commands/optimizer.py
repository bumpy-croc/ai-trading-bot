from __future__ import annotations

import argparse

from cli.core.forward import forward_to_module_main


def _handle(ns: argparse.Namespace) -> int:
    tail = ns.args or []
    return forward_to_module_main("scripts.run_optimizer", tail)


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("optimizer", help="Run optimizer (proxies to scripts.run_optimizer)")
    p.add_argument("args", nargs=argparse.REMAINDER, help="Arguments passed through to runner")
    p.set_defaults(func=_handle)


