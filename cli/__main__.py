from __future__ import annotations

import argparse
import sys
from typing import Optional


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AI Trading Bot unified CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Import and register command groups
    from cli.commands import dashboards, live, backtest, optimizer, scripts as scripts_cmd

    dashboards.register(subparsers)
    live.register(subparsers)
    backtest.register(subparsers)
    optimizer.register(subparsers)
    scripts_cmd.register(subparsers)

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    ns = parser.parse_args(argv)

    func = getattr(ns, "func", None)
    if callable(func):
        return int(func(ns) or 0)

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())


