from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AI Trading Bot unified CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Import and register command groups
    from cli.commands import (
        backtest,
        dashboards,
        data,
        db,
        live,
        live_health,
        optimizer,
        tests,
        train,
    )
    from cli.commands import scripts as scripts_cmd

    dashboards.register(subparsers)
    live.register(subparsers)
    backtest.register(subparsers)
    optimizer.register(subparsers)
    scripts_cmd.register(subparsers)
    live_health.register(subparsers)
    data.register(subparsers)
    db.register(subparsers)
    train.register(subparsers)
    tests.register(subparsers)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    ns = parser.parse_args(argv)

    func = getattr(ns, "func", None)
    if callable(func):
        return int(func(ns) or 0)

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
