from __future__ import annotations

import argparse
import os
from typing import Any

from cli.core.discovery import DiscoveredDashboard, call_with_supported_params, discover_dashboards, _import_module


def _run_dashboard(name: str, host: str | None, port: int | None, debug: bool, extra_init: dict[str, Any] | None = None) -> int:
    dashboards = discover_dashboards()
    if name not in dashboards:
        available = ", ".join(sorted(dashboards)) or "(none found)"
        print(f"Unknown dashboard: {name}. Available: {available}")
        return 1

    d: DiscoveredDashboard = dashboards[name]
    mod = _import_module(d.module_name)

    if d.object_name is not None:
        # Instantiate dashboard class (pass only supported init kwargs if provided)
        cls = getattr(mod, d.object_name)
        init_kwargs = extra_init or {}
        try:
            instance = call_with_supported_params(cls, init_kwargs)
        except TypeError:
            # If passing kwargs failed, try default constructor
            instance = cls()

        # Call run(host, port, debug) with supported params only
        run_kwargs = {"host": host, "port": port, "debug": debug}
        call_with_supported_params(getattr(instance, "run"), run_kwargs)
        return 0

    # Fallback to module main()
    main_func = getattr(mod, "main")
    main_func()
    return 0


def _handle_list(_ns: argparse.Namespace) -> int:
    discovered = discover_dashboards()
    if not discovered:
        print("No dashboards found under src/dashboards")
        return 0
    print("Available dashboards:")
    for name, d in sorted(discovered.items()):
        line = f"- {name}"
        if d.summary:
            short = d.summary.strip().splitlines()[0]
            if len(short) > 120:
                short = short[:117] + "..."
            line += f": {short}"
        print(line)
    return 0


def _handle_run(ns: argparse.Namespace) -> int:
    extra_init = {k: getattr(ns, k) for k in ["db_url", "update_interval", "logs_dir"]}
    extra_init = {k: v for k, v in extra_init.items() if v is not None}
    return _run_dashboard(ns.name, ns.host, ns.port, ns.debug, extra_init)


def register(subparsers: argparse._SubParsersAction) -> None:
    p_dash = subparsers.add_parser("dashboards", help="List and run available dashboards")
    sub_dash = p_dash.add_subparsers(dest="dash_cmd", required=True)

    p_list = sub_dash.add_parser("list", help="List discovered dashboards")
    p_list.set_defaults(func=_handle_list)

    p_run = sub_dash.add_parser("run", help="Run a dashboard by name")
    p_run.add_argument("name", help="Dashboard name (e.g., monitoring, backtesting)")
    p_run.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    p_run.add_argument("--port", type=int, default=None)
    p_run.add_argument("--debug", action="store_true")
    # Optional init overrides commonly accepted by dashboards
    p_run.add_argument("--db-url", dest="db_url", default=None)
    p_run.add_argument("--update-interval", dest="update_interval", type=int, default=None)
    p_run.add_argument("--logs-dir", dest="logs_dir", default=None)
    p_run.set_defaults(func=_handle_run)


