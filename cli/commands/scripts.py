from __future__ import annotations

import argparse
import importlib
from pathlib import Path

from cli.core.forward import forward_to_module_main


def _list_scripts() -> list[str]:
    root = Path(__file__).resolve().parents[3]
    out: list[str] = []
    for fp in sorted((root / "scripts").glob("*.py")):
        if fp.name in {"postgres-init.sql"}:
            continue
        mod_name = f"scripts.{fp.stem}"
        try:
            mod = importlib.import_module(mod_name)
            if hasattr(mod, "main") and callable(mod.main):
                out.append(fp.stem)
        except Exception:
            continue
    return out


def _handle_list(_ns: argparse.Namespace) -> int:
    items = _list_scripts()
    if not items:
        print("No runnable script modules found in scripts/")
        return 0
    print("Runnable scripts:")
    for name in items:
        print(f"- {name}")
    return 0


def _handle_run(ns: argparse.Namespace) -> int:
    mod = f"scripts.{ns.name}"
    tail = ns.args or []
    return forward_to_module_main(mod, tail)


def register(subparsers: argparse._SubParsersAction) -> None:
    p_scripts = subparsers.add_parser("scripts", help="Run any script module in scripts/ by name")
    sub_scripts = p_scripts.add_subparsers(dest="scripts_cmd", required=True)

    p_list = sub_scripts.add_parser("list", help="List runnable script modules")
    p_list.set_defaults(func=_handle_list)

    p_run = sub_scripts.add_parser("run", help="Run a script module by name")
    p_run.add_argument("name", help="Module name under scripts/ (without .py)")
    p_run.add_argument("args", nargs=argparse.REMAINDER, help="Args passed to the module")
    p_run.set_defaults(func=_handle_run)
