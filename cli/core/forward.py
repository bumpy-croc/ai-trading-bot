from __future__ import annotations

import importlib
import sys


def forward_to_module_main(module_name: str, argv_tail: list[str]) -> int:
    """Forward execution to another script module's main(), preserving its CLI.

    We temporarily replace sys.argv so the target module's argparse parses the tail as intended.
    """
    module = importlib.import_module(module_name)
    if not hasattr(module, "main"):
        print(f"Module '{module_name}' does not expose a main() function")
        return 1

    original_argv = sys.argv[:]
    try:
        sys.argv = [module_name.split(".")[-1]] + argv_tail
        return int(module.main() or 0)
    finally:
        sys.argv = original_argv
