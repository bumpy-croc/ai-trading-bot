"""Install (or refresh) the worktree import shim into the active interpreter's site-packages.

Run automatically by ``make install``; safe to re-run at any time.

Copies ``tools/atb_worktree_shim.py`` next to the venv's other site-packages modules and writes
a ``.pth`` file that imports it on interpreter start. See GH #1070 for the defect this closes.

Usage:
    python tools/install_worktree_shim.py [--check] [--uninstall]

``--check`` exits non-zero when the installed copy is missing or out of date, so CI and
bootstrap scripts can assert the shim is current without mutating anything.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import sysconfig
from pathlib import Path

MODULE_NAME = "atb_worktree_shim"
# 'zz_' keeps the filename late in site.py's sorted .pth processing order, after the editable
# install's own .pth. Ordering is not load-bearing (the shim inserts at meta_path[0] either
# way) but it keeps the intent obvious to anyone reading site-packages.
PTH_NAME = "zz_atb_worktree_shim.pth"
PTH_CONTENT = f"import {MODULE_NAME}; {MODULE_NAME}.install_quietly()\n"

SOURCE = Path(__file__).resolve().parent / f"{MODULE_NAME}.py"


def site_packages() -> Path:
    return Path(sysconfig.get_paths()["purelib"])


def _installed_paths() -> tuple[Path, Path]:
    target = site_packages()
    return target / f"{MODULE_NAME}.py", target / PTH_NAME


def is_current() -> bool:
    module_dst, pth_dst = _installed_paths()
    if not module_dst.is_file() or not pth_dst.is_file():
        return False
    if pth_dst.read_text(encoding="utf-8") != PTH_CONTENT:
        return False
    return module_dst.read_text(encoding="utf-8") == SOURCE.read_text(encoding="utf-8")


def install() -> None:
    module_dst, pth_dst = _installed_paths()
    module_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(SOURCE, module_dst)
    pth_dst.write_text(PTH_CONTENT, encoding="utf-8")
    print(f"atb worktree import shim installed -> {module_dst}")


def uninstall() -> None:
    for path in _installed_paths():
        path.unlink(missing_ok=True)
    print("atb worktree import shim removed")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify without installing")
    parser.add_argument("--uninstall", action="store_true", help="remove the shim")
    args = parser.parse_args(argv)

    if args.uninstall:
        uninstall()
        return 0
    if args.check:
        if is_current():
            print("atb worktree import shim is installed and current")
            return 0
        print(
            "atb worktree import shim is MISSING or STALE. Run: python tools/install_worktree_shim.py",
            file=sys.stderr,
        )
        return 1
    install()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
