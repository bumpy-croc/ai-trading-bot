"""CI check: a PR that records a "NO CHANGE" model promotion must not move any ``latest``.

``docs/research/model-promotions.md`` gets a ``NO CHANGE`` entry when a retrain was
evaluated and the incumbent retained. If the same PR also adds or repoints a
``src/ml/models/**/latest`` symlink, the registry would serve the rejected candidate as
production while the document says nothing moved (GH #1135, #1242).

Usage (from a checkout with the base ref fetched)::

    python tools/check_promotion_log.py --base origin/develop
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys

PROMOTION_LOG = "docs/research/model-promotions.md"
NO_CHANGE_HEADING = re.compile(r"^\+#{1,6}\s.*\bNO CHANGE\b")
LATEST_POINTER = re.compile(r"(^|/)models/.+/latest$")


def added_no_change_headings(log_diff: str) -> list[str]:
    """Return the ``NO CHANGE`` headings a unified diff of the promotion log adds."""
    return [line[1:].strip() for line in log_diff.splitlines() if NO_CHANGE_HEADING.match(line)]


def moved_latest_pointers(changed_paths: list[str]) -> list[str]:
    """Return the changed paths that are registry ``latest`` pointers."""
    return [path for path in changed_paths if LATEST_POINTER.search(path)]


def violations(log_diff: str, changed_paths: list[str]) -> list[str]:
    """Describe each contradiction between the log's NO CHANGE claims and the diff."""
    if not added_no_change_headings(log_diff):
        return []
    return [
        f"{path} changed in a PR that records a NO CHANGE promotion"
        for path in moved_latest_pointers(changed_paths)
    ]


def _git(*args: str) -> str:
    return subprocess.run(  # nosec B603 B607 — fixed git argv, ref comes from the CI workflow
        ["git", *args], check=True, capture_output=True, text=True
    ).stdout


def main(argv: list[str] | None = None) -> int:
    """Check the current checkout against ``--base``; non-zero means a contradiction."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="Base ref to diff against")
    args = parser.parse_args(argv)

    log_diff = _git("diff", f"{args.base}...HEAD", "--unified=0", "--", PROMOTION_LOG)
    changed = _git("diff", "--name-only", f"{args.base}...HEAD").splitlines()
    problems = violations(log_diff, changed)
    for problem in problems:
        print(f"ERROR: {problem}", file=sys.stderr)
    if problems:
        print(
            f"{PROMOTION_LOG} says no model was promoted, but a latest pointer moved. "
            "Drop the latest change from this PR or rewrite the entry as a promotion.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
