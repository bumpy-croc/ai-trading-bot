"""Randomised adversarial merges against the append-only driver (GH #1079).

This harness ships with the driver deliberately. Across three review rounds, seven
content-loss paths were found in `merge_append_only.py`; the seventh — two sides quoting the
same header, whose identical blocks collapse to one token and let the diff thread one
contribution through the middle of the other — was found by this harness after careful code
review by two people had missed it. Both reviewers independently concluded that reading the
code is not a reliable filter for this design, so the thing that *was* reliable should outlive
the session that wrote it.

Run it after any change to the driver's parsing, tokenisation or resolution:

    python tools/fuzz_append_only_merge.py                  # 400 hostile cases
    python tools/fuzz_append_only_merge.py --realistic      # ordinary prose, for the toil rate
    python tools/fuzz_append_only_merge.py --cases 2000     # longer soak

`tests/unit/test_append_only_merge_driver.py` runs a small fixed-seed subset in CI; this is the
soak version. Exit status is non-zero if any case corrupts content.

Four oracles, checked on every case. The last two exist because the driver's own post-condition
structurally cannot see them — it checks that each side's added runs are *present*, which says
nothing about how many times, nor about ancestor content that no side added:

1. a clean merge preserves each side's added text verbatim and unbroken;
2. a clean merge keeps every ancestor line that neither side deleted;
3. no line appears more often than the side that had the most copies of it (de-duplication is
   allowed; duplication is not);
4. a conflict leaves real markers, with theirs' content recoverable.
"""

from __future__ import annotations

import argparse
import collections
import random
import subprocess
import sys
import tempfile
from pathlib import Path

LOG_PATH = ".claude/state/log.md"
PREAMBLE = "# Log\n\nAppend-only. Newest last.\n\n"

HEADERS = [
    "## 2026-08-{d:02d} {h:02d}:00 · note · {who}\n",
    "## [D-2026-08-{d:02d}-01] 2026-08-{d:02d} ~{h:02d}:30 · decision · {who}\n",
    "## 2026-08-{d:02d} · track-record · {who}\n",
]

ORDINARY_BODIES = [
    "Shipped the thing.\nRef: GH #1234\n",
    "Verified clean on prod.\nBoot checks passed.\nRef: PR #999\n",
    "Branch review: verdict=approve, confidence=high\n",
    "Investigated the anomaly; root cause isolated.\nRef: docs/research/x.md\n",
]

HOSTILE_BODIES = [
    # The charter's own correction pattern: a quoted header that satisfies every shape rule.
    "Corrects the entry below, quoted verbatim:\n\n## [D-2026-07-08-01] 2026-07-08 20:33 · deploy-verify · daemon(PM)\n\nand the correction.\n",
    "Quoted mid-paragraph:\n## 2026-02-01 00:00 · note · old\nend.\n",
    "Forward-dated quote:\n\n## 2099-12-31 23:59 · note · future\n\nend.\n",
    "Markers in prose:\n<<<<<<< ours\n=======\n>>>>>>> theirs\nend.\n",
    "- column-0 bullet\n- another bullet\n## 2026-03-03 03:03 · note · quoted\n",
    "Body with no trailing newline.",
    "Multi\n\nparagraph\n\nbody.\n",
]


def git(cwd: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    if check and proc.returncode:
        raise AssertionError(f"git {' '.join(args)}: {proc.stderr}")
    return proc


def make_entry(rng: random.Random, who: str, hostile_rate: float) -> str:
    header = rng.choice(HEADERS).format(d=rng.randint(1, 28), h=rng.randint(0, 23), who=who)
    pool = HOSTILE_BODIES if rng.random() < hostile_rate else ORDINARY_BODIES
    body = rng.choice(pool)
    tail = "" if body.endswith("\n\n") else "\n" if body.endswith("\n") else "\n\n"
    return header + body + tail


def counts(text: str) -> collections.Counter[str]:
    return collections.Counter(line for line in text.splitlines() if line.strip())


def check(ancestor: str, ours: str, theirs: str, merged: str, conflicted: bool) -> str | None:
    if conflicted:
        if "<<<<<<<" not in merged:
            return "conflict without markers"
        theirs_lines = [line for line in theirs.splitlines() if line.strip()]
        if theirs_lines and not any(line in merged for line in theirs_lines):
            return "conflict lost theirs entirely"
        return None

    for name, side in (("ours", ours), ("theirs", theirs)):
        added = side[len(ancestor) :] if side.startswith(ancestor) else side
        if added.strip() and added.strip() not in merged:
            return f"clean merge altered or split {name}'s contribution"

    merged_counts = counts(merged)
    ours_counts, theirs_counts = counts(ours), counts(theirs)
    ancestor_counts = counts(ancestor)

    for line in ancestor_counts:
        if ours_counts[line] and theirs_counts[line] and not merged_counts[line]:
            return f"clean merge dropped ancestor line {line[:60]!r}"

    for line, n in merged_counts.items():
        # Inclusion-exclusion: copies shared through the ancestor must not be counted twice.
        # A plain max() bound is wrong — two *different* entries may legitimately share a line,
        # and both survive. (Getting this wrong is how a fuzz oracle manufactures false
        # positives; the independent review hit the same trap from the other direction.)
        allowed = max(
            ours_counts[line],
            theirs_counts[line],
            ours_counts[line] + theirs_counts[line] - ancestor_counts[line],
        )
        if allowed and n > allowed:
            return f"clean merge duplicated {line[:60]!r} ({n} > {allowed})"

    return None


def run_case(seed: int, hostile_rate: float, driver: Path) -> tuple[str | None, str]:
    rng = random.Random(seed)
    sys.path.insert(0, str(driver.parent))
    try:
        import install_merge_drivers
    finally:
        sys.path.pop(0)

    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp) / "repo"
        (repo / LOG_PATH).parent.mkdir(parents=True)
        git(Path(tmp), "init", "-q", "-b", "main", str(repo))
        git(repo, "config", "user.email", "fuzz@example.com")
        git(repo, "config", "user.name", "Fuzz")
        git(repo, "config", "merge.append-only.name", "fuzz")
        git(
            repo,
            "config",
            "merge.append-only.driver",
            install_merge_drivers.driver_command(str(driver)),
        )
        (repo / ".gitattributes").write_text(f"{LOG_PATH} merge=append-only\n")

        ancestor = PREAMBLE + "".join(
            make_entry(rng, "base", hostile_rate) for _ in range(rng.randint(1, 3))
        )
        (repo / LOG_PATH).write_text(ancestor)
        git(repo, "add", "-A")
        git(repo, "commit", "-qm", "seed")

        ours = ancestor + "".join(
            make_entry(rng, "ours", hostile_rate) for _ in range(rng.randint(1, 3))
        )
        theirs = ancestor + "".join(
            make_entry(rng, "theirs", hostile_rate) for _ in range(rng.randint(1, 3))
        )
        git(repo, "checkout", "-q", "-b", "ours")
        (repo / LOG_PATH).write_text(ours)
        git(repo, "add", "-A")
        git(repo, "commit", "-qm", "ours")
        git(repo, "checkout", "-q", "-b", "theirs", "main")
        (repo / LOG_PATH).write_text(theirs)
        git(repo, "add", "-A")
        git(repo, "commit", "-qm", "theirs")
        git(repo, "checkout", "-q", "ours")

        merge = git(repo, "merge", "theirs", "-m", "merge", check=False)
        result = (repo / LOG_PATH).read_text()
        conflicted = merge.returncode != 0
        problem = check(ancestor, ours, theirs, result, conflicted)
        return (
            f"seed {seed}: {problem}" if problem else None,
            "conflict" if conflicted else "clean",
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cases", type=int, default=400)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument(
        "--realistic",
        action="store_true",
        help="ordinary prose rather than hostile bodies; measures the conflict rate",
    )
    args = parser.parse_args(argv)

    driver = Path(__file__).resolve().parent / "merge_append_only.py"
    hostile_rate = 0.1 if args.realistic else 0.7
    failures: list[str] = []
    outcomes: collections.Counter[str] = collections.Counter()

    for seed in range(args.start_seed, args.start_seed + args.cases):
        problem, outcome = run_case(seed, hostile_rate, driver)
        outcomes[outcome] += 1
        if problem:
            failures.append(problem)

    kind = "realistic" if args.realistic else "hostile"
    print(f"{args.cases} {kind} merges: {dict(outcomes)}, corruptions: {len(failures)}")
    for failure in failures[:20]:
        print("  ", failure)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
