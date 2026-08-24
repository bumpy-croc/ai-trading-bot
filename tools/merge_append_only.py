"""Git merge driver for genuinely append-only files.

`.claude/state/log.md` is written by every agent and the PM, and is append-only by charter.
Two branches that each append an entry therefore conflict at EOF on almost every PR — a
conflict whose resolution is always the same ("keep both, chronological"). Doing that by hand
is toil, and each hand-resolution is a chance to silently drop an entry from a record whose
entire point is that entries are never dropped (GH #1079, #1090).

This driver resolves that one case and *only* that one case.

## Semantics

The file is split into **blocks** (an entry and its body). The three sides are reduced to
sequences of block tokens and handed to ``git merge-file``, so the actual three-way merge is
git's own — with the usual guarantees that nothing present on either side is dropped and the
common ancestor's content is not duplicated. The driver then post-processes git's result:

Each conflict region git reports is then re-merged at *entry* granularity. An entry is
identified by its first line, so the same entry appearing on both sides with different bodies
is an **edit**, not two appends:

* entries only one side touched are taken from that side — including deletions, so the weekly
  retro clearing `AGENDA.md` still clears it, while an item another branch appended in the
  meantime survives the clear (GH #1090);
* entries neither side had are appends and are all kept, de-duplicated, in timestamp order
  when every entry in the region carries a date;
* **anything else stops the merge with real markers** — both sides editing one entry, an edit
  racing a deletion, two different bodies under the same entry header, or repeated entry
  headers that make the region ambiguous. This is the difference between this driver and git's
  built-in ``union``, which cannot tell an append from an edit and would silently ship both
  halves of a rewrite.

Because identity *is* the first line, an edit to that line would otherwise present as a
delete plus an unrelated add. A vanished entry and an arrived one are therefore also paired by
**body** similarity, and the pair conflicts. Residual carve-out, stated rather than hidden: an
entry whose first line and body are both rewritten substantially, or a single-line entry whose
only line changes, remains indistinguishable from a delete plus an append and merges as one.

## Which files qualify

Only the paths in ``APPEND_ONLY_PATHS`` below. The driver is handed the pathname as ``%P`` and
**refuses (exits with a conflict) for any path it does not recognise**, so listing a file in
`.gitattributes` is necessary but not sufficient. That is deliberate: union-style resolution is
correct for a file whose existing lines are never edited and flatly wrong for anything else, and
a wrong entry here would corrupt merges silently. Adding a path requires adding it in both
places, with a reason.

Invoked by git as::

    merge_append_only.py %O %A %B %L %P

where %O is the ancestor, %A ours (and the file the result must be written to), %B theirs,
%L the conflict-marker size and %P the pathname in the repository. Exit 0 means resolved.
"""

from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

DRIVER_NAME = "append-only"

# Body-similarity above which a vanished and an arrived entry are judged to be one edit.
# Chosen with a wide margin: measured must-conflict cases score 1.00, unrelated ones <= 0.38.
_EDIT_SIMILARITY = 0.8


@dataclass(frozen=True)
class BlockRule:
    """How to cut one append-only file into independently-mergeable blocks.

    ``start`` matches the first line of a block; everything up to the next match (or EOF)
    belongs to it. Text before the first match is the preamble and is merged as a single
    block, so edits to a file's header still conflict normally.
    """

    start: re.Pattern[str]
    reorder_by_timestamp: bool


# The allowlist. A file belongs here only if its existing content is never edited or removed
# in the ordinary course of work — appended to, and otherwise left alone.
#
# NOT included, on purpose:
#   docs/changelog.md — not append-only. New entries are *prepended* inside shared
#     `### Added`/`### Fixed` sections and the `[Unreleased]` section is rewritten wholesale at
#     release time. Two branches adding entries touch the same section, which is an edit to
#     shared structure, not an append; git's normal line merge already handles the common case
#     and should conflict on the rest.
#   .claude/state/incidents/*.md, proposals/*.md — `status:` frontmatter is edited in place as
#     they move through their lifecycle. An edit/edit collision there must be seen by a human.
APPEND_ONLY_PATHS: dict[str, BlockRule] = {
    # Charter rule: "Never rewrite history in log.md — append-only; corrections are new
    # entries referencing the earlier one." Entries are H2 sections.
    ".claude/state/log.md": BlockRule(start=re.compile(r"^## "), reorder_by_timestamp=True),
    # Items are top-level bullets, appended by any agent the moment something is noticed, and
    # cleared wholesale by the retro that actions them. The clear is a deletion, which this
    # driver merges correctly; GH #1090 is an item that a hand-resolution nearly lost.
    ".claude/skills/weekly-retro/AGENDA.md": BlockRule(
        start=re.compile(r"^- "), reorder_by_timestamp=True
    ),
}

# (year, month, day, hour, minute) — the sort key for chronological placement.
Timestamp = tuple[int, int, int, int, int]

_DATE = re.compile(r"(\d{4})-(\d{2})-(\d{2})")
_TIME = re.compile(r"~?\b([01]\d|2[0-3]):([0-5]\d)\b")


def rule_for(pathname: str) -> BlockRule | None:
    normalized = pathname.replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return APPEND_ONLY_PATHS.get(normalized)


def starts_entry(line: str, previous: str | None, rule: BlockRule) -> bool:
    """Whether ``line`` opens a new entry, given the line before it.

    Three conditions, all needed. The marker (``## ``/``- ``) alone is not enough: entries are
    encouraged to *quote* an earlier entry's header, and `log.md` bodies routinely carry
    column-0 markup (107 column-0 ``- `` bullets sit inside bodies today). Splitting on a
    quoted header hoisted a fabricated entry above the real one and truncated the real one
    mid-sentence, so the shape test is deliberately narrow:

    * the marker matches;
    * the line carries a date, which every real header in both files does (95/95 in `log.md`);
    * it is preceded by a blank line or by nothing, which is likewise true of every real
      header, while a quoted one sits mid-paragraph.

    A header that fails any of these folds into the preceding entry, which makes that entry
    look *edited* — a conflict, never a silent split.
    """
    if not rule.start.match(line) or not _DATE.search(line):
        return False
    return previous is None or not previous.strip()


def split_blocks(text: str, rule: BlockRule) -> list[str]:
    """Cut ``text`` into blocks, preserving every byte: ``"".join(result) == text``."""
    lines = text.splitlines(keepends=True)
    blocks: list[str] = []
    current: list[str] = []
    for line in lines:
        if current and starts_entry(line, current[-1], rule):
            blocks.append("".join(current))
            current = []
        current.append(line)
    if current:
        blocks.append("".join(current))
    return blocks


def is_entry(block: str, rule: BlockRule) -> bool:
    """Whether a block is a well-formed entry (rather than the preamble or a malformed one)."""
    return starts_entry(block.split("\n", 1)[0], None, rule)


def timestamp_of(block: str) -> Timestamp | None:
    """Sort key from a block's first line, or None when it carries no date."""
    head = block.split("\n", 1)[0]
    date = _DATE.search(head)
    if not date:
        return None
    time = _TIME.search(head[date.end() :])
    hour, minute = (int(time.group(1)), int(time.group(2))) if time else (0, 0)
    return (int(date.group(1)), int(date.group(2)), int(date.group(3)), hour, minute)


def _tokenize(
    sides: list[list[str]],
) -> tuple[list[list[str]], dict[str, str]]:
    """Map each distinct block to a short token line. Identical blocks share a token."""
    token_of: dict[str, str] = {}
    block_of: dict[str, str] = {}
    tokenized: list[list[str]] = []
    for blocks in sides:
        stream = []
        for block in blocks:
            token = token_of.get(block)
            if token is None:
                token = f"b{len(token_of)}\n"
                token_of[block] = token
                block_of[token] = block
            stream.append(token)
        tokenized.append(stream)
    return tokenized, block_of


def _merge_file(
    ours: Path, base: Path, theirs: Path, *, marker_size: int, to_stdout: bool
) -> subprocess.CompletedProcess[str]:
    cmd = ["git", "merge-file", f"--marker-size={marker_size}"]
    if to_stdout:
        cmd += ["-p", "--diff3"]
    cmd += ["-L", "ours", "-L", "base", "-L", "theirs", str(ours), str(base), str(theirs)]
    return subprocess.run(cmd, capture_output=True, text=True)


def _key(block: str) -> str:
    """Identity of an entry: its first line. Same key on two sides means the *same* entry,
    so a differing body is an edit — which must conflict — rather than an unrelated append."""
    return block.split("\n", 1)[0]


def _sorted_chronologically(tokens: list[str], block_of: dict[str, str]) -> list[str]:
    """Stable timestamp sort, applied only when every block carries a date."""
    dated: list[tuple[Timestamp, str]] = []
    for token in tokens:
        stamp = timestamp_of(block_of[token])
        if stamp is None:
            return tokens
        dated.append((stamp, token))
    return [token for _, token in sorted(dated, key=lambda pair: pair[0])]


def _pad(tokens: list[str], block_of: dict[str, str]) -> list[str]:
    """Keep a blank line between entries the driver has just reordered.

    An entry's separator blank line belongs to the block *before* it, so moving a block that
    happens to end without one would butt it against its new neighbour. Confined to blocks
    coming out of a conflict region, so untouched parts of the file keep their exact bytes.
    """
    padded: list[str] = []
    for token in tokens[:-1]:
        block = block_of[token]
        if not block.endswith("\n\n"):
            # A block ending mid-line needs both newlines, or the next entry's heading is
            # glued onto it and stops being a heading at all — in the file and on re-split.
            token = f"pad{len(block_of)}\n"
            block_of[token] = block + ("\n" if block.endswith("\n") else "\n\n")
        padded.append(token)
    return padded + tokens[-1:]


def _body(block: str) -> str:
    return block.split("\n", 1)[1] if "\n" in block else ""


def _looks_like_an_edit(removed: str, added: str) -> bool:
    """Whether an entry that vanished and one that appeared are the same entry, reworded.

    Entries are keyed on their first line, so editing that line reads as delete-old +
    add-new and slips past the edit/edit and edit/delete checks entirely. The tell is the
    body: a first-line edit leaves it intact. Comparing *bodies* rather than whole blocks is
    what makes this safe — on real data the must-conflict cases score 1.00 while unrelated
    entries score 0.06-0.38, whereas whole-block comparison puts an unrelated pair at 0.66
    against 0.89 for a genuine edit, far too close to separate.

    Both bodies must be non-blank, so single-line entries never pair with each other.
    """
    removed_body, added_body = _body(removed).strip(), _body(added).strip()
    if not removed_body or not added_body:
        return False
    return SequenceMatcher(None, removed_body, added_body).ratio() >= _EDIT_SIMILARITY


def _resolve_hunk(
    ours: list[str],
    base: list[str],
    theirs: list[str],
    block_of: dict[str, str],
    reorder: bool,
    rule: BlockRule,
) -> list[str] | None:
    """Three-way merge one conflict region at entry granularity.

    Returns the resolved token list, or None when the region holds a real disagreement that a
    human must settle. Because entries are identified by their first line, an edit is visible
    as *the same key with a different body* and is never mistaken for an append.
    """

    if not all(is_entry(block_of[t], rule) for t in (*base, *ours, *theirs)):
        # The preamble, or a malformed/quoted header. Never resolve these automatically.
        return None

    def by_key(tokens: list[str]) -> dict[str, str] | None:
        keyed: dict[str, str] = {}
        for token in tokens:
            key = _key(block_of[token])
            if key in keyed:
                return None  # Repeated entry headers in one region: too ambiguous to resolve.
            keyed[key] = token
        return keyed

    keyed_base = by_key(base)
    keyed_ours = by_key(ours)
    keyed_theirs = by_key(theirs)
    if keyed_base is None or keyed_ours is None or keyed_theirs is None:
        return None

    resolved: dict[str, str | None] = {}
    for key in set(keyed_base) | set(keyed_ours) | set(keyed_theirs):
        b = keyed_base.get(key)
        a = keyed_ours.get(key)
        t = keyed_theirs.get(key)
        if b is not None:
            if a == b and t == b:
                resolved[key] = b
            elif a == b:
                resolved[key] = t  # only theirs touched it (edited or deleted)
            elif t == b:
                resolved[key] = a  # only ours touched it
            elif a == t:
                resolved[key] = a  # both made the same change
            else:
                return None  # edit/edit, or edit racing a delete
        elif a is not None and t is not None and a != t:
            return None  # both sides added the same entry with different bodies
        else:
            resolved[key] = a if a is not None else t

    # A first-line edit is invisible to the checks above: it presents as one key leaving and
    # another arriving. Pair them by body and conflict, or the "both sides edited one entry"
    # and "edit racing a deletion" rows of the contract quietly keep both copies.
    removed = [keyed_base[key] for key in keyed_base if resolved.get(key) is None]
    added = [
        token for key, token in resolved.items() if token is not None and key not in keyed_base
    ]
    for gone in removed:
        for arrival in added:
            if _looks_like_an_edit(block_of[gone], block_of[arrival]):
                return None

    def surviving(tokens: list[str], skip: set[str]) -> list[str]:
        out: list[str] = []
        for token in tokens:
            key = _key(block_of[token])
            survivor = resolved.get(key)
            if key in skip or survivor is None:
                continue
            skip.add(key)
            out.append(survivor)
        return out

    seen: set[str] = set()
    existing = surviving(base, seen)
    added = surviving(ours, seen) + surviving(theirs, seen)
    merged = existing + added
    return _sorted_chronologically(merged, block_of) if reorder else merged


def _resolve_tokens(
    merged: str, block_of: dict[str, str], marker_size: int, rule: BlockRule
) -> list[str] | None:
    """Walk git's --diff3 output. Returns None if any non-append conflict remains."""
    ours_m, base_m, theirs_m, end_m = (c * marker_size for c in ("<", "|", "=", ">"))
    result: list[str] = []
    section: str | None = None
    ours: list[str] = []
    base: list[str] = []
    theirs: list[str] = []

    for line in merged.splitlines(keepends=True):
        if line.startswith(ours_m):
            section, ours, base, theirs = "ours", [], [], []
        elif section and line.startswith(base_m):
            section = "base"
        elif section and line.startswith(theirs_m):
            section = "theirs"
        elif section and line.startswith(end_m):
            hunk = _resolve_hunk(ours, base, theirs, block_of, rule.reorder_by_timestamp, rule)
            if hunk is None:
                return None
            result.extend(_pad(hunk, block_of))
            section = None
        elif section == "ours":
            ours.append(line)
        elif section == "base":
            base.append(line)
        elif section == "theirs":
            theirs.append(line)
        else:
            result.append(line)

    if section is not None:
        return None  # Unterminated conflict block; refuse rather than guess.
    return result


def merge(ancestor: Path, ours: Path, theirs: Path, marker_size: int, pathname: str) -> int:
    rule = rule_for(pathname)
    if rule is None:
        print(
            f"merge-{DRIVER_NAME}: '{pathname}' is not a registered append-only file "
            f"(see APPEND_ONLY_PATHS in tools/merge_append_only.py). "
            "Falling back to a normal conflict.",
            file=sys.stderr,
        )
        return _fallback(ancestor, ours, theirs, marker_size)

    try:
        texts = [p.read_text(encoding="utf-8") for p in (ancestor, ours, theirs)]
    except (OSError, UnicodeDecodeError) as exc:
        print(f"merge-{DRIVER_NAME}: {exc}", file=sys.stderr)
        return _fallback(ancestor, ours, theirs, marker_size)

    sides = [split_blocks(t, rule) for t in texts]
    (tok_o, tok_a, tok_b), block_of = _tokenize(sides)

    tmp = ours.parent
    paths = {}
    for name, stream in (("o", tok_o), ("a", tok_a), ("b", tok_b)):
        paths[name] = tmp / f"{ours.name}.{DRIVER_NAME}.{name}"
        paths[name].write_text("".join(stream), encoding="utf-8")
    try:
        proc = _merge_file(
            paths["a"], paths["o"], paths["b"], marker_size=marker_size, to_stdout=True
        )
        if proc.returncode < 0 or proc.returncode > 127:
            return _fallback(ancestor, ours, theirs, marker_size)
        resolved = _resolve_tokens(proc.stdout, block_of, marker_size, rule)
    finally:
        for path in paths.values():
            path.unlink(missing_ok=True)

    if resolved is None:
        return _fallback(ancestor, ours, theirs, marker_size)

    ours.write_text("".join(block_of[t] for t in resolved), encoding="utf-8")
    return 0


def _fallback(ancestor: Path, ours: Path, theirs: Path, marker_size: int) -> int:
    """Leave the standard three-way conflict in %A, exactly as no driver at all would."""
    proc = _merge_file(ours, ancestor, theirs, marker_size=marker_size, to_stdout=False)
    # git merge-file returns the number of conflicts, so 0 here means the plain line merge
    # succeeded where the block merge would not have — the correct answer, not a failure.
    return 0 if proc.returncode == 0 else 1


def main(argv: list[str]) -> int:
    if len(argv) < 4:
        print("usage: merge_append_only.py %O %A %B %L %P", file=sys.stderr)
        return 2
    ancestor, ours, theirs = (Path(a) for a in argv[:3])
    try:
        marker_size = int(argv[3])
    except (IndexError, ValueError):
        marker_size = 7
    pathname = argv[4] if len(argv) > 4 else ""
    marker_size = max(marker_size, 7)
    try:
        return merge(ancestor, ours, theirs, marker_size, pathname)
    except Exception as exc:  # noqa: BLE001 - a crash must not leave %A marker-free
        # git does not write conflict markers for us; if we bail without doing so, %A keeps
        # ours' content verbatim and reads as a clean merge, silently dropping theirs.
        print(f"merge-{DRIVER_NAME}: unexpected failure: {exc!r}", file=sys.stderr)
        return _fallback(ancestor, ours, theirs, marker_size)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
