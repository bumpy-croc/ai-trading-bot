"""Refuse agent writes into the primary checkout's working tree (GH #1082).

Third guard in the "am I touching the right checkout?" family, after
``tools/atb_worktree_shim.py`` (makes the right import happen) and ``src/_source_root.py``
(makes the wrong import loud). Those two cover the *import* system. This one covers the
*filesystem*, which they cannot: a ``sed -i .claude/LESSONS.md`` never imports anything.

The hazard (GH #1082, ``.claude/LESSONS.md`` §1.10a / §3): an agent shell's working directory
is silently reset to the primary checkout mid-task. Every subsequent *relative* path then
resolves against the primary checkout instead of the agent's worktree — a read returns stale
content (recorded: 247 lines where the worktree's copy had 578, no error), and a write mutates
the tree that is supposed to stay pinned to ``main``.

Mechanism: a Claude Code ``PreToolUse`` hook. That choice is the whole design — the hook runs
**only inside a Claude Code session**, so it separates agent writes from human writes exactly,
with no filesystem permission change. Alex editing in his editor, or running git in his own
terminal, never invokes this code. A ``chmod``/ACL scheme cannot make that distinction: agents
run as his uid on his machine.

What is protected: the primary checkout's *working tree*. Explicitly NOT protected, because
agent work legitimately writes there:

* ``<primary>/.git/**`` — every worktree's git operations write to the shared git dir, and
  ``git worktree add`` is how agent isolation is created in the first place;
* ``<primary>/.claude/worktrees/**`` — the worktrees themselves live inside the primary
  checkout's directory but are not its working tree;
* anything git-ignored in the primary checkout — the shared ``.venv``, ``logs/``, caches.

Failure mode is fail-OPEN: any unexpected error allows the tool call. A guard that bricks every
tool call would be disabled within the hour, and a protection that gets switched off is worth
nothing. The hazard it exists for is a mistake, not an attacker.

Human override, in order of preference:
    1. launch with ``ATB_ALLOW_PRIMARY_WRITE=1 claude`` — the hook process inherits Claude Code's
       environment, not the Bash tool's, so an agent cannot set this for itself;
    2. ``touch ~/.claude/atb-allow-primary-write`` from your own terminal for a live session.

Protocol: reads the hook payload as JSON on stdin, exits 0 to allow, exits 2 to block with the
reason on stderr (Claude Code feeds a PreToolUse exit-2 stderr back to the model).
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

__all__ = [
    "ALLOW_ENV",
    "ALLOW_SENTINEL",
    "PrimaryCheckoutWriteError",
    "evaluate",
    "find_primary_checkout",
    "is_protected",
    "main",
]

ALLOW_ENV = "ATB_ALLOW_PRIMARY_WRITE"
ALLOW_SENTINEL = Path.home() / ".claude" / "atb-allow-primary-write"
PIN_DIR = Path.home() / ".cache" / "atb-primary-guard"
# A pin older than this is assumed to belong to a finished session that reused the id
# (`--resume`/`--continue` keep it), so it must not guard a fresh one forever.
PIN_MAX_AGE_SECONDS = 12 * 60 * 60
# The weak override tier is self-expiring: one stray `touch` must not disable the guard
# for the rest of the machine's life.
SENTINEL_MAX_AGE_SECONDS = 30 * 60

# Paths under the primary checkout that agent work legitimately writes to.
UNPROTECTED_PREFIXES = (".git", ".claude/worktrees")

# Always-ignored trees, short-circuited ahead of `git check-ignore` so the common case
# (the shared venv, logs) costs no subprocess and survives git being slow or absent.
ALWAYS_IGNORED_PREFIXES = (
    ".venv",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "logs",
    "node_modules",
)

FILE_TOOLS = ("Edit", "Write", "NotebookEdit", "MultiEdit")

# Commands whose non-flag operands are all write targets.
_WRITE_ALL_OPERANDS = {
    "mkdir",
    "rm",
    "rmdir",
    "tee",
    "touch",
    "truncate",
    "unlink",
}
# `chmod 755 f` / `chown me f`: the first operand is a mode or an owner, not a path.
# `dd`, `chown` and `shred` are deliberately absent — they buy almost nothing here and their
# operand grammars (`of=`, `if=`) produced false "would write to <primary>/if=/dev/zero".
_WRITE_ALL_OPERANDS_AFTER_FIRST = {"chmod"}
# Commands where only the final operand is the write target (source args are reads).
_WRITE_LAST_OPERAND = {"cp", "install", "ln", "mv", "rsync"}

# git subcommands that mutate the working tree or HEAD of the repo they run in.
_GIT_MUTATORS = {
    "add",
    "am",
    "apply",
    "checkout",
    "cherry-pick",
    "clean",
    "commit",
    "merge",
    "mv",
    "rebase",
    "reset",
    "restore",
    "revert",
    "rm",
    "stash",
    "switch",
}

# Commands that read files; used only for the cwd-reset detector.
_READ_COMMANDS = {
    "awk",
    "black",
    "cat",
    "cut",
    "diff",
    "egrep",
    "fgrep",
    "find",
    "grep",
    "head",
    "jq",
    "less",
    "more",
    "mypy",
    "nl",
    "python",
    "python3",
    "rg",
    "ruff",
    "sed",
    "sort",
    "stat",
    "tail",
    "tr",
    "uniq",
    "wc",
}
# For these the first non-flag operand is a pattern/script, not a path.
_FIRST_OPERAND_IS_NOT_A_PATH = {"awk", "egrep", "fgrep", "grep", "rg", "sed"}
# Flags whose value is an inline expression rather than a path.
_FLAGS_TAKING_EXPRESSION = {"-c", "-e", "-m", "--expression", "-E", "--regexp"}


class PrimaryCheckoutWriteError(RuntimeError):
    """Raised when a tool call would write into the primary checkout's working tree."""


_SHIM_CACHE: list = []


def _load_shim_helpers():
    """Reuse the #1070 shim's root-finding rather than growing a third copy of it."""
    if _SHIM_CACHE:
        return _SHIM_CACHE[0]
    path = Path(__file__).resolve().with_name("atb_worktree_shim.py")
    spec = importlib.util.spec_from_file_location("_atb_worktree_shim_for_guard", path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _SHIM_CACHE.append(module)
    return module


def find_repo_root(start: str | os.PathLike[str] | None = None) -> Path | None:
    """Delegate to ``tools/atb_worktree_shim.find_repo_root`` (single implementation)."""
    return _load_shim_helpers().find_repo_root(start)


def find_primary_checkout(root: Path) -> Path | None:
    """Resolve the primary checkout that owns ``root``.

    A primary checkout has a ``.git`` **directory**; a linked worktree has a ``.git`` **file**
    holding ``gitdir: <primary>/.git/worktrees/<name>``. Read rather than shell out to git: the
    hook runs on every tool call and must stay cheap.
    """
    git_entry = root / ".git"
    if git_entry.is_dir():
        return root
    if not git_entry.is_file():
        return None
    try:
        text = git_entry.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return None
    if not text.startswith("gitdir:"):
        return None
    gitdir = Path(text.split(":", 1)[1].strip())
    if not gitdir.is_absolute():
        gitdir = (root / gitdir).resolve()
    for parent in (gitdir, *gitdir.parents):
        if parent.name == ".git":
            return parent.parent
    return None


def _git_ignores(primary: Path, path: Path) -> bool:
    """True when git ignores ``path`` — i.e. it is safe to write.

    Fails **open** (returns True) when git is missing, slow, or errors. Every other failure
    path in this module allows the tool call; this one used to be the exception, so a sick
    machine turned the shared ``.venv`` into protected territory and broke ``make install``
    exactly when things were already going wrong.
    """
    try:
        result = subprocess.run(
            ["git", "-C", str(primary), "check-ignore", "-q", "--", str(path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return True
    if result.returncode not in (0, 1):
        return True
    return result.returncode == 0


def is_protected(path: Path, primary: Path) -> bool:
    """True when ``path`` is inside the primary checkout's protected working tree."""
    try:
        relative = path.resolve().relative_to(primary.resolve())
    except (ValueError, OSError):
        try:
            relative = Path(os.path.normpath(str(path))).relative_to(primary)
        except ValueError:
            return False
    rel_posix = relative.as_posix()
    if rel_posix == ".":
        # The checkout root itself — what a `git checkout`/`git reset` in the primary targets.
        return True
    for prefix in (*UNPROTECTED_PREFIXES, *ALWAYS_IGNORED_PREFIXES):
        if rel_posix == prefix or rel_posix.startswith(prefix + "/"):
            return False
    return not _git_ignores(primary, primary / relative)


def _override_active() -> str | None:
    if os.environ.get(ALLOW_ENV) == "1":
        return f"{ALLOW_ENV}=1"
    try:
        age = time.time() - ALLOW_SENTINEL.stat().st_mtime
    except OSError:
        return None
    if age > SENTINEL_MAX_AGE_SECONDS:
        return None
    return f"{ALLOW_SENTINEL} (expires {int((SENTINEL_MAX_AGE_SECONDS - age) / 60)} min from now)"


# --------------------------------------------------------------------------------------
# shell command analysis
# --------------------------------------------------------------------------------------


_HEREDOC_RE = re.compile(r"<<-?\s*(['\"]?)([A-Za-z_][A-Za-z0-9_]*)\1")


def _strip_heredoc_bodies(command: str) -> str:
    """Remove heredoc bodies before parsing.

    A heredoc body is data, not shell. Left in, its prose becomes commands: a line reading
    ``if a > b then`` inside a ``cat <<'EOF'`` block parses as a redirect into a file named
    ``b``. That shape is routine here — every `gh pr create --body "$(cat <<EOF …)"` has one.
    """
    lines = command.split("\n")
    kept: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        kept.append(line)
        delimiters = [match.group(2) for match in _HEREDOC_RE.finditer(line)]
        index += 1
        for delimiter in delimiters:
            while index < len(lines) and lines[index].strip() != delimiter:
                index += 1
            index += 1  # consume the delimiter line itself
    return "\n".join(kept)


def _split_segments(command: str) -> list[str]:
    """Split on shell control operators, ignoring ones inside quotes.

    Quote awareness is not pedantry: a regex split turns ``grep -E "a|b" f`` into nonsense and,
    worse, made ``python -c "print(1 > 2)"`` look like a redirect into a file named ``2)``.
    """
    segments: list[str] = []
    current: list[str] = []
    quote: str | None = None
    escaped = False
    index = 0
    while index < len(command):
        char = command[index]
        if escaped:
            current.append(char)
            escaped = False
        elif char == "\\" and quote != "'":
            current.append(char)
            escaped = True
        elif quote is not None:
            current.append(char)
            if char == quote:
                quote = None
        elif char in "'\"":
            quote = char
            current.append(char)
        elif char in ";\n|&":
            segments.append("".join(current))
            current = []
            if command.startswith(("&&", "||"), index):
                index += 1
        else:
            current.append(char)
        index += 1
    segments.append("".join(current))
    return [segment for segment in segments if segment.strip()]


def _redirect_targets(segment: str) -> list[str]:
    """Output-redirection targets in one segment, quote-aware."""
    targets: list[str] = []
    quote: str | None = None
    escaped = False
    index = 0
    while index < len(segment):
        char = segment[index]
        if escaped:
            escaped = False
        elif char == "\\" and quote != "'":
            escaped = True
        elif quote is not None:
            if char == quote:
                quote = None
        elif char in "'\"":
            quote = char
        elif char == ">":
            index += 1
            if index < len(segment) and segment[index] == ">":
                index += 1
            while index < len(segment) and segment[index] in " \t":
                index += 1
            start = index
            while index < len(segment) and segment[index] not in " \t":
                index += 1
            token = segment[start:index]
            if token and not token.startswith("&"):
                targets.append(token)
            continue
        index += 1
    return targets


def _tokenize(segment: str) -> list[str]:
    try:
        return shlex.split(segment, comments=True)
    except ValueError:
        return segment.split()


def _strip_env_prefix(tokens: list[str]) -> list[str]:
    """Drop ``FOO=bar`` assignments and ``env``/``sudo`` wrappers preceding the command."""
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token in ("env", "sudo", "command", "nohup", "time"):
            index += 1
        elif re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", token):
            index += 1
        else:
            break
    return tokens[index:]


def _operands(tokens: list[str], *, skip_first: bool = False) -> list[str]:
    """Non-flag arguments, minus values of flags that take an inline expression.

    Empty operands are dropped: BSD's ``sed -i '' 's/a/b/' f`` passes an empty backup suffix,
    which otherwise absorbed ``skip_first`` and made the banner name the sed script instead of
    the file. After ``--`` every remaining token is an operand, even a ``-``-prefixed one.
    """
    operands: list[str] = []
    skip_next = False
    end_of_flags = False
    for token in tokens[1:]:
        if skip_next:
            skip_next = False
            continue
        if end_of_flags:
            if token:
                operands.append(token)
            continue
        if token == "--":
            end_of_flags = True
            continue
        if token.startswith("-") and token != "-":
            if token in _FLAGS_TAKING_EXPRESSION:
                skip_next = True
            continue
        if token:
            operands.append(token)
    if skip_first and operands:
        operands = operands[1:]
    return operands


def _unquote(token: str) -> str:
    """Strip one layer of surrounding quotes left by the raw (non-shlex) redirect scanner."""
    if len(token) >= 2 and token[0] == token[-1] and token[0] in "'\"":
        return token[1:-1]
    return token


def _resolve(token: str, cwd: Path) -> Path | None:
    """Absolute path for ``token``, or None when it cannot be resolved with confidence.

    An unresolvable token — one carrying a shell variable or a command substitution — must
    NOT be guessed as primary-relative. Guessing inverted this module's fail-open polarity:
    ``OUT=/tmp/o.txt; echo hi > $OUT`` was refused as a write to ``<primary>/$OUT``.
    """
    token = _unquote(token)
    if not token or "$" in token or "`" in token:
        return None
    expanded = os.path.expanduser(token)
    candidate = Path(expanded)
    if not candidate.is_absolute():
        candidate = cwd / candidate
    return Path(os.path.normpath(str(candidate)))


def _cd_target(tokens: list[str], cwd: Path) -> Path | None:
    if not tokens or tokens[0] != "cd":
        return None
    operands = [token for token in tokens[1:] if not token.startswith("-")]
    if not operands:
        return None
    return _resolve(operands[0], cwd)


def _git_cwd(tokens: list[str], cwd: Path) -> Path | None:
    for index, token in enumerate(tokens):
        if token == "-C" and index + 1 < len(tokens):
            return _resolve(tokens[index + 1], cwd)
    return cwd


def _git_subcommand(tokens: list[str]) -> str | None:
    index = 1
    while index < len(tokens):
        token = tokens[index]
        if token == "-C" or token == "-c":
            index += 2
            continue
        if token.startswith("-"):
            index += 1
            continue
        return token
    return None


def _sed_in_place_targets(tokens: list[str]) -> list[str]:
    """Files a ``sed`` invocation edits in place, or [] when it is not an in-place edit.

    ``sed -i -e 's/a/b/' f`` is the common portable form and used to slip through entirely:
    ``-e`` consumes the script as its flag value, so skipping "the first operand" ate the
    FILE instead. Only skip the first operand when no ``-e``/``--expression`` was given.
    """
    flags = tokens[1:]
    in_place = any(token.startswith("-i") or token.startswith("--in-place") for token in flags)
    if not in_place:
        return []
    script_given_by_flag = any(
        token in ("-e", "--expression") or token.startswith("--expression=") for token in flags
    )
    return _operands(tokens, skip_first=not script_given_by_flag)


def _write_targets(segment: str, tokens: list[str], cwd: Path) -> list[Path]:
    """Paths this shell segment would write to, best-effort."""
    candidates = list(_redirect_targets(segment))
    name = Path(tokens[0]).name if tokens else ""
    if name == "sed":
        candidates += _sed_in_place_targets(tokens)
    elif name in _WRITE_ALL_OPERANDS:
        candidates += _operands(tokens)
    elif name in _WRITE_ALL_OPERANDS_AFTER_FIRST:
        candidates += _operands(tokens, skip_first=True)
    elif name in _WRITE_LAST_OPERAND:
        operands = _operands(tokens)
        if operands:
            candidates.append(operands[-1])
    elif name == "git" and _git_subcommand(tokens) in _GIT_MUTATORS:
        git_cwd = _git_cwd(tokens, cwd)
        return [path for path in (git_cwd,) if path is not None]
    resolved = (_resolve(candidate, cwd) for candidate in candidates)
    return [path for path in resolved if path is not None]


def _relative_read_operands(tokens: list[str], cwd: Path) -> list[str]:
    """Relative path operands of a file-reading command (cwd-reset detector)."""
    if not tokens:
        return []
    name = Path(tokens[0]).name
    if name not in _READ_COMMANDS:
        return []
    operands = _operands(tokens, skip_first=name in _FIRST_OPERAND_IS_NOT_A_PATH)
    found = []
    for token in operands:
        if token.startswith("/") or token.startswith("~") or token == "-":
            continue
        if "/" in token or (cwd / token).exists():
            found.append(token)
    return found


# --------------------------------------------------------------------------------------
# banners
# --------------------------------------------------------------------------------------


def _banner(title: str, rows: list[tuple[str, str]], body: str) -> str:
    width = 78
    lines = ["", f" {title} ".center(width, "="), ""]
    label_width = max(len(label) for label, _ in rows)
    for label, value in rows:
        lines.append(f"  {label.ljust(label_width)} : {value}")
    lines.append("")
    lines.append(body.rstrip("\n"))
    lines.append("")
    lines.append(f"  {'guard'.ljust(label_width)} : {Path(__file__).resolve()}")
    lines.append(f"  {'human override'.ljust(label_width)} : launch with {ALLOW_ENV}=1, or")
    lines.append(f"  {' ' * label_width}   touch {ALLOW_SENTINEL}")
    lines.append("=" * width)
    lines.append("")
    return "\n".join(lines)


def _write_banner(target: Path, primary: Path, cwd: Path, detail: str) -> str:
    return _banner(
        "ATB PRIMARY CHECKOUT WRITE REFUSED (GH #1082)",
        [
            ("would write to", str(target)),
            ("primary checkout", str(primary)),
            ("tool call cwd", str(cwd)),
            ("what", detail),
        ],
        "The primary checkout is Alex's working copy and the production reference; it is\n"
        "pinned to `main` and nothing in an agent session may write to it. This is usually\n"
        "a RELATIVE path resolving against a cwd that was silently reset to the primary\n"
        "checkout mid-task (GH #1082) — the write you intended is almost certainly meant\n"
        "for your worktree.\n"
        "\n"
        "Fix: re-run the command with an ABSOLUTE path under your own worktree, e.g.\n"
        "    <your-worktree>/path/to/file\n"
        "If you have no worktree yet, create one:\n"
        "    git -C <primary> worktree add <primary>/.claude/worktrees/<name> origin/develop\n"
        "and `touch .agent-active` in it.",
    )


def _cwd_reset_banner(pinned: Path, primary: Path, operands: list[str]) -> str:
    return _banner(
        "ATB WORKING-DIRECTORY RESET DETECTED (GH #1082)",
        [
            ("session worktree", str(pinned)),
            ("tool call cwd", str(primary)),
            ("relative operands", ", ".join(operands)),
        ],
        "This session has been working in the worktree above, but this command's working\n"
        "directory is the PRIMARY checkout — the documented cwd-reset failure. The relative\n"
        "paths listed would read the primary checkout's copy (pinned to `main`), not yours,\n"
        "and would do so silently: a stale read looks exactly like a correct one. The\n"
        "2026-08-17 retro read 247 lines of a 578-line file this way.\n"
        "\n"
        "Fix: use ABSOLUTE paths under your worktree, or prefix the command with\n"
        f"    cd {pinned} && ...",
    )


# --------------------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------------------


def _pin_path(session_id: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", session_id)[:128]
    return PIN_DIR / f"{safe}.worktree"


def _read_pin(session_id: str | None) -> Path | None:
    """The worktree this session is working in, or None.

    A pin is discarded when it is stale — the worktree was deleted (the nightly pruner does
    this routinely) or the pin is old enough that it likely belongs to an earlier run of a
    resumed session id. A stale pin used to refuse reads forever with the un-followable
    remedy ``cd <deleted path>``.
    """
    if not session_id:
        return None
    path = _pin_path(session_id)
    try:
        if time.time() - path.stat().st_mtime > PIN_MAX_AGE_SECONDS:
            path.unlink(missing_ok=True)
            return None
        text = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not text:
        return None
    pinned = Path(text)
    if not pinned.is_dir():
        try:
            path.unlink(missing_ok=True)
        except OSError:  # pragma: no cover - defensive
            pass
        return None
    return pinned


def _write_pin(session_id: str | None, worktree: Path) -> None:
    if not session_id:
        return
    try:
        PIN_DIR.mkdir(parents=True, exist_ok=True)
        _pin_path(session_id).write_text(str(worktree), encoding="utf-8")
    except OSError:  # pragma: no cover - defensive
        pass


def evaluate(payload: dict) -> str | None:
    """Return a refusal banner for this hook payload, or None to allow the tool call.

    Everything hangs off one signal: **is this session pinned to a worktree?** A session is
    pinned once any of its tool calls has run with a cwd inside a linked worktree, which for
    a dispatched agent is its first call. Reads and writes are both guarded on that signal.

    Guarding writes unconditionally instead — the obvious reading of "nothing may write to the
    primary" — breaks two real users, and both breakages are certain rather than theoretical:

    * an ordinary single clone with no worktrees IS structurally the primary, so every write
      into it is refused. `.claude/settings.json` is checked in, so that reaches every
      contributor and every Claude Code Web session (which clones into one checkout);
    * the PM daemon runs in the primary by design and must append to the tracked, append-only
      `.claude/state/log.md` on every material action.

    Neither is ever pinned — a fresh clone has no worktrees, and the daemon never works inside
    one — so pinning is the signal that separates "an agent that has a worktree and is writing
    to the wrong tree" from "somebody legitimately working in the only tree they have".

    The cost is a cross-checkout write issued before a session's first worktree-cwd tool call,
    which is no longer caught. That window is narrow and traded against two guaranteed breakages.
    """
    tool_name = payload.get("tool_name") or ""
    tool_input = payload.get("tool_input") or {}
    cwd = Path(payload.get("cwd") or os.getcwd())
    session_id = payload.get("session_id")

    root = find_repo_root(cwd)
    if root is None:
        return None
    primary = find_primary_checkout(root)
    if primary is None:
        return None

    if root != primary:
        # cwd is inside a linked worktree: remember it, so a later cwd reset is detectable.
        _write_pin(session_id, root)

    pinned = _read_pin(session_id)
    if pinned is None or pinned == primary:
        return None

    if tool_name in FILE_TOOLS:
        target = tool_input.get("file_path") or tool_input.get("notebook_path")
        if not target:
            return None
        resolved = _resolve(str(target), cwd)
        if resolved is not None and is_protected(resolved, primary):
            return _write_banner(resolved, primary, cwd, f"{tool_name} tool")
        return None

    if tool_name != "Bash":
        return None

    command = tool_input.get("command") or ""
    if not command.strip():
        return None

    effective_cwd = cwd
    for segment in _split_segments(_strip_heredoc_bodies(command)):
        tokens = _strip_env_prefix(_tokenize(segment))
        if tokens and tokens[0] == "cd":
            moved = _cd_target(tokens, effective_cwd)
            if moved is None:
                # `cd "$WORKTREE" && …`: the rest of the command runs somewhere we cannot
                # name, so guarding it would mean guessing. Fail open, as everywhere else.
                return None
            effective_cwd = moved
            continue

        for target in _write_targets(segment, tokens, effective_cwd):
            if is_protected(target, primary):
                return _write_banner(
                    target, primary, effective_cwd, f"shell command `{segment.strip()}`"
                )

        if effective_cwd == primary:
            operands = _relative_read_operands(tokens, effective_cwd)
            if operands:
                return _cwd_reset_banner(pinned, primary, operands)

    return None


def main(argv: list[str] | None = None) -> int:
    raw = sys.stdin.read()
    try:
        payload = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError:
        return 0
    if not isinstance(payload, dict):
        return 0

    override = _override_active()
    try:
        banner = evaluate(payload)
    except Exception as exc:  # noqa: BLE001 - a broken guard must never brick a session
        print(f"[primary-checkout-guard] disabled for this call: {exc!r}", file=sys.stderr)
        return 0

    if banner is None:
        return 0
    if override is not None:
        print(
            f"[primary-checkout-guard] override active ({override}); allowing:\n{banner}",
            file=sys.stderr,
        )
        return 0
    print(banner, file=sys.stderr)
    return 2


if __name__ == "__main__":  # pragma: no cover - process entry point
    sys.exit(main())
