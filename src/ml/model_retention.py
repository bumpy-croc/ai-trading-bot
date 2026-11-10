"""Retention policy for versioned model artifacts in the model registry.

Registry layout: ``{registry}/{SYMBOL}/{TYPE}/{VERSION}/`` with an optional
``latest`` symlink per ``{SYMBOL}/{TYPE}``. A version directory is a removal
candidate unless it is protected by one of these rules:

* it is the target of a ``latest`` (or any other) symlink;
* it is among the newest ``keep_newest`` versions of its ``{SYMBOL}/{TYPE}``;
* its version id appears in a reference file (config, research notes,
  model-promotion records) -- something pins or documents it;
* its name does not follow the dated naming convention, so its age is unknown.

Planning never touches the filesystem beyond reading. Deletion is a separate
step (``apply_plan``) that only ever removes directories the plan lists, and
never a symlink.
"""

from __future__ import annotations

import re
import shutil
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_KEEP_NEWEST = 3
LATEST_LINK = "latest"
# Repo-relative places where config, research notes and model-promotion records pin versions.
DEFAULT_REFERENCE_PATHS = ("src/config", "feature_flags.json", "docs/research")

_REFERENCE_SUFFIXES = frozenset({".json", ".md", ".py", ".yml", ".yaml", ".toml", ".txt"})
_MAX_REFERENCE_FILE_BYTES = 5 * 1024 * 1024

_DATE_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})")
_TIME_RE = re.compile(r"_(\d+)h(?:(\d+)m)?(?:(\d+)s)?(?=_|$)")
_SEQ_RE = re.compile(r"_v(\d+)$")


class RegistryStateError(RuntimeError):
    """The registry is in a state the retention policy refuses to act on."""


@dataclass(frozen=True)
class ModelVersion:
    """One version directory in the registry."""

    symbol: str
    model_type: str
    version: str
    path: Path
    size_bytes: int

    @property
    def key(self) -> str:
        return f"{self.symbol}/{self.model_type}/{self.version}"


@dataclass
class RetentionPlan:
    """Outcome of applying the policy to a registry."""

    candidates: list[ModelVersion] = field(default_factory=list)
    protected: dict[str, list[str]] = field(default_factory=dict)

    @property
    def bytes_freed(self) -> int:
        return sum(v.size_bytes for v in self.candidates)


def version_sort_key(version: str) -> tuple[int, ...] | None:
    """Chronological sort key parsed from a version id, or None if undated.

    ``2026-07-05_10h30m00s_v1`` -> (2026, 7, 5, 10, 30, 0, 1). Date-only ids such as
    ``2025-09-16_legacy`` sort at midnight with sequence 0.
    """
    date = _DATE_RE.match(version)
    if date is None:
        return None
    time = _TIME_RE.search(version)
    seq = _SEQ_RE.search(version)
    clock = tuple(int(g) if g else 0 for g in time.groups()) if time else (0, 0, 0)
    return (
        *(int(g) for g in date.groups()),
        *clock,
        int(seq.group(1)) if seq else 0,
    )


def directory_size(path: Path) -> int:
    """Total bytes of regular files under ``path`` (symlinks are not followed)."""
    total = 0
    for item in path.rglob("*"):
        if item.is_symlink():
            continue
        try:
            if item.is_file():
                total += item.stat().st_size
        except OSError:
            continue
    return total


def _read_reference_texts(reference_paths: Iterable[Path]) -> list[tuple[Path, str]]:
    files: list[Path] = []
    for ref in reference_paths:
        if ref.is_dir():
            files.extend(p for p in sorted(ref.rglob("*")) if p.is_file())
        elif ref.is_file():
            files.append(ref)
    texts: list[tuple[Path, str]] = []
    for f in files:
        if f.suffix.lower() not in _REFERENCE_SUFFIXES:
            continue
        try:
            if f.stat().st_size > _MAX_REFERENCE_FILE_BYTES:
                continue
            texts.append((f, f.read_text(encoding="utf-8", errors="ignore")))
        except OSError:
            continue
    return texts


def _referenced_in(version: str, texts: Sequence[tuple[Path, str]]) -> list[Path]:
    # Boundary-aware so ``..._v1`` does not match a mention of ``..._v12``.
    pattern = re.compile(rf"(?<![0-9A-Za-z_-]){re.escape(version)}(?![0-9A-Za-z_])")
    return [path for path, text in texts if pattern.search(text)]


def _newest_versions(version_dirs: Sequence[Path], keep: int) -> set[str]:
    keyed = [(key, d.name) for d in version_dirs if (key := version_sort_key(d.name)) is not None]
    keyed.sort(reverse=True)
    return {name for _, name in keyed[:keep]}


def _scan_type_dir(symbol: str, type_dir: Path) -> tuple[list[Path], dict[str, str]]:
    """Return (version dirs, protected-by-symlink map) for one ``{SYMBOL}/{TYPE}`` dir.

    Raises:
        RegistryStateError: if a symlink is dangling, escapes the type directory, or
            ``latest`` exists as a real directory.
    """
    versions: list[Path] = []
    link_targets: dict[str, str] = {}
    for entry in sorted(type_dir.iterdir()):
        if entry.is_symlink():
            target = entry.resolve()
            if not target.is_dir():
                raise RegistryStateError(
                    f"Dangling symlink {entry} -> {entry.readlink()}; fix it before cleaning up."
                )
            if target.parent != type_dir.resolve():
                raise RegistryStateError(
                    f"Symlink {entry} points outside {type_dir}; refusing to clean up."
                )
            link_targets[target.name] = f"target of '{entry.name}' symlink"
        elif entry.is_dir():
            if entry.name == LATEST_LINK:
                raise RegistryStateError(
                    f"{entry} is a directory, not a symlink; refusing to clean up."
                )
            versions.append(entry)
    return versions, link_targets


def plan_cleanup(
    registry: Path,
    *,
    keep_newest: int = DEFAULT_KEEP_NEWEST,
    reference_paths: Sequence[Path] = (),
) -> RetentionPlan:
    """Decide which version directories may be removed. Reads only.

    Args:
        registry: Model registry root (``{registry}/{SYMBOL}/{TYPE}/{VERSION}``).
        keep_newest: Newest versions to keep per ``{SYMBOL}/{TYPE}``.
        reference_paths: Files or directories whose text pins or documents versions.

    Raises:
        ValueError: if ``keep_newest`` is negative.
        RegistryStateError: if the registry is inconsistent (see ``_scan_type_dir``).
    """
    if keep_newest < 0:
        raise ValueError("keep_newest must be >= 0")
    if not registry.is_dir():
        raise RegistryStateError(f"Model registry directory does not exist: {registry}")

    texts = _read_reference_texts(reference_paths)
    plan = RetentionPlan()

    for symbol_dir in sorted(p for p in registry.iterdir() if p.is_dir() and not p.is_symlink()):
        for type_dir in sorted(
            p for p in symbol_dir.iterdir() if p.is_dir() and not p.is_symlink()
        ):
            version_dirs, link_targets = _scan_type_dir(symbol_dir.name, type_dir)
            newest = _newest_versions(version_dirs, keep_newest)
            for vdir in version_dirs:
                mv = ModelVersion(
                    symbol=symbol_dir.name,
                    model_type=type_dir.name,
                    version=vdir.name,
                    path=vdir,
                    size_bytes=directory_size(vdir),
                )
                reasons: list[str] = []
                if vdir.name in link_targets:
                    reasons.append(link_targets[vdir.name])
                if vdir.name in newest:
                    reasons.append(f"one of the newest {keep_newest}")
                if version_sort_key(vdir.name) is None:
                    reasons.append("unrecognised version name (age unknown)")
                refs = _referenced_in(vdir.name, texts)
                if refs:
                    reasons.append(f"referenced in {len(refs)} file(s), e.g. {refs[0].name}")
                if reasons:
                    plan.protected[mv.key] = reasons
                else:
                    plan.candidates.append(mv)
    return plan


def apply_plan(registry: Path, plan: RetentionPlan) -> list[ModelVersion]:
    """Delete the plan's candidates and return what was removed.

    Each path is re-checked right before removal: it must still be a real directory
    (never a symlink) at ``{registry}/{SYMBOL}/{TYPE}/{VERSION}``.
    """
    root = registry.resolve()
    removed: list[ModelVersion] = []
    for mv in plan.candidates:
        path = mv.path
        expected = root / mv.symbol / mv.model_type / mv.version
        if path.is_symlink() or not path.is_dir() or path.resolve() != expected:
            raise RegistryStateError(f"Refusing to delete {path}: not a plain version directory.")
        shutil.rmtree(path)
        removed.append(mv)
    return removed
