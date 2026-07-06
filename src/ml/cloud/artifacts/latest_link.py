"""Atomic 'latest' symlink updates for the local model registry."""

from __future__ import annotations

import os
from pathlib import Path


def update_latest_symlink(type_dir: Path, version_id: str) -> None:
    """Atomically point ``{type_dir}/latest`` at ``version_id``.

    Creates the symlink under a temporary name and swaps it in with
    ``os.replace`` (atomic on POSIX), so readers never observe a missing
    ``latest`` during the update.

    Raises:
        OSError: If the symlink cannot be created or swapped in.
    """
    latest_link = type_dir / "latest"
    temp_link = type_dir / f".latest.{version_id}.tmp"

    # Clean up any stale temp link from an interrupted earlier update
    if temp_link.exists() or temp_link.is_symlink():
        temp_link.unlink()

    try:
        temp_link.symlink_to(version_id)
        os.replace(str(temp_link), str(latest_link))
    except OSError:
        if temp_link.exists() or temp_link.is_symlink():
            try:
                temp_link.unlink()
            except OSError:
                # Best-effort cleanup — the original error takes precedence
                pass
        raise
