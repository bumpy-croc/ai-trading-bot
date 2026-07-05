"""Explicit promotion of model bundles between registry namespaces.

Cloud training syncs bundles into ``{SYMBOL}/price/`` (the ``model_type``
recorded by the training pipeline), while live strategies load
``{SYMBOL}/basic/latest``. Promoting a cloud-trained model into the live
namespace is therefore a deliberate, human-triggered copy — never automatic.
The ``basic/latest`` symlink is only touched when ``set_latest`` is passed.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
from pathlib import Path

from src.infrastructure.runtime.paths import get_project_root
from src.ml.cloud.exceptions import ModelPromotionError

logger = logging.getLogger(__name__)

# Model bundle is only usable with at least one of these files present.
_MODEL_FILES = ("model.onnx", "model.keras")


def _validate_component(value: str, field: str) -> str:
    """Validate a user-supplied path component used inside the registry.

    Symbol, model type, and version id become directory names under the
    registry root; anything containing separators or ``..`` could escape it.
    """
    text = str(value)
    if text in (".", "..") or not re.match(r"^[\w\-.]+$", text):
        raise ModelPromotionError(f"Unsafe {field}: {value!r}")
    return text


def _update_latest_symlink(type_dir: Path, version_id: str) -> None:
    """Atomically point ``{type_dir}/latest`` at ``version_id``."""
    latest_link = type_dir / "latest"
    temp_link = type_dir / f".latest.{version_id}.tmp"

    if temp_link.exists() or temp_link.is_symlink():
        temp_link.unlink()
    temp_link.symlink_to(version_id)
    # os.replace is atomic on POSIX, so readers never see a missing symlink
    os.replace(str(temp_link), str(latest_link))


def promote_model_version(
    symbol: str,
    version_id: str,
    source_type: str = "price",
    target_type: str = "basic",
    set_latest: bool = False,
    registry_root: Path | None = None,
) -> Path:
    """Copy a model bundle from one registry namespace to another.

    Args:
        symbol: Trading symbol (e.g., BTCUSDT)
        version_id: Version directory name to promote
        source_type: Namespace the bundle currently lives in (default: price)
        target_type: Namespace to copy the bundle into (default: basic)
        set_latest: Also point ``{target_type}/latest`` at the promoted version.
            Without this flag the live-loading symlink is never touched.
        registry_root: Model registry root (default: ``src/ml/models``)

    Returns:
        Path to the promoted bundle directory

    Raises:
        ModelPromotionError: If the source is missing/incomplete, the target
            already exists, or any path component is unsafe.
    """
    symbol = _validate_component(symbol, "symbol").upper()
    version_id = _validate_component(version_id, "version_id")
    source_type = _validate_component(source_type, "source_type")
    target_type = _validate_component(target_type, "target_type")

    root = registry_root or (get_project_root() / "src" / "ml" / "models")
    source_dir = root / symbol / source_type / version_id
    target_dir = root / symbol / target_type / version_id

    # Defense in depth: validated components cannot escape, but check anyway
    # before any filesystem mutation.
    root_resolved = root.resolve()
    for candidate in (source_dir, target_dir):
        if not candidate.resolve().is_relative_to(root_resolved):
            raise ModelPromotionError(f"Path escapes registry root: {candidate}")

    if not source_dir.is_dir():
        raise ModelPromotionError(f"Source version not found: {source_dir}")
    if not any((source_dir / name).exists() for name in _MODEL_FILES):
        raise ModelPromotionError(
            f"Source bundle has no model file ({' or '.join(_MODEL_FILES)}): {source_dir}"
        )
    if target_dir.exists():
        raise ModelPromotionError(
            f"Target version already exists, refusing to overwrite: {target_dir}"
        )

    target_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_dir, target_dir)
    logger.info("Promoted %s/%s/%s -> %s", symbol, source_type, version_id, target_dir)

    if set_latest:
        _update_latest_symlink(target_dir.parent, version_id)
        logger.info("Updated %s/%s/latest -> %s", symbol, target_type, version_id)

    return target_dir
