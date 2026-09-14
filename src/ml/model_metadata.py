"""Shared bundle-metadata contract for every training path.

Local (``atb train``) and cloud (``atb train cloud``) training must emit the
same keys, because the prediction path reads them to decide whether a model's
raw output still needs denormalizing. Until #1049 the cloud path omitted
``price_normalization`` and both denormalization sites fell through to
"return as-is", so a normalized ~[0,1] output was compared against a real
price -- structurally valid, numerically nonsense, and silent.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

ROLLING_MINMAX = "rolling_minmax"
PRICE_ONLY_FEATURE_STRATEGY = "price_only_rolling_minmax"
NORMALIZED_SUFFIX = "_normalized"
DEFAULT_TARGET_FEATURE = "close"

# Keys the live prediction path reads off a regression bundle. Absence of any
# of these used to degrade silently rather than raise.
REQUIRED_PREDICTION_KEYS = ("price_normalization", "model_file", "framework")


def build_price_normalization(
    sequence_length: int, target_feature: str = DEFAULT_TARGET_FEATURE
) -> dict[str, Any]:
    """Build the ``price_normalization`` block for a rolling-minmax bundle."""
    return {
        "method": ROLLING_MINMAX,
        "window": int(sequence_length),
        "target_feature": target_feature,
    }


def uses_rolling_minmax_features(metadata: dict[str, Any]) -> bool:
    """Whether this bundle's model emits rolling-minmax-normalized price output.

    Detected from the bundle itself rather than from which code path wrote it:
    a regression model whose target feature was fed in normalized form is,
    by construction, producing normalized output.
    """
    task_type = str(metadata.get("task_type") or "regression").lower()
    if task_type != "regression":
        return False

    feature_names = metadata.get("feature_names") or []
    if not isinstance(feature_names, list | tuple):
        return False

    target = str(
        (metadata.get("price_normalization") or {}).get("target_feature", DEFAULT_TARGET_FEATURE)
    )
    return f"{target}{NORMALIZED_SUFFIX}" in {str(f) for f in feature_names}


def missing_prediction_keys(metadata: dict[str, Any]) -> list[str]:
    """Required keys absent from a bundle that needs denormalization."""
    if not uses_rolling_minmax_features(metadata):
        return []
    return [key for key in REQUIRED_PREDICTION_KEYS if not metadata.get(key)]


def enrich_bundle_metadata(metadata: dict[str, Any]) -> list[str]:
    """Fill in the prediction-path keys a rolling-minmax bundle is missing.

    Mutates ``metadata`` in place and returns the keys that were added. Only
    ever adds what is derivable from the bundle itself -- it never overwrites
    a value the training path already wrote.
    """
    missing = missing_prediction_keys(metadata)
    if not missing:
        return []

    defaults: dict[str, Any] = {
        "price_normalization": build_price_normalization(
            int(metadata.get("sequence_length") or 120)
        ),
        "model_file": "model.onnx",
        "framework": "onnx",
    }
    for key in missing:
        metadata[key] = defaults[key]
    metadata.setdefault("feature_strategy", PRICE_ONLY_FEATURE_STRATEGY)
    return missing


def validate_bundle_metadata(metadata: dict[str, Any], *, bundle_id: str) -> None:
    """Raise if a bundle needing denormalization cannot be served correctly.

    Raises:
        ValueError: The bundle emits normalized price output but does not
            declare how to denormalize it.
    """
    missing = missing_prediction_keys(metadata)
    if not missing:
        return
    raise ValueError(
        f"Model bundle {bundle_id} has rolling-minmax normalized features but is "
        f"missing required metadata: {', '.join(missing)}. Its raw output is in "
        f"normalized space and would be compared against real prices. Re-train or "
        f"repair the bundle metadata (see #1049)."
    )


def ensure_bundle_metadata_complete(bundle_dir: Path) -> list[str]:
    """Enrich a synced bundle's ``metadata.json`` on disk.

    Returns the keys added, or an empty list when nothing was missing.
    """
    metadata_path = Path(bundle_dir) / "metadata.json"
    if not metadata_path.exists():
        return []

    with open(metadata_path, encoding="utf-8") as handle:
        metadata = json.load(handle)
    if not isinstance(metadata, dict):
        return []

    added = enrich_bundle_metadata(metadata)
    if added:
        # Atomic write (write-temp-then-rename, matching gate.py/meta_labels.py's
        # checkpoint pattern) so a crash mid-write never leaves a torn
        # metadata.json -- this file gates whether the registry will serve the
        # bundle at all.
        tmp_path = metadata_path.with_suffix(f"{metadata_path.suffix}.tmp")
        tmp_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        os.replace(tmp_path, metadata_path)
        logger.warning(
            "Backfilled prediction metadata %s into %s -- the training path should "
            "have written these (see #1049)",
            added,
            metadata_path,
        )
    return added
