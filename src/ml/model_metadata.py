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

from src.ml.training_pipeline.task_types import PRICE_SCALE_TARGET_TYPES, TARGET_TASK_TYPES

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
    by construction, producing normalized output -- PROVIDED the target
    itself is the normalized price. A normalized price feature can also feed
    a model trained on a different target (e.g. ``--force-price-only
    --target-type smoothed_return``): same input, wholly different output
    scale. ``task_type`` alone can't tell these apart -- "regression" covers
    both "regression" (predicts close_normalized) and "smoothed_return"
    (predicts a small-scale return) target types (see
    ``training_pipeline/task_types.py::TARGET_TASK_TYPES``) -- so this also
    consults the recorded ``training_params.target_type`` (GH #1145).

    When ``target_type`` is absent altogether (bundles predating this field,
    including the still-served ETHUSDT/basic/2026-07-04_22h_v1), we fall
    back to the old feature-based inference rather than rejecting the
    bundle: every target_type-less bundle was trained before smoothed_return
    existed, so the ambiguity this function exists to resolve cannot arise
    for it, and refusing it here would break `PredictionModelRegistry` bundle
    loads (`validate_bundle_metadata`) for models that already carry a
    correct, explicitly-written `price_normalization` block.
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
    if f"{target}{NORMALIZED_SUFFIX}" not in {str(f) for f in feature_names}:
        return False

    training_params = metadata.get("training_params")
    if not isinstance(training_params, dict):
        training_params = {}
    target_type = training_params.get("target_type")
    if target_type is None:
        return True
    normalized_target = str(target_type).lower()
    if normalized_target not in TARGET_TASK_TYPES:
        logger.warning(
            "Unknown target_type %r in bundle metadata; treating as non-price output. "
            "Register it in task_types.TARGET_TASK_TYPES (and PRICE_SCALE_TARGET_TYPES if "
            "it is price-scale).",
            target_type,
        )
    return normalized_target in PRICE_SCALE_TARGET_TYPES


def missing_prediction_keys(metadata: dict[str, Any]) -> list[str]:
    """Required keys absent from a bundle that needs denormalization."""
    if not uses_rolling_minmax_features(metadata):
        return []
    return [key for key in REQUIRED_PREDICTION_KEYS if not metadata.get(key)]


def has_contradictory_price_normalization(metadata: dict[str, Any]) -> bool:
    """True when a ``price_normalization`` block is already stamped but the
    bundle's own ``target_type`` says it should not be (GH #1145 follow-up).

    ``uses_rolling_minmax_features`` decides whether a bundle NEEDS the block
    going forward; this catches the case where one was already written and
    is now wrong -- a bundle enriched by ``ensure_bundle_metadata_complete``
    before the #1145 fix landed (or edited by hand), whose ``target_type`` is
    a non-price target (e.g. ``smoothed_return``) but which still carries a
    stale ``rolling_minmax`` block from before that distinction existed.
    ``PredictionEngine._apply_rolling_denormalization`` reads the stamped
    block directly, not this classifier, so a bundle in this state is
    denormalized as a price regardless of what the classifier would say
    about it today -- this check is what actually stops it from being served.
    """
    price_norm = metadata.get("price_normalization")
    if not isinstance(price_norm, dict) or price_norm.get("method") != ROLLING_MINMAX:
        return False
    return not uses_rolling_minmax_features(metadata)


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
    if missing:
        raise ValueError(
            f"Model bundle {bundle_id} has rolling-minmax normalized features but is "
            f"missing required metadata: {', '.join(missing)}. Its raw output is in "
            f"normalized space and would be compared against real prices. Re-train or "
            f"repair the bundle metadata (see #1049)."
        )
    if has_contradictory_price_normalization(metadata):
        target_type = (metadata.get("training_params") or {}).get("target_type")
        raise ValueError(
            f"Model bundle {bundle_id} carries a price_normalization block but its "
            f"training_params.target_type ({target_type!r}) is not a price-scale target. "
            f"Its raw output would be silently denormalized as a price when it is not one "
            f"(likely enriched before GH #1145's fix, or edited by hand). Re-train or "
            f"repair the bundle metadata (see #1049, #1145)."
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
