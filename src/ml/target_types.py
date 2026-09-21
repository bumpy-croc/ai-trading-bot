"""Target-type to task-type declarations shared by training and inference.

Neutral module (no pipeline imports) so ``model_metadata`` can classify bundles
without depending on the training pipeline package.
"""

from __future__ import annotations

from enum import Enum


class TaskType(Enum):
    """What kind of target a model's compiled head expects."""

    REGRESSION = "regression"
    BINARY_CLASSIFICATION = "binary_classification"
    TERNARY_CLASSIFICATION = "ternary_classification"


# Every target_type the training pipeline can build a label for, by task
# type. "regression" is the incumbent next-bar price-regression target
# (pipeline.py's current unconditional behavior); "meta_label" is built by
# meta_labels.py rather than labels.py but is listed here so the guard
# covers it too.
TARGET_TASK_TYPES: dict[str, TaskType] = {
    "regression": TaskType.REGRESSION,
    "binary_direction": TaskType.BINARY_CLASSIFICATION,
    "triple_barrier": TaskType.TERNARY_CLASSIFICATION,
    "smoothed_return": TaskType.REGRESSION,
    "meta_label": TaskType.BINARY_CLASSIFICATION,
}

# Regression target_types whose label is the rolling-min-max-normalized PRICE
# (output must be denormalized before comparing with a real price). Every
# other REGRESSION target (e.g. smoothed_return) is small-scale and must not
# be. Kept beside TARGET_TASK_TYPES so a new regression target forces an
# explicit decision here.
PRICE_SCALE_TARGET_TYPES: frozenset[str] = frozenset({"regression"})


__all__ = ["PRICE_SCALE_TARGET_TYPES", "TARGET_TASK_TYPES", "TaskType"]
