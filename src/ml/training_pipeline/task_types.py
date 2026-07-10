"""Model/target task-type declarations and the #947 compatibility guard.

`create_model()` (`models.py`) dispatches to architectures with fixed,
already-compiled heads: every family except `tft` compiles a regression head
(`loss="mse"`, an `rmse` metric); `tft` compiles a binary-classification head
(`loss="binary_crossentropy"`, sigmoid output). Before this module existed,
`pipeline.py` built the regression close/close_normalized target
unconditionally regardless of `model_type` — training `tft` silently fit a
classification head against a continuous price target: no crash, no
warning, garbage model (GH #947, the 4th instance of the per-architecture
contract-drift family after #928/#931/#936).

This module is the single source of truth both `pipeline.py` (to derive the
right target for a given `model_type`) and its regression-guard tests
consult, so the two can never drift out of sync silently.
"""

from __future__ import annotations

from enum import Enum


class TaskType(Enum):
    """What kind of target a model's compiled head expects."""

    REGRESSION = "regression"
    BINARY_CLASSIFICATION = "binary_classification"
    TERNARY_CLASSIFICATION = "ternary_classification"


# Every model_type selectable via `create_model()` (models.py), by task type.
# "adaptive"/"default" are cnn_lstm aliases (models.py:200).
MODEL_TASK_TYPES: dict[str, TaskType] = {
    "lstm": TaskType.REGRESSION,
    "cnn_lstm": TaskType.REGRESSION,
    "adaptive": TaskType.REGRESSION,
    "default": TaskType.REGRESSION,
    "attention_lstm": TaskType.REGRESSION,
    "tcn": TaskType.REGRESSION,
    "tcn_attention": TaskType.REGRESSION,
    "tft": TaskType.BINARY_CLASSIFICATION,
}

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


def get_model_task_type(model_type: str) -> TaskType:
    """Return the task type a model_type's compiled head expects.

    Raises:
        ValueError: model_type is not a recognized `create_model()` architecture.
    """
    task_type = MODEL_TASK_TYPES.get(model_type.lower())
    if task_type is None:
        raise ValueError(
            f"Unknown model_type: {model_type!r}. " f"Known model types: {sorted(MODEL_TASK_TYPES)}"
        )
    return task_type


def get_target_task_type(target_type: str) -> TaskType:
    """Return the task type a target_type's label produces.

    Raises:
        ValueError: target_type is not a recognized label type.
    """
    task_type = TARGET_TASK_TYPES.get(target_type)
    if task_type is None:
        raise ValueError(
            f"Unknown target_type: {target_type!r}. "
            f"Known target types: {sorted(TARGET_TASK_TYPES)}"
        )
    return task_type


def validate_target_head_compatibility(model_type: str, target_type: str) -> None:
    """Refuse loudly when a model_type's head is incompatible with the target.

    This is the #947 guard: a model_type's compiled head (regression MSE
    loss, binary sigmoid/BCE, ...) must match the task type of the label
    being built for it. Call this before any data download / training work
    starts, so an incompatible combination fails fast and cheap.

    Args:
        model_type: Architecture selector passed to `create_model()`.
        target_type: Label type selector (see TARGET_TASK_TYPES).

    Raises:
        ValueError: Either type is unrecognized, or they're incompatible.
    """
    model_task = get_model_task_type(model_type)
    target_task = get_target_task_type(target_type)

    if model_task is not target_task:
        raise ValueError(
            f"model_type={model_type!r} has a {model_task.value} head, incompatible "
            f"with target_type={target_type!r} (produces a {target_task.value} label). "
            f"Pick a model_type/target_type pair with matching task types."
        )


__all__ = [
    "MODEL_TASK_TYPES",
    "TARGET_TASK_TYPES",
    "TaskType",
    "get_model_task_type",
    "get_target_task_type",
    "validate_target_head_compatibility",
]
