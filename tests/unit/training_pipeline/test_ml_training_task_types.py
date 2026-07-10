"""Unit tests for the model/target task-type compatibility guard (#947).

`models_tft.py` compiles a binary-classification head (sigmoid, BCE loss),
but before this guard existed `pipeline.py` built the regression close
target unconditionally -- training tft silently fit a classification head
against a continuous price target (no crash, no warning, garbage model).
This guard must refuse loudly on any (model_type, target_type) mismatch.
"""

import pytest

from src.ml.training_pipeline.task_types import (
    TARGET_TASK_TYPES,
    TaskType,
    get_model_task_type,
    get_target_class_labels,
    get_target_task_type,
    validate_target_head_compatibility,
)


class TestGetModelTaskType:
    @pytest.mark.parametrize(
        "model_type",
        ["lstm", "cnn_lstm", "adaptive", "default", "attention_lstm", "tcn", "tcn_attention"],
    )
    def test_regression_architectures(self, model_type):
        assert get_model_task_type(model_type) is TaskType.REGRESSION

    def test_tft_is_binary_classification(self):
        assert get_model_task_type("tft") is TaskType.BINARY_CLASSIFICATION

    def test_tft_ternary_is_ternary_classification(self):
        """tft_ternary (entrant (c)'s 3-class softmax sibling to tft) must
        declare TERNARY_CLASSIFICATION so validate_target_head_compatibility
        accepts it against triple_barrier."""
        assert get_model_task_type("tft_ternary") is TaskType.TERNARY_CLASSIFICATION

    def test_case_insensitive(self):
        assert get_model_task_type("TFT") is TaskType.BINARY_CLASSIFICATION
        assert get_model_task_type("CNN_LSTM") is TaskType.REGRESSION

    def test_unknown_model_type_raises(self):
        with pytest.raises(ValueError, match="model_type"):
            get_model_task_type("not_a_real_architecture")


class TestGetTargetTaskType:
    def test_regression(self):
        assert get_target_task_type("regression") is TaskType.REGRESSION

    def test_binary_direction(self):
        assert get_target_task_type("binary_direction") is TaskType.BINARY_CLASSIFICATION

    def test_triple_barrier(self):
        assert get_target_task_type("triple_barrier") is TaskType.TERNARY_CLASSIFICATION

    def test_smoothed_return(self):
        assert get_target_task_type("smoothed_return") is TaskType.REGRESSION

    def test_meta_label(self):
        assert get_target_task_type("meta_label") is TaskType.BINARY_CLASSIFICATION

    def test_unknown_target_type_raises(self):
        with pytest.raises(ValueError, match="target_type"):
            get_target_task_type("not_a_real_target")

    def test_every_target_type_covered(self):
        # Every key in TARGET_TASK_TYPES must resolve via get_target_task_type
        # (defends against the two falling out of sync).
        for target_type in TARGET_TASK_TYPES:
            assert get_target_task_type(target_type) == TARGET_TASK_TYPES[target_type]


class TestValidateTargetHeadCompatibility:
    def test_regression_architecture_with_regression_target_is_compatible(self):
        validate_target_head_compatibility("cnn_lstm", "regression")  # no raise

    def test_regression_architecture_with_smoothed_return_is_compatible(self):
        # smoothed_return is REGRESSION task type -- compatible with any
        # regression-headed architecture.
        validate_target_head_compatibility("attention_lstm", "smoothed_return")  # no raise

    def test_tft_with_binary_direction_is_compatible(self):
        validate_target_head_compatibility("tft", "binary_direction")  # no raise

    def test_tft_with_default_regression_target_raises(self):
        """This is the exact #947 bug: tft (binary_classification head) trained
        against the regression target with no error today. Must raise now."""
        with pytest.raises(ValueError, match="incompatible"):
            validate_target_head_compatibility("tft", "regression")

    def test_regression_architecture_with_binary_direction_raises(self):
        with pytest.raises(ValueError, match="incompatible"):
            validate_target_head_compatibility("cnn_lstm", "binary_direction")

    def test_regression_architecture_with_triple_barrier_raises(self):
        with pytest.raises(ValueError, match="incompatible"):
            validate_target_head_compatibility("lstm", "triple_barrier")

    def test_tft_with_triple_barrier_raises(self):
        """tft is a binary head; triple_barrier needs a 3-class head, which
        is tft_ternary, not tft."""
        with pytest.raises(ValueError, match="incompatible"):
            validate_target_head_compatibility("tft", "triple_barrier")

    def test_tft_ternary_with_triple_barrier_is_compatible(self):
        validate_target_head_compatibility("tft_ternary", "triple_barrier")  # no raise

    def test_tft_ternary_with_regression_raises(self):
        with pytest.raises(ValueError, match="incompatible"):
            validate_target_head_compatibility("tft_ternary", "regression")

    def test_unknown_model_type_raises(self):
        with pytest.raises(ValueError, match="model_type"):
            validate_target_head_compatibility("not_a_real_architecture", "regression")

    def test_unknown_target_type_raises(self):
        with pytest.raises(ValueError, match="target_type"):
            validate_target_head_compatibility("cnn_lstm", "not_a_real_target")

    def test_error_message_names_both_types(self):
        with pytest.raises(ValueError) as exc_info:
            validate_target_head_compatibility("tft", "regression")
        message = str(exc_info.value)
        assert "tft" in message
        assert "regression" in message


class TestGetTargetClassLabels:
    """class_labels ARE the direction values (see onnx_runner.py's
    _process_classification_output) -- ordered to match the probability
    vector's class order."""

    def test_binary_direction(self):
        assert get_target_class_labels("binary_direction") == [-1, 1]

    def test_triple_barrier(self):
        assert get_target_class_labels("triple_barrier") == [-1, 0, 1]

    def test_regression_has_no_class_labels(self):
        assert get_target_class_labels("regression") is None

    def test_smoothed_return_has_no_class_labels(self):
        assert get_target_class_labels("smoothed_return") is None

    def test_every_classification_target_type_has_class_labels(self):
        for target_type, task_type in TARGET_TASK_TYPES.items():
            if task_type is TaskType.REGRESSION:
                continue
            if target_type == "meta_label":
                # meta_label's classification output is a profitability gate
                # on the primary signal's own direction, not a direction
                # prediction itself -- it has no canonical class_labels
                # ordering in the direction-value sense this function covers.
                continue
            assert get_target_class_labels(target_type) is not None, target_type
