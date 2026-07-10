"""Classification-native prediction path (TARGET-REDESIGN tournament, #933 Phase 2).

Covers OnnxRunner's handling of classification model_metadata (task_type=
"binary_classification"/"ternary_classification"), which is purely additive:
every existing regression bundle has no "task_type" key in its metadata, so
none of this code executes for them -- verified explicitly below.
"""

from unittest.mock import Mock, mock_open, patch

import numpy as np
import pytest

from src.prediction.config import PredictionConfig
from src.prediction.models.onnx_runner import ModelPrediction, OnnxRunner


@pytest.fixture
def config():
    return PredictionConfig(
        prediction_horizons=[1],
        min_confidence_threshold=0.6,
        max_prediction_latency=0.1,
        model_registry_path="src/ml/models",
    )


@pytest.fixture(autouse=True)
def _mock_providers():
    with patch(
        "src.prediction.models.onnx_runner.get_preferred_providers",
        return_value=["CPUExecutionProvider"],
    ):
        yield


class TestBinaryClassificationOutput:
    def _make_runner(self, config, metadata_json: str) -> OnnxRunner:
        with (
            patch("onnxruntime.InferenceSession", return_value=Mock()),
            patch("builtins.open", mock_open(read_data=metadata_json)),
        ):
            return OnnxRunner("/tmp/test_model.onnx", config)

    def test_high_up_probability_maps_to_up_direction(self, config):
        metadata = (
            '{"sequence_length": 120, "task_type": "binary_classification", '
            '"class_labels": [-1, 1]}'
        )
        runner = self._make_runner(config, metadata)

        output = np.array([[0.9]])  # P(up) = 0.9
        result = runner._process_output(output)

        assert result["direction"] == 1
        assert result["confidence"] == pytest.approx(0.9)
        assert result["probabilities"] == pytest.approx((0.1, 0.9))
        # price is not a real output for a classifier -- NaN, never a
        # spurious-looking real price a naive consumer could misread.
        assert np.isnan(result["price"])

    def test_low_up_probability_maps_to_down_direction(self, config):
        metadata = (
            '{"sequence_length": 120, "task_type": "binary_classification", '
            '"class_labels": [-1, 1]}'
        )
        runner = self._make_runner(config, metadata)

        output = np.array([[0.2]])  # P(up) = 0.2 -> P(down) = 0.8
        result = runner._process_output(output)

        assert result["direction"] == -1
        assert result["confidence"] == pytest.approx(0.8)
        assert result["probabilities"] == pytest.approx((0.8, 0.2))

    def test_missing_class_labels_raises(self, config):
        metadata = '{"sequence_length": 120, "task_type": "binary_classification"}'
        runner = self._make_runner(config, metadata)

        with pytest.raises(ValueError, match="class_labels"):
            runner._process_output(np.array([[0.7]]))

    def test_wrong_class_label_count_raises(self, config):
        metadata = (
            '{"sequence_length": 120, "task_type": "binary_classification", '
            '"class_labels": [-1, 0, 1]}'
        )
        runner = self._make_runner(config, metadata)

        with pytest.raises(ValueError, match="2 class_labels"):
            runner._process_output(np.array([[0.7]]))

    def test_non_finite_output_raises(self, config):
        metadata = (
            '{"sequence_length": 120, "task_type": "binary_classification", '
            '"class_labels": [-1, 1]}'
        )
        runner = self._make_runner(config, metadata)

        with pytest.raises(ValueError, match="[Nn]on-finite"):
            runner._process_output(np.array([[float("nan")]]))


class TestTernaryClassificationOutput:
    def _make_runner(self, config, metadata_json: str) -> OnnxRunner:
        with (
            patch("onnxruntime.InferenceSession", return_value=Mock()),
            patch("builtins.open", mock_open(read_data=metadata_json)),
        ):
            return OnnxRunner("/tmp/test_model.onnx", config)

    def test_softmax_output_argmax_and_confidence(self, config):
        metadata = (
            '{"sequence_length": 120, "task_type": "ternary_classification", '
            '"class_labels": [-1, 0, 1]}'
        )
        runner = self._make_runner(config, metadata)

        # [P(down)=0.1, P(hold)=0.15, P(up)=0.75]
        output = np.array([[0.1, 0.15, 0.75]])
        result = runner._process_output(output)

        assert result["direction"] == 1
        assert result["confidence"] == pytest.approx(0.75)
        assert result["probabilities"] == pytest.approx((0.1, 0.15, 0.75))

    def test_hold_class_wins(self, config):
        metadata = (
            '{"sequence_length": 120, "task_type": "ternary_classification", '
            '"class_labels": [-1, 0, 1]}'
        )
        runner = self._make_runner(config, metadata)

        output = np.array([[0.2, 0.6, 0.2]])
        result = runner._process_output(output)

        assert result["direction"] == 0
        assert result["confidence"] == pytest.approx(0.6)

    def test_output_class_count_mismatch_raises(self, config):
        metadata = (
            '{"sequence_length": 120, "task_type": "ternary_classification", '
            '"class_labels": [-1, 0, 1]}'
        )
        runner = self._make_runner(config, metadata)

        with pytest.raises(ValueError, match="class_labels"):
            runner._process_output(np.array([[0.5, 0.5]]))


class TestRegressionUnaffected:
    """The classification path is purely additive -- every existing
    regression bundle has no task_type key, and must behave byte-identically
    to before this feature existed."""

    def test_no_task_type_key_uses_regression_path(self, config):
        with (
            patch("onnxruntime.InferenceSession", return_value=Mock()),
            patch("builtins.open", mock_open(read_data='{"sequence_length": 120}')),
        ):
            runner = OnnxRunner("/tmp/test_model.onnx", config)

        output = np.array([[[0.05]]])
        result = runner._process_output(output)

        assert result["price"] > 0
        assert result["direction"] == 1
        assert "probabilities" not in result or result.get("probabilities") is None

    def test_explicit_regression_task_type_uses_regression_path(self, config):
        with (
            patch("onnxruntime.InferenceSession", return_value=Mock()),
            patch(
                "builtins.open",
                mock_open(read_data='{"sequence_length": 120, "task_type": "regression"}'),
            ),
        ):
            runner = OnnxRunner("/tmp/test_model.onnx", config)

        output = np.array([[[0.05]]])
        result = runner._process_output(output)

        assert result["price"] > 0
        assert result["direction"] == 1


class TestModelPredictionProbabilitiesField:
    def test_default_none(self):
        pred = ModelPrediction(
            price=100.0, confidence=0.5, direction=1, model_name="m", inference_time=0.01
        )
        assert pred.probabilities is None

    def test_accepts_probabilities(self):
        pred = ModelPrediction(
            price=float("nan"),
            confidence=0.8,
            direction=1,
            model_name="m",
            inference_time=0.01,
            probabilities=(0.2, 0.8),
        )
        assert pred.probabilities == (0.2, 0.8)


class TestFullPredictionFlowClassification:
    def test_predict_propagates_probabilities(self, config):
        mock_session_instance = Mock()
        mock_session_instance.run.return_value = [np.array([[0.85]])]
        mock_input = Mock()
        mock_input.name = "input"
        mock_session_instance.get_inputs.return_value = [mock_input]

        metadata = (
            '{"sequence_length": 120, "task_type": "binary_classification", '
            '"class_labels": [-1, 1]}'
        )
        with (
            patch("onnxruntime.InferenceSession", return_value=mock_session_instance),
            patch("builtins.open", mock_open(read_data=metadata)),
        ):
            runner = OnnxRunner("/tmp/test_model.onnx", config)
            features = np.random.rand(120, 5).astype(np.float32)
            prediction = runner.predict(features)

        assert isinstance(prediction, ModelPrediction)
        assert prediction.probabilities == pytest.approx((0.15, 0.85))
        assert prediction.direction == 1
        assert prediction.confidence == pytest.approx(0.85)

    def test_classification_predictions_bypass_cache(self, config):
        """No caching support for classification bundles yet (probabilities
        would be lost on a cache hit, since the cache only stores
        price/confidence/direction) -- every call must hit the session."""
        mock_session_instance = Mock()
        mock_session_instance.run.return_value = [np.array([[0.85]])]
        mock_input = Mock()
        mock_input.name = "input"
        mock_session_instance.get_inputs.return_value = [mock_input]

        metadata = (
            '{"sequence_length": 120, "task_type": "binary_classification", '
            '"class_labels": [-1, 1]}'
        )
        cache_manager = Mock()
        config.prediction_cache_enabled = True
        with (
            patch("onnxruntime.InferenceSession", return_value=mock_session_instance),
            patch("builtins.open", mock_open(read_data=metadata)),
        ):
            runner = OnnxRunner("/tmp/test_model.onnx", config, cache_manager=cache_manager)
            features = np.random.rand(120, 5).astype(np.float32)
            runner.predict(features)
            runner.predict(features)

        cache_manager.get.assert_not_called()
        cache_manager.set.assert_not_called()
        assert mock_session_instance.run.call_count == 2
