"""Unit tests for the hand-rolled sklearn LogisticRegression -> ONNX exporter.

entrant (a) (meta-labeling)'s working classifier is a plain sklearn
LogisticRegression (lightgbm/onnxmltools/skl2onnx are not installed or
declared dependencies in this repo -- see Phase 2b item 3's plan). This
exporter builds the minimal ONNX equivalent (MatMul + Add + Sigmoid) by
hand via the `onnx` package directly, which IS a declared dependency.

Round-trip numerical correctness is verified via onnx.reference.
ReferenceEvaluator (part of the `onnx` package itself, pure Python) rather
than onnxruntime.InferenceSession -- tests/conftest.py stubs the entire
onnxruntime module for the whole suite (always returns a fixed constant),
so it cannot verify real inference. ReferenceEvaluator sidesteps that stub
entirely and gives a genuine correctness check.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from src.ml.training_pipeline.sklearn_onnx_export import export_logistic_regression_to_onnx

pytestmark = pytest.mark.fast


def _fit_toy_model(n_features: int = 4) -> LogisticRegression:
    rng = np.random.default_rng(42)
    X = rng.normal(size=(200, n_features))
    # A separable-ish rule so the fitted model has real, non-degenerate weights.
    y = (X[:, 0] + 0.5 * X[:, 1] - 0.3 * X[:, 2] > 0).astype(int)
    model = LogisticRegression()
    model.fit(X, y)
    return model


class TestExportLogisticRegressionToOnnx:
    def test_writes_a_valid_onnx_file(self, tmp_path: Path):
        import onnx

        model = _fit_toy_model(n_features=4)
        output_path = tmp_path / "model.onnx"

        export_logistic_regression_to_onnx(model, n_features=4, output_path=output_path)

        assert output_path.exists()
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)

    def test_output_matches_sklearn_predict_proba(self, tmp_path: Path):
        """Round-trip: ONNX graph's sigmoid(Wx+b) must equal sklearn's own
        predict_proba(...)[:, 1] (P(class=1)) for the same input, within
        float32 tolerance."""
        from onnx.reference import ReferenceEvaluator

        n_features = 5
        model = _fit_toy_model(n_features=n_features)
        output_path = tmp_path / "model.onnx"
        export_logistic_regression_to_onnx(model, n_features=n_features, output_path=output_path)

        rng = np.random.default_rng(7)
        # OnnxRunner._prepare_input reshapes a flat (n_features,) array to
        # (1, 1, n_features) -- feed exactly that shape here.
        sample = rng.normal(size=(1, 1, n_features)).astype(np.float32)

        evaluator = ReferenceEvaluator(str(output_path))
        (onnx_output,) = evaluator.run(None, {"input": sample})

        expected = model.predict_proba(sample.reshape(1, n_features))[:, 1]

        np.testing.assert_allclose(onnx_output.flatten(), expected, atol=1e-5)

    def test_output_is_a_single_scalar_matching_binary_classification_contract(
        self, tmp_path: Path
    ):
        """OnnxRunner._process_classification_output's binary_classification
        branch requires flattened.size == 1 -- a 2-column probability
        vector (sklearn's predict_proba shape) would violate that contract."""
        from onnx.reference import ReferenceEvaluator

        n_features = 3
        model = _fit_toy_model(n_features=n_features)
        output_path = tmp_path / "model.onnx"
        export_logistic_regression_to_onnx(model, n_features=n_features, output_path=output_path)

        sample = np.random.default_rng(1).normal(size=(1, 1, n_features)).astype(np.float32)
        evaluator = ReferenceEvaluator(str(output_path))
        (onnx_output,) = evaluator.run(None, {"input": sample})

        assert np.asarray(onnx_output).flatten().size == 1

    def test_output_bounded_in_unit_interval(self, tmp_path: Path):
        from onnx.reference import ReferenceEvaluator

        n_features = 4
        model = _fit_toy_model(n_features=n_features)
        output_path = tmp_path / "model.onnx"
        export_logistic_regression_to_onnx(model, n_features=n_features, output_path=output_path)

        rng = np.random.default_rng(3)
        evaluator = ReferenceEvaluator(str(output_path))
        for _ in range(10):
            sample = rng.normal(size=(1, 1, n_features)).astype(np.float32) * 10
            (onnx_output,) = evaluator.run(None, {"input": sample})
            value = float(np.asarray(onnx_output).flatten()[0])
            assert 0.0 <= value <= 1.0

    def test_rejects_feature_count_mismatch(self, tmp_path: Path):
        model = _fit_toy_model(n_features=4)
        output_path = tmp_path / "model.onnx"

        with pytest.raises(ValueError, match="coef_"):
            export_logistic_regression_to_onnx(model, n_features=5, output_path=output_path)

    def test_rejects_multiclass_model(self, tmp_path: Path):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(60, 3))
        y = rng.integers(0, 3, size=60)  # 3 classes -- not binary
        model = LogisticRegression()
        model.fit(X, y)
        output_path = tmp_path / "model.onnx"

        with pytest.raises(ValueError, match="binary"):
            export_logistic_regression_to_onnx(model, n_features=3, output_path=output_path)
