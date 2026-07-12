"""Hand-rolled ONNX export for a fitted sklearn LogisticRegression.

entrant (a) (meta-labeling) of the TARGET-REDESIGN tournament trains a
sklearn LogisticRegression as its working classifier -- `lightgbm`,
`onnxmltools`, and `skl2onnx` are not installed or declared dependencies in
this repo (checked pyproject.toml/requirements*.txt), so this module builds
the ONNX graph directly via the `onnx` package (already a declared
dependency: `onnx>=1.14.0`) instead of pulling in a new dependency for a
single binary classifier.

The graph is the minimal equivalent of
``LogisticRegression.predict_proba(X)[:, 1]``: Reshape (squeeze the
sequence axis OnnxRunner._prepare_input always adds) -> MatMul -> Add ->
Sigmoid. Matches this repo's existing tf2onnx convention of opset 13
(artifacts.py's convert_to_onnx); ir_version is pinned to 9 for
onnxruntime 1.17.0 compatibility (onnx's own default IR_VERSION at the time
of writing is ahead of what onnxruntime 1.17.0 declares support for).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sklearn.linear_model import LogisticRegression

_OPSET_VERSION = 13
_IR_VERSION = 9


def export_logistic_regression_to_onnx(
    model: LogisticRegression,
    n_features: int,
    output_path: Path,
) -> None:
    """Export a fitted BINARY sklearn LogisticRegression to an ONNX file.

    Args:
        model: A fitted sklearn LogisticRegression. Must be binary
            (``model.coef_.shape == (1, n_features)``) -- multi-class
            (one-vs-rest/multinomial) models are not supported.
        n_features: Number of input features; must match
            ``model.coef_.shape[1]``.
        output_path: Destination ``.onnx`` file path.

    Raises:
        ValueError: ``model`` is not a fitted binary LogisticRegression, or
            ``n_features`` doesn't match ``model.coef_``.
    """
    import onnx
    from onnx import TensorProto, helper

    coef = np.asarray(model.coef_, dtype=np.float32)
    intercept = np.asarray(model.intercept_, dtype=np.float32)

    if coef.ndim != 2 or coef.shape[0] != 1:
        raise ValueError(
            f"export_logistic_regression_to_onnx only supports a binary "
            f"LogisticRegression (coef_.shape == (1, n_features)), got coef_ "
            f"shape {coef.shape} -- multi-class models are not supported."
        )
    if coef.shape[1] != n_features:
        raise ValueError(
            f"model.coef_ has {coef.shape[1]} features but n_features={n_features} "
            f"was requested -- these must match."
        )
    if intercept.shape != (1,):
        raise ValueError(f"Expected intercept_ shape (1,), got {intercept.shape}")

    # Weight matrix for MatMul: (n_features, 1) so a (batch, n_features)
    # input produces a (batch, 1) output.
    weight = np.ascontiguousarray(coef.T)

    # OnnxRunner._prepare_input always reshapes a flat feature vector to
    # (1, 1, n_features) (the (batch, sequence, features) convention every
    # other model in this registry uses) -- meta-label features are NOT
    # sequence-shaped, so the graph itself squeezes the size-1 sequence axis
    # back out via Reshape before the MatMul, rather than requiring a
    # special-cased caller.
    input_tensor = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, ["batch", "sequence", n_features]
    )
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, ["batch", 1])

    weight_initializer = helper.make_tensor(
        "weight", TensorProto.FLOAT, list(weight.shape), weight.flatten().tolist()
    )
    bias_initializer = helper.make_tensor(
        "bias", TensorProto.FLOAT, list(intercept.shape), intercept.tolist()
    )
    reshape_shape_initializer = helper.make_tensor(
        "reshape_shape", TensorProto.INT64, [2], [-1, n_features]
    )

    reshape_node = helper.make_node(
        "Reshape", inputs=["input", "reshape_shape"], outputs=["flattened"]
    )
    matmul_node = helper.make_node("MatMul", inputs=["flattened", "weight"], outputs=["logits"])
    add_node = helper.make_node("Add", inputs=["logits", "bias"], outputs=["biased_logits"])
    sigmoid_node = helper.make_node("Sigmoid", inputs=["biased_logits"], outputs=["output"])

    graph = helper.make_graph(
        nodes=[reshape_node, matmul_node, add_node, sigmoid_node],
        name="meta_label_logistic_regression",
        inputs=[input_tensor],
        outputs=[output_tensor],
        initializer=[weight_initializer, bias_initializer, reshape_shape_initializer],
    )

    onnx_model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", _OPSET_VERSION)],
        ir_version=_IR_VERSION,
    )
    onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model, str(output_path))


__all__ = ["export_logistic_regression_to_onnx"]
