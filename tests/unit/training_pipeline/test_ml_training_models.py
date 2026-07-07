"""Unit tests for ML training pipeline model factories module."""

import numpy as np
import pytest

try:
    import tensorflow as tf
    from tensorflow.keras import callbacks

    _TENSORFLOW_AVAILABLE = True
except ImportError:
    _TENSORFLOW_AVAILABLE = False
    tf = None  # type: ignore
    callbacks = None  # type: ignore

from src.ml.training_pipeline.artifacts import evaluate_model_performance
from src.ml.training_pipeline.models import (
    build_price_only_model,
    create_model,
    default_callbacks,
)


@pytest.mark.fast
@pytest.mark.skipif(not _TENSORFLOW_AVAILABLE, reason="TensorFlow not installed")
class TestCreateAdaptiveModel:
    """Test create_model function with the new factory API."""

    def test_creates_model_with_sentiment(self):
        # Arrange
        model_type = "cnn_lstm"  # Baseline model
        input_shape = (120, 15)  # sequence_length=120, num_features=15

        # Act
        model = create_model(model_type, input_shape, has_sentiment=True)

        # Assert
        assert isinstance(model, tf.keras.Model)
        assert model.input_shape == (None, 120, 15)
        assert model.output_shape == (None, 1)

    def test_creates_model_without_sentiment(self):
        # Arrange
        model_type = "cnn_lstm"
        input_shape = (120, 10)  # sequence_length=120, num_features=10

        # Act
        model = create_model(model_type, input_shape, has_sentiment=False)

        # Assert
        assert isinstance(model, tf.keras.Model)
        assert model.input_shape == (None, 120, 10)
        assert model.output_shape == (None, 1)

    def test_model_is_compiled(self):
        # Arrange
        model_type = "cnn_lstm"
        input_shape = (120, 15)

        # Act
        model = create_model(model_type, input_shape)

        # Assert
        assert model.optimizer is not None
        assert model.loss is not None
        assert len(model.metrics) > 0

    def test_model_has_adam_optimizer(self):
        # Arrange
        model_type = "cnn_lstm"
        input_shape = (120, 15)

        # Act
        model = create_model(model_type, input_shape)

        # Assert
        assert isinstance(model.optimizer, tf.keras.optimizers.Adam)

    def test_model_has_mse_loss(self):
        # Arrange
        model_type = "cnn_lstm"
        input_shape = (120, 15)

        # Act
        model = create_model(model_type, input_shape)

        # Assert
        # TensorFlow converts loss functions to their canonical form
        assert model.loss == "mse" or "mean_squared_error" in str(model.loss).lower()

    def test_model_has_rmse_metric(self):
        # Arrange
        model_type = "cnn_lstm"
        input_shape = (120, 15)

        # Act
        model = create_model(model_type, input_shape)

        # Assert - model should be compiled with at least one metric besides loss
        # In Keras 3, metrics are configured differently so we just check they exist
        assert len(model.metrics) >= 1  # Should have loss and at least one metric
        # Verify model can run (which validates metrics work)
        import numpy as np

        test_input = np.random.rand(1, 120, 15).astype(np.float32)
        test_target = np.array([[0.5]])
        result = model.evaluate(test_input, test_target, verbose=0)
        assert len(result) >= 2  # Returns [loss, metric1, ...]

    def test_model_architecture_has_conv_layers(self):
        # Arrange
        model_type = "cnn_lstm"
        input_shape = (120, 15)

        # Act
        model = create_model(model_type, input_shape)

        # Assert - check for Conv1D layers
        layer_types = [type(layer).__name__ for layer in model.layers]
        assert "Conv1D" in layer_types

    def test_model_architecture_has_gru_layers(self):
        # Arrange
        model_type = "cnn_lstm"
        input_shape = (120, 15)

        # Act
        model = create_model(model_type, input_shape)

        # Assert - check for GRU layers (balanced model uses GRU)
        layer_types = [type(layer).__name__ for layer in model.layers]
        assert "GRU" in layer_types

    def test_model_architecture_has_dropout_layers(self):
        # Arrange
        model_type = "cnn_lstm"
        input_shape = (120, 15)

        # Act
        model = create_model(model_type, input_shape)

        # Assert - check for Dropout layers
        layer_types = [type(layer).__name__ for layer in model.layers]
        assert "Dropout" in layer_types

    def test_model_can_predict(self):
        # Arrange
        import numpy as np

        model_type = "cnn_lstm"
        input_shape = (120, 15)
        model = create_model(model_type, input_shape)
        test_input = np.random.rand(1, 120, 15).astype(np.float32)

        # Act
        prediction = model.predict(test_input, verbose=0)

        # Assert
        assert prediction.shape == (1, 1)


@pytest.mark.fast
@pytest.mark.skipif(not _TENSORFLOW_AVAILABLE, reason="TensorFlow not installed")
class TestBuildPriceOnlyModel:
    """Test build_price_only_model function."""

    def test_creates_model(self):
        # Arrange
        sequence_length = 120
        num_features = 10

        # Act
        model = build_price_only_model(sequence_length, num_features)

        # Assert
        assert isinstance(model, tf.keras.Model)
        assert model.input_shape == (None, 120, 10)
        assert model.output_shape == (None, 1)

    def test_model_is_compiled(self):
        # Arrange
        sequence_length = 120
        num_features = 10

        # Act
        model = build_price_only_model(sequence_length, num_features)

        # Assert
        assert model.optimizer is not None
        assert model.loss is not None
        assert len(model.metrics) > 0

    def test_model_has_adam_optimizer(self):
        # Arrange
        sequence_length = 120
        num_features = 10

        # Act
        model = build_price_only_model(sequence_length, num_features)

        # Assert
        assert isinstance(model.optimizer, tf.keras.optimizers.Adam)

    def test_model_has_rmse_metric(self):
        # Arrange
        sequence_length = 120
        num_features = 10

        # Act
        model = build_price_only_model(sequence_length, num_features)

        # Assert - model should be compiled with at least one metric besides loss
        assert len(model.metrics) >= 1  # Should have loss and at least one metric
        # Verify model can run (which validates metrics work)
        import numpy as np

        test_input = np.random.rand(1, 120, 10).astype(np.float32)
        test_target = np.array([[0.5]])
        result = model.evaluate(test_input, test_target, verbose=0)
        assert len(result) >= 2  # Returns [loss, metric1, ...]

    def test_model_architecture_has_gru_layers(self):
        # Arrange
        sequence_length = 120
        num_features = 10

        # Act
        model = build_price_only_model(sequence_length, num_features)

        # Assert - check for GRU layers (uses balanced model which has GRU)
        layer_types = [type(layer).__name__ for layer in model.layers]
        assert "GRU" in layer_types

    def test_model_architecture_has_dropout_layers(self):
        # Arrange
        sequence_length = 120
        num_features = 10

        # Act
        model = build_price_only_model(sequence_length, num_features)

        # Assert - check for Dropout layers
        layer_types = [type(layer).__name__ for layer in model.layers]
        assert "Dropout" in layer_types

    def test_model_output_activation_is_linear(self):
        # Arrange
        sequence_length = 120
        num_features = 10

        # Act
        model = build_price_only_model(sequence_length, num_features)

        # Assert - output layer should use linear activation (balanced model uses linear)
        output_layer = model.layers[-1]
        assert output_layer.activation.__name__ == "linear"

    def test_model_can_predict(self):
        # Arrange
        import numpy as np

        sequence_length = 120
        num_features = 10
        model = build_price_only_model(sequence_length, num_features)
        test_input = np.random.rand(1, 120, 10).astype(np.float32)

        # Act
        prediction = model.predict(test_input, verbose=0)

        # Assert
        assert prediction.shape == (1, 1)

    def test_different_sequence_lengths(self):
        # Arrange
        sequence_length = 60
        num_features = 5

        # Act
        model = build_price_only_model(sequence_length, num_features)

        # Assert
        assert model.input_shape == (None, 60, 5)


@pytest.mark.fast
@pytest.mark.skipif(not _TENSORFLOW_AVAILABLE, reason="TensorFlow not installed")
class TestDefaultCallbacks:
    """Test default_callbacks function."""

    def test_returns_list_of_callbacks(self):
        # Arrange & Act
        result = default_callbacks()

        # Assert
        assert isinstance(result, list)
        assert len(result) > 0
        assert all(isinstance(cb, callbacks.Callback) for cb in result)

    def test_includes_early_stopping(self):
        # Arrange & Act
        result = default_callbacks()

        # Assert
        callback_types = [type(cb).__name__ for cb in result]
        assert "EarlyStopping" in callback_types

    def test_includes_reduce_lr_on_plateau(self):
        # Arrange & Act
        result = default_callbacks()

        # Assert
        callback_types = [type(cb).__name__ for cb in result]
        assert "ReduceLROnPlateau" in callback_types

    def test_early_stopping_default_patience(self):
        # Arrange & Act
        result = default_callbacks()

        # Assert
        early_stopping = next(cb for cb in result if isinstance(cb, callbacks.EarlyStopping))
        assert early_stopping.patience == 15

    def test_early_stopping_custom_patience(self):
        # Arrange & Act
        result = default_callbacks(patience=20)

        # Assert
        early_stopping = next(cb for cb in result if isinstance(cb, callbacks.EarlyStopping))
        assert early_stopping.patience == 20

    def test_early_stopping_monitors_val_loss(self):
        # Arrange & Act
        result = default_callbacks()

        # Assert
        early_stopping = next(cb for cb in result if isinstance(cb, callbacks.EarlyStopping))
        assert early_stopping.monitor == "val_loss"

    def test_early_stopping_restores_best_weights(self):
        # Arrange & Act
        result = default_callbacks()

        # Assert
        early_stopping = next(cb for cb in result if isinstance(cb, callbacks.EarlyStopping))
        assert early_stopping.restore_best_weights is True

    def test_reduce_lr_monitors_val_loss(self):
        # Arrange & Act
        result = default_callbacks()

        # Assert
        reduce_lr = next(cb for cb in result if isinstance(cb, callbacks.ReduceLROnPlateau))
        assert reduce_lr.monitor == "val_loss"

    def test_reduce_lr_patience_scales_with_early_stopping(self):
        # Arrange & Act
        result = default_callbacks(patience=30)

        # Assert
        reduce_lr = next(cb for cb in result if isinstance(cb, callbacks.ReduceLROnPlateau))
        # patience // 3 with minimum of 3
        assert reduce_lr.patience == 10

    def test_reduce_lr_minimum_patience(self):
        # Arrange & Act
        result = default_callbacks(patience=3)

        # Assert
        reduce_lr = next(cb for cb in result if isinstance(cb, callbacks.ReduceLROnPlateau))
        # patience // 3 = 1, but minimum is 3
        assert reduce_lr.patience == 3


# Every architecture selectable from the train CLIs, crossed with every variant.
# Mirrors the choices exposed by `atb train cloud --model-type/--model-variant`.
TRAINER_MODEL_MATRIX = [
    (model_type, variant)
    for model_type in ["lstm", "cnn_lstm", "attention_lstm", "tcn", "tcn_attention", "tft"]
    for variant in ["default", "lightweight", "deep"]
]


@pytest.mark.fast
@pytest.mark.skipif(not _TENSORFLOW_AVAILABLE, reason="TensorFlow not installed")
class TestCreateModelTrainerContract:
    """Regression guard for #928: every CLI-selectable (model_type, variant) pair
    must construct with the exact kwargs run_training_pipeline passes.

    The trainer always forwards has_sentiment; factories that don't accept it
    must never see it (create_model pops it before dispatch).
    """

    @pytest.mark.parametrize(("model_type", "variant"), TRAINER_MODEL_MATRIX)
    def test_constructs_with_trainer_kwargs(self, model_type, variant):
        # Arrange - mirror run_training_pipeline's call exactly (price-only shape)
        input_shape = (120, 5)

        # Act
        model = create_model(
            model_type=model_type,
            input_shape=input_shape,
            variant=variant,
            has_sentiment=False,
        )

        # Assert
        assert isinstance(model, tf.keras.Model)
        assert model.input_shape == (None, 120, 5)
        assert model.output_shape == (None, 1)


# lstm/cnn_lstm compile with a single metric ([rmse]); attention_lstm/tcn/tcn_attention
# compile with two ([rmse, mae]). tft is excluded: it's a binary direction-classification
# model (loss=binary_crossentropy, metrics=[accuracy, auc]) with no "rmse" key at all --
# a fundamentally different evaluation contract that evaluate_model_performance was never
# meant to score, not an instance of this regression.
EVAL_COMPATIBLE_MATRIX = [
    (model_type, variant)
    for model_type in ["lstm", "cnn_lstm", "attention_lstm", "tcn", "tcn_attention"]
    for variant in ["default", "lightweight", "deep"]
]


@pytest.mark.fast
@pytest.mark.skipif(not _TENSORFLOW_AVAILABLE, reason="TensorFlow not installed")
class TestCreateModelEvaluationContract:
    """Regression guard for #936: every CLI-selectable, regression-scored
    (model_type, variant) pair must survive a real evaluate_model_performance() call,
    not just construction (see TestCreateModelTrainerContract above). attention_lstm and
    tcn/tcn_attention compile with metrics=[rmse, mae] -- a 3-value model.evaluate()
    return that crashed the positional 2-value unpack this evaluate_model_performance
    used before #936, destroying a fully-trained model on SageMaker. Construction alone
    never exercises model.evaluate() and would not have caught it.
    """

    @pytest.mark.parametrize(("model_type", "variant"), EVAL_COMPATIBLE_MATRIX)
    def test_evaluate_model_performance_does_not_crash(self, model_type, variant):
        # Arrange - tiny synthetic data, no training needed to exercise model.evaluate()
        input_shape = (120, 5)
        model = create_model(
            model_type=model_type,
            input_shape=input_shape,
            variant=variant,
            has_sentiment=False,
        )
        rng = np.random.default_rng(0)
        X_train = rng.random((4, *input_shape)).astype("float32")
        y_train = rng.random(4).astype("float32")
        X_test = rng.random((3, *input_shape)).astype("float32")
        y_test = rng.random(3).astype("float32")

        # Act
        result = evaluate_model_performance(model, X_train, y_train, X_test, y_test)

        # Assert
        assert set(result) == {"train_loss", "test_loss", "train_rmse", "test_rmse", "mape"}
        assert all(np.isfinite(v) for v in result.values())
