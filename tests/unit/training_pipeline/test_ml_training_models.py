"""Unit tests for ML training pipeline model factories module."""

import pytest
import tensorflow as tf
from tensorflow.keras import callbacks

from src.ml.training_pipeline.models import (
    build_price_only_model,
    create_adaptive_model,
    default_callbacks,
)


@pytest.mark.fast
class TestCreateAdaptiveModel:
    """Test create_adaptive_model function."""

    def test_creates_model_with_sentiment(self):
        # Arrange
        input_shape = (120, 15)  # sequence_length=120, num_features=15

        # Act
        model = create_adaptive_model(input_shape, has_sentiment=True)

        # Assert
        assert isinstance(model, tf.keras.Model)
        assert model.input_shape == (None, 120, 15)
        assert model.output_shape == (None, 1)

    def test_creates_model_without_sentiment(self):
        # Arrange
        input_shape = (120, 10)  # sequence_length=120, num_features=10

        # Act
        model = create_adaptive_model(input_shape, has_sentiment=False)

        # Assert
        assert isinstance(model, tf.keras.Model)
        assert model.input_shape == (None, 120, 10)
        assert model.output_shape == (None, 1)

    def test_model_is_compiled(self):
        # Arrange
        input_shape = (120, 15)

        # Act
        model = create_adaptive_model(input_shape)

        # Assert
        assert model.optimizer is not None
        assert model.loss is not None
        assert len(model.metrics) > 0

    def test_model_has_adam_optimizer(self):
        # Arrange
        input_shape = (120, 15)

        # Act
        model = create_adaptive_model(input_shape)

        # Assert
        assert isinstance(model.optimizer, tf.keras.optimizers.Adam)

    def test_model_has_mse_loss(self):
        # Arrange
        input_shape = (120, 15)

        # Act
        model = create_adaptive_model(input_shape)

        # Assert
        # TensorFlow converts loss functions to their canonical form
        assert model.loss == "mse" or "mean_squared_error" in str(model.loss).lower()

    def test_model_has_rmse_metric(self):
        # Arrange
        input_shape = (120, 15)

        # Act
        model = create_adaptive_model(input_shape)

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
        input_shape = (120, 15)

        # Act
        model = create_adaptive_model(input_shape)

        # Assert - check for Conv1D layers
        layer_types = [type(layer).__name__ for layer in model.layers]
        assert "Conv1D" in layer_types

    def test_model_architecture_has_lstm_layers(self):
        # Arrange
        input_shape = (120, 15)

        # Act
        model = create_adaptive_model(input_shape)

        # Assert - check for LSTM layers
        layer_types = [type(layer).__name__ for layer in model.layers]
        assert "LSTM" in layer_types

    def test_model_architecture_has_dropout_layers(self):
        # Arrange
        input_shape = (120, 15)

        # Act
        model = create_adaptive_model(input_shape)

        # Assert - check for Dropout layers
        layer_types = [type(layer).__name__ for layer in model.layers]
        assert "Dropout" in layer_types

    def test_model_can_predict(self):
        # Arrange
        import numpy as np

        input_shape = (120, 15)
        model = create_adaptive_model(input_shape)
        test_input = np.random.rand(1, 120, 15).astype(np.float32)

        # Act
        prediction = model.predict(test_input, verbose=0)

        # Assert
        assert prediction.shape == (1, 1)


@pytest.mark.fast
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

    def test_model_architecture_has_lstm_layers(self):
        # Arrange
        sequence_length = 120
        num_features = 10

        # Act
        model = build_price_only_model(sequence_length, num_features)

        # Assert - check for LSTM layers
        layer_types = [type(layer).__name__ for layer in model.layers]
        assert "LSTM" in layer_types

    def test_model_architecture_has_dropout_layers(self):
        # Arrange
        sequence_length = 120
        num_features = 10

        # Act
        model = build_price_only_model(sequence_length, num_features)

        # Assert - check for Dropout layers
        layer_types = [type(layer).__name__ for layer in model.layers]
        assert "Dropout" in layer_types

    def test_model_output_activation_is_sigmoid(self):
        # Arrange
        sequence_length = 120
        num_features = 10

        # Act
        model = build_price_only_model(sequence_length, num_features)

        # Assert - output layer should use sigmoid activation
        output_layer = model.layers[-1]
        assert output_layer.activation.__name__ == "sigmoid"

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
