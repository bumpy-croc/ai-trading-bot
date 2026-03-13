"""Tests for Temporal Fusion Transformer (TFT) model architecture."""

import numpy as np
import pytest

try:
    import tensorflow as tf

    _TENSORFLOW_AVAILABLE = True
except ImportError:
    _TENSORFLOW_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _TENSORFLOW_AVAILABLE,
    reason="TensorFlow not available",
)


@pytest.fixture
def input_shape():
    """Standard input shape for tests."""
    return (30, 10)


@pytest.fixture
def sample_input(input_shape):
    """Sample input tensor for forward pass tests."""
    batch_size = 4
    return np.random.randn(batch_size, *input_shape).astype(np.float32)


class TestGatedResidualNetwork:
    """Tests for the GRN layer."""

    def test_output_shape_same_dims(self):
        """GRN preserves shape when input and output dimensions match."""
        from src.ml.training_pipeline.models_tft import GatedResidualNetwork

        grn = GatedResidualNetwork(hidden_size=32, output_size=10)
        x = tf.random.normal((4, 30, 10))
        output = grn(x)
        assert output.shape == (4, 30, 10)

    def test_output_shape_different_dims(self):
        """GRN projects to output_size when different from input dim."""
        from src.ml.training_pipeline.models_tft import GatedResidualNetwork

        grn = GatedResidualNetwork(hidden_size=32, output_size=64)
        x = tf.random.normal((4, 30, 10))
        output = grn(x)
        assert output.shape == (4, 30, 64)

    def test_output_shape_2d_input(self):
        """GRN works with 2D input (batch, features)."""
        from src.ml.training_pipeline.models_tft import GatedResidualNetwork

        grn = GatedResidualNetwork(hidden_size=16, output_size=8)
        x = tf.random.normal((4, 10))
        output = grn(x)
        assert output.shape == (4, 8)

    def test_serialization(self):
        """GRN config can be serialized and deserialized."""
        from src.ml.training_pipeline.models_tft import GatedResidualNetwork

        grn = GatedResidualNetwork(hidden_size=32, output_size=16, dropout=0.2)
        config = grn.get_config()
        assert config["hidden_size"] == 32
        assert config["output_size"] == 16
        assert config["dropout"] == 0.2


class TestVariableSelectionNetwork:
    """Tests for the VSN layer."""

    def test_output_shape(self):
        """VSN produces correct output shape."""
        from src.ml.training_pipeline.models_tft import VariableSelectionNetwork

        vsn = VariableSelectionNetwork(n_features=10, hidden_size=32)
        x = tf.random.normal((4, 30, 10))
        output = vsn(x)
        assert output.shape == (4, 30, 32)

    def test_feature_weights_sum_to_one(self):
        """VSN selection weights sum to 1 across features (softmax)."""
        from src.ml.training_pipeline.models_tft import VariableSelectionNetwork

        vsn = VariableSelectionNetwork(n_features=10, hidden_size=32)
        x = tf.random.normal((4, 30, 10))
        _, weights = vsn(x, return_weights=True)

        # Weights shape: (batch, timesteps, n_features)
        assert weights.shape == (4, 30, 10)

        # Weights should sum to 1 across features dimension
        weight_sums = tf.reduce_sum(weights, axis=-1)
        np.testing.assert_allclose(weight_sums.numpy(), 1.0, atol=1e-5)

    def test_weights_are_positive(self):
        """VSN weights are non-negative (softmax output)."""
        from src.ml.training_pipeline.models_tft import VariableSelectionNetwork

        vsn = VariableSelectionNetwork(n_features=5, hidden_size=16)
        x = tf.random.normal((2, 10, 5))
        _, weights = vsn(x, return_weights=True)
        assert tf.reduce_all(weights >= 0).numpy()

    def test_serialization(self):
        """VSN config can be serialized."""
        from src.ml.training_pipeline.models_tft import VariableSelectionNetwork

        vsn = VariableSelectionNetwork(n_features=10, hidden_size=32, dropout=0.15)
        config = vsn.get_config()
        assert config["n_features"] == 10
        assert config["hidden_size"] == 32
        assert config["dropout"] == 0.15


class TestTemporalFusionDecoder:
    """Tests for the attention decoder layer."""

    def test_output_shape(self):
        """Decoder preserves input shape."""
        from src.ml.training_pipeline.models_tft import TemporalFusionDecoder

        decoder = TemporalFusionDecoder(n_heads=4, hidden_size=64)
        x = tf.random.normal((4, 30, 64))
        output = decoder(x)
        assert output.shape == (4, 30, 64)

    def test_attention_scores_shape(self):
        """Decoder returns attention scores with correct shape."""
        from src.ml.training_pipeline.models_tft import TemporalFusionDecoder

        decoder = TemporalFusionDecoder(n_heads=4, hidden_size=64)
        x = tf.random.normal((4, 30, 64))
        output, scores = decoder(x, return_attention_scores=True)
        assert output.shape == (4, 30, 64)
        # Attention scores: (batch, n_heads, timesteps, timesteps)
        assert scores.shape == (4, 4, 30, 30)

    def test_serialization(self):
        """Decoder config can be serialized."""
        from src.ml.training_pipeline.models_tft import TemporalFusionDecoder

        decoder = TemporalFusionDecoder(n_heads=8, hidden_size=128, dropout=0.2)
        config = decoder.get_config()
        assert config["n_heads"] == 8
        assert config["hidden_size"] == 128
        assert config["dropout"] == 0.2


class TestCreateTFTModel:
    """Tests for the full TFT model factory function."""

    def test_default_params(self, input_shape):
        """Model creates successfully with default parameters."""
        from src.ml.training_pipeline.models_tft import create_tft_model

        model = create_tft_model(input_shape)
        assert model is not None
        assert model.name == "tft"

    def test_custom_params(self):
        """Model creates successfully with custom parameters."""
        from src.ml.training_pipeline.models_tft import create_tft_model

        model = create_tft_model(
            input_shape=(20, 8),
            n_heads=2,
            hidden_size=32,
            dropout=0.2,
            num_lstm_layers=2,
        )
        assert model is not None

    def test_forward_pass_output_shape(self, input_shape, sample_input):
        """Model forward pass produces correct output shape."""
        from src.ml.training_pipeline.models_tft import create_tft_model

        model = create_tft_model(input_shape)
        output = model.predict(sample_input, verbose=0)

        # Output: (batch_size, 1) - directional prediction
        assert output.shape == (4, 1)

    def test_output_in_sigmoid_range(self, input_shape, sample_input):
        """Model output is in [0, 1] range (sigmoid activation)."""
        from src.ml.training_pipeline.models_tft import create_tft_model

        model = create_tft_model(input_shape)
        output = model.predict(sample_input, verbose=0)

        assert np.all(output >= 0.0)
        assert np.all(output <= 1.0)

    def test_invalid_n_features(self):
        """Raise ValueError for non-positive n_features."""
        from src.ml.training_pipeline.models_tft import create_tft_model

        with pytest.raises(ValueError, match="n_features must be positive"):
            create_tft_model((30, 0))

    def test_invalid_n_heads(self, input_shape):
        """Raise ValueError for non-positive n_heads."""
        from src.ml.training_pipeline.models_tft import create_tft_model

        with pytest.raises(ValueError, match="n_heads must be positive"):
            create_tft_model(input_shape, n_heads=0)

    def test_invalid_hidden_size(self, input_shape):
        """Raise ValueError for non-positive hidden_size."""
        from src.ml.training_pipeline.models_tft import create_tft_model

        with pytest.raises(ValueError, match="hidden_size must be positive"):
            create_tft_model(input_shape, hidden_size=0)

    def test_hidden_size_not_divisible_by_heads(self, input_shape):
        """Raise ValueError when hidden_size not divisible by n_heads."""
        from src.ml.training_pipeline.models_tft import create_tft_model

        with pytest.raises(ValueError, match="divisible by n_heads"):
            create_tft_model(input_shape, n_heads=3, hidden_size=64)

    def test_invalid_dropout(self, input_shape):
        """Raise ValueError for dropout outside [0, 1)."""
        from src.ml.training_pipeline.models_tft import create_tft_model

        with pytest.raises(ValueError, match="dropout"):
            create_tft_model(input_shape, dropout=1.0)

    def test_invalid_lstm_layers(self, input_shape):
        """Raise ValueError for non-positive num_lstm_layers."""
        from src.ml.training_pipeline.models_tft import create_tft_model

        with pytest.raises(ValueError, match="num_lstm_layers must be positive"):
            create_tft_model(input_shape, num_lstm_layers=0)

    def test_model_is_compiled(self, input_shape):
        """Model is compiled with optimizer and loss."""
        from src.ml.training_pipeline.models_tft import create_tft_model

        model = create_tft_model(input_shape)
        assert model.optimizer is not None
        assert model.loss is not None


class TestModelFactoryIntegration:
    """Test TFT integration with the model factory."""

    def test_create_model_tft(self, input_shape):
        """TFT is accessible through create_model factory."""
        from src.ml.training_pipeline.models import create_model

        model = create_model("tft", input_shape)
        assert model is not None
        assert model.name == "tft"

    def test_create_model_tft_case_insensitive(self, input_shape):
        """Factory handles case-insensitive model type."""
        from src.ml.training_pipeline.models import create_model

        model = create_model("TFT", input_shape)
        assert model is not None

    def test_available_models_includes_tft(self):
        """TFT is listed in AVAILABLE_MODELS registry."""
        from src.ml.training_pipeline.models import AVAILABLE_MODELS

        assert "tft" in AVAILABLE_MODELS

    def test_get_model_callbacks_tft(self):
        """TFT callbacks are returned by get_model_callbacks."""
        from src.ml.training_pipeline.models import get_model_callbacks

        cbs = get_model_callbacks("tft", patience=15)
        assert len(cbs) == 2  # EarlyStopping + ReduceLROnPlateau


class TestTFTCallbacks:
    """Tests for TFT-specific callbacks."""

    def test_tft_callbacks_default(self):
        """Default TFT callbacks create correctly."""
        from src.ml.training_pipeline.models_tft import tft_callbacks

        cbs = tft_callbacks()
        assert len(cbs) == 2

    def test_tft_callbacks_custom_patience(self):
        """TFT callbacks respect custom patience."""
        from src.ml.training_pipeline.models_tft import tft_callbacks

        cbs = tft_callbacks(patience=30)
        # EarlyStopping should have patience=30
        early_stopping = cbs[0]
        assert early_stopping.patience == 30
