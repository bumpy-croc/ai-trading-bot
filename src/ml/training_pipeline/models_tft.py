"""Temporal Fusion Transformer (TFT) model for cryptocurrency price prediction.

This module implements a simplified TFT architecture using TensorFlow/Keras,
providing interpretable predictions via variable selection and attention weights.

Architecture:
    - Gated Residual Networks (GRN) for information flow control
    - Variable Selection Networks (VSN) for feature importance
    - LSTM encoder for temporal processing
    - Multi-head self-attention decoder for long-range dependencies
    - Sigmoid output for directional prediction (up/down)

Performance expectations:
    - Interpretable feature importance via learned selection weights
    - Long-range temporal dependencies via self-attention
    - Robust gradient flow via gated residual connections

References:
    - "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series
      Forecasting" (Lim et al., 2021)
    - Gated mechanisms inspired by GRU/Highway Networks
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

try:
    import tensorflow as tf
    from tensorflow.keras import Model, callbacks, layers

    _TENSORFLOW_AVAILABLE = True
except ImportError:
    _TENSORFLOW_AVAILABLE = False
    tf = None  # type: ignore
    Model = None  # type: ignore
    callbacks = None  # type: ignore
    layers = None  # type: ignore

if TYPE_CHECKING:
    from tensorflow.keras import callbacks as CallbacksType
    from tensorflow.keras.models import Model as ModelType
else:
    ModelType = Any  # type: ignore
    CallbacksType = Any  # type: ignore


def _ensure_tensorflow_available() -> None:
    """Ensure tensorflow is available, raising ImportError with helpful message if not."""
    if not _TENSORFLOW_AVAILABLE:
        raise ImportError(
            "tensorflow is required for model training but is not installed. "
            "Install it with: pip install tensorflow"
        )


class GatedResidualNetwork(layers.Layer):
    """Gated Residual Network for controlled information flow.

    Applies a gated transformation with residual connection and layer
    normalization. The GLU gate learns which information to pass through,
    allowing the network to suppress irrelevant inputs.

    Architecture:
        Input -> Dense -> ELU -> Dense -> Dropout -> GLU gate -> Add residual -> LayerNorm

    Args:
        hidden_size: Dimension of the hidden layers
        output_size: Dimension of the output (defaults to hidden_size)
        dropout: Dropout rate for regularization
    """

    def __init__(
        self,
        hidden_size: int,
        output_size: int | None = None,
        dropout: float = 0.1,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.output_size = output_size or hidden_size
        self.dropout_rate = dropout

        # Layers initialized in build()
        self.dense_hidden = None
        self.dense_gate_linear = None
        self.dense_gate_sigmoid = None
        self.dense_skip = None
        self.dropout_layer = None
        self.layer_norm = None
        self.multiply = None
        self.add = None

    def build(self, input_shape: Any) -> None:
        """Build GRN layers based on input shape."""
        input_dim = input_shape[-1]

        # Hidden transformation
        self.dense_hidden = layers.Dense(self.hidden_size, activation="elu", name="grn_hidden")

        # GLU gate: split into linear and sigmoid paths
        self.dense_gate_linear = layers.Dense(self.output_size, name="grn_gate_linear")
        self.dense_gate_sigmoid = layers.Dense(
            self.output_size, activation="sigmoid", name="grn_gate_sigmoid"
        )

        self.dropout_layer = layers.Dropout(self.dropout_rate)

        # Skip connection projection if input and output dimensions differ
        if input_dim != self.output_size:
            self.dense_skip = layers.Dense(self.output_size, name="grn_skip")
        else:
            self.dense_skip = None

        self.layer_norm = layers.LayerNormalization(name="grn_layer_norm")
        self.multiply = layers.Multiply(name="grn_glu_gate")
        self.add = layers.Add(name="grn_residual_add")

        super().build(input_shape)

    def call(self, inputs: Any, training: bool | None = None) -> Any:
        """Apply gated residual transformation.

        Args:
            inputs: Input tensor of arbitrary shape with last dim matching input_dim
            training: Whether in training mode (for dropout)

        Returns:
            Transformed tensor with same shape except last dim is output_size
        """
        # Hidden transformation: Dense -> ELU
        hidden = self.dense_hidden(inputs)
        hidden = self.dropout_layer(hidden, training=training)

        # GLU gate: element-wise product of linear and sigmoid paths
        gate_linear = self.dense_gate_linear(hidden)
        gate_sigmoid = self.dense_gate_sigmoid(hidden)
        gated = self.multiply([gate_linear, gate_sigmoid])

        # Residual connection
        if self.dense_skip is not None:
            residual = self.dense_skip(inputs)
        else:
            residual = inputs

        output = self.add([gated, residual])
        output = self.layer_norm(output)

        return output

    def get_config(self) -> dict[str, Any]:
        """Get layer configuration for serialization."""
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "output_size": self.output_size,
                "dropout": self.dropout_rate,
            }
        )
        return config


class VariableSelectionNetwork(layers.Layer):
    """Variable Selection Network for learning feature importance.

    Applies independent GRNs to each input feature, then uses a softmax-weighted
    combination to select the most important features. Provides interpretable
    feature weights.

    Args:
        n_features: Number of input features
        hidden_size: Hidden dimension for GRN transformations
        dropout: Dropout rate for regularization
    """

    def __init__(
        self,
        n_features: int,
        hidden_size: int,
        dropout: float = 0.1,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.dropout_rate = dropout

        # Layers initialized in build()
        self.feature_grns: list[Any] = []
        self.selection_grn = None
        self.selection_dense = None

    def build(self, input_shape: Any) -> None:
        """Build VSN layers."""
        # Per-feature GRN transformations
        self.feature_grns = [
            GatedResidualNetwork(
                hidden_size=self.hidden_size,
                output_size=self.hidden_size,
                dropout=self.dropout_rate,
                name=f"feature_grn_{i}",
            )
            for i in range(self.n_features)
        ]

        # Selection weights GRN: processes flattened features to produce weights
        self.selection_grn = GatedResidualNetwork(
            hidden_size=self.hidden_size,
            output_size=self.hidden_size,
            dropout=self.dropout_rate,
            name="selection_grn",
        )
        self.selection_dense = layers.Dense(
            self.n_features, activation="softmax", name="selection_weights"
        )

        super().build(input_shape)

    def call(
        self,
        inputs: Any,
        training: bool | None = None,
        return_weights: bool = False,
    ) -> Any:
        """Apply variable selection to input features.

        Args:
            inputs: Tensor of shape (batch, timesteps, n_features)
            training: Whether in training mode
            return_weights: Whether to return selection weights

        Returns:
            Selected features tensor of shape (batch, timesteps, hidden_size).
            If return_weights=True, returns tuple of (output, weights).
        """
        # Transform each feature independently through its own GRN
        # Each feature is a scalar per timestep, expanded to hidden_size
        transformed = []
        for i in range(self.n_features):
            # Extract single feature: (batch, timesteps, 1)
            feature_i = inputs[:, :, i : i + 1]
            # Transform through GRN: (batch, timesteps, hidden_size)
            transformed_i = self.feature_grns[i](feature_i, training=training)
            transformed.append(transformed_i)

        # Stack transformed features: (batch, timesteps, n_features, hidden_size)
        stacked = tf.stack(transformed, axis=2)

        # Compute selection weights from flattened input
        # Use the raw input to determine which features matter
        selection_input = self.selection_grn(inputs, training=training)
        # Weights: (batch, timesteps, n_features)
        weights = self.selection_dense(selection_input)

        # Weighted combination: expand weights for broadcasting
        # weights: (batch, timesteps, n_features, 1)
        weights_expanded = tf.expand_dims(weights, axis=-1)
        # Weighted sum: (batch, timesteps, hidden_size)
        selected = tf.reduce_sum(stacked * weights_expanded, axis=2)

        if return_weights:
            return selected, weights
        return selected

    def get_config(self) -> dict[str, Any]:
        """Get layer configuration for serialization."""
        config = super().get_config()
        config.update(
            {
                "n_features": self.n_features,
                "hidden_size": self.hidden_size,
                "dropout": self.dropout_rate,
            }
        )
        return config


class TemporalFusionDecoder(layers.Layer):
    """Self-attention decoder with causal masking for temporal fusion.

    Applies multi-head self-attention over time steps with causal masking
    to prevent information leakage from future timesteps, followed by
    a GRN for post-attention processing.

    Args:
        n_heads: Number of attention heads
        hidden_size: Dimension for attention keys/queries/values
        dropout: Dropout rate for regularization
    """

    def __init__(
        self,
        n_heads: int = 4,
        hidden_size: int = 64,
        dropout: float = 0.1,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.n_heads = n_heads
        self.hidden_size = hidden_size
        self.dropout_rate = dropout

        # Layers initialized in build()
        self.attention = None
        self.attention_dropout = None
        self.attention_norm = None
        self.post_attention_grn = None
        self.residual_add = None

    def build(self, input_shape: Any) -> None:
        """Build decoder layers."""
        self.attention = layers.MultiHeadAttention(
            num_heads=self.n_heads,
            key_dim=self.hidden_size // self.n_heads,
            dropout=self.dropout_rate,
            name="temporal_attention",
        )
        self.attention_dropout = layers.Dropout(self.dropout_rate)
        self.attention_norm = layers.LayerNormalization(name="attention_norm")
        self.post_attention_grn = GatedResidualNetwork(
            hidden_size=self.hidden_size,
            dropout=self.dropout_rate,
            name="post_attention_grn",
        )
        self.residual_add = layers.Add(name="attention_residual_add")

        super().build(input_shape)

    def call(
        self,
        inputs: Any,
        training: bool | None = None,
        return_attention_scores: bool = False,
    ) -> Any:
        """Apply temporal self-attention with causal masking.

        Args:
            inputs: Tensor of shape (batch, timesteps, hidden_size)
            training: Whether in training mode
            return_attention_scores: Whether to return attention weights

        Returns:
            Processed tensor. If return_attention_scores=True, returns tuple.
        """
        # Self-attention with causal mask (prevents future peeking)
        attended, attention_scores = self.attention(
            query=inputs,
            value=inputs,
            key=inputs,
            use_causal_mask=True,
            return_attention_scores=True,
            training=training,
        )

        attended = self.attention_dropout(attended, training=training)

        # Add & Norm residual connection
        x = self.residual_add([inputs, attended])
        x = self.attention_norm(x)

        # Post-attention GRN for additional processing
        x = self.post_attention_grn(x, training=training)

        if return_attention_scores:
            return x, attention_scores
        return x

    def get_config(self) -> dict[str, Any]:
        """Get layer configuration for serialization."""
        config = super().get_config()
        config.update(
            {
                "n_heads": self.n_heads,
                "hidden_size": self.hidden_size,
                "dropout": self.dropout_rate,
            }
        )
        return config


def create_tft_model(
    input_shape: tuple[int, int],
    n_heads: int = 4,
    hidden_size: int = 64,
    dropout: float = 0.1,
    num_lstm_layers: int = 1,
    learning_rate: float = 1e-3,
) -> Any:
    """Create a Temporal Fusion Transformer model for directional prediction.

    Architecture:
        1. Variable Selection Network - learns feature importance
        2. LSTM encoder - captures temporal patterns
        3. Multi-head self-attention decoder - models long-range dependencies
        4. Dense output with sigmoid - predicts direction (up/down)

    Args:
        input_shape: Shape of input sequences (sequence_length, n_features)
        n_heads: Number of attention heads in the decoder
        hidden_size: Hidden dimension for GRN and attention layers
        dropout: Dropout rate for regularization (0.0-0.5)
        num_lstm_layers: Number of stacked LSTM layers (1-3)
        learning_rate: Adam optimizer learning rate

    Returns:
        Compiled Keras model

    Raises:
        ValueError: If input parameters are invalid

    Example:
        >>> model = create_tft_model((60, 15), n_heads=4, hidden_size=64)
        >>> model.fit(train_ds, validation_data=val_ds, epochs=50)
    """
    _ensure_tensorflow_available()

    sequence_length, n_features = input_shape

    # Validate parameters
    if sequence_length <= 0:
        raise ValueError(f"sequence_length must be positive, got {sequence_length}")
    if n_features <= 0:
        raise ValueError(f"n_features must be positive, got {n_features}")
    if n_heads <= 0:
        raise ValueError(f"n_heads must be positive, got {n_heads}")
    if hidden_size <= 0:
        raise ValueError(f"hidden_size must be positive, got {hidden_size}")
    if hidden_size % n_heads != 0:
        raise ValueError(f"hidden_size ({hidden_size}) must be divisible by n_heads ({n_heads})")
    if not 0.0 <= dropout < 1.0:
        raise ValueError(f"dropout must be in [0.0, 1.0), got {dropout}")
    if num_lstm_layers <= 0:
        raise ValueError(f"num_lstm_layers must be positive, got {num_lstm_layers}")
    if learning_rate <= 0:
        raise ValueError(f"learning_rate must be positive, got {learning_rate}")

    inputs = layers.Input(shape=input_shape, name="sequence_input")

    # 1. Variable Selection Network - select important features
    x = VariableSelectionNetwork(
        n_features=n_features,
        hidden_size=hidden_size,
        dropout=dropout,
        name="variable_selection",
    )(inputs)

    # 2. LSTM encoder for temporal processing
    for i in range(num_lstm_layers):
        x = layers.LSTM(
            hidden_size,
            return_sequences=True,
            dropout=dropout,
            name=f"lstm_encoder_{i + 1}",
        )(x)

    # 3. Multi-head self-attention decoder
    x = TemporalFusionDecoder(
        n_heads=n_heads,
        hidden_size=hidden_size,
        dropout=dropout,
        name="temporal_decoder",
    )(x)

    # 4. Global pooling and output
    avg_pool = layers.GlobalAveragePooling1D(name="global_avg_pool")(x)
    max_pool = layers.GlobalMaxPooling1D(name="global_max_pool")(x)
    pooled = layers.Concatenate(name="pooling_concat")([avg_pool, max_pool])

    x = layers.Dense(hidden_size, activation="relu", name="dense_1")(pooled)
    x = layers.Dropout(dropout, name="dropout_dense")(x)

    # Sigmoid for binary directional prediction (up/down)
    outputs = layers.Dense(1, activation="sigmoid", name="direction_prediction")(x)

    model = Model(inputs=inputs, outputs=outputs, name="tft")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )

    return model


def tft_callbacks(patience: int = 20, min_delta: float = 1e-4) -> list[Any]:
    """Get optimized callbacks for TFT training.

    TFT benefits from longer patience due to complex architecture.

    Args:
        patience: Epochs to wait for improvement before early stopping
        min_delta: Minimum change to qualify as improvement

    Returns:
        List of Keras callbacks
    """
    _ensure_tensorflow_available()

    return [
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            min_delta=min_delta,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=max(patience // 3, 5),
            min_lr=1e-6,
            verbose=1,
        ),
    ]


# Recommended hyperparameters for cryptocurrency directional prediction
RECOMMENDED_HYPERPARAMETERS = {
    "default": {
        "n_heads": 4,
        "hidden_size": 64,
        "dropout": 0.1,
        "num_lstm_layers": 1,
        "learning_rate": 1e-3,
        "batch_size": 32,
        "epochs": 100,
        "patience": 20,
    },
    "lightweight": {
        "n_heads": 2,
        "hidden_size": 32,
        "dropout": 0.1,
        "num_lstm_layers": 1,
        "learning_rate": 1e-3,
        "batch_size": 64,
        "epochs": 80,
        "patience": 15,
    },
    "deep": {
        "n_heads": 8,
        "hidden_size": 128,
        "dropout": 0.2,
        "num_lstm_layers": 2,
        "learning_rate": 1e-3,
        "batch_size": 16,
        "epochs": 150,
        "patience": 25,
    },
}
