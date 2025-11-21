"""Temporal Convolutional Network (TCN) for cryptocurrency price prediction.

This module implements TCN architecture with dilated causal convolutions,
offering faster training than LSTM while maintaining competitive accuracy.

Architecture:
    - Dilated causal 1D convolutions (no future leakage)
    - Residual connections for deep networks
    - Same input/output sequence length
    - Parallel training (much faster than LSTM)

Performance expectations:
    - Comparable or better accuracy than LSTM
    - 3-5x faster training (parallelizable)
    - Excellent for real-time inference (streaming capable)
    - Large receptive field via dilation

References:
    - "Temporal Convolutional Networks and Forecasting" (Unit8)
    - "Temporal Convolutional Attention Neural Networks" (IEEE)
    - Research showing TCN outperforms LSTM on many time series tasks
    - TCAN beats DeepAR, LogSparse Transformer, N-BEATS on benchmarks
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
    from tensorflow.keras.models import Model as ModelType
    from tensorflow.keras import callbacks as CallbacksType
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


class ResidualBlock(layers.Layer):
    """Residual block for TCN with dilated causal convolutions.

    Implements a residual block with:
    - Dilated causal convolution (no future leakage)
    - Weight normalization (optional)
    - ReLU activation
    - Dropout for regularization
    - Skip connection for gradient flow

    Args:
        filters: Number of convolutional filters
        kernel_size: Size of convolutional kernel (3-7 typical)
        dilation_rate: Dilation rate for temporal receptive field
        dropout: Dropout rate (0.1-0.3)
        use_weight_norm: Whether to use weight normalization
    """

    def __init__(
        self,
        filters: int,
        kernel_size: int = 3,
        dilation_rate: int = 1,
        dropout: float = 0.2,
        use_weight_norm: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.dropout_rate = dropout
        self.use_weight_norm = use_weight_norm

        # Layers will be built in build() method
        self.conv1 = None
        self.conv2 = None
        self.dropout1 = None
        self.dropout2 = None
        self.downsample = None

    def build(self, input_shape):
        """Build the residual block layers."""
        # First causal convolution
        self.conv1 = layers.Conv1D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            padding="causal",
            dilation_rate=self.dilation_rate,
            activation="relu",
            name=f"conv1_dilation_{self.dilation_rate}",
        )

        # Second causal convolution
        self.conv2 = layers.Conv1D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            padding="causal",
            dilation_rate=self.dilation_rate,
            activation="relu",
            name=f"conv2_dilation_{self.dilation_rate}",
        )

        # Dropout layers
        self.dropout1 = layers.Dropout(self.dropout_rate)
        self.dropout2 = layers.Dropout(self.dropout_rate)

        # Downsample residual connection if input/output channels differ
        if input_shape[-1] != self.filters:
            self.downsample = layers.Conv1D(
                filters=self.filters, kernel_size=1, padding="same", name="residual_downsample"
            )
        else:
            self.downsample = None

        super().build(input_shape)

    def call(self, inputs, training=None):
        """Apply residual block with dilated convolutions.

        Args:
            inputs: Input tensor of shape (batch, timesteps, features)
            training: Whether in training mode (for dropout)

        Returns:
            Output tensor with residual connection
        """
        # First conv block
        x = self.conv1(inputs)
        x = self.dropout1(x, training=training)

        # Second conv block
        x = self.conv2(x)
        x = self.dropout2(x, training=training)

        # Residual connection
        if self.downsample is not None:
            residual = self.downsample(inputs)
        else:
            residual = inputs

        # Add residual and apply activation
        output = layers.Add()([x, residual])
        output = layers.Activation("relu")(output)

        return output

    def get_config(self):
        """Get layer configuration for serialization."""
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "dilation_rate": self.dilation_rate,
                "dropout": self.dropout_rate,
                "use_weight_norm": self.use_weight_norm,
            }
        )
        return config


def create_tcn_model(
    input_shape: tuple[int, int],
    num_filters: int = 64,
    kernel_size: int = 5,
    num_layers: int = 5,
    dilation_base: int = 2,
    dropout: float = 0.2,
    dense_units: int = 64,
    learning_rate: float = 1e-3,
) -> Any:
    """Create Temporal Convolutional Network for cryptocurrency price prediction.

    Architecture:
        - Stacked residual blocks with exponentially increasing dilation
        - Dilated causal convolutions (dilation: 1, 2, 4, 8, 16, ...)
        - Skip connections for gradient flow
        - Global pooling and dense layers for prediction

    Receptive Field Calculation:
        receptive_field = 1 + (kernel_size - 1) * sum(dilation_rates)
        Example: kernel=5, layers=5, dilation=[1,2,4,8,16]
        receptive_field = 1 + (5-1) * (1+2+4+8+16) = 1 + 4*31 = 125 timesteps

    Args:
        input_shape: Shape of input sequences (sequence_length, num_features)
        num_filters: Number of filters per conv layer (32-128 typical)
        kernel_size: Convolution kernel size (3-7 typical)
        num_layers: Number of residual blocks (4-6 typical)
        dilation_base: Base for exponential dilation (typically 2)
        dropout: Dropout rate (0.1-0.3)
        dense_units: Units in dense layer before output
        learning_rate: Adam optimizer learning rate

    Returns:
        Compiled Keras model

    Example:
        >>> model = create_tcn_model((60, 15), num_filters=64, num_layers=5)
        >>> model.fit(train_ds, validation_data=val_ds, epochs=50)
    """
    _ensure_tensorflow_available()

    inputs = layers.Input(shape=input_shape, name="sequence_input")

    # Initial convolution to expand channels
    x = layers.Conv1D(
        filters=num_filters, kernel_size=1, padding="same", activation="relu", name="input_conv"
    )(inputs)

    # Stack residual blocks with exponentially increasing dilation
    # This creates a large receptive field efficiently
    for i in range(num_layers):
        dilation_rate = dilation_base**i  # 1, 2, 4, 8, 16, ...
        x = ResidualBlock(
            filters=num_filters,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            dropout=dropout,
            name=f"residual_block_{i+1}",
        )(x)

    # Global pooling to aggregate temporal information
    # Use both average and max for richer representation
    avg_pool = layers.GlobalAveragePooling1D(name="global_avg_pool")(x)
    max_pool = layers.GlobalMaxPooling1D(name="global_max_pool")(x)
    pooled = layers.Concatenate(name="pooling_concat")([avg_pool, max_pool])

    # Dense layers for final prediction
    x = layers.Dense(dense_units, activation="relu", name="dense_1")(pooled)
    x = layers.Dropout(dropout, name="dropout_dense")(x)

    # Output layer (linear activation for regression)
    outputs = layers.Dense(1, activation="linear", name="price_prediction")(x)

    # Build and compile model
    model = Model(inputs=inputs, outputs=outputs, name="tcn")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=[
            tf.keras.metrics.RootMeanSquaredError(name="rmse"),
            tf.keras.metrics.MeanAbsoluteError(name="mae"),
        ],
    )

    return model


def create_lightweight_tcn(input_shape: tuple[int, int]) -> Any:
    """Create lightweight TCN for faster training/inference.

    Suitable for:
    - Resource-constrained environments (Railway)
    - Real-time inference
    - Fast experimentation

    Args:
        input_shape: Shape of input sequences (sequence_length, num_features)

    Returns:
        Compiled Keras model
    """
    return create_tcn_model(
        input_shape=input_shape,
        num_filters=32,  # Fewer filters
        kernel_size=3,  # Smaller kernel
        num_layers=4,  # Fewer layers
        dropout=0.2,
        dense_units=32,
        learning_rate=1e-3,
    )


def create_deep_tcn(input_shape: tuple[int, int]) -> Any:
    """Create deep TCN for maximum performance.

    Suitable for:
    - Large training datasets
    - Maximum accuracy
    - When computational resources available

    Args:
        input_shape: Shape of input sequences (sequence_length, num_features)

    Returns:
        Compiled Keras model
    """
    return create_tcn_model(
        input_shape=input_shape,
        num_filters=128,  # More filters
        kernel_size=7,  # Larger kernel
        num_layers=6,  # More layers
        dropout=0.3,  # More dropout
        dense_units=128,
        learning_rate=1e-3,
    )


def create_tcn_with_attention(
    input_shape: tuple[int, int],
    num_filters: int = 64,
    kernel_size: int = 5,
    num_layers: int = 5,
    num_attention_heads: int = 4,
    dropout: float = 0.2,
) -> Any:
    """Create TCN with multi-head attention for enhanced performance.

    Combines strengths of TCN (fast, large receptive field) with
    attention (interpretability, feature importance).

    Args:
        input_shape: Shape of input sequences
        num_filters: TCN filters
        kernel_size: TCN kernel size
        num_layers: Number of TCN blocks
        num_attention_heads: Number of attention heads
        dropout: Dropout rate

    Returns:
        Compiled Keras model
    """
    _ensure_tensorflow_available()

    inputs = layers.Input(shape=input_shape, name="sequence_input")

    # Initial convolution
    x = layers.Conv1D(
        filters=num_filters, kernel_size=1, padding="same", activation="relu", name="input_conv"
    )(inputs)

    # TCN blocks
    for i in range(num_layers):
        dilation_rate = 2**i
        x = ResidualBlock(
            filters=num_filters,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            dropout=dropout,
            name=f"residual_block_{i+1}",
        )(x)

    # Multi-head attention (from models_attention_lstm.py)
    x = layers.MultiHeadAttention(num_heads=num_attention_heads, key_dim=64, dropout=dropout)(
        query=x, value=x, key=x
    )

    # Global pooling
    avg_pool = layers.GlobalAveragePooling1D()(x)
    max_pool = layers.GlobalMaxPooling1D()(x)
    pooled = layers.Concatenate()([avg_pool, max_pool])

    # Dense layers
    x = layers.Dense(64, activation="relu")(pooled)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(1, activation="linear")(x)

    model = Model(inputs=inputs, outputs=outputs, name="tcn_attention")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=[
            tf.keras.metrics.RootMeanSquaredError(name="rmse"),
            tf.keras.metrics.MeanAbsoluteError(name="mae"),
        ],
    )

    return model


def tcn_callbacks(patience: int = 15, min_delta: float = 1e-4) -> list[Any]:
    """Get optimized callbacks for TCN training.

    TCN typically trains faster than LSTM, so can use lower patience.

    Args:
        patience: Epochs to wait for improvement
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
            monitor="val_loss", factor=0.5, patience=max(patience // 3, 3), min_lr=1e-6, verbose=1
        ),
    ]


def calculate_receptive_field(kernel_size: int, num_layers: int, dilation_base: int = 2) -> int:
    """Calculate receptive field of TCN.

    Receptive field determines how far back in time the model can see.
    Larger receptive field captures longer-term dependencies.

    Args:
        kernel_size: Convolution kernel size
        num_layers: Number of residual blocks
        dilation_base: Base for exponential dilation

    Returns:
        Receptive field in timesteps

    Example:
        >>> calculate_receptive_field(kernel_size=5, num_layers=5)
        125  # Can see 125 timesteps back
    """
    # Sum of geometric series: sum(base^i for i in range(n)) = (base^n - 1) / (base - 1)
    if dilation_base == 1:
        total_dilation = num_layers
    else:
        total_dilation = (dilation_base**num_layers - 1) // (dilation_base - 1)

    receptive_field = 1 + (kernel_size - 1) * total_dilation
    return receptive_field


# Recommended hyperparameters based on research and experimentation
RECOMMENDED_HYPERPARAMETERS = {
    "default": {
        "num_filters": 64,
        "kernel_size": 5,
        "num_layers": 5,
        "dilation_base": 2,
        "dropout": 0.2,
        "dense_units": 64,
        "learning_rate": 1e-3,
        "batch_size": 32,
        "epochs": 80,
        "patience": 15,
        "receptive_field": 125,  # With kernel=5, layers=5
    },
    "lightweight": {
        "num_filters": 32,
        "kernel_size": 3,
        "num_layers": 4,
        "dilation_base": 2,
        "dropout": 0.2,
        "dense_units": 32,
        "learning_rate": 1e-3,
        "batch_size": 64,
        "epochs": 60,
        "patience": 12,
        "receptive_field": 31,  # With kernel=3, layers=4
    },
    "deep": {
        "num_filters": 128,
        "kernel_size": 7,
        "num_layers": 6,
        "dilation_base": 2,
        "dropout": 0.3,
        "dense_units": 128,
        "learning_rate": 1e-3,
        "batch_size": 16,
        "epochs": 120,
        "patience": 20,
        "receptive_field": 381,  # With kernel=7, layers=6
    },
}
