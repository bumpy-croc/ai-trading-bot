"""Model factory for the training pipeline.

Optimized architecture with:
- Efficient Conv1D + GRU with batch normalization
- Good speed/accuracy tradeoff
- Batch normalization for faster convergence
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

try:
    import tensorflow as tf
    from tensorflow.keras import callbacks
    from tensorflow.keras.layers import (
        BatchNormalization,
        Conv1D,
        Dense,
        Dropout,
        GRU,
        Input,
        LayerNormalization,
        MaxPooling1D,
    )
    from tensorflow.keras.models import Model

    _TENSORFLOW_AVAILABLE = True
except ImportError:
    _TENSORFLOW_AVAILABLE = False
    # Create placeholder types for type checking
    tf = None  # type: ignore
    callbacks = None  # type: ignore
    Conv1D = None  # type: ignore
    Dense = None  # type: ignore
    Dropout = None  # type: ignore
    Input = None  # type: ignore
    GRU = None  # type: ignore
    MaxPooling1D = None  # type: ignore
    Model = None  # type: ignore
    BatchNormalization = None  # type: ignore
    LayerNormalization = None  # type: ignore

if TYPE_CHECKING:
    # For type checking, assume tensorflow is available
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


def create_model(input_shape, has_sentiment: bool = True) -> Any:
    """Create an optimized model with good speed/accuracy tradeoff.

    Architecture: Efficient Conv1D + GRU with batch normalization
    Optimized for fast training while maintaining good accuracy.

    Args:
        input_shape: Tuple of (sequence_length, num_features)
        has_sentiment: Whether sentiment features are included (unused, for compatibility)

    Returns:
        Compiled Keras model
    """
    _ensure_tensorflow_available()

    inputs = Input(shape=input_shape)

    # Two convolutional blocks with batch norm
    x = Conv1D(filters=48, kernel_size=3, activation=None, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = MaxPooling1D(pool_size=2)(x)

    x = Conv1D(filters=96, kernel_size=3, activation=None, padding="same")(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = MaxPooling1D(pool_size=2)(x)

    # GRU layers with layer normalization
    x = GRU(80, return_sequences=True)(x)
    x = LayerNormalization()(x)
    x = Dropout(0.2)(x)
    x = GRU(40, return_sequences=False)(x)
    x = Dropout(0.2)(x)

    # Output layers
    x = Dense(40, activation="relu")(x)
    x = Dropout(0.2)(x)
    outputs = Dense(1, activation="linear")(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmse")],
    )
    return model




def default_callbacks(patience: int = 15, reduce_lr_patience: int | None = None) -> list[Any]:
    """Create default training callbacks with optimized parameters.

    Args:
        patience: Epochs to wait before early stopping
        reduce_lr_patience: Epochs to wait before reducing LR (default: patience // 3)

    Returns:
        List of Keras callbacks
    """
    _ensure_tensorflow_available()

    if reduce_lr_patience is None:
        reduce_lr_patience = max(patience // 3, 3)

    return [
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=reduce_lr_patience,
            min_lr=1e-6,
            verbose=1,
        ),
    ]


# Legacy function for backwards compatibility
def build_price_only_model(sequence_length: int, num_features: int) -> Any:
    """Build a price-only model (legacy compatibility function).

    Args:
        sequence_length: Length of input sequences
        num_features: Number of features per timestep

    Returns:
        Compiled Keras model
    """
    return create_model((sequence_length, num_features), has_sentiment=False)
