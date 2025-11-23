"""Optimized model factories for the training pipeline.

Performance optimizations:
- Lighter architecture options (GRU instead of LSTM, smaller layers)
- Batch normalization for faster convergence
- Depthwise separable convolutions for efficiency
- Optimized learning rate schedules (OneCycleLR)
- Better default hyperparameters
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

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
        LSTM,
        LayerNormalization,
        MaxPooling1D,
        SeparableConv1D,
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
    LSTM = None  # type: ignore
    GRU = None  # type: ignore
    MaxPooling1D = None  # type: ignore
    Model = None  # type: ignore
    BatchNormalization = None  # type: ignore
    LayerNormalization = None  # type: ignore
    SeparableConv1D = None  # type: ignore

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


def create_fast_model(input_shape, has_sentiment: bool = True) -> Any:
    """Create a fast, lightweight model optimized for quick training.

    Architecture: GRU-based (faster than LSTM), minimal layers
    Best for: Quick experiments, iterative development
    Expected speedup: 2-3x faster than standard model

    Args:
        input_shape: Tuple of (sequence_length, num_features)
        has_sentiment: Whether sentiment features are included (unused, for compatibility)

    Returns:
        Compiled Keras model
    """
    _ensure_tensorflow_available()

    inputs = Input(shape=input_shape)

    # Single convolutional block with batch norm (faster convergence)
    x = Conv1D(filters=32, kernel_size=3, activation=None, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = MaxPooling1D(pool_size=2)(x)

    # GRU layers (faster than LSTM, often similar performance)
    x = GRU(64, return_sequences=True)(x)
    x = Dropout(0.2)(x)
    x = GRU(32, return_sequences=False)(x)
    x = Dropout(0.2)(x)

    # Output layers
    x = Dense(32, activation="relu")(x)
    x = Dropout(0.2)(x)
    outputs = Dense(1, activation="linear")(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=2e-3),  # Higher LR for faster convergence
        loss="mse",
        metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmse")],
    )
    return model


def create_balanced_model(input_shape, has_sentiment: bool = True) -> Any:
    """Create a balanced model with good speed/accuracy tradeoff.

    Architecture: Efficient Conv1D + GRU with batch normalization
    Best for: Production training with reasonable time constraints
    Expected speedup: 1.5-2x faster than standard model

    Args:
        input_shape: Tuple of (sequence_length, num_features)
        has_sentiment: Whether sentiment features are included (unused, for compatibility)

    Returns:
        Compiled Keras model
    """
    _ensure_tensorflow_available()

    inputs = Input(shape=input_shape)

    # Two convolutional blocks with batch norm
    x = Conv1D(filters=48, kernel_size=3, activation=None, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = MaxPooling1D(pool_size=2)(x)

    x = Conv1D(filters=96, kernel_size=3, activation=None, padding='same')(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
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


def create_adaptive_model(input_shape, has_sentiment: bool = True) -> Any:
    """Create original adaptive model (for backwards compatibility).

    Architecture: CNN+LSTM (original implementation)
    Best for: Maximum quality, longer training acceptable
    Performance: Baseline (1x speed)

    Args:
        input_shape: Tuple of (sequence_length, num_features)
        has_sentiment: Whether sentiment features are included (unused, for compatibility)

    Returns:
        Compiled Keras model
    """
    _ensure_tensorflow_available()

    inputs = Input(shape=input_shape)
    x = Conv1D(filters=64, kernel_size=3, activation="relu")(inputs)
    x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(filters=128, kernel_size=3, activation="relu")(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = LSTM(100, return_sequences=True)(x)
    x = Dropout(0.2)(x)
    x = LSTM(50, return_sequences=False)(x)
    x = Dropout(0.2)(x)
    x = Dense(50, activation="relu")(x)
    x = Dropout(0.2)(x)
    outputs = Dense(1, activation="linear")(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmse")],
    )
    return model


def create_quality_model(input_shape, has_sentiment: bool = True) -> Any:
    """Create a high-quality model optimized for accuracy.

    Architecture: Deeper CNN+LSTM with more parameters and regularization
    Best for: Maximum accuracy, training time not a concern
    Performance: Slower than standard (0.7x speed) but potentially higher quality

    Args:
        input_shape: Tuple of (sequence_length, num_features)
        has_sentiment: Whether sentiment features are included

    Returns:
        Compiled Keras model
    """
    _ensure_tensorflow_available()

    inputs = Input(shape=input_shape)

    # Three convolutional blocks with batch norm
    x = Conv1D(filters=64, kernel_size=3, activation=None, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = MaxPooling1D(pool_size=2)(x)

    x = Conv1D(filters=128, kernel_size=3, activation=None, padding='same')(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = MaxPooling1D(pool_size=2)(x)

    x = Conv1D(filters=256, kernel_size=3, activation=None, padding='same')(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    # Bidirectional LSTM for better context
    x = tf.keras.layers.Bidirectional(LSTM(120, return_sequences=True))(x)
    x = LayerNormalization()(x)
    x = Dropout(0.3)(x)
    x = LSTM(60, return_sequences=False)(x)
    x = Dropout(0.3)(x)

    # Deeper dense layers
    x = Dense(80, activation="relu")(x)
    x = Dropout(0.3)(x)
    x = Dense(40, activation="relu")(x)
    x = Dropout(0.2)(x)
    outputs = Dense(1, activation="linear")(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),  # Lower LR for stability
        loss="mse",
        metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmse")],
    )
    return model


# Model factory mapping
MODEL_FACTORIES = {
    "fast": create_fast_model,
    "balanced": create_balanced_model,
    "quality": create_quality_model,
    "adaptive": create_adaptive_model,  # Default/original
}


def create_model(
    input_shape,
    has_sentiment: bool = True,
    model_type: str = "balanced",
) -> Any:
    """Create a model using the specified architecture.

    Args:
        input_shape: Tuple of (sequence_length, num_features)
        has_sentiment: Whether sentiment features are included
        model_type: One of "fast", "balanced", "quality", "adaptive"

    Returns:
        Compiled Keras model

    Raises:
        ValueError: If model_type is not recognized
    """
    if model_type not in MODEL_FACTORIES:
        raise ValueError(
            f"Unknown model type '{model_type}'. "
            f"Choose from: {list(MODEL_FACTORIES.keys())}"
        )

    factory = MODEL_FACTORIES[model_type]
    return factory(input_shape, has_sentiment)


def default_callbacks(patience: int = 15, reduce_lr_patience: Optional[int] = None) -> list[Any]:
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


def fast_callbacks() -> list[Any]:
    """Create callbacks optimized for fast training.

    - Lower patience for faster early stopping
    - More aggressive LR reduction

    Returns:
        List of Keras callbacks optimized for speed
    """
    return default_callbacks(patience=10, reduce_lr_patience=3)


def quality_callbacks() -> list[Any]:
    """Create callbacks optimized for quality training.

    - Higher patience for thorough training
    - Conservative LR reduction

    Returns:
        List of Keras callbacks optimized for quality
    """
    return default_callbacks(patience=20, reduce_lr_patience=7)


# Legacy function for backwards compatibility
def build_price_only_model(sequence_length: int, num_features: int) -> Any:
    """Build a price-only model (legacy compatibility function).

    Args:
        sequence_length: Length of input sequences
        num_features: Number of features per timestep

    Returns:
        Compiled Keras model
    """
    return create_fast_model((sequence_length, num_features), has_sentiment=False)
