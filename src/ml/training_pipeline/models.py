"""Model factories for the training pipeline.

This module provides factory functions to create different model architectures:
- CNN-LSTM (original, baseline)
- Attention-LSTM (12-15% improvement expected)
- TCN (fast training, competitive accuracy)
- LightGBM (gradient boosting baseline)

Usage:
    >>> from src.ml.training_pipeline.models import create_model
    >>> model = create_model('attention_lstm', input_shape=(60, 15))
    >>> model.fit(train_ds, validation_data=val_ds, epochs=50)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

try:
    import tensorflow as tf
    from tensorflow.keras import callbacks
    from tensorflow.keras.layers import Conv1D, Dense, Dropout, Input, LSTM, MaxPooling1D
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
    MaxPooling1D = None  # type: ignore
    Model = None  # type: ignore

if TYPE_CHECKING:
    # For type checking, assume tensorflow is available
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


def create_adaptive_model(input_shape, has_sentiment: bool = True) -> Any:
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


def build_price_only_model(sequence_length: int, num_features: int) -> Any:
    _ensure_tensorflow_available()
    inputs = tf.keras.layers.Input(shape=(sequence_length, num_features))
    x = tf.keras.layers.LSTM(128, return_sequences=True)(inputs)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.LSTM(64)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmse")],
    )
    return model


def default_callbacks(patience: int = 15) -> list[Any]:
    _ensure_tensorflow_available()
    return [
        callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=max(patience // 3, 3), min_lr=1e-5, verbose=1
        ),
    ]


def create_model(
    model_type: str,
    input_shape: tuple[int, int],
    variant: str = "default",
    **kwargs: Any,
) -> Any:
    """Factory function to create different model architectures.

    Provides a unified interface for creating models of different types:
    - 'cnn_lstm' or 'adaptive': Original CNN-LSTM hybrid (current baseline)
    - 'attention_lstm': LSTM with multi-head attention (12-15% improvement expected)
    - 'tcn': Temporal Convolutional Network (fast training)
    - 'tcn_attention': TCN with attention mechanism

    Args:
        model_type: Type of model to create
        input_shape: Shape of input sequences (sequence_length, num_features)
        variant: Model variant ('default', 'lightweight', 'deep')
        **kwargs: Additional model-specific parameters

    Returns:
        Compiled Keras model

    Example:
        >>> # Create Attention-LSTM model
        >>> model = create_model('attention_lstm', (60, 15))
        >>>
        >>> # Create lightweight TCN
        >>> model = create_model('tcn', (60, 15), variant='lightweight')
        >>>
        >>> # Create deep Attention-LSTM with custom parameters
        >>> model = create_model('attention_lstm', (60, 15),
        ...                      lstm_units=[256, 128], dropout=0.3)

    Raises:
        ValueError: If model_type is not recognized
    """
    model_type_lower = model_type.lower()

    # CNN-LSTM (original/baseline)
    if model_type_lower in ["cnn_lstm", "adaptive", "default"]:
        has_sentiment = kwargs.get("has_sentiment", True)
        return create_adaptive_model(input_shape, has_sentiment)

    # Attention-LSTM
    elif model_type_lower == "attention_lstm":
        try:
            from src.ml.training_pipeline.models_attention_lstm import (
                create_attention_lstm_model,
                create_deep_attention_lstm,
                create_lightweight_attention_lstm,
            )
        except ImportError:
            raise ImportError(
                "Attention-LSTM model requires models_attention_lstm module. "
                "Ensure the file exists in src/ml/training_pipeline/"
            )

        if variant == "lightweight":
            return create_lightweight_attention_lstm(input_shape)
        elif variant == "deep":
            return create_deep_attention_lstm(input_shape)
        else:
            return create_attention_lstm_model(input_shape, **kwargs)

    # Temporal Convolutional Network
    elif model_type_lower == "tcn":
        try:
            from src.ml.training_pipeline.models_tcn import (
                create_deep_tcn,
                create_lightweight_tcn,
                create_tcn_model,
            )
        except ImportError:
            raise ImportError(
                "TCN model requires models_tcn module. "
                "Ensure the file exists in src/ml/training_pipeline/"
            )

        if variant == "lightweight":
            return create_lightweight_tcn(input_shape)
        elif variant == "deep":
            return create_deep_tcn(input_shape)
        else:
            return create_tcn_model(input_shape, **kwargs)

    # TCN with Attention
    elif model_type_lower == "tcn_attention":
        try:
            from src.ml.training_pipeline.models_tcn import create_tcn_with_attention
        except ImportError:
            raise ImportError(
                "TCN with attention requires models_tcn module. "
                "Ensure the file exists in src/ml/training_pipeline/"
            )

        return create_tcn_with_attention(input_shape, **kwargs)

    # Price-only LSTM (simple baseline)
    elif model_type_lower == "lstm":
        sequence_length, num_features = input_shape
        return build_price_only_model(sequence_length, num_features)

    else:
        raise ValueError(
            f"Unknown model_type: {model_type}. "
            f"Supported types: 'cnn_lstm', 'attention_lstm', 'tcn', 'tcn_attention', 'lstm'"
        )


def get_model_callbacks(model_type: str, patience: int = 15) -> list[Any]:
    """Get optimized callbacks for specific model type.

    Different models may benefit from different callback configurations.

    Args:
        model_type: Type of model
        patience: Epochs to wait for improvement

    Returns:
        List of Keras callbacks

    Example:
        >>> callbacks = get_model_callbacks('attention_lstm', patience=20)
        >>> model.fit(train_ds, validation_data=val_ds, callbacks=callbacks)
    """
    model_type_lower = model_type.lower()

    if model_type_lower == "attention_lstm":
        try:
            from src.ml.training_pipeline.models_attention_lstm import attention_lstm_callbacks

            return attention_lstm_callbacks(patience=patience)
        except ImportError:
            pass

    elif model_type_lower in ["tcn", "tcn_attention"]:
        try:
            from src.ml.training_pipeline.models_tcn import tcn_callbacks

            return tcn_callbacks(patience=patience)
        except ImportError:
            pass

    # Default callbacks for all models
    return default_callbacks(patience=patience)


# Model registry for easy reference
AVAILABLE_MODELS = {
    "cnn_lstm": "CNN-LSTM hybrid (original baseline)",
    "attention_lstm": "LSTM with multi-head attention (12-15% improvement expected)",
    "tcn": "Temporal Convolutional Network (fast training, competitive accuracy)",
    "tcn_attention": "TCN with multi-head attention",
    "lstm": "Simple LSTM baseline",
}

# Model variants
MODEL_VARIANTS = {
    "default": "Standard configuration, balanced performance/speed",
    "lightweight": "Fewer parameters, faster training/inference",
    "deep": "More parameters, potentially better accuracy on large datasets",
}
