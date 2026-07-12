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
    from tensorflow.keras.layers import (
        GRU,
        BatchNormalization,
        Conv1D,
        Dense,
        Dropout,
        Input,
        LayerNormalization,
        MaxPooling1D,
    )
    from tensorflow.keras.models import Model

    _TENSORFLOW_AVAILABLE = True
except ImportError:
    _TENSORFLOW_AVAILABLE = False
    if not TYPE_CHECKING:
        tf = None
        callbacks = None
        Conv1D = None
        Dense = None
        Dropout = None
        Input = None
        GRU = None
        MaxPooling1D = None
        Model = None
        BatchNormalization = None
        LayerNormalization = None


def _ensure_tensorflow_available() -> None:
    """Ensure tensorflow is available, raising ImportError with helpful message if not."""
    if not _TENSORFLOW_AVAILABLE:
        raise ImportError(
            "tensorflow is required for model training but is not installed. "
            "Install it with: pip install tensorflow"
        )


def create_adaptive_model(input_shape: tuple[int, int], has_sentiment: bool = True) -> Any:
    """Create an optimized model with good speed/accuracy tradeoff.

    Architecture: Efficient Conv1D + GRU with batch normalization
    Optimized for fast training while maintaining good accuracy.

    Args:
        input_shape: Tuple of (sequence_length, num_features)
        has_sentiment: Whether sentiment features are included (unused, for compatibility)

    Returns:
        Compiled Keras model

    Raises:
        ValueError: If input_shape is invalid for model architecture
    """
    _ensure_tensorflow_available()

    # Validate input shape for model architecture requirements
    # Two MaxPooling1D(pool_size=2) layers require sequence_length >= 4
    sequence_length, num_features = input_shape
    if sequence_length < 4:
        raise ValueError(
            f"sequence_length must be >= 4 for model architecture (two pooling layers), "
            f"got {sequence_length}"
        )
    if num_features <= 0:
        raise ValueError(f"num_features must be positive, got {num_features}")

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

    # Avoid reassigning function parameter (CODE.md line 77)
    lr_patience = reduce_lr_patience if reduce_lr_patience is not None else max(patience // 3, 3)

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
            patience=lr_patience,
            min_lr=1e-6,
            verbose=1,
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

    # Only the CNN-LSTM baseline consumes has_sentiment. Pop it here so the other
    # architecture factories — which don't accept it — never receive it via the
    # **kwargs forwarding below (the trainer always passes it; see #928).
    has_sentiment = kwargs.pop("has_sentiment", True)

    # CNN-LSTM (original/baseline) - use optimized adaptive model
    if model_type_lower in ["cnn_lstm", "adaptive", "default"]:
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
            ) from None

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
            ) from None

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
            ) from None

        return create_tcn_with_attention(input_shape, **kwargs)

    # Temporal Fusion Transformer
    elif model_type_lower == "tft":
        try:
            from src.ml.training_pipeline.models_tft import (
                RECOMMENDED_HYPERPARAMETERS as TFT_HYPERPARAMS,
            )
            from src.ml.training_pipeline.models_tft import (
                create_tft_model,
            )
        except ImportError:
            raise ImportError(
                "TFT model requires models_tft module. "
                "Ensure the file exists in src/ml/training_pipeline/"
            ) from None

        if variant != "default" and variant in TFT_HYPERPARAMS:
            # Hyperparameter values are heterogeneous (int and float)
            variant_params: dict[str, Any] = {
                k: v
                for k, v in TFT_HYPERPARAMS[variant].items()
                if k not in ("batch_size", "epochs", "patience")
            }
            variant_params.update(kwargs)
            return create_tft_model(input_shape, **variant_params)
        return create_tft_model(input_shape, **kwargs)

    # Temporal Fusion Transformer, ternary (3-class) head -- entrant (c) of
    # the TARGET-REDESIGN tournament (triple_barrier target). Same backbone
    # as 'tft', softmax(3)/sparse_categorical_crossentropy head instead of
    # sigmoid/BCE. Deliberately reuses TFT's variant hyperparameter presets
    # (batch_size/epochs/patience excluded exactly like the 'tft' branch
    # above) since the two share every layer except the output head.
    elif model_type_lower == "tft_ternary":
        try:
            from src.ml.training_pipeline.models_tft import (
                RECOMMENDED_HYPERPARAMETERS as TFT_HYPERPARAMS,
            )
            from src.ml.training_pipeline.models_tft import (
                create_tft_ternary_model,
            )
        except ImportError:
            raise ImportError(
                "tft_ternary model requires models_tft module. "
                "Ensure the file exists in src/ml/training_pipeline/"
            ) from None

        if variant != "default" and variant in TFT_HYPERPARAMS:
            variant_params = {
                k: v
                for k, v in TFT_HYPERPARAMS[variant].items()
                if k not in ("batch_size", "epochs", "patience")
            }
            variant_params.update(kwargs)
            return create_tft_ternary_model(input_shape, **variant_params)
        return create_tft_ternary_model(input_shape, **kwargs)

    # Price-only LSTM (simple baseline)
    elif model_type_lower == "lstm":
        sequence_length, num_features = input_shape
        return build_price_only_model(sequence_length, num_features)

    # LightGBM directional classifier. Reachable from this factory (per
    # Phase 2b item 3 -- "models_lightgbm.py's classifier isn't reachable
    # from create_model or the CLI"), but lightgbm is NOT an installed or
    # declared dependency in this repo -- _ensure_lightgbm_available() raises
    # ImportError here, same as it always has. input_shape/variant are
    # unused (LightGBM is tabular, not sequence-shaped); ONNX export for
    # this architecture is an explicit follow-up, not built here.
    elif model_type_lower == "lightgbm":
        from src.ml.training_pipeline.models_lightgbm import create_directional_classifier

        return create_directional_classifier(**kwargs)

    else:
        raise ValueError(
            f"Unknown model_type: {model_type}. "
            f"Supported types: 'cnn_lstm', 'attention_lstm', 'tcn', 'tcn_attention', "
            f"'tft', 'tft_ternary', 'lstm', 'lightgbm'"
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

    elif model_type_lower in ("tft", "tft_ternary"):
        try:
            from src.ml.training_pipeline.models_tft import tft_callbacks

            return tft_callbacks(patience=patience)
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
    "tft": "Temporal Fusion Transformer (interpretable, attention-based)",
    "tft_ternary": "TFT with 3-class softmax head (triple-barrier target)",
    "lstm": "Simple LSTM baseline",
    "lightgbm": "LightGBM directional classifier (requires optional lightgbm dependency)",
}

# Model variants
MODEL_VARIANTS = {
    "default": "Standard configuration, balanced performance/speed",
    "lightweight": "Fewer parameters, faster training/inference",
    "deep": "More parameters, potentially better accuracy on large datasets",
}


# Backwards compatibility function
def build_price_only_model(sequence_length: int, num_features: int) -> Any:
    """Provides backwards compatibility for code using the old function signature.

    Args:
        sequence_length: Length of input sequences
        num_features: Number of features per timestep

    Returns:
        Compiled Keras model
    """
    return create_model("cnn_lstm", (sequence_length, num_features), has_sentiment=False)
