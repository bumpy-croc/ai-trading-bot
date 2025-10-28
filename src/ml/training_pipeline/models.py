"""Model factories for the training pipeline."""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import callbacks
from tensorflow.keras.layers import Conv1D, Dense, Dropout, Input, LSTM, MaxPooling1D
from tensorflow.keras.models import Model


def create_adaptive_model(input_shape, has_sentiment: bool = True) -> tf.keras.Model:
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


def build_price_only_model(sequence_length: int, num_features: int) -> tf.keras.Model:
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


def default_callbacks(patience: int = 15) -> list[callbacks.Callback]:
    return [
        callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=max(patience // 3, 3), min_lr=1e-5, verbose=1
        ),
    ]
