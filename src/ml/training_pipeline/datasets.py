"""Dataset preparation utilities for training."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import tensorflow as tf
from numpy.lib.stride_tricks import sliding_window_view


def create_sequences(feature_data: np.ndarray, target_data: np.ndarray, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
    if len(feature_data) <= sequence_length:
        raise ValueError("Insufficient rows for the requested sequence length")

    window_shape = (sequence_length, feature_data.shape[1])
    sequences = sliding_window_view(feature_data, window_shape)
    sequences = sequences[:, 0, :, :]
    usable = len(feature_data) - sequence_length
    sequences = sequences[:usable]
    targets = target_data[sequence_length:]

    return sequences.astype(np.float32), targets.astype(np.float32)


def split_sequences(
    sequences: np.ndarray,
    targets: np.ndarray,
    split_ratio: float = 0.8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    split_index = max(int(len(sequences) * split_ratio), 1)
    X_train = sequences[:split_index]
    y_train = targets[:split_index]
    X_val = sequences[split_index:]
    y_val = targets[split_index:]
    return X_train, y_train, X_val, y_val


def build_tf_datasets(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int,
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_ds = train_ds.cache().shuffle(min(len(X_train), 2048)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return train_ds, val_ds
