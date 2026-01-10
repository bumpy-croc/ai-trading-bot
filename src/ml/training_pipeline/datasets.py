"""Dataset preparation utilities for training."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

try:
    import tensorflow as tf

    _TENSORFLOW_AVAILABLE = True
except ImportError:
    _TENSORFLOW_AVAILABLE = False
    tf = None  # type: ignore

if TYPE_CHECKING:
    from tensorflow.data import Dataset as DatasetType
else:
    DatasetType = Any  # type: ignore

# Training dataset constants
DEFAULT_SHUFFLE_BUFFER_SIZE = (
    2048  # Buffer size for shuffle operation (balances memory vs randomness)
)


def create_sequences(
    feature_data: np.ndarray, target_data: np.ndarray, sequence_length: int
) -> tuple[np.ndarray, np.ndarray]:
    """Create sequences from feature and target data using sliding windows.

    Args:
        feature_data: 2D array of features (rows=timesteps, cols=features)
        target_data: 1D array of targets aligned with feature_data
        sequence_length: Number of timesteps in each sequence (must be positive)

    Returns:
        Tuple of (sequences, targets) as float32 arrays

    Raises:
        ValueError: If inputs are invalid (mismatched lengths, wrong shapes, non-positive sequence_length)
    """
    # Validate sequence_length is positive
    if sequence_length <= 0:
        raise ValueError(f"sequence_length must be positive, got {sequence_length}")

    # Validate feature_data is 2D
    if feature_data.ndim != 2:
        raise ValueError(
            f"feature_data must be 2D (timesteps, features), got shape {feature_data.shape}"
        )

    # Validate target_data is 1D
    if target_data.ndim != 1:
        raise ValueError(f"target_data must be 1D, got shape {target_data.shape}")

    # Validate matching lengths
    if len(feature_data) != len(target_data):
        raise ValueError(
            f"feature_data and target_data must have same length, "
            f"got {len(feature_data)} and {len(target_data)}"
        )

    # Validate sufficient data for sequences
    if len(feature_data) <= sequence_length:
        raise ValueError(
            f"Insufficient data: need at least {sequence_length + 1} rows, got {len(feature_data)}"
        )

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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split sequences into training and validation sets.

    Args:
        sequences: 3D array of sequences (samples, timesteps, features)
        targets: 1D array of target values
        split_ratio: Ratio of data to use for training (default 0.8, range: 0 < ratio < 1)

    Returns:
        Tuple of (X_train, y_train, X_val, y_val)

    Raises:
        ValueError: If arrays are empty or split_ratio is out of range
    """
    # Validate inputs
    if len(sequences) == 0 or len(targets) == 0:
        raise ValueError("Cannot split empty sequences or targets")

    if not 0 < split_ratio < 1:
        raise ValueError(f"split_ratio must be between 0 and 1, got {split_ratio}")

    if len(sequences) != len(targets):
        raise ValueError(
            f"sequences and targets must have same length, got {len(sequences)} and {len(targets)}"
        )

    # Ensure at least 1 sample in training set (if possible)
    if len(sequences) == 1:
        split_index = 1  # Put the single sample in training
    else:
        split_index = max(int(len(sequences) * split_ratio), 1)
        split_index = min(split_index, len(sequences) - 1)
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
) -> tuple[Any, Any]:
    """Build TensorFlow datasets for training and validation.

    Args:
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        batch_size: Batch size for training (must be positive)

    Returns:
        Tuple of (train_dataset, validation_dataset)

    Raises:
        ImportError: If TensorFlow is not installed
        ValueError: If batch_size is invalid
    """
    if not _TENSORFLOW_AVAILABLE:
        raise ImportError(
            "tensorflow is required for dataset building but is not installed. "
            "Install it with: pip install tensorflow"
        )

    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_ds = (
        train_ds.cache()
        .shuffle(min(len(X_train), DEFAULT_SHUFFLE_BUFFER_SIZE))
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return train_ds, val_ds
