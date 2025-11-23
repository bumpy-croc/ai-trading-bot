"""Optimized dataset preparation utilities for training.

Performance optimizations:
- Use float32 instead of float64 for all arrays
- Optimized sliding window creation
- Better TensorFlow dataset configuration
- Cached preprocessing where possible
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Tuple

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

# Optimized dataset constants
DEFAULT_SHUFFLE_BUFFER_SIZE = 2048
PREFETCH_BUFFER_SIZE = tf.data.AUTOTUNE if _TENSORFLOW_AVAILABLE else None


def create_sequences(
    feature_data: np.ndarray, target_data: np.ndarray, sequence_length: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Create sequences from feature and target data using optimized sliding windows.

    Optimizations:
    - Ensure float32 dtype throughout
    - Use numpy's efficient sliding_window_view
    - Pre-allocate output arrays where beneficial

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

    # Ensure float32 dtype for memory efficiency
    if feature_data.dtype != np.float32:
        feature_data = feature_data.astype(np.float32)
    if target_data.dtype != np.float32:
        target_data = target_data.astype(np.float32)

    # Create sliding windows (efficient view-based operation)
    window_shape = (sequence_length, feature_data.shape[1])
    sequences = sliding_window_view(feature_data, window_shape)
    sequences = sequences[:, 0, :, :]  # Remove extra dimension

    # Calculate number of usable sequences
    usable = len(feature_data) - sequence_length
    sequences = sequences[:usable]
    targets = target_data[sequence_length:]

    # Ensure contiguous arrays for better performance
    if not sequences.flags['C_CONTIGUOUS']:
        sequences = np.ascontiguousarray(sequences, dtype=np.float32)
    if not targets.flags['C_CONTIGUOUS']:
        targets = np.ascontiguousarray(targets, dtype=np.float32)

    return sequences, targets


def split_sequences(
    sequences: np.ndarray,
    targets: np.ndarray,
    split_ratio: float = 0.8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split sequences into training and validation sets.

    Args:
        sequences: 3D array of sequences (samples, timesteps, features)
        targets: 1D array of target values
        split_ratio: Ratio of data to use for training (default 0.8)

    Returns:
        Tuple of (X_train, y_train, X_val, y_val)
    """
    # Ensure at least 1 sample in each split
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
    cache: bool = True,
) -> Tuple[Any, Any]:
    """Build optimized TensorFlow datasets for training and validation.

    Optimizations:
    - Cache preprocessed data in memory
    - Use AUTOTUNE for dynamic prefetch tuning
    - Optimize shuffle buffer size based on dataset size
    - Use parallel map operations where applicable

    Args:
        X_train: Training features (3D array)
        y_train: Training targets (1D array)
        X_val: Validation features (3D array)
        y_val: Validation targets (1D array)
        batch_size: Batch size for training
        cache: Whether to cache the dataset in memory (default True)

    Returns:
        Tuple of (train_dataset, validation_dataset)
    """
    if not _TENSORFLOW_AVAILABLE:
        raise ImportError(
            "tensorflow is required for dataset building but is not installed. "
            "Install it with: pip install tensorflow"
        )

    # Ensure float32 dtype
    X_train = X_train.astype(np.float32)
    y_train = y_train.astype(np.float32)
    X_val = X_val.astype(np.float32)
    y_val = y_val.astype(np.float32)

    # Build training dataset with optimizations
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))

    # Cache before shuffling for better performance
    if cache:
        train_ds = train_ds.cache()

    # Optimize shuffle buffer size (use smaller of dataset size or default)
    shuffle_buffer = min(len(X_train), DEFAULT_SHUFFLE_BUFFER_SIZE)
    train_ds = train_ds.shuffle(shuffle_buffer, reshuffle_each_iteration=True)

    # Batch and prefetch
    train_ds = train_ds.batch(batch_size, drop_remainder=False)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

    # Build validation dataset (no shuffling needed)
    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    if cache:
        val_ds = val_ds.cache()
    val_ds = val_ds.batch(batch_size, drop_remainder=False)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds


def create_datasets_fast(
    feature_array: np.ndarray,
    target_array: np.ndarray,
    sequence_length: int,
    batch_size: int,
    split_ratio: float = 0.8,
) -> Tuple[Any, Any, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create TensorFlow datasets from raw features and targets (convenience function).

    This combines sequence creation, splitting, and TF dataset building into one call.

    Args:
        feature_array: 2D array of features
        target_array: 1D array of targets
        sequence_length: Length of sequences to create
        batch_size: Batch size for training
        split_ratio: Train/validation split ratio

    Returns:
        Tuple of (train_ds, val_ds, X_train, y_train, X_val, y_val)
    """
    # Create sequences
    sequences, targets = create_sequences(feature_array, target_array, sequence_length)

    # Split into train/validation
    X_train, y_train, X_val, y_val = split_sequences(sequences, targets, split_ratio)

    # Build TensorFlow datasets
    train_ds, val_ds = build_tf_datasets(X_train, y_train, X_val, y_val, batch_size)

    return train_ds, val_ds, X_train, y_train, X_val, y_val
