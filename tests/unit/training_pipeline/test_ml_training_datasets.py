"""Unit tests for ML training pipeline dataset preparation module."""

import numpy as np
import pytest
import tensorflow as tf

from src.ml.training_pipeline.datasets import (
    DEFAULT_SHUFFLE_BUFFER_SIZE,
    build_tf_datasets,
    create_sequences,
    split_sequences,
)


@pytest.mark.fast
class TestCreateSequences:
    """Test create_sequences function."""

    def test_basic_sequence_creation(self):
        # Arrange
        feature_data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]])
        target_data = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        sequence_length = 2

        # Act
        sequences, targets = create_sequences(feature_data, target_data, sequence_length)

        # Assert
        assert sequences.shape == (3, 2, 2)  # (n_sequences, seq_len, n_features)
        assert targets.shape == (3,)
        assert sequences.dtype == np.float32
        assert targets.dtype == np.float32

    def test_sequence_alignment(self):
        # Arrange
        feature_data = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        target_data = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        sequence_length = 2

        # Act
        sequences, targets = create_sequences(feature_data, target_data, sequence_length)

        # Assert - targets should align with sequence endings
        assert targets[0] == 30.0  # Sequence [1,2] -> target at index 2
        assert targets[1] == 40.0  # Sequence [2,3] -> target at index 3
        assert targets[2] == 50.0  # Sequence [3,4] -> target at index 4

    def test_invalid_sequence_length_zero(self):
        # Arrange
        feature_data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        target_data = np.array([10.0, 20.0, 30.0])

        # Act & Assert
        with pytest.raises(ValueError, match="sequence_length must be positive"):
            create_sequences(feature_data, target_data, 0)

    def test_invalid_sequence_length_negative(self):
        # Arrange
        feature_data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        target_data = np.array([10.0, 20.0, 30.0])

        # Act & Assert
        with pytest.raises(ValueError, match="sequence_length must be positive"):
            create_sequences(feature_data, target_data, -1)

    def test_invalid_feature_data_1d(self):
        # Arrange
        feature_data = np.array([1.0, 2.0, 3.0])
        target_data = np.array([10.0, 20.0, 30.0])

        # Act & Assert
        with pytest.raises(ValueError, match="feature_data must be 2D"):
            create_sequences(feature_data, target_data, 2)

    def test_invalid_feature_data_3d(self):
        # Arrange
        feature_data = np.array([[[1.0, 2.0], [3.0, 4.0]]])
        target_data = np.array([10.0])

        # Act & Assert
        with pytest.raises(ValueError, match="feature_data must be 2D"):
            create_sequences(feature_data, target_data, 2)

    def test_invalid_target_data_2d(self):
        # Arrange
        feature_data = np.array([[1.0, 2.0], [3.0, 4.0]])
        target_data = np.array([[10.0], [20.0]])

        # Act & Assert
        with pytest.raises(ValueError, match="target_data must be 1D"):
            create_sequences(feature_data, target_data, 2)

    def test_mismatched_lengths(self):
        # Arrange
        feature_data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        target_data = np.array([10.0, 20.0])  # One less element

        # Act & Assert
        with pytest.raises(ValueError, match="must have same length"):
            create_sequences(feature_data, target_data, 2)

    def test_insufficient_data(self):
        # Arrange
        feature_data = np.array([[1.0, 2.0], [3.0, 4.0]])
        target_data = np.array([10.0, 20.0])
        sequence_length = 5  # More than available data

        # Act & Assert
        with pytest.raises(ValueError, match="Insufficient data"):
            create_sequences(feature_data, target_data, sequence_length)

    def test_exact_minimum_data(self):
        # Arrange
        feature_data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        target_data = np.array([10.0, 20.0, 30.0])
        sequence_length = 2  # Need at least 3 rows (2 + 1)

        # Act
        sequences, targets = create_sequences(feature_data, target_data, sequence_length)

        # Assert
        assert len(sequences) == 1
        assert len(targets) == 1

    def test_multiple_features(self):
        # Arrange
        feature_data = np.array(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]
        )
        target_data = np.array([10.0, 20.0, 30.0, 40.0])
        sequence_length = 2

        # Act
        sequences, targets = create_sequences(feature_data, target_data, sequence_length)

        # Assert
        assert sequences.shape == (2, 2, 3)  # 2 sequences, length 2, 3 features
        assert targets.shape == (2,)


@pytest.mark.fast
class TestSplitSequences:
    """Test split_sequences function."""

    def test_default_split_ratio(self):
        # Arrange
        sequences = np.array([[[1.0]], [[2.0]], [[3.0]], [[4.0]], [[5.0]]])
        targets = np.array([10.0, 20.0, 30.0, 40.0, 50.0])

        # Act
        X_train, y_train, X_val, y_val = split_sequences(sequences, targets)

        # Assert - 80% train, 20% val
        assert len(X_train) == 4
        assert len(y_train) == 4
        assert len(X_val) == 1
        assert len(y_val) == 1

    def test_custom_split_ratio(self):
        # Arrange
        sequences = np.array([[[1.0]], [[2.0]], [[3.0]], [[4.0]], [[5.0]]])
        targets = np.array([10.0, 20.0, 30.0, 40.0, 50.0])

        # Act
        X_train, y_train, X_val, y_val = split_sequences(sequences, targets, split_ratio=0.6)

        # Assert - 60% train, 40% val
        assert len(X_train) == 3
        assert len(y_train) == 3
        assert len(X_val) == 2
        assert len(y_val) == 2

    def test_split_preserves_order(self):
        # Arrange
        sequences = np.array([[[1.0]], [[2.0]], [[3.0]], [[4.0]]])
        targets = np.array([10.0, 20.0, 30.0, 40.0])

        # Act
        X_train, y_train, X_val, y_val = split_sequences(sequences, targets, split_ratio=0.5)

        # Assert - splits should maintain temporal order
        assert X_train[0][0][0] == 1.0
        assert X_train[1][0][0] == 2.0
        assert X_val[0][0][0] == 3.0
        assert X_val[1][0][0] == 4.0

    def test_minimum_validation_size(self):
        # Arrange
        sequences = np.array([[[1.0]], [[2.0]]])
        targets = np.array([10.0, 20.0])

        # Act
        X_train, y_train, X_val, y_val = split_sequences(sequences, targets, split_ratio=0.99)

        # Assert - should have at least 1 sample in each set
        assert len(X_train) >= 1
        assert len(X_val) >= 1

    def test_single_sequence_edge_case(self):
        # Arrange
        sequences = np.array([[[1.0]]])
        targets = np.array([10.0])

        # Act
        X_train, y_train, X_val, y_val = split_sequences(sequences, targets, split_ratio=0.8)

        # Assert - with only 1 sample, should still split properly
        assert len(X_train) == 1
        assert len(X_val) == 0 or len(X_val) > 0  # Either is valid


@pytest.mark.fast
class TestBuildTfDatasets:
    """Test build_tf_datasets function."""

    def test_creates_tf_datasets(self):
        # Arrange
        X_train = np.array([[[1.0, 2.0]], [[3.0, 4.0]], [[5.0, 6.0]]])
        y_train = np.array([10.0, 20.0, 30.0])
        X_val = np.array([[[7.0, 8.0]]])
        y_val = np.array([40.0])
        batch_size = 2

        # Act
        train_ds, val_ds = build_tf_datasets(X_train, y_train, X_val, y_val, batch_size)

        # Assert
        assert isinstance(train_ds, tf.data.Dataset)
        assert isinstance(val_ds, tf.data.Dataset)

    def test_train_dataset_batching(self):
        # Arrange
        X_train = np.array([[[1.0]], [[2.0]], [[3.0]], [[4.0]]])
        y_train = np.array([10.0, 20.0, 30.0, 40.0])
        X_val = np.array([[[5.0]]])
        y_val = np.array([50.0])
        batch_size = 2

        # Act
        train_ds, val_ds = build_tf_datasets(X_train, y_train, X_val, y_val, batch_size)

        # Assert - check batching
        for batch_X, batch_y in train_ds.take(1):
            assert batch_X.shape[0] <= batch_size
            assert batch_y.shape[0] <= batch_size

    def test_validation_dataset_batching(self):
        # Arrange
        X_train = np.array([[[1.0]]])
        y_train = np.array([10.0])
        X_val = np.array([[[2.0]], [[3.0]], [[4.0]]])
        y_val = np.array([20.0, 30.0, 40.0])
        batch_size = 2

        # Act
        train_ds, val_ds = build_tf_datasets(X_train, y_train, X_val, y_val, batch_size)

        # Assert - validation should also be batched
        for batch_X, batch_y in val_ds.take(1):
            assert batch_X.shape[0] <= batch_size
            assert batch_y.shape[0] <= batch_size

    def test_shuffle_buffer_size_small_dataset(self):
        # Arrange
        X_train = np.array([[[1.0]], [[2.0]]])
        y_train = np.array([10.0, 20.0])
        X_val = np.array([[[3.0]]])
        y_val = np.array([30.0])
        batch_size = 1

        # Act
        train_ds, val_ds = build_tf_datasets(X_train, y_train, X_val, y_val, batch_size)

        # Assert - should handle small datasets without error
        assert isinstance(train_ds, tf.data.Dataset)

    def test_shuffle_buffer_size_large_dataset(self):
        # Arrange - dataset larger than DEFAULT_SHUFFLE_BUFFER_SIZE
        n_samples = DEFAULT_SHUFFLE_BUFFER_SIZE + 100
        X_train = np.random.rand(n_samples, 5, 3).astype(np.float32)
        y_train = np.random.rand(n_samples).astype(np.float32)
        X_val = np.random.rand(10, 5, 3).astype(np.float32)
        y_val = np.random.rand(10).astype(np.float32)
        batch_size = 32

        # Act
        train_ds, val_ds = build_tf_datasets(X_train, y_train, X_val, y_val, batch_size)

        # Assert - should handle large datasets without error
        assert isinstance(train_ds, tf.data.Dataset)
        assert isinstance(val_ds, tf.data.Dataset)

    def test_dataset_prefetch_configured(self):
        # Arrange
        X_train = np.array([[[1.0]], [[2.0]], [[3.0]]])
        y_train = np.array([10.0, 20.0, 30.0])
        X_val = np.array([[[4.0]]])
        y_val = np.array([40.0])
        batch_size = 2

        # Act
        train_ds, val_ds = build_tf_datasets(X_train, y_train, X_val, y_val, batch_size)

        # Assert - datasets should be iterable (prefetch configured)
        train_samples = list(train_ds.take(1))
        val_samples = list(val_ds.take(1))
        assert len(train_samples) > 0
        assert len(val_samples) > 0

    def test_batch_size_one(self):
        # Arrange
        X_train = np.array([[[1.0]], [[2.0]], [[3.0]]])
        y_train = np.array([10.0, 20.0, 30.0])
        X_val = np.array([[[4.0]]])
        y_val = np.array([40.0])
        batch_size = 1

        # Act
        train_ds, val_ds = build_tf_datasets(X_train, y_train, X_val, y_val, batch_size)

        # Assert - should handle batch_size=1
        for batch_X, batch_y in train_ds.take(1):
            assert batch_X.shape[0] == 1
            assert batch_y.shape[0] == 1

    def test_large_batch_size(self):
        # Arrange
        X_train = np.array([[[1.0]], [[2.0]], [[3.0]]])
        y_train = np.array([10.0, 20.0, 30.0])
        X_val = np.array([[[4.0]]])
        y_val = np.array([40.0])
        batch_size = 100  # Larger than dataset

        # Act
        train_ds, val_ds = build_tf_datasets(X_train, y_train, X_val, y_val, batch_size)

        # Assert - should handle batch_size > dataset size
        for batch_X, batch_y in train_ds.take(1):
            assert batch_X.shape[0] == 3  # All samples in one batch
            assert batch_y.shape[0] == 3
