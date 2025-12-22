import numpy as np

from src.ml.training_pipeline.datasets import build_tf_datasets, create_sequences, split_sequences


def test_create_sequences_matches_manual_loop():
    feature_data = np.arange(50, dtype=np.float32).reshape(10, 5)
    target_data = np.linspace(0.1, 1.0, num=10, dtype=np.float32)
    sequences, targets = create_sequences(feature_data, target_data, sequence_length=3)

    manual_sequences = []
    manual_targets = []
    for idx in range(3, len(feature_data)):
        manual_sequences.append(feature_data[idx - 3 : idx])
        manual_targets.append(target_data[idx])

    manual_sequences = np.stack(manual_sequences)
    manual_targets = np.array(manual_targets)

    assert sequences.shape == manual_sequences.shape
    assert targets.shape == manual_targets.shape
    np.testing.assert_allclose(sequences, manual_sequences)
    np.testing.assert_allclose(targets, manual_targets)


def test_build_tf_datasets_batches_correctly():
    features = np.random.random((20, 5, 3)).astype(np.float32)
    targets = np.random.random(20).astype(np.float32)
    X_train, y_train, X_val, y_val = split_sequences(features, targets)
    train_ds, val_ds = build_tf_datasets(X_train, y_train, X_val, y_val, batch_size=4)

    train_batches = list(train_ds.as_numpy_iterator())
    val_batches = list(val_ds.as_numpy_iterator())

    assert train_batches, "train dataset should yield batches"
    assert val_batches, "validation dataset should yield batches"
    first_batch = train_batches[0]
    assert first_batch[0].shape[0] <= 4
    assert first_batch[0].shape[1:] == (features.shape[1], features.shape[2])
