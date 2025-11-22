#!/usr/bin/env python3
"""Quick validation script for new ML architectures.

Tests that all implemented models can be created, compiled, and trained
on synthetic data without errors.

Usage:
    python scripts/validate_models.py
"""

import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import tensorflow as tf
    print("✓ TensorFlow available:", tf.__version__)
except ImportError:
    print("✗ TensorFlow not available - install with: pip install tensorflow")
    sys.exit(1)

from src.ml.training_pipeline.models import create_model, AVAILABLE_MODELS, MODEL_VARIANTS


def generate_synthetic_data(num_samples=100, sequence_length=60, num_features=15):
    """Generate synthetic data for testing."""
    X = np.random.randn(num_samples, sequence_length, num_features).astype(np.float32)
    y = np.random.randn(num_samples, 1).astype(np.float32)
    return X, y


def test_model(model_type, variant="default"):
    """Test a single model configuration."""
    print(f"\n{'='*60}")
    print(f"Testing {model_type} ({variant})")
    print(f"{'='*60}")

    try:
        # Generate synthetic data
        X_train, y_train = generate_synthetic_data(num_samples=100)
        X_val, y_val = generate_synthetic_data(num_samples=20)

        # Create model
        print(f"  Creating model...")
        model = create_model(model_type, input_shape=(60, 15), variant=variant)

        # Check model
        print(f"  Model input shape: {model.input_shape}")
        print(f"  Model output shape: {model.output_shape}")
        print(f"  Total parameters: {model.count_params():,}")

        # Train for 1 epoch
        print(f"  Training for 1 epoch...")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=1,
            verbose=0
        )

        # Inference
        print(f"  Testing inference...")
        predictions = model.predict(X_val[:5], verbose=0)

        print(f"  ✓ Model works correctly!")
        print(f"    - Loss: {history.history['loss'][0]:.4f}")
        print(f"    - Val Loss: {history.history['val_loss'][0]:.4f}")
        print(f"    - Predictions shape: {predictions.shape}")

        return True

    except Exception as e:
        print(f"  ✗ Model failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run validation for all models."""
    print("\n" + "="*60)
    print("ML MODEL ARCHITECTURE VALIDATION")
    print("="*60)
    print("\nTesting all implemented model architectures...")

    results = {}

    # Test each model type
    for model_type in ["cnn_lstm", "attention_lstm", "tcn", "tcn_attention", "lstm"]:
        # Test default variant
        success = test_model(model_type, "default")
        results[f"{model_type}_default"] = success

        # Test variants for architectures that support them
        if model_type in ["attention_lstm", "tcn"]:
            # Test lightweight
            success = test_model(model_type, "lightweight")
            results[f"{model_type}_lightweight"] = success

            # Test deep
            success = test_model(model_type, "deep")
            results[f"{model_type}_deep"] = success

    # Summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)

    total = len(results)
    passed = sum(results.values())
    failed = total - passed

    print(f"\nTotal tests: {total}")
    print(f"Passed: {passed} ✓")
    print(f"Failed: {failed} ✗")

    if failed > 0:
        print("\nFailed tests:")
        for name, success in results.items():
            if not success:
                print(f"  - {name}")

    print("\n" + "="*60)

    if failed == 0:
        print("✓ ALL MODELS VALIDATED SUCCESSFULLY!")
        print("\nNext steps:")
        print("  1. Train models on real data:")
        print("     atb train model BTCUSDT --model-type attention_lstm --days 30 --epochs 20")
        print("     atb train model BTCUSDT --model-type tcn --days 30 --epochs 20")
        print("  2. Run comprehensive benchmarks:")
        print("     pytest tests/benchmark/test_model_architectures.py -v")
        return 0
    else:
        print("✗ SOME MODELS FAILED VALIDATION")
        print("\nPlease fix the errors above before proceeding.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
