"""Diagnostics and artifact helpers for training."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict

import matplotlib.pyplot as plt
import numpy as np

try:
    import tensorflow as tf

    _TENSORFLOW_AVAILABLE = True
except ImportError:
    _TENSORFLOW_AVAILABLE = False
    tf = None  # type: ignore

if TYPE_CHECKING:
    from tensorflow.keras.models import Model as ModelType
else:
    ModelType = Any  # type: ignore

logger = logging.getLogger(__name__)

# Plotting and performance metric constants
PLOT_SAMPLE_SIZE = 100  # Number of predictions to show in training plot (balances detail vs readability)
MIN_MAPE_DENOMINATOR = 1e-8  # Minimum denominator for MAPE calculation (prevents division by near-zero values)
MAX_PERCENTAGE_ERROR_CAP = 1000.0  # Cap MAPE errors at 1000% to prevent outliers from dominating metrics


class PerformanceMetrics(TypedDict):
    """Performance metrics with MSE and RMSE."""

    mse: float
    rmse: float


class SentimentPerformanceMetrics(TypedDict):
    """Sentiment ablation metrics with degradation percentage."""

    mse: float
    rmse: float
    degradation_pct: float


class RobustnessValidationResult(TypedDict, total=False):
    """Result of model robustness validation.

    Attributes:
        base_performance: Baseline metrics with all features
        no_sentiment_performance: Optional metrics without sentiment features
    """

    base_performance: PerformanceMetrics
    no_sentiment_performance: SentimentPerformanceMetrics


@dataclass
class ArtifactPaths:
    directory: Path
    keras_path: Path
    onnx_path: Path | None
    metadata_path: Path
    plot_path: Path | None


def create_training_plots(
    history: Any,
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    symbol: str,
    model_type: str,
    output_dir: Path,
    enable_plots: bool,
) -> Path | None:
    """Create training plots."""
    if not _TENSORFLOW_AVAILABLE:
        raise ImportError(
            "tensorflow is required for creating training plots but is not installed. "
            "Install it with: pip install tensorflow"
        )
    if not enable_plots:
        return None
    try:
        plt.figure(figsize=(15, 10))
        plt.subplot(2, 2, 1)
        plt.plot(history.history["loss"], label="Train Loss")
        plt.plot(history.history["val_loss"], label="Validation Loss")
        plt.title(f"Model Loss - {symbol} ({model_type})")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)

        if "rmse" in history.history:
            plt.subplot(2, 2, 2)
            plt.plot(history.history["rmse"], label="Train RMSE")
            plt.plot(history.history["val_rmse"], label="Validation RMSE")
            plt.title(f"Model RMSE - {symbol} ({model_type})")
            plt.xlabel("Epochs")
            plt.ylabel("RMSE")
            plt.legend()
            plt.grid(True, alpha=0.3)

        plt.subplot(2, 2, 4)
        test_predictions = model.predict(X_test, verbose=0)
        sample = min(PLOT_SAMPLE_SIZE, len(y_test))
        plt.plot(y_test[:sample], label="Actual", alpha=0.8, linewidth=2)
        plt.plot(test_predictions[:sample].flatten(), label="Predicted", alpha=0.8, linewidth=2)
        plt.title("Prediction Sample (Test Set)")
        plt.xlabel("Time Steps")
        plt.ylabel("Normalized Price")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_path = output_dir / f"{symbol}_{model_type}_training.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        return plot_path
    except Exception:  # noqa: BLE001 - Catch all matplotlib/display errors
        # Plot generation is diagnostic only - training should continue if plotting fails
        # (e.g., missing display, matplotlib backend issues, file write errors)
        logger.warning("Failed to generate training plots", exc_info=True)
        return None


def validate_model_robustness(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    has_sentiment: bool,
) -> RobustnessValidationResult:
    """Validate model robustness."""
    if not _TENSORFLOW_AVAILABLE:
        raise ImportError(
            "tensorflow is required for model robustness validation but is not installed. "
            "Install it with: pip install tensorflow"
        )
    # Validate input tensor shape
    if len(X_test.shape) != 3:
        raise ValueError(f"Expected 3D tensor (batch, sequence, features), got shape {X_test.shape}")

    results = {"base_performance": {}}
    base_pred = model.predict(X_test)
    base_mse = np.mean((base_pred.flatten() - y_test) ** 2)
    results["base_performance"] = {"mse": float(base_mse), "rmse": float(np.sqrt(base_mse))}

    if has_sentiment:
        sentiment_indices = [i for i, name in enumerate(feature_names) if "sentiment" in name]
        if sentiment_indices:
            X_no_sentiment = X_test.copy()
            # Validate sentiment feature indices are within bounds
            if not all(i < X_test.shape[2] for i in sentiment_indices):
                raise ValueError(
                    f"Sentiment feature indices {sentiment_indices} exceed feature dimension {X_test.shape[2]}"
                )
            X_no_sentiment[:, :, sentiment_indices] = 0
            no_sentiment_pred = model.predict(X_no_sentiment)
            no_sentiment_mse = np.mean((no_sentiment_pred.flatten() - y_test) ** 2)
            results["no_sentiment_performance"] = {
                "mse": float(no_sentiment_mse),
                "rmse": float(np.sqrt(no_sentiment_mse)),
                "degradation_pct": float(((no_sentiment_mse - base_mse) / base_mse) * 100),
            }
    return results


def evaluate_model_performance(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    close_scaler: Any | None = None,
) -> dict[str, float]:
    """Evaluate model performance."""
    if not _TENSORFLOW_AVAILABLE:
        raise ImportError(
            "tensorflow is required for model performance evaluation but is not installed. "
            "Install it with: pip install tensorflow"
        )
    train_loss, train_rmse = model.evaluate(X_train, y_train, verbose=0)
    test_loss, test_rmse = model.evaluate(X_test, y_test, verbose=0)
    test_predictions = model.predict(X_test, verbose=0)

    if close_scaler is not None:
        y_test_denorm = close_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        pred_denorm = close_scaler.inverse_transform(
            test_predictions.flatten().reshape(-1, 1)
        ).flatten()
        # Guard against division by near-zero values in MAPE calculation
        denominator = np.maximum(np.abs(y_test_denorm), MIN_MAPE_DENOMINATOR)
        percentage_errors = np.abs((y_test_denorm - pred_denorm) / denominator) * 100
        # Cap individual errors to prevent extreme outliers from dominating MAPE
        percentage_errors = np.minimum(percentage_errors, MAX_PERCENTAGE_ERROR_CAP)
    else:
        # Guard against division by near-zero values in MAPE calculation
        denominator = np.maximum(np.abs(y_test), MIN_MAPE_DENOMINATOR)
        percentage_errors = np.abs((y_test - test_predictions.flatten()) / denominator) * 100
        # Cap individual errors to prevent extreme outliers from dominating MAPE
        percentage_errors = np.minimum(percentage_errors, MAX_PERCENTAGE_ERROR_CAP)
    mape = np.mean(percentage_errors)

    return {
        "train_loss": float(train_loss),
        "test_loss": float(test_loss),
        "train_rmse": float(train_rmse),
        "test_rmse": float(test_rmse),
        "mape": float(mape),
    }


def convert_to_onnx(model: Any, output_path: Path) -> Path | None:
    """Convert Keras model to ONNX format.

    Args:
        model: Keras model to convert
        output_path: Path where ONNX model will be saved

    Returns:
        Path to ONNX file if successful, None otherwise
    """
    if not _TENSORFLOW_AVAILABLE:
        raise ImportError(
            "tensorflow is required for ONNX conversion but is not installed. "
            "Install it with: pip install tensorflow"
        )
    tmp_dir = None
    try:
        tmp_dir = tempfile.mkdtemp()
        saved_model_path = Path(tmp_dir) / "saved_model"
        model.export(saved_model_path)
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "tf2onnx.convert",
                "--saved-model",
                str(saved_model_path),
                "--output",
                str(output_path),
                "--opset",
                "13",
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode == 0:
            return output_path
        logger.warning("ONNX conversion failed: %s", result.stderr)
        return None
    except subprocess.TimeoutExpired as exc:
        # ONNX conversion timeout - training should continue with Keras model
        logger.warning("ONNX conversion timed out after 300 seconds")
        if exc.stdout:
            logger.debug("Partial stdout: %s", exc.stdout[:500])
        if exc.stderr:
            logger.debug("Partial stderr: %s", exc.stderr[:500])
        return None
    except Exception as exc:  # noqa: BLE001 - Catch all ONNX conversion errors
        # ONNX export is optional - training should continue with Keras model if conversion fails
        # (e.g., missing tf2onnx, unsupported ops, file system errors)
        logger.warning("ONNX conversion failed: %s", exc)
        return None
    finally:
        # Always clean up temporary directory
        if tmp_dir and Path(tmp_dir).exists():
            try:
                shutil.rmtree(tmp_dir)
            except Exception as exc:
                logger.warning("Failed to clean up temporary ONNX directory: %s", exc)


def save_artifacts(
    model: Any,
    symbol: str,
    model_type: str,
    registry_root: Path,
    metadata: dict,
    version_id: str,
    enable_onnx: bool,
) -> ArtifactPaths:
    """Save training artifacts."""
    if not _TENSORFLOW_AVAILABLE:
        raise ImportError(
            "tensorflow is required for saving artifacts but is not installed. "
            "Install it with: pip install tensorflow"
        )
    symbol_dir = registry_root / symbol.upper()
    type_dir = symbol_dir / model_type
    version_dir = type_dir / version_id
    version_dir.mkdir(parents=True, exist_ok=True)

    keras_path = version_dir / "model.keras"
    model.save(keras_path)

    onnx_path: Path | None = None
    if enable_onnx:
        candidate = version_dir / "model.onnx"
        onnx_path = convert_to_onnx(model, candidate)

    metadata_path = version_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)

    # Atomic symlink update to avoid race conditions (TOCTOU vulnerability)
    # Create temporary symlink, then atomically replace the old one
    latest_link = type_dir / "latest"
    temp_link = type_dir / f".latest.{version_dir.name}.tmp"

    try:
        # Clean up any stale temp symlink
        if temp_link.exists() or temp_link.is_symlink():
            temp_link.unlink()

        # Create new symlink with temporary name
        temp_link.symlink_to(version_dir.name)

        # Atomically replace old symlink (rename is atomic on POSIX systems)
        temp_link.replace(latest_link)

    except OSError as e:
        # Clean up temp symlink on failure
        if temp_link.exists() or temp_link.is_symlink():
            try:
                temp_link.unlink()
            except OSError:
                # Ignore cleanup errors - primary error already logged
                pass
        logger.error(f"Failed to update 'latest' symlink: {e}")
        raise RuntimeError(f"Failed to update 'latest' symlink at {latest_link}") from e

    return ArtifactPaths(
        directory=version_dir,
        keras_path=keras_path,
        onnx_path=onnx_path,
        metadata_path=metadata_path,
        plot_path=None,
    )
