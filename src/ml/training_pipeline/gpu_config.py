"""GPU configuration utilities for TensorFlow training on Mac and other platforms."""

from __future__ import annotations

import logging
import platform

try:
    import tensorflow as tf

    _TENSORFLOW_AVAILABLE = True
except ImportError:
    _TENSORFLOW_AVAILABLE = False
    tf = None  # type: ignore

# Check for tensorflow-metal plugin (required for Apple Silicon GPU support)
try:
    import tensorflow_metal  # noqa: F401

    _TENSORFLOW_METAL_AVAILABLE = True
except ImportError:
    _TENSORFLOW_METAL_AVAILABLE = False

logger = logging.getLogger(__name__)


def configure_gpu() -> str | None:
    """Configure TensorFlow to use available GPU devices.

    On Apple Silicon Macs, configures Metal Performance Shaders (MPS) backend.
    On other platforms, detects and configures NVIDIA/AMD GPUs.

    Returns:
        Device name string (e.g., "GPU:0", "mps:0") if GPU is available, None otherwise

    Note:
        Must be called before any TensorFlow operations to ensure device placement.
    """
    if not _TENSORFLOW_AVAILABLE:
        logger.warning("TensorFlow not available - GPU configuration skipped")
        return None

    # Check for Apple Silicon Mac and configure MPS
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        if not _TENSORFLOW_METAL_AVAILABLE:
            logger.warning(
                "⚠️  tensorflow-metal plugin not installed. Install it for GPU acceleration:"
            )
            logger.warning("   pip install tensorflow-metal")
            logger.info("   Training will continue with CPU (slower)")
            return None
        try:
            # MPS devices appear as "GPU" in TensorFlow when tensorflow-metal is installed
            mps_devices = tf.config.list_physical_devices("GPU")
            if mps_devices:
                logger.info("✅ Apple Silicon GPU detected (Metal Performance Shaders)")
                logger.info(f"   Using device: {mps_devices[0].name}")
                # TensorFlow automatically uses MPS when available, no explicit config needed
                return mps_devices[0].name
            else:
                logger.info("ℹ️  No Apple Silicon GPU detected, using CPU")
                return None
        except (RuntimeError, ValueError, ImportError, AttributeError) as exc:
            logger.warning(f"Failed to configure Apple Silicon GPU: {exc}")
            logger.info("Falling back to CPU")
            return None

    # Check for standard GPU devices (NVIDIA/AMD)
    try:
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            logger.info(f"✅ GPU detected: {len(gpus)} device(s)")
            for gpu in gpus:
                logger.info(f"   Device: {gpu.name}")
                # Enable memory growth to avoid allocating all GPU memory at once
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                except (RuntimeError, ValueError, AttributeError) as exc:
                    logger.warning(f"Failed to set memory growth for {gpu.name}: {exc}")
            return gpus[0].name
        else:
            logger.info("ℹ️  No GPU detected, using CPU")
            return None
    except (RuntimeError, ValueError, ImportError, AttributeError) as exc:
        logger.warning(f"Failed to detect GPU devices: {exc}")
        logger.info("Falling back to CPU")
        return None


def get_compute_device() -> str:
    """Get the compute device currently in use by TensorFlow.

    Returns:
        Device name string (e.g., "GPU:0", "CPU:0", "mps:0")
    """
    if not _TENSORFLOW_AVAILABLE:
        return "CPU"

    try:
        # Check if GPU is available
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            return gpus[0].name
        return "CPU"
    except (RuntimeError, ValueError, ImportError, AttributeError):
        return "CPU"
