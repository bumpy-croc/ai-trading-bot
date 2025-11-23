"""Training configuration presets for different use cases.

This module provides pre-configured settings for:
- Fast: Quick iterations, experimental development (2-3x faster)
- Balanced: Production training with good speed/quality tradeoff (default, 1.5-2x faster)
- Quality: Maximum accuracy, longer training acceptable (baseline speed)
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from typing import Optional

from src.ml.training_pipeline.config import DiagnosticsOptions, TrainingConfig


@dataclass
class TrainingPreset:
    """Preset configuration for training pipeline."""

    name: str
    description: str
    model_type: str
    epochs: int
    batch_size: int
    sequence_length: int
    mixed_precision: bool
    early_stopping_patience: int
    generate_plots: bool
    evaluate_robustness: bool
    convert_to_onnx: bool
    expected_speedup: str


# Fast preset - optimized for quick iterations
FAST_PRESET = TrainingPreset(
    name="fast",
    description="Quick training for experiments and iteration (2-3x faster)",
    model_type="fast",
    epochs=100,  # Fewer epochs
    batch_size=64,  # Larger batch size for faster training
    sequence_length=60,  # Shorter sequences
    mixed_precision=True,
    early_stopping_patience=10,  # More aggressive early stopping
    generate_plots=False,  # Skip plots for speed
    evaluate_robustness=False,  # Skip robustness checks
    convert_to_onnx=True,  # Still export ONNX for deployment
    expected_speedup="2-3x",
)

# Balanced preset - good speed/quality tradeoff (recommended default)
BALANCED_PRESET = TrainingPreset(
    name="balanced",
    description="Balanced speed and quality for production (1.5-2x faster, recommended)",
    model_type="balanced",
    epochs=200,  # Moderate epochs
    batch_size=48,  # Balanced batch size
    sequence_length=90,  # Moderate sequence length
    mixed_precision=True,
    early_stopping_patience=15,  # Standard early stopping
    generate_plots=False,  # Skip plots by default (can enable with flag)
    evaluate_robustness=False,  # Skip robustness by default
    convert_to_onnx=True,
    expected_speedup="1.5-2x",
)

# Quality preset - maximum accuracy
QUALITY_PRESET = TrainingPreset(
    name="quality",
    description="Maximum quality, longer training time (baseline speed)",
    model_type="quality",
    epochs=300,  # More epochs for thorough training
    batch_size=32,  # Smaller batch size for better generalization
    sequence_length=120,  # Longer sequences for more context
    mixed_precision=True,
    early_stopping_patience=20,  # Patient early stopping
    generate_plots=True,  # Generate diagnostic plots
    evaluate_robustness=True,  # Full robustness evaluation
    convert_to_onnx=True,
    expected_speedup="1x (baseline)",
)

# Legacy preset - original configuration for backwards compatibility
LEGACY_PRESET = TrainingPreset(
    name="legacy",
    description="Original configuration (adaptive model, full diagnostics)",
    model_type="adaptive",
    epochs=300,
    batch_size=32,
    sequence_length=120,
    mixed_precision=True,
    early_stopping_patience=15,
    generate_plots=True,
    evaluate_robustness=True,
    convert_to_onnx=True,
    expected_speedup="1x (original)",
)


# Preset mapping
PRESETS = {
    "fast": FAST_PRESET,
    "balanced": BALANCED_PRESET,
    "quality": QUALITY_PRESET,
    "legacy": LEGACY_PRESET,
}


def get_preset(name: str) -> TrainingPreset:
    """Get a training preset by name.

    Args:
        name: Preset name ("fast", "balanced", "quality", or "legacy")

    Returns:
        TrainingPreset configuration

    Raises:
        ValueError: If preset name is not recognized
    """
    if name not in PRESETS:
        raise ValueError(f"Unknown preset '{name}'. Available presets: {list(PRESETS.keys())}")
    return PRESETS[name]


def create_config_from_preset(
    preset_name: str,
    symbol: str,
    timeframe: str = "1h",
    days: Optional[int] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    force_sentiment: bool = False,
    force_price_only: bool = False,
    **overrides,
) -> TrainingConfig:
    """Create a TrainingConfig from a preset with optional overrides.

    Args:
        preset_name: Name of the preset to use
        symbol: Trading symbol (e.g., "BTCUSDT")
        timeframe: Timeframe for training data (default "1h")
        days: Number of days of historical data (optional, default 365 for quality, 180 for others)
        start_date: Start date for training data (optional)
        end_date: End date for training data (optional, defaults to now)
        force_sentiment: Force sentiment inclusion
        force_price_only: Force price-only training
        **overrides: Additional overrides for TrainingConfig fields

    Returns:
        TrainingConfig configured from preset

    Examples:
        # Use fast preset with defaults
        config = create_config_from_preset("fast", "BTCUSDT")

        # Use balanced preset with custom epochs
        config = create_config_from_preset("balanced", "ETHUSDT", epochs=250)

        # Use quality preset with 2 years of data
        config = create_config_from_preset("quality", "BTCUSDT", days=730)
    """
    preset = get_preset(preset_name)

    # Calculate default date range if not provided
    if end_date is None:
        end_date = datetime.now()

    if start_date is None:
        if days is None:
            # Default days based on preset
            days = 365 if preset_name == "quality" else 180
        start_date = end_date - timedelta(days=days)

    # Create diagnostics from preset
    diagnostics = DiagnosticsOptions(
        generate_plots=overrides.pop("generate_plots", preset.generate_plots),
        evaluate_robustness=overrides.pop("evaluate_robustness", preset.evaluate_robustness),
        convert_to_onnx=overrides.pop("convert_to_onnx", preset.convert_to_onnx),
    )

    # Create config from preset
    config = TrainingConfig(
        symbol=symbol,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
        epochs=overrides.pop("epochs", preset.epochs),
        batch_size=overrides.pop("batch_size", preset.batch_size),
        sequence_length=overrides.pop("sequence_length", preset.sequence_length),
        force_sentiment=force_sentiment,
        force_price_only=force_price_only,
        mixed_precision=overrides.pop("mixed_precision", preset.mixed_precision),
        diagnostics=diagnostics,
    )

    # Apply any additional overrides
    if overrides:
        config = replace(config, **overrides)

    return config


def list_presets() -> str:
    """Get a formatted list of available presets.

    Returns:
        Formatted string describing all presets
    """
    lines = ["Available Training Presets:", ""]
    for name, preset in PRESETS.items():
        lines.append(f"  {name:10s} - {preset.description}")
        lines.append(
            f"             Model: {preset.model_type}, "
            f"Epochs: {preset.epochs}, "
            f"Batch: {preset.batch_size}, "
            f"Speedup: {preset.expected_speedup}"
        )
        lines.append("")
    return "\n".join(lines)
