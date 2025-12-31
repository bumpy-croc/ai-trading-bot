"""Shared utility functions for the trading system.

This module provides common utilities used across the codebase:
- bounds: Value clamping and validation functions
"""

from src.utils.bounds import (
    clamp,
    clamp_fraction,
    clamp_multiplier,
    clamp_percentage,
    clamp_position_size,
    clamp_positive,
    clamp_risk_amount,
    clamp_stop_loss_pct,
    validate_fraction,
    validate_non_negative,
    validate_positive,
    validate_range,
)

__all__ = [
    # Core clamping
    "clamp",
    "clamp_fraction",
    "clamp_percentage",
    "clamp_positive",
    # Domain-specific clamping
    "clamp_position_size",
    "clamp_stop_loss_pct",
    "clamp_risk_amount",
    "clamp_multiplier",
    # Validation
    "validate_fraction",
    "validate_positive",
    "validate_non_negative",
    "validate_range",
]
