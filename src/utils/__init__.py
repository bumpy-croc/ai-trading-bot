"""Shared utility functions for the trading system.

This module provides common utilities used across the codebase:
- bounds: Value clamping and validation functions
- price_targets: Stop loss and take profit price calculations
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
from src.utils.price_targets import (
    PriceTargetCalculator,
    PriceTargets,
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
    # Price targets
    "PriceTargetCalculator",
    "PriceTargets",
]
