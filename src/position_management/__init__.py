"""Position management module for dynamic risk and position sizing."""

import logging
from importlib import import_module
from typing import Any

from .dynamic_risk import DynamicRiskConfig, DynamicRiskManager

logger = logging.getLogger(__name__)


def _optional_import(module_name: str, class_name: str) -> Any | None:
    """Import a module and class, returning None if unavailable.

    This helper consolidates duplicate exception handling for optional imports
    and ensures only ImportError (the relevant exception for imports) is caught.
    """
    try:
        module = import_module(f".{module_name}", package="src.position_management")
        return getattr(module, class_name)
    except ImportError:
        logger.debug(f"{class_name} unavailable (expected in minimal environments)")
        return None


# Re-export new MFE/MAE modules if present
MFEMAETracker = _optional_import("mfe_mae_tracker", "MFEMAETracker")
MFEMAEAnalyzer = _optional_import("mfe_mae_analyzer", "MFEMAEAnalyzer")

# Conditionally build __all__ to only export available modules
__all__ = ["DynamicRiskConfig", "DynamicRiskManager"]
if MFEMAETracker is not None:
    __all__.append("MFEMAETracker")
if MFEMAEAnalyzer is not None:
    __all__.append("MFEMAEAnalyzer")
