"""Position management module for dynamic risk and position sizing."""

import logging

from .dynamic_risk import DynamicRiskConfig, DynamicRiskManager

logger = logging.getLogger(__name__)

# Re-export new MFE/MAE modules if present
try:
    from .mfe_mae_tracker import MFEMAETracker
except ImportError as e:
    logger.debug(f"MFEMAETracker unavailable: {e}")
    MFEMAETracker = None  # type: ignore[assignment] - Optional module, set to None if unavailable
except Exception as e:
    logger.warning(f"Unexpected error importing MFEMAETracker: {e}")
    MFEMAETracker = None  # type: ignore[assignment] - Optional module, set to None if unavailable

try:
    from .mfe_mae_analyzer import MFEMAEAnalyzer
except ImportError as e:
    logger.debug(f"MFEMAEAnalyzer unavailable: {e}")
    MFEMAEAnalyzer = None  # type: ignore[assignment] - Optional module, set to None if unavailable
except Exception as e:
    logger.warning(f"Unexpected error importing MFEMAEAnalyzer: {e}")
    MFEMAEAnalyzer = None  # type: ignore[assignment] - Optional module, set to None if unavailable

__all__ = ["DynamicRiskConfig", "DynamicRiskManager", "MFEMAETracker", "MFEMAEAnalyzer"]
