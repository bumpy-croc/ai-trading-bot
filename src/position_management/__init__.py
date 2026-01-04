"""Position management module for dynamic risk and position sizing."""

import logging

from .dynamic_risk import DynamicRiskConfig, DynamicRiskManager

logger = logging.getLogger(__name__)

# Re-export new MFE/MAE modules if present
try:
    from .mfe_mae_tracker import MFEMAETracker
except ImportError as e:
    logger.debug(f"MFEMAETracker unavailable (expected in minimal environments): {e}")
    MFEMAETracker = None  # type: ignore[assignment] - Optional module, set to None if unavailable
except (SyntaxError, AttributeError) as e:
    # Re-raise critical errors that indicate code bugs
    logger.error(f"Critical error importing MFEMAETracker: {e}")
    raise
except Exception as e:
    # Only catch truly unexpected errors
    logger.error(f"Unexpected error importing MFEMAETracker: {e}", exc_info=True)
    MFEMAETracker = None  # type: ignore[assignment] - Optional module, set to None if unavailable

try:
    from .mfe_mae_analyzer import MFEMAEAnalyzer
except ImportError as e:
    logger.debug(f"MFEMAEAnalyzer unavailable (expected in minimal environments): {e}")
    MFEMAEAnalyzer = None  # type: ignore[assignment] - Optional module, set to None if unavailable
except (SyntaxError, AttributeError) as e:
    # Re-raise critical errors that indicate code bugs
    logger.error(f"Critical error importing MFEMAEAnalyzer: {e}")
    raise
except Exception as e:
    # Only catch truly unexpected errors
    logger.error(f"Unexpected error importing MFEMAEAnalyzer: {e}", exc_info=True)
    MFEMAEAnalyzer = None  # type: ignore[assignment] - Optional module, set to None if unavailable

# Conditionally build __all__ to only export available modules
__all__ = ["DynamicRiskConfig", "DynamicRiskManager"]
if MFEMAETracker is not None:
    __all__.append("MFEMAETracker")
if MFEMAEAnalyzer is not None:
    __all__.append("MFEMAEAnalyzer")
